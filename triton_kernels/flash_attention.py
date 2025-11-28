
import torch
import triton
import triton.language as tl
import math


@triton.jit
def _fwd_kernel(
    Q, K, V, Out,
    L,  # Logsumexp for numerical stability
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_ok,
    Z, H, M, N, K,
    scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """
    FlashAttention-2 Forward Kernel
    
    Args:
        Q: Query tensor [batch, heads, seq_len, head_dim]
        K: Key tensor [batch, heads, seq_len, head_dim]
        V: Value tensor [batch, heads, seq_len, head_dim]
        Out: Output tensor [batch, heads, seq_len, head_dim]
        L: Logsumexp tensor for numerical stability [batch, heads, seq_len]
        stride_*: Strides for each dimension
        Z, H, M, N, K: Batch size, num heads, seq_len (Q), seq_len (K), head_dim
        scale: Attention scale factor (1/sqrt(head_dim))
        BLOCK_M, BLOCK_N, BLOCK_K: Tile sizes
    """
    # Program ID
    pid_m = tl.program_id(0)
    pid_z = tl.program_id(1)
    pid_h = tl.program_id(2)
    
    # Batch and head offsets
    q_offset = pid_z * stride_qz + pid_h * stride_qh
    k_offset = pid_z * stride_kz + pid_h * stride_kh
    v_offset = pid_z * stride_vz + pid_h * stride_vh
    o_offset = pid_z * stride_oz + pid_h * stride_oh
    
    # Block pointers for Q (BLOCK_M x BLOCK_K)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_K)
    
    q_ptrs = Q + q_offset + offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk
    
    # Initialize accumulator and statistics
    acc = tl.zeros([BLOCK_M, BLOCK_K], dtype=tl.float32)
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")  # Max value for softmax
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)  # Sum exp for softmax
    
    # Load Q block (BLOCK_M x BLOCK_K)
    q = tl.load(q_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0.0)
    
    # Loop over K, V blocks
    for start_n in range(0, N, BLOCK_N):
        # Block pointers for K (BLOCK_N x BLOCK_K)
        offs_n = start_n + tl.arange(0, BLOCK_N)
        k_ptrs = K + k_offset + offs_n[:, None] * stride_kn + offs_k[None, :] * stride_kk
        v_ptrs = V + v_offset + offs_n[:, None] * stride_vn + offs_k[None, :] * stride_vk
        
        # Load K block (BLOCK_N x BLOCK_K) and transpose
        k = tl.load(k_ptrs, mask=(offs_n[:, None] < N) & (offs_k[None, :] < K), other=0.0)
        
        # Compute QK^T (BLOCK_M x BLOCK_N)
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, tl.trans(k))
        qk *= scale
        
        # Online softmax - update statistics
        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
        p = tl.exp(qk - m_ij[:, None])
        l_ij = tl.sum(p, axis=1)
        
        # Rescale previous accumulator
        alpha = tl.exp(m_i - m_ij)
        acc = acc * alpha[:, None]
        
        # Load V block (BLOCK_N x BLOCK_K)
        v = tl.load(v_ptrs, mask=(offs_n[:, None] < N) & (offs_k[None, :] < K), other=0.0)
        
        # Accumulate attention * V
        acc += tl.dot(p.to(v.dtype), v)
        
        # Update statistics
        l_i = l_i * alpha + l_ij
        m_i = m_ij
    
    # Final rescaling
    acc = acc / l_i[:, None]
    
    # Store output
    offs_k = tl.arange(0, BLOCK_K)
    o_ptrs = Out + o_offset + offs_m[:, None] * stride_om + offs_k[None, :] * stride_ok
    tl.store(o_ptrs, acc.to(Out.dtype.element_ty), mask=(offs_m[:, None] < M) & (offs_k[None, :] < K))
    
    # Store logsumexp for backward pass
    l_ptrs = L + pid_z * H * M + pid_h * M + offs_m
    tl.store(l_ptrs, m_i + tl.log(l_i), mask=offs_m < M)


class TritonFlashAttention(torch.autograd.Function):
    """
    Triton FlashAttention-2 Implementation
    Forward-only for inference optimization
    """
    
    @staticmethod
    def forward(ctx, q, k, v, causal=False):
        """
        Args:
            q: Query tensor [batch, heads, seq_len, head_dim]
            k: Key tensor [batch, heads, seq_len, head_dim]
            v: Value tensor [batch, heads, seq_len, head_dim]
            causal: Whether to apply causal masking (not implemented in this version)
        
        Returns:
            out: Attention output [batch, heads, seq_len, head_dim]
        """
        # Shape validation
        assert q.dim() == 4 and k.dim() == 4 and v.dim() == 4
        assert q.shape == k.shape == v.shape
        
        batch, heads, seq_len, head_dim = q.shape
        
        # Validate contiguous
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        
        # Allocate output
        out = torch.empty_like(q)
        
        # Logsumexp for numerical stability (needed for backward, but we compute anyway)
        L = torch.empty((batch, heads, seq_len), device=q.device, dtype=torch.float32)
        
        # Attention scale
        scale = 1.0 / math.sqrt(head_dim)
        
        # Tile sizes - tuned for RTX 4090 / A100
        BLOCK_M = 128
        BLOCK_N = 128
        BLOCK_K = head_dim
        
        # Grid size
        grid = (triton.cdiv(seq_len, BLOCK_M), batch, heads)
        
        # Launch kernel
        _fwd_kernel[grid](
            q, k, v, out, L,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            batch, heads, seq_len, seq_len, head_dim,
            scale,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        )
        
        return out
    
    @staticmethod
    def backward(ctx, grad_out):
        """Backward pass - not implemented for inference-only use"""
        raise NotImplementedError("Backward pass not implemented for inference-only kernel")


def triton_flash_attention(q, k, v, causal=False):
    """
    User-facing API for Triton FlashAttention
    
    Args:
        q: Query [batch, heads, seq_len, head_dim]
        k: Key [batch, heads, seq_len, head_dim]
        v: Value [batch, heads, seq_len, head_dim]
        causal: Apply causal masking (not yet implemented)
    
    Returns:
        Attention output [batch, heads, seq_len, head_dim]
    """
    return TritonFlashAttention.apply(q, k, v, causal)


# Alternative: Optimized kernel with better memory access pattern
@triton.jit
def _optimized_fwd_kernel(
    Q, K, V, Out,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_ok,
    Z, H, M, N, K,
    scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """
    Optimized version with improved coalescing and reduced register pressure
    """
    pid_m = tl.program_id(0)
    pid_z = tl.program_id(1)
    pid_h = tl.program_id(2)
    
    # Calculate base offsets
    q_base = pid_z * stride_qz + pid_h * stride_qh
    k_base = pid_z * stride_kz + pid_h * stride_kh
    v_base = pid_z * stride_vz + pid_h * stride_vh
    o_base = pid_z * stride_oz + pid_h * stride_oh
    
    # Row indices for this block
    row_idx = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    row_mask = row_idx < M
    
    # Initialize output accumulator
    acc = tl.zeros([BLOCK_M, BLOCK_K], dtype=tl.float32)
    
    # Load Q for this block
    q_offs = q_base + row_idx[:, None] * stride_qm + tl.arange(0, BLOCK_K)[None, :] * stride_qk
    q = tl.load(Q + q_offs, mask=row_mask[:, None], other=0.0)
    
    # Initialize softmax statistics
    row_max = tl.full([BLOCK_M], -float('inf'), dtype=tl.float32)
    row_sum = tl.zeros([BLOCK_M], dtype=tl.float32)
    
    # Iterate over K, V blocks
    for block_n in range(0, tl.cdiv(N, BLOCK_N)):
        col_idx = block_n * BLOCK_N + tl.arange(0, BLOCK_N)
        col_mask = col_idx < N
        
        # Load K^T for this block
        k_offs = k_base + col_idx[:, None] * stride_kn + tl.arange(0, BLOCK_K)[None, :] * stride_kk
        k = tl.load(K + k_offs, mask=col_mask[:, None], other=0.0)
        
        # Compute attention scores: QK^T
        scores = tl.dot(q, tl.trans(k)) * scale
        
        # Update softmax statistics (online algorithm)
        block_max = tl.max(scores, axis=1)
        new_max = tl.maximum(row_max, block_max)
        
        # Rescale and compute exp
        scores = tl.exp(scores - new_max[:, None])
        
        # Rescale previous accumulator
        scale_factor = tl.exp(row_max - new_max)
        acc *= scale_factor[:, None]
        row_sum *= scale_factor
        
        # Load V for this block
        v_offs = v_base + col_idx[:, None] * stride_vn + tl.arange(0, BLOCK_K)[None, :] * stride_vk
        v = tl.load(V + v_offs, mask=col_mask[:, None], other=0.0)
        
        # Accumulate weighted V
        acc += tl.dot(scores, v)
        row_sum += tl.sum(scores, axis=1)
        row_max = new_max
    
    # Normalize by softmax denominator
    acc /= row_sum[:, None]
    
    # Store output
    o_offs = o_base + row_idx[:, None] * stride_om + tl.arange(0, BLOCK_K)[None, :] * stride_ok
    tl.store(Out + o_offs, acc, mask=row_mask[:, None])


def triton_flash_attention_v2(q, k, v):
    """Optimized version of FlashAttention"""
    batch, heads, seq_len, head_dim = q.shape
    
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    out = torch.empty_like(q)
    
    scale = 1.0 / math.sqrt(head_dim)
    
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = head_dim
    
    grid = (triton.cdiv(seq_len, BLOCK_M), batch, heads)
    
    _optimized_fwd_kernel[grid](
        q, k, v, out,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        batch, heads, seq_len, seq_len, head_dim,
        scale,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )
    
    return out


if __name__ == "__main__":
    # Quick test
    print("Testing Triton FlashAttention kernel...")
    
    batch, heads, seq_len, head_dim = 2, 8, 512, 64
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if device == "cuda":
        q = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=torch.float16)
        k = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=torch.float16)
        v = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=torch.float16)
        
        # Test forward pass
        out = triton_flash_attention(q, k, v)
        print(f" Forward pass successful: {out.shape}")
        
        # Test v2
        out_v2 = triton_flash_attention_v2(q, k, v)
        print(f"Optimized version successful: {out_v2.shape}")
    else:
        print("CUDA not available, skipping test")
