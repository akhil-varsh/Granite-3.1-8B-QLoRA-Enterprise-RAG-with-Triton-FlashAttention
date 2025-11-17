"""
Benchmark Script: Compare Triton FlashAttention vs PyTorch/xFormers
Measures inference speed, memory usage, and accuracy
"""

import torch
import time
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import json
import sys
from typing import Dict, List

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from triton_kernels import triton_flash_attention, triton_flash_attention_v2

# Try to import xformers
try:
    import xformers.ops as xops
    HAS_XFORMERS = True
except ImportError:
    HAS_XFORMERS = False
    print("⚠️ xFormers not available")

# Check for Flash Attention
try:
    from flash_attn import flash_attn_func
    HAS_FLASH_ATTN = True
except ImportError:
    HAS_FLASH_ATTN = False
    print("⚠️ Flash Attention not available")


class AttentionBenchmark:
    """Benchmark different attention implementations"""
    
    def __init__(self, device="cuda", dtype=torch.float16):
        self.device = device
        self.dtype = dtype
        self.results = {}
        
    def pytorch_attention(self, q, k, v):
        """Standard PyTorch scaled dot-product attention"""
        scale = 1.0 / np.sqrt(q.shape[-1])
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn = torch.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        return out
    
    def pytorch_sdpa(self, q, k, v):
        """PyTorch 2.0+ scaled_dot_product_attention"""
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        return out
    
    def xformers_attention(self, q, k, v):
        """xFormers memory-efficient attention"""
        if not HAS_XFORMERS:
            return None
        # xFormers expects [B, M, H, K] format
        b, h, m, k = q.shape
        q = q.transpose(1, 2)  # [B, M, H, K]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        out = xops.memory_efficient_attention(q, k, v)
        return out.transpose(1, 2)  # Back to [B, H, M, K]
    
    def flash_attn_attention(self, q, k, v):
        """Flash Attention 2"""
        if not HAS_FLASH_ATTN:
            return None
        # Flash Attention expects [B, M, H, K] format
        b, h, m, k = q.shape
        q = q.transpose(1, 2).contiguous()  # [B, M, H, K]
        k = k.transpose(1, 2).contiguous()
        v = v.transpose(1, 2).contiguous()
        out = flash_attn_func(q, k, v)
        return out.transpose(1, 2)  # Back to [B, H, M, K]
    
    def benchmark_implementation(
        self,
        name: str,
        func,
        q, k, v,
        warmup_iters: int = 10,
        test_iters: int = 100
    ) -> Dict:
        """Benchmark a single attention implementation"""
        
        if func is None:
            return None
        
        # Warmup
        for _ in range(warmup_iters):
            _ = func(q, k, v)
            if self.device == "cuda":
                torch.cuda.synchronize()
        
        # Benchmark
        if self.device == "cuda":
            torch.cuda.reset_peak_memory_stats()
            start_mem = torch.cuda.memory_allocated()
        
        times = []
        for _ in range(test_iters):
            start = time.perf_counter()
            out = func(q, k, v)
            if self.device == "cuda":
                torch.cuda.synchronize()
            end = time.perf_counter()
            times.append(end - start)
        
        if self.device == "cuda":
            peak_mem = torch.cuda.max_memory_allocated()
            end_mem = torch.cuda.memory_allocated()
            mem_used = (peak_mem - start_mem) / 1024**2  # MB
        else:
            mem_used = 0
        
        return {
            "name": name,
            "mean_time": np.mean(times) * 1000,  # ms
            "std_time": np.std(times) * 1000,
            "min_time": np.min(times) * 1000,
            "max_time": np.max(times) * 1000,
            "memory_mb": mem_used,
            "output": out,
        }
    
    def run_benchmark(
        self,
        batch_sizes: List[int] = [1, 2, 4, 8],
        num_heads: List[int] = [8, 16, 32],
        seq_lengths: List[int] = [128, 256, 512, 1024, 2048],
        head_dim: int = 64,
    ):
        """Run comprehensive benchmark across different configurations"""
        
        print("="*80)
        print("ATTENTION BENCHMARK")
        print("="*80)
        
        all_results = []
        
        for batch in batch_sizes:
            for heads in num_heads:
                for seq_len in seq_lengths:
                    config = f"B{batch}_H{heads}_S{seq_len}_D{head_dim}"
                    print(f"\n🔧 Config: {config}")
                    
                    # Create random tensors
                    q = torch.randn(batch, heads, seq_len, head_dim, 
                                   device=self.device, dtype=self.dtype)
                    k = torch.randn(batch, heads, seq_len, head_dim,
                                   device=self.device, dtype=self.dtype)
                    v = torch.randn(batch, heads, seq_len, head_dim,
                                   device=self.device, dtype=self.dtype)
                    
                    config_results = {
                        "batch_size": batch,
                        "num_heads": heads,
                        "seq_length": seq_len,
                        "head_dim": head_dim,
                        "implementations": {}
                    }
                    
                    # Benchmark PyTorch
                    print("  📊 PyTorch (naive)...", end=" ")
                    result = self.benchmark_implementation("pytorch", self.pytorch_attention, q, k, v)
                    if result:
                        config_results["implementations"]["pytorch"] = result
                        print(f"{result['mean_time']:.2f}ms")
                    
                    # Benchmark PyTorch SDPA
                    print("  📊 PyTorch SDPA...", end=" ")
                    result = self.benchmark_implementation("pytorch_sdpa", self.pytorch_sdpa, q, k, v)
                    if result:
                        config_results["implementations"]["pytorch_sdpa"] = result
                        print(f"{result['mean_time']:.2f}ms")
                    
                    # Benchmark xFormers
                    if HAS_XFORMERS:
                        print("  📊 xFormers...", end=" ")
                        result = self.benchmark_implementation("xformers", self.xformers_attention, q, k, v)
                        if result:
                            config_results["implementations"]["xformers"] = result
                            print(f"{result['mean_time']:.2f}ms")
                    
                    # Benchmark Flash Attention
                    if HAS_FLASH_ATTN:
                        print("  📊 Flash Attention 2...", end=" ")
                        result = self.benchmark_implementation("flash_attn", self.flash_attn_attention, q, k, v)
                        if result:
                            config_results["implementations"]["flash_attn"] = result
                            print(f"{result['mean_time']:.2f}ms")
                    
                    # Benchmark Triton v1
                    print("  📊 Triton FlashAttn (v1)...", end=" ")
                    result = self.benchmark_implementation("triton_v1", triton_flash_attention, q, k, v)
                    if result:
                        config_results["implementations"]["triton_v1"] = result
                        print(f"{result['mean_time']:.2f}ms")
                    
                    # Benchmark Triton v2
                    print("  📊 Triton FlashAttn (v2)...", end=" ")
                    result = self.benchmark_implementation("triton_v2", triton_flash_attention_v2, q, k, v)
                    if result:
                        config_results["implementations"]["triton_v2"] = result
                        print(f"{result['mean_time']:.2f}ms")
                    
                    all_results.append(config_results)
        
        self.results = all_results
        return all_results
    
    def calculate_speedups(self, baseline="pytorch"):
        """Calculate speedup relative to baseline"""
        for config in self.results:
            impls = config["implementations"]
            if baseline not in impls:
                continue
            
            baseline_time = impls[baseline]["mean_time"]
            
            for name, result in impls.items():
                speedup = baseline_time / result["mean_time"]
                result["speedup"] = speedup
    
    def print_summary(self):
        """Print benchmark summary"""
        print("\n" + "="*80)
        print("BENCHMARK SUMMARY")
        print("="*80)
        
        # Find best configuration for detailed comparison
        target_config = None
        for config in self.results:
            if (config["batch_size"] == 2 and 
                config["num_heads"] == 8 and 
                config["seq_length"] == 512):
                target_config = config
                break
        
        if not target_config:
            target_config = self.results[len(self.results)//2]
        
        print(f"\nConfiguration: B={target_config['batch_size']}, "
              f"H={target_config['num_heads']}, "
              f"S={target_config['seq_length']}, "
              f"D={target_config['head_dim']}")
        print("\n{:<20} {:>15} {:>15} {:>15}".format(
            "Implementation", "Time (ms)", "Memory (MB)", "Speedup"))
        print("-"*70)
        
        baseline_time = None
        for name, result in target_config["implementations"].items():
            if baseline_time is None:
                baseline_time = result["mean_time"]
            
            speedup = baseline_time / result["mean_time"]
            print("{:<20} {:>15.2f} {:>15.2f} {:>15.2f}x".format(
                name,
                result["mean_time"],
                result.get("memory_mb", 0),
                speedup
            ))
    
    def plot_results(self, output_dir="results"):
        """Generate visualization plots"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        sns.set_style("whitegrid")
        
        # Plot 1: Speedup comparison across sequence lengths
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Filter for specific config
        filtered = [r for r in self.results if r["batch_size"] == 2 and r["num_heads"] == 8]
        
        implementations = list(filtered[0]["implementations"].keys())
        seq_lengths = [r["seq_length"] for r in filtered]
        
        for impl in implementations:
            times = [r["implementations"][impl]["mean_time"] for r in filtered]
            ax.plot(seq_lengths, times, marker='o', label=impl, linewidth=2)
        
        ax.set_xlabel("Sequence Length", fontsize=12)
        ax.set_ylabel("Time (ms)", fontsize=12)
        ax.set_title("Attention Implementation Comparison (B=2, H=8)", fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path / "benchmark_comparison.png", dpi=300, bbox_inches='tight')
        print(f"📊 Saved: {output_path / 'benchmark_comparison.png'}")
        
        # Plot 2: Speedup bars
        fig, ax = plt.subplots(figsize=(10, 6))
        
        target_config = filtered[len(filtered)//2]  # Middle seq length
        impl_names = list(target_config["implementations"].keys())
        baseline_time = target_config["implementations"][impl_names[0]]["mean_time"]
        speedups = [baseline_time / target_config["implementations"][name]["mean_time"] 
                   for name in impl_names]
        
        bars = ax.bar(impl_names, speedups, color=sns.color_palette("husl", len(impl_names)))
        ax.axhline(y=1.0, color='red', linestyle='--', label='Baseline')
        ax.set_ylabel("Speedup", fontsize=12)
        ax.set_title(f"Speedup vs PyTorch (S={target_config['seq_length']})", 
                    fontsize=14, fontweight='bold')
        ax.set_xticklabels(impl_names, rotation=45, ha='right')
        
        # Add value labels on bars
        for bar, speedup in zip(bars, speedups):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{speedup:.2f}x',
                   ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(output_path / "speedup_comparison.png", dpi=300, bbox_inches='tight')
        print(f"📊 Saved: {output_path / 'speedup_comparison.png'}")
        
        plt.close('all')
    
    def save_results(self, output_path="results/benchmark_results.json"):
        """Save results to JSON"""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert tensors to lists for JSON serialization
        serializable_results = []
        for config in self.results:
            config_copy = {
                "batch_size": config["batch_size"],
                "num_heads": config["num_heads"],
                "seq_length": config["seq_length"],
                "head_dim": config["head_dim"],
                "implementations": {}
            }
            for name, result in config["implementations"].items():
                config_copy["implementations"][name] = {
                    k: v for k, v in result.items() if k != "output"
                }
            serializable_results.append(config_copy)
        
        with open(output_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"💾 Saved results to {output_file}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Benchmark Triton FlashAttention")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--dtype", type=str, default="float16", help="Data type")
    parser.add_argument("--output_dir", type=str, default="results", help="Output directory")
    
    args = parser.parse_args()
    
    if args.device == "cuda" and not torch.cuda.is_available():
        print("❌ CUDA not available!")
        return
    
    dtype = torch.float16 if args.dtype == "float16" else torch.float32
    
    # Run benchmark
    benchmark = AttentionBenchmark(device=args.device, dtype=dtype)
    
    # Quick benchmark for demo
    results = benchmark.run_benchmark(
        batch_sizes=[2, 4],
        num_heads=[8, 16],
        seq_lengths=[256, 512, 1024, 2048],
        head_dim=64,
    )
    
    # Calculate speedups
    benchmark.calculate_speedups(baseline="pytorch")
    
    # Print summary
    benchmark.print_summary()
    
    # Save and plot
    benchmark.save_results(f"{args.output_dir}/benchmark_results.json")
    benchmark.plot_results(args.output_dir)
    
    print("\n✅ Benchmark complete!")


if __name__ == "__main__":
    main()
