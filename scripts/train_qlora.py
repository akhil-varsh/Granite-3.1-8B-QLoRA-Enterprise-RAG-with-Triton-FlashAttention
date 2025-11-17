"""
QLoRA Fine-tuning Script for Granite-3.1-8B-Instruct
Optimized for single 24GB GPU with full enterprise dataset

Features:
- 4-bit quantization with bitsandbytes
- LoRA adapters (rank=64, alpha=16, dropout=0.1)
- Target all linear layers for comprehensive adaptation
- Gradient checkpointing for memory efficiency
- WandB logging and checkpointing
- <20 hour training on RTX 4090/A100
"""

import os
import json
import torch
import transformers
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
from datasets import load_dataset, Dataset
from typing import Dict, List, Optional
import logging
from datetime import datetime
from pathlib import Path
import wandb
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class EnterpriseDatasetProcessor:
    """Process enterprise dataset for QLoRA training"""
    
    def __init__(self, tokenizer, max_length: int = 2048):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Alpaca-style prompt template
        self.prompt_template = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:
{output}"""
    
    def format_example(self, example: Dict) -> str:
        """Format example using Alpaca template"""
        instruction = example.get('instruction', '').strip()
        input_text = example.get('input', '').strip()
        output_text = example.get('output', '').strip()
        
        return self.prompt_template.format(
            instruction=instruction,
            input=input_text if input_text else "None",
            output=output_text
        )
    
    def tokenize_function(self, examples: Dict) -> Dict:
        """Tokenize examples for training"""
        # Format all examples
        texts = [self.format_example(ex) for ex in examples['examples']]
        
        # Tokenize with padding and truncation
        tokenized = self.tokenizer(
            texts,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors=None,
        )
        
        # Labels are the same as input_ids for causal LM
        tokenized['labels'] = tokenized['input_ids'].copy()
        
        return tokenized
    
    def prepare_dataset(self, dataset_path: str):
        """Load and prepare dataset"""
        logger.info(f"Loading dataset from {dataset_path}")
        
        # Load JSON dataset
        with open(dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        logger.info(f"Loaded {len(data)} examples")
        
        # Convert to HF Dataset
        dataset = Dataset.from_dict({'examples': data})
        
        # Split into train/val
        dataset = dataset.train_test_split(test_size=0.05, seed=42)
        
        logger.info(f"Train: {len(dataset['train'])}, Val: {len(dataset['test'])}")
        
        # Tokenize
        logger.info("Tokenizing dataset...")
        tokenized_dataset = dataset.map(
            self.tokenize_function,
            batched=True,
            remove_columns=dataset['train'].column_names,
            desc="Tokenizing"
        )
        
        return tokenized_dataset['train'], tokenized_dataset['test']


def setup_model_and_tokenizer(
    model_name: str = "ibm-granite/granite-3.1-8b-instruct",
    use_4bit: bool = True,
    use_flash_attention: bool = True
):
    """
    Load model with 4-bit quantization and setup tokenizer
    """
    logger.info(f"Loading model: {model_name}")
    
    # BitsAndBytes config for 4-bit quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=use_4bit,
        bnb_4bit_quant_type="nf4",  # Normal Float 4-bit
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,  # Nested quantization
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        use_fast=True,
    )
    
    # Set padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2" if use_flash_attention else "eager",
    )
    
    # Enable gradient checkpointing for memory efficiency
    model.gradient_checkpointing_enable()
    
    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)
    
    logger.info("✅ Model and tokenizer loaded successfully")
    
    return model, tokenizer


def setup_lora_config():
    """
    Configure LoRA parameters
    - rank=64 for higher capacity
    - alpha=16 for learning rate scaling
    - dropout=0.1 for regularization
    - Target all linear layers
    """
    lora_config = LoraConfig(
        r=64,  # LoRA rank
        lora_alpha=16,  # Scaling factor
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        # Target all linear layers for comprehensive adaptation
        target_modules=[
            "q_proj",
            "k_proj", 
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        inference_mode=False,
    )
    
    logger.info("LoRA Config:")
    logger.info(f"  Rank: {lora_config.r}")
    logger.info(f"  Alpha: {lora_config.lora_alpha}")
    logger.info(f"  Dropout: {lora_config.lora_dropout}")
    logger.info(f"  Target modules: {lora_config.target_modules}")
    
    return lora_config


def train(
    model_name: str = "ibm-granite/granite-3.1-8b-instruct",
    dataset_path: str = "data/enterprise_dataset.json",
    output_dir: str = "outputs",
    num_epochs: int = 3,
    batch_size: int = 4,
    gradient_accumulation_steps: int = 4,
    learning_rate: float = 2e-4,
    max_seq_length: int = 2048,
    warmup_ratio: float = 0.03,
    logging_steps: int = 10,
    save_steps: int = 500,
    use_wandb: bool = True,
):
    """Main training function"""
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    Path("logs").mkdir(exist_ok=True)
    
    # Initialize WandB
    if use_wandb and os.getenv("WANDB_API_KEY"):
        wandb.init(
            project=os.getenv("WANDB_PROJECT", "enterprise-rag-qlora"),
            name=f"qlora-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            config={
                "model": model_name,
                "epochs": num_epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "lora_rank": 64,
                "max_seq_length": max_seq_length,
            }
        )
    
    # Setup model and tokenizer
    model, tokenizer = setup_model_and_tokenizer(model_name)
    
    # Setup LoRA
    lora_config = setup_lora_config()
    model = get_peft_model(model, lora_config)
    
    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable params: {trainable_params:,} || "
                f"Total params: {total_params:,} || "
                f"Trainable %: {100 * trainable_params / total_params:.2f}%")
    
    # Prepare dataset
    processor = EnterpriseDatasetProcessor(tokenizer, max_seq_length)
    train_dataset, eval_dataset = processor.prepare_dataset(dataset_path)
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        logging_steps=logging_steps,
        save_steps=save_steps,
        eval_steps=save_steps,
        evaluation_strategy="steps",
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=True,
        tf32=True,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",  # Memory-efficient optimizer
        lr_scheduler_type="cosine",
        report_to="wandb" if use_wandb else "none",
        logging_dir=f"{output_dir}/logs",
        save_safetensors=True,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
    )
    
    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )
    
    # Train
    logger.info("🚀 Starting training...")
    logger.info(f"  Total optimization steps: {len(train_dataset) // (batch_size * gradient_accumulation_steps) * num_epochs}")
    
    train_result = trainer.train()
    
    # Save model
    logger.info(f"💾 Saving model to {output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
    # Save training metrics
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()
    
    logger.info("✅ Training complete!")
    logger.info(f"📊 Final train loss: {metrics['train_loss']:.4f}")
    
    # Final evaluation
    logger.info("📊 Running final evaluation...")
    eval_metrics = trainer.evaluate()
    trainer.log_metrics("eval", eval_metrics)
    trainer.save_metrics("eval", eval_metrics)
    logger.info(f"📊 Final eval loss: {eval_metrics['eval_loss']:.4f}")
    
    if use_wandb:
        wandb.finish()
    
    return trainer, model, tokenizer


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="QLoRA Fine-tuning for Granite-3.1-8B")
    parser.add_argument("--model_name", type=str, default="ibm-granite/granite-3.1-8b-instruct",
                       help="Base model name")
    parser.add_argument("--dataset_path", type=str, default="data/enterprise_dataset.json",
                       help="Path to training dataset")
    parser.add_argument("--output_dir", type=str, default="outputs/qlora_model",
                       help="Output directory")
    parser.add_argument("--num_epochs", type=int, default=3,
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Training batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                       help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=2e-4,
                       help="Learning rate")
    parser.add_argument("--max_seq_length", type=int, default=2048,
                       help="Maximum sequence length")
    parser.add_argument("--no_wandb", action="store_true",
                       help="Disable WandB logging")
    
    args = parser.parse_args()
    
    # Train model
    train(
        model_name=args.model_name,
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        max_seq_length=args.max_seq_length,
        use_wandb=not args.no_wandb,
    )


if __name__ == "__main__":
    main()
