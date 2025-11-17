"""
Dataset Preparation Script for Enterprise RAG
Combines multiple high-quality datasets into a unified training corpus:
- 10,000 samples from databricks-dolly-15k
- 5,000 finance QA from Finance-Alpaca + ConvFinQA
- 3,000 SQL + Python code from Spider + CodeAlpaca

Output format: Alpaca-style JSON with instruction, input, output fields
"""

import json
import random
from pathlib import Path
from typing import List, Dict
from datasets import load_dataset
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatasetPreparator:
    def __init__(self, output_dir: str = "data"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.final_dataset = []
        
    def prepare_dolly(self, num_samples: int = 10000) -> List[Dict]:
        """Load and format Databricks Dolly-15k dataset"""
        logger.info(f"Loading {num_samples} samples from Dolly-15k...")
        
        dataset = load_dataset("databricks/databricks-dolly-15k", split="train")
        
        # Sample diverse categories
        formatted_samples = []
        for item in tqdm(dataset.shuffle(seed=42).select(range(num_samples))):
            formatted_samples.append({
                "instruction": item["instruction"],
                "input": item.get("context", ""),
                "output": item["response"],
                "source": "dolly-15k",
                "category": item.get("category", "general")
            })
        
        logger.info(f"Prepared {len(formatted_samples)} Dolly samples")
        return formatted_samples
    
    def prepare_finance_qa(self, num_samples: int = 5000) -> List[Dict]:
        """Load and format Finance datasets"""
        logger.info(f"Loading {num_samples} finance QA samples...")
        
        formatted_samples = []
        
        # Finance Alpaca (3000 samples)
        try:
            finance_alpaca = load_dataset("gbharti/finance-alpaca", split="train")
            for item in tqdm(finance_alpaca.shuffle(seed=42).select(range(min(3000, len(finance_alpaca)))), 
                           desc="Finance Alpaca"):
                formatted_samples.append({
                    "instruction": item.get("instruction", ""),
                    "input": item.get("input", ""),
                    "output": item.get("output", ""),
                    "source": "finance-alpaca",
                    "category": "finance"
                })
        except Exception as e:
            logger.warning(f"Could not load Finance Alpaca: {e}")
        
        # ConvFinQA (2000 samples) - financial reasoning
        try:
            convfinqa = load_dataset("MU-NLPC/Calc-convfinqa", split="train")
            for item in tqdm(convfinqa.shuffle(seed=42).select(range(min(2000, len(convfinqa)))),
                           desc="ConvFinQA"):
                # Format as instruction-following
                question = item.get("question", "")
                context = item.get("pre_text", "") + " " + item.get("post_text", "")
                answer = item.get("answer", "")
                
                formatted_samples.append({
                    "instruction": f"Answer this financial question based on the provided context: {question}",
                    "input": context.strip(),
                    "output": str(answer),
                    "source": "convfinqa",
                    "category": "finance"
                })
        except Exception as e:
            logger.warning(f"Could not load ConvFinQA: {e}")
        
        # If we don't have enough, create synthetic finance QA
        while len(formatted_samples) < num_samples:
            formatted_samples.append(self._generate_synthetic_finance())
        
        logger.info(f"Prepared {len(formatted_samples[:num_samples])} finance samples")
        return formatted_samples[:num_samples]
    
    def prepare_code_datasets(self, num_samples: int = 3000) -> List[Dict]:
        """Load and format SQL and Python coding datasets"""
        logger.info(f"Loading {num_samples} code samples...")
        
        formatted_samples = []
        
        # Spider SQL dataset (1500 samples)
        try:
            spider = load_dataset("spider", split="train")
            for item in tqdm(spider.shuffle(seed=42).select(range(min(1500, len(spider)))),
                           desc="Spider SQL"):
                formatted_samples.append({
                    "instruction": f"Generate a SQL query to answer: {item['question']}",
                    "input": f"Database schema: {item.get('db_id', '')}",
                    "output": item["query"],
                    "source": "spider",
                    "category": "sql"
                })
        except Exception as e:
            logger.warning(f"Could not load Spider: {e}")
        
        # CodeAlpaca Python (1500 samples)
        try:
            code_alpaca = load_dataset("sahil2801/CodeAlpaca-20k", split="train")
            python_samples = [item for item in code_alpaca if "python" in item.get("instruction", "").lower()]
            
            for item in tqdm(python_samples[:1500], desc="CodeAlpaca Python"):
                formatted_samples.append({
                    "instruction": item["instruction"],
                    "input": item.get("input", ""),
                    "output": item["output"],
                    "source": "code-alpaca",
                    "category": "python"
                })
        except Exception as e:
            logger.warning(f"Could not load CodeAlpaca: {e}")
        
        # Generate synthetic code samples if needed
        while len(formatted_samples) < num_samples:
            formatted_samples.append(self._generate_synthetic_code())
        
        logger.info(f"Prepared {len(formatted_samples[:num_samples])} code samples")
        return formatted_samples[:num_samples]
    
    def _generate_synthetic_finance(self) -> Dict:
        """Generate synthetic finance QA"""
        templates = [
            {
                "instruction": "Calculate the compound annual growth rate (CAGR) for an investment.",
                "input": "Initial value: $10,000, Final value: $15,000, Period: 5 years",
                "output": "CAGR = ((15000/10000)^(1/5) - 1) × 100 = 8.45%"
            },
            {
                "instruction": "Explain the difference between EBITDA and net income.",
                "input": "",
                "output": "EBITDA (Earnings Before Interest, Taxes, Depreciation, and Amortization) measures operational profitability, while net income is the bottom-line profit after all expenses including interest, taxes, and non-cash charges."
            }
        ]
        return random.choice(templates)
    
    def _generate_synthetic_code(self) -> Dict:
        """Generate synthetic code examples"""
        templates = [
            {
                "instruction": "Write a Python function to calculate factorial recursively.",
                "input": "",
                "output": "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n - 1)"
            },
            {
                "instruction": "Generate SQL to find top 5 customers by revenue.",
                "input": "Tables: customers (id, name), orders (customer_id, amount)",
                "output": "SELECT c.name, SUM(o.amount) as revenue\nFROM customers c\nJOIN orders o ON c.id = o.customer_id\nGROUP BY c.id, c.name\nORDER BY revenue DESC\nLIMIT 5;"
            }
        ]
        return random.choice(templates)
    
    def combine_and_save(self):
        """Combine all datasets and save to JSON"""
        logger.info("Combining all datasets...")
        
        # Gather all samples
        dolly_samples = self.prepare_dolly(10000)
        finance_samples = self.prepare_finance_qa(5000)
        code_samples = self.prepare_code_datasets(3000)
        
        # Combine
        self.final_dataset = dolly_samples + finance_samples + code_samples
        
        # Shuffle
        random.seed(42)
        random.shuffle(self.final_dataset)
        
        # Save full dataset
        output_path = self.output_dir / "enterprise_dataset.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.final_dataset, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ Saved {len(self.final_dataset)} samples to {output_path}")
        
        # Create train/val split
        split_idx = int(len(self.final_dataset) * 0.95)
        train_data = self.final_dataset[:split_idx]
        val_data = self.final_dataset[split_idx:]
        
        with open(self.output_dir / "train.json", 'w', encoding='utf-8') as f:
            json.dump(train_data, f, indent=2, ensure_ascii=False)
        
        with open(self.output_dir / "validation.json", 'w', encoding='utf-8') as f:
            json.dump(val_data, f, indent=2, ensure_ascii=False)
        
        # Print statistics
        self._print_statistics()
    
    def _print_statistics(self):
        """Print dataset statistics"""
        logger.info("\n" + "="*60)
        logger.info("DATASET STATISTICS")
        logger.info("="*60)
        logger.info(f"Total samples: {len(self.final_dataset)}")
        
        # By source
        sources = {}
        categories = {}
        for item in self.final_dataset:
            source = item.get("source", "unknown")
            category = item.get("category", "unknown")
            sources[source] = sources.get(source, 0) + 1
            categories[category] = categories.get(category, 0) + 1
        
        logger.info("\nBy Source:")
        for source, count in sorted(sources.items()):
            logger.info(f"  {source}: {count}")
        
        logger.info("\nBy Category:")
        for category, count in sorted(categories.items()):
            logger.info(f"  {category}: {count}")
        
        logger.info("="*60 + "\n")


def main():
    preparator = DatasetPreparator(output_dir="data")
    preparator.combine_and_save()
    
    logger.info("✅ Dataset preparation complete!")
    logger.info("📁 Files created:")
    logger.info("   - data/enterprise_dataset.json (full dataset)")
    logger.info("   - data/train.json (95% split)")
    logger.info("   - data/validation.json (5% split)")


if __name__ == "__main__":
    main()
