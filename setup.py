"""
Quick start script to set up the entire project
Run this first after cloning the repository
"""

import os
import subprocess
import sys
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def run_command(cmd, description):
    """Run a shell command with progress indication"""
    logger.info(f"\n{'='*60}")
    logger.info(f"🔄 {description}")
    logger.info(f"{'='*60}")
    logger.info(f"Command: {cmd}\n")
    
    result = subprocess.run(cmd, shell=True)
    
    if result.returncode != 0:
        logger.error(f"❌ Failed: {description}")
        return False
    
    logger.info(f"✅ Completed: {description}")
    return True


def main():
    logger.info("""
╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║   Enterprise RAG with Granite-3.1-8B-Instruct                ║
║   Quick Setup Script                                          ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝
    """)
    
    project_root = Path(__file__).parent
    os.chdir(project_root)
    
    # Step 1: Create directories
    logger.info("\n📁 Step 1/5: Creating directories...")
    dirs = ['data', 'data/rag_documents', 'outputs', 'results', 'logs']
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
        logger.info(f"   Created: {d}")
    
    # Step 2: Check Python version
    logger.info("\n🐍 Step 2/5: Checking Python version...")
    if sys.version_info < (3, 10):
        logger.error("❌ Python 3.10+ required!")
        return
    logger.info(f"   ✅ Python {sys.version_info.major}.{sys.version_info.minor}")
    
    # Step 3: Check CUDA
    logger.info("\n🎮 Step 3/5: Checking CUDA availability...")
    try:
        import torch
        if torch.cuda.is_available():
            logger.info(f"   ✅ CUDA available: {torch.cuda.get_device_name(0)}")
            logger.info(f"   ✅ CUDA version: {torch.version.cuda}")
            logger.info(f"   ✅ GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        else:
            logger.warning("   ⚠️ CUDA not available - CPU training will be very slow!")
    except ImportError:
        logger.warning("   ⚠️ PyTorch not installed yet")
    
    # Step 4: Install dependencies
    logger.info("\n📦 Step 4/5: Installing dependencies...")
    install_choice = input("Install dependencies now? (y/n): ").lower()
    
    if install_choice == 'y':
        if not run_command(
            f"{sys.executable} -m pip install -r requirements.txt",
            "Installing Python packages"
        ):
            logger.error("Failed to install dependencies")
            return
    else:
        logger.info("   ⏭️ Skipped dependency installation")
    
    # Step 5: Set up environment
    logger.info("\n⚙️ Step 5/5: Setting up environment...")
    
    if not Path('.env').exists():
        import shutil
        shutil.copy('.env.example', '.env')
        logger.info("   ✅ Created .env file from template")
        logger.warning("   ⚠️ Please edit .env and add your HF_TOKEN!")
    else:
        logger.info("   ✅ .env file already exists")
    
    # Print next steps
    logger.info("""
╔═══════════════════════════════════════════════════════════════╗
║                  ✅ SETUP COMPLETE!                           ║
╚═══════════════════════════════════════════════════════════════╝

📝 NEXT STEPS:

1️⃣  Edit .env and add your Hugging Face token:
   HF_TOKEN=your_token_here

2️⃣  Prepare dataset (~10 minutes):
   python scripts/prepare_dataset.py
   python scripts/generate_rag_docs.py

3️⃣  Create evaluation benchmark:
   python scripts/evaluate_rag.py --create_benchmark

4️⃣  Train model (~18 hours on A100):
   python scripts/train_qlora.py \\
       --model_name ibm-granite/granite-3.1-8b-instruct \\
       --dataset_path data/enterprise_dataset.json \\
       --output_dir outputs/qlora_model

5️⃣  Merge and deploy:
   python scripts/merge_and_push.py \\
       --adapter_path outputs/qlora_model \\
       --output_path outputs/merged_model

6️⃣  Launch demo:
   python inference/gradio_demo.py \\
       --model_path outputs/merged_model

📚 For more details, see README.md

🌟 Star the repo if you find it useful!
    """)


if __name__ == "__main__":
    main()
