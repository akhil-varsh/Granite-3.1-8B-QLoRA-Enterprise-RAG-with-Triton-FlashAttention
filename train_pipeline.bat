@echo off
REM Complete training pipeline for Windows
REM Run this script to train from scratch

echo ==========================================
echo Enterprise RAG - Complete Training Pipeline
echo ==========================================

REM Step 1: Prepare datasets
echo.
echo Step 1/6: Preparing training dataset...
python scripts\prepare_dataset.py

echo.
echo Step 2/6: Generating RAG documents...
python scripts\generate_rag_docs.py

echo.
echo Step 3/6: Creating evaluation benchmark...
python scripts\evaluate_rag.py --create_benchmark

REM Step 2: Train model
echo.
echo Step 4/6: Training QLoRA model (~18 hours)...
python scripts\train_qlora.py ^
    --model_name ibm-granite/granite-3.1-8b-instruct ^
    --dataset_path data/enterprise_dataset.json ^
    --output_dir outputs/qlora_model ^
    --num_epochs 3 ^
    --batch_size 4 ^
    --gradient_accumulation_steps 4

REM Step 3: Merge adapters
echo.
echo Step 5/6: Merging LoRA adapters...
python scripts\merge_and_push.py ^
    --adapter_path outputs/qlora_model ^
    --output_path outputs/merged_model

REM Step 4: Evaluate
echo.
echo Step 6/6: Evaluating model...
python scripts\evaluate_rag.py ^
    --model_path outputs/merged_model ^
    --base_model_path ibm-granite/granite-3.1-8b-instruct ^
    --output_dir results

echo.
echo ==========================================
echo SUCCESS: Training pipeline complete!
echo ==========================================
echo.
echo Model saved to: outputs\merged_model
echo Results saved to: results\
echo.
echo To launch demo:
echo   python inference\gradio_demo.py --model_path outputs\merged_model
pause
