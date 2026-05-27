@echo off
:: Windows Batch script for OmniVoice PEFT/LoRA fine-tuning.
:: Optimized for consumer GPUs using SDPA and 1 GPU.

setlocal enabledelayedexpansion

:: ====== Modify as needed ======
:: GPU to use (usually "0" for single consumer GPU)
set GPU_IDS=0
set NUM_GPUS=1

:: Path to your input JSONL files
set TRAIN_JSONL=megu.jsonl
set DEV_JSONL=

:: Directory to write tokenized WebDataset shards
set TOKEN_DIR=..\..\..\omnivoice_finetune\tokens

:: Audio tokenizer model (HuggingFace repo or local path)
set TOKENIZER_PATH=eustlb/higgs-audio-v2-tokenizer

:: Training config file
set TRAIN_CONFIG=config\train_config_finetune_lora.json

:: Data config file
set DATA_CONFIG=config\data_config_finetune.json

:: Output directory for fine-tuned checkpoints
set OUTPUT_DIR=exp\omnivoice_finetune_lora

:: Set stages to run (0: Tokenize, 1: Fine-tune)
set STAGE=0
set STOP_STAGE=1
:: =================================

:: Set PYTHONPATH to project root
set SCRIPT_DIR=%~dp0
set PROJECT_ROOT=%SCRIPT_DIR%..
set PYTHONPATH=%PROJECT_ROOT%;%PYTHONPATH%

echo ========================================================
echo Starting OmniVoice LoRA Fine-Tuning Pipeline (Windows)
echo ========================================================

:: Stage 0: Tokenize audio into WebDataset shards
if %STAGE% LEQ 0 (
    if %STOP_STAGE% GEQ 0 (
        echo Stage 0: Tokenizing audio...
        
        if exist "%TRAIN_JSONL%" (
            echo   Tokenizing train split from %TRAIN_JSONL%
            set CUDA_VISIBLE_DEVICES=%GPU_IDS%
            python -m omnivoice.scripts.extract_audio_tokens ^
                --input_jsonl "%TRAIN_JSONL%" ^
                --tar_output_pattern "%TOKEN_DIR%\train\audios\shard-%%06d.tar" ^
                --jsonl_output_pattern "%TOKEN_DIR%\train\txts\shard-%%06d.jsonl" ^
                --tokenizer_path "%TOKENIZER_PATH%" ^
                --nj_per_gpu 3 ^
                --shuffle True
            echo   Done. Train manifest written to %TOKEN_DIR%\train\data.lst
        ) else (
            echo   WARNING: Train JSONL file not found at %TRAIN_JSONL%. Skipping tokenization.
        )
        
        if exist "%DEV_JSONL%" (
            echo   Tokenizing dev split from %DEV_JSONL%
            set CUDA_VISIBLE_DEVICES=%GPU_IDS%
            python -m omnivoice.scripts.extract_audio_tokens ^
                --input_jsonl "%DEV_JSONL%" ^
                --tar_output_pattern "%TOKEN_DIR%\dev\audios\shard-%%06d.tar" ^
                --jsonl_output_pattern "%TOKEN_DIR%\dev\txts\shard-%%06d.jsonl" ^
                --tokenizer_path "%TOKENIZER_PATH%" ^
                --nj_per_gpu 3 ^
                --shuffle True
            echo   Done. Dev manifest written to %TOKEN_DIR%\dev\data.lst
        )
    )
)

:: Stage 1: Fine-tune using LoRA CLI entry point
if %STAGE% LEQ 1 (
    if %STOP_STAGE% GEQ 1 (
        echo Stage 1: Fine-tuning with LoRA on LLM Backbone...
        
        set CUDA_VISIBLE_DEVICES=%GPU_IDS%
        accelerate launch ^
            --gpu_ids "%GPU_IDS%" ^
            --num_processes %NUM_GPUS% ^
            -m omnivoice.cli.train_lora ^
            --train_config "%TRAIN_CONFIG%" ^
            --data_config "%DATA_CONFIG%" ^
            --output_dir "%OUTPUT_DIR%"
    )
)

echo Pipeline finished.
pause
