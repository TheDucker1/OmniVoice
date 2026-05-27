#!/usr/bin/env bash

# This script demonstrates how to fine-tune OmniVoice with PEFT/LoRA from a JSONL manifest.
# Optimized for consumer GPUs using SDPA and 1 GPU.

set -euo pipefail

stage=0
stop_stage=1

# ====== Modify as needed ======
# GPUs to use (optimised for single consumer GPU by default)
GPU_IDS="0"
NUM_GPUS=1

# Path to your input JSONL file
# (each line: {"id": ..., "audio_path": ..., "text": ..., "language_id": ...})
TRAIN_JSONL="data/my_data_train.jsonl"

# Path to your dev JSONL file. Set to empty string to skip dev set.
DEV_JSONL="data/my_data_dev.jsonl"

# Directory to write tokenized WebDataset shards
TOKEN_DIR="data/finetune/tokens"

# Audio tokenizer model (HuggingFace repo or local path)
TOKENIZER_PATH="eustlb/higgs-audio-v2-tokenizer"

# Training config file (pointing to our new LoRA config template)
TRAIN_CONFIG="config/train_config_finetune_lora.json"

# Data config file
data_config="config/data_config_finetune.json"

# Output directory for fine-tuned checkpoints
OUTPUT_DIR="exp/omnivoice_finetune_lora"
# =================================

export PYTHONPATH="$(cd "$(dirname "$0")/.." && pwd):${PYTHONPATH:-}"


# Stage 0: Tokenize audio into WebDataset shards
if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then
    echo "Stage 0: Tokenizing audio"

    for split_jsonl_path in ${TRAIN_JSONL} ${DEV_JSONL}; do
        if [ -z "${split_jsonl_path}" ]; then
            continue
        fi

        if [ "${split_jsonl_path}" = "${TRAIN_JSONL}" ]; then
            split="train"
        else
            split="dev"
        fi

        echo "  Tokenizing ${split} from ${split_jsonl_path}"

        CUDA_VISIBLE_DEVICES=${GPU_IDS} \
            python -m omnivoice.scripts.extract_audio_tokens \
            --input_jsonl "${split_jsonl_path}" \
            --tar_output_pattern "${TOKEN_DIR}/${split}/audios/shard-%06d.tar" \
            --jsonl_output_pattern "${TOKEN_DIR}/${split}/txts/shard-%06d.jsonl" \
            --tokenizer_path "${TOKENIZER_PATH}" \
            --nj_per_gpu 3 \
            --shuffle True

        echo "  Done. Manifest written to ${TOKEN_DIR}/${split}/data.lst"
    done
fi


# Stage 1: Fine-tune using LoRA CLI entry point
if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
    echo "Stage 1: Fine-tuning with LoRA on LLM Backbone"

    accelerate launch \
        --gpu_ids "${GPU_IDS}" \
        --num_processes ${NUM_GPUS} \
        -m omnivoice.cli.train_lora \
        --train_config ${TRAIN_CONFIG} \
        --data_config ${data_config} \
        --output_dir ${OUTPUT_DIR}
fi
