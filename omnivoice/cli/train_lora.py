#!/usr/bin/env python3
# Copyright    2026  Xiaomi Corp.        (authors:  Han Zhu)
#
# See ../../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""LoRA Fine-tuning CLI for OmniVoice.

Launches distributed fine-tuning with PEFT LoRA on the LLM backbone
using HuggingFace Accelerate.

Usage:
    accelerate launch --gpu_ids 0 --num_processes 1 \
        -m omnivoice.cli.train_lora \
        --train_config train_config_finetune_sdpa.json \
        --data_config data_config_finetune.json \
        --output_dir output/
"""

import argparse
import json
import logging
import sys

from omnivoice.training.builder import build_dataloaders, build_model_and_tokenizer
from omnivoice.training.config import TrainingConfig
from omnivoice.training.trainer import OmniTrainer

logger = logging.getLogger("omnivoice.cli.train_lora")


def main():
    parser = argparse.ArgumentParser(description="OmniVoice LoRA Training Entry Point")
    parser.add_argument(
        "--train_config", type=str, required=True, help="Path to config JSON"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Where to save checkpoints"
    )
    parser.add_argument(
        "--data_config", type=str, required=True, help="Path to data config JSON"
    )
    # LoRA config overrides
    parser.add_argument(
        "--lora_r", type=int, default=16, help="Rank size of the LoRA adapter"
    )
    parser.add_argument(
        "--lora_alpha", type=int, default=32, help="Alpha multiplier for the LoRA adapter"
    )
    parser.add_argument(
        "--lora_dropout", type=float, default=0.05, help="Dropout rate for LoRA layers"
    )
    args = parser.parse_args()

    # 1. Load Configuration
    config = TrainingConfig.from_json(args.train_config)
    config.output_dir = args.output_dir
    config.data_config = args.data_config

    # Ensure SDPA is enforced for consumer GPUs if user configures it or we override
    if config.attn_implementation != "sdpa":
        logger.warning(
            "Flexible attention ('flex_attention') is usually aimed at datacenter-grade hardware. "
            "Forcing attention implementation to 'sdpa' for consumer-grade GPU compatibility."
        )
        config.attn_implementation = "sdpa"

    # 2. Build Base Model and Tokenizer (Loads original pre-trained weights)
    model, tokenizer = build_model_and_tokenizer(config)

    # 3. Apply PEFT LoRA
    logger.info("Initializing PEFT LoRA setup...")
    try:
        from peft import LoraConfig, get_peft_model
    except ImportError:
        logger.critical(
            "The 'peft' library is required for LoRA training. "
            "Please run: pip install peft"
        )
        sys.exit(1)

    # Load custom settings from the train config JSON if present, otherwise use CLI arguments
    with open(args.train_config, "r") as f:
        raw_config = json.load(f)

    lora_r = raw_config.get("lora_r", args.lora_r)
    lora_alpha = raw_config.get("lora_alpha", args.lora_alpha)
    lora_dropout = raw_config.get("lora_dropout", args.lora_dropout)

    # Target modules for Qwen3 backbone projection matrices
    target_modules = ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias="none",
    )

    logger.info(
        f"Applying LoRA wrapper to LLM backbone with rank={lora_r}, alpha={lora_alpha}."
    )
    model.llm = get_peft_model(model.llm, lora_config)

    # Print trainable parameters to verify
    model.llm.print_trainable_parameters()

    # 4. Build Data Loaders
    train_loader, eval_loader = build_dataloaders(config, tokenizer)

    # 5. Initialize Trainer and Start training
    trainer = OmniTrainer(
        model=model,
        config=config,
        train_dataloader=train_loader,
        eval_dataloader=eval_loader,
        tokenizer=tokenizer,
    )
    trainer.train()


if __name__ == "__main__":
    main()
