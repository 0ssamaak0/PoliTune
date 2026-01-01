# Portions Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the BSD-style license found in LICENSE-BSD.
#
# Modifications Copyright (c) SCALE Lab, Brown University.
# These modifications are licensed under the MIT license (see LICENSE).

"""
LoRA fine-tuning script using HuggingFace Transformers, PEFT, and TRL.
"""

import argparse
import csv
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    set_seed,
)
from trl import SFTTrainer, SFTConfig

from finetune.utils import (
    pc_instruction,
    pc_questions_txt_file,
    custom_prompts,
    format_instruction_alpaca,
    eval_pc,
    eval_custom_prompts,
)
from data.datasets import prepare_sft_dataset

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """Arguments for model configuration."""
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    torch_dtype: Optional[str] = field(
        default="bfloat16",
        metadata={"help": "Torch dtype for model (float32, float16, bfloat16)"}
    )
    use_flash_attention_2: bool = field(
        default=False,
        metadata={"help": "Whether to use Flash Attention 2"}
    )


@dataclass
class LoRAArguments:
    """Arguments for LoRA configuration."""
    lora_r: int = field(default=16, metadata={"help": "LoRA attention dimension"})
    lora_alpha: int = field(default=32, metadata={"help": "LoRA alpha parameter"})
    lora_dropout: float = field(default=0.05, metadata={"help": "LoRA dropout"})
    lora_target_modules: Optional[str] = field(
        default="q_proj,v_proj,k_proj,o_proj,gate_proj,up_proj,down_proj",
        metadata={"help": "Comma-separated list of target modules for LoRA"}
    )


@dataclass
class DataArguments:
    """Arguments for data configuration."""
    dataset_name: str = field(
        default="tatsu-lab/alpaca",
        metadata={"help": "HuggingFace dataset name"}
    )
    max_seq_length: int = field(
        default=1024,
        metadata={"help": "Maximum sequence length"}
    )
    max_samples: Optional[int] = field(
        default=None,
        metadata={"help": "Maximum number of samples to use"}
    )


@dataclass
class EvalArguments:
    """Arguments for evaluation configuration."""
    eval_freq: int = field(default=512, metadata={"help": "Evaluation frequency in steps"})
    max_generated_tokens: int = field(default=300, metadata={"help": "Max tokens to generate"})
    temperature: float = field(default=0.3, metadata={"help": "Sampling temperature"})
    top_k: int = field(default=200, metadata={"help": "Top-k sampling"})


def get_torch_dtype(dtype_str: str) -> torch.dtype:
    """Convert string dtype to torch dtype."""
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }
    return dtype_map.get(dtype_str.lower(), torch.bfloat16)


def setup_pc_evaluation(output_dir: str, tokenizer):
    """Setup political compass evaluation."""
    pc_questions = []
        with open(pc_questions_txt_file, "r") as f:
            for line in f:
            question = line.strip()
            if question:
                formatted = format_instruction_alpaca(pc_instruction, question)
                pc_questions.append(formatted)
    
    pc_csv_file = os.path.join(output_dir, "pc.csv")
    pc_headers = ['iteration', 'step'] + [f"question_{i}" for i in range(len(pc_questions))]
    with open(pc_csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(pc_headers)
    
    return pc_questions, pc_csv_file


def setup_custom_prompts_evaluation(output_dir: str, tokenizer):
    """Setup custom prompts evaluation."""
    formatted_prompts = [format_instruction_alpaca(q) for q in custom_prompts]
    
    custom_prompts_file = os.path.join(output_dir, "custom_instrs.csv")
    headers = ['iteration', 'step'] + [f"prompt_{i}" for i in range(len(formatted_prompts))]
    with open(custom_prompts_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
    
    return formatted_prompts, custom_prompts_file


def load_config(config_path: str) -> dict:
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="LoRA Fine-tuning with HuggingFace")
    parser.add_argument("--config", type=str, required=True, help="Path to config JSON file")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory (overrides config)")
    parser.add_argument("--dataset", type=str, default=None, help="Dataset name (overrides config)")
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Override config with command line args
    if args.output_dir:
        config["output_dir"] = args.output_dir
    if args.dataset:
        config["dataset"]["name"] = args.dataset
    
    # Extract config sections
    model_config = config.get("model", {})
    lora_config = config.get("lora", {})
    training_config = config.get("training", {})
    data_config = config.get("dataset", {})
    eval_config = config.get("evaluation", {})
    
    output_dir = config.get("output_dir", "outputs/sft")
    os.makedirs(output_dir, exist_ok=True)
    
    # Set seed
    seed = config.get("seed", 42)
    if seed is not None:
        set_seed(seed)
    
    log.info(f"Loading model: {model_config['name_or_path']}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_config["name_or_path"],
        trust_remote_code=True,
    )
    
    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Load model
    torch_dtype = get_torch_dtype(model_config.get("torch_dtype", "bfloat16"))
    model = AutoModelForCausalLM.from_pretrained(
        model_config["name_or_path"],
        torch_dtype=torch_dtype,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="flash_attention_2" if model_config.get("use_flash_attention_2", False) else None,
    )
    
    # Setup LoRA
    target_modules = lora_config.get("target_modules", ["q_proj", "v_proj"])
    if isinstance(target_modules, str):
        target_modules = [m.strip() for m in target_modules.split(",")]
    
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_config.get("r", 16),
        lora_alpha=lora_config.get("alpha", 32),
        lora_dropout=lora_config.get("dropout", 0.05),
        target_modules=target_modules,
        bias="none",
    )
    
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    # Load dataset
    dataset_name = data_config.get("name", "tatsu-lab/alpaca")
    max_seq_length = data_config.get("max_seq_length", 1024)
    max_samples = data_config.get("max_samples", None)
    
    log.info(f"Loading dataset: {dataset_name}")
    train_dataset = prepare_sft_dataset(
        source=dataset_name,
        tokenizer=tokenizer,
        max_seq_len=max_seq_length,
        max_samples=max_samples,
    )
    
    # Setup evaluation
    pc_questions, pc_csv_file = setup_pc_evaluation(output_dir, tokenizer)
    formatted_custom_prompts, custom_prompts_file = setup_custom_prompts_evaluation(output_dir, tokenizer)
    
    # Training arguments
    training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=training_config.get("epochs", 2),
        per_device_train_batch_size=training_config.get("batch_size", 4),
        gradient_accumulation_steps=training_config.get("gradient_accumulation_steps", 8),
        learning_rate=training_config.get("learning_rate", 3e-4),
        weight_decay=training_config.get("weight_decay", 0.01),
        warmup_steps=training_config.get("warmup_steps", 100),
        lr_scheduler_type=training_config.get("lr_scheduler_type", "cosine"),
        logging_steps=training_config.get("logging_steps", 10),
        save_steps=training_config.get("save_steps", 500),
        save_total_limit=training_config.get("save_total_limit", 3),
        bf16=torch_dtype == torch.bfloat16,
        fp16=torch_dtype == torch.float16,
        gradient_checkpointing=training_config.get("gradient_checkpointing", True),
        max_seq_length=max_seq_length,
        packing=False,
        dataset_text_field="text",
        seed=seed if seed is not None else 42,
        report_to=training_config.get("report_to", "none"),
    )
    
    # Initialize trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        peft_config=None,  # Already applied
    )
    
    # Run evaluation before training
    device = next(model.parameters()).device
    log.info("Running initial evaluation...")
    eval_pc(
        pc_questions=pc_questions,
        pc_csv_file=pc_csv_file,
        model=model,
        tokenizer=tokenizer,
        max_generated_tokens=eval_config.get("max_generated_tokens", 300),
        temperature=eval_config.get("temperature", 0.3),
        top_k=eval_config.get("top_k", 200),
        iteration=0,
        step=0,
        device=str(device),
    )
    eval_custom_prompts(
        custom_prompts=formatted_custom_prompts,
        custom_prompts_file=custom_prompts_file,
        model=model,
        tokenizer=tokenizer,
        max_generated_tokens=eval_config.get("max_generated_tokens", 300),
        temperature=eval_config.get("temperature", 0.3),
        top_k=eval_config.get("top_k", 200),
        iteration=0,
        step=0,
        device=str(device),
    )
    
    # Train
    log.info("Starting training...")
    trainer.train()
    
    # Save final model
    log.info("Saving final model...")
    trainer.save_model(os.path.join(output_dir, "final_model"))
    tokenizer.save_pretrained(os.path.join(output_dir, "final_model"))
    
    # Run final evaluation
    log.info("Running final evaluation...")
    eval_pc(
        pc_questions=pc_questions,
        pc_csv_file=pc_csv_file,
        model=model,
        tokenizer=tokenizer,
        max_generated_tokens=eval_config.get("max_generated_tokens", 300),
        temperature=eval_config.get("temperature", 0.3),
        top_k=eval_config.get("top_k", 200),
        iteration=training_config.get("epochs", 2),
        step=-1,
        device=str(device),
    )
    eval_custom_prompts(
        custom_prompts=formatted_custom_prompts,
        custom_prompts_file=custom_prompts_file,
        model=model,
        tokenizer=tokenizer,
        max_generated_tokens=eval_config.get("max_generated_tokens", 300),
        temperature=eval_config.get("temperature", 0.3),
        top_k=eval_config.get("top_k", 200),
        iteration=training_config.get("epochs", 2),
        step=-1,
        device=str(device),
    )
    
    log.info("Training complete!")


if __name__ == "__main__":
    main()
