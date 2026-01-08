# Copyright (c) 2024 SCALE Lab, Brown University
# Licensed under the MIT License (see LICENSE for details).

import os
import sys
import csv
import argparse
import torch
import logging
from typing import Optional
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTConfig, SFTTrainer

from finetune.utils import (
    pc_instruction,
    pc_questions_txt_file,
    custom_prompts,
    eval_pc_hf,
    eval_custom_prompts_hf,
    setup_logging,
)

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="SFT Fine-tuning with LoRA")
    
    # Model arguments
    parser.add_argument("--model_name_or_path", type=str, required=True,
                        help="Path to pretrained model or model identifier from huggingface.co/models")
    parser.add_argument("--tokenizer_name", type=str, default=None,
                        help="Pretrained tokenizer name or path if different from model")
    
    # Dataset arguments
    parser.add_argument("--dataset_name", type=str, default="yahma/alpaca-cleaned",
                        help="The dataset to use for SFT")
    parser.add_argument("--dataset_text_field", type=str, default="text",
                        help="The field containing the text to train on")
    parser.add_argument("--max_seq_length", type=int, default=1024,
                        help="Maximum sequence length")
    
    # LoRA arguments
    parser.add_argument("--lora_r", type=int, default=16,
                        help="LoRA attention dimension (rank)")
    parser.add_argument("--lora_alpha", type=int, default=32,
                        help="LoRA alpha parameter")
    parser.add_argument("--lora_dropout", type=float, default=0.05,
                        help="LoRA dropout probability")
    parser.add_argument("--lora_target_modules", type=str, nargs="+",
                        default=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                        help="Target modules for LoRA")
    
    # Training arguments
    parser.add_argument("--output_dir", type=str, default="./outputs",
                        help="Output directory for model and logs")
    parser.add_argument("--num_train_epochs", type=int, default=2,
                        help="Number of training epochs")
    parser.add_argument("--max_steps", type=int, default=-1,
                        help="Maximum number of training steps (-1 for full training)")
    parser.add_argument("--per_device_train_batch_size", type=int, default=8,
                        help="Batch size per device during training")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8,
                        help="Number of gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=3e-4,
                        help="Initial learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="Weight decay coefficient")
    parser.add_argument("--warmup_steps", type=int, default=100,
                        help="Number of warmup steps")
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine",
                        help="Learning rate scheduler type")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    
    # Precision and optimization
    parser.add_argument("--bf16", action="store_true",
                        help="Use bfloat16 precision")
    parser.add_argument("--fp16", action="store_true",
                        help="Use float16 precision")
    parser.add_argument("--gradient_checkpointing", action="store_true",
                        help="Enable gradient checkpointing")
    parser.add_argument("--use_4bit", action="store_true",
                        help="Use 4-bit quantization (QLoRA)")
    parser.add_argument("--use_8bit", action="store_true",
                        help="Use 8-bit quantization")
    
    # Logging and saving
    parser.add_argument("--logging_steps", type=int, default=1,
                        help="Log every N steps")
    parser.add_argument("--save_steps", type=int, default=500,
                        help="Save checkpoint every N steps")
    parser.add_argument("--save_total_limit", type=int, default=3,
                        help="Limit total number of checkpoints")
    
    # Evaluation
    parser.add_argument("--eval_freq", type=int, default=64,
                        help="Evaluation frequency for political compass")
    parser.add_argument("--max_generated_tokens", type=int, default=300,
                        help="Maximum tokens to generate during evaluation")
    parser.add_argument("--temperature", type=float, default=0.3,
                        help="Temperature for generation")
    parser.add_argument("--top_k", type=int, default=200,
                        help="Top-k for generation")
    
    # Resume training
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                        help="Path to checkpoint to resume from")
    
    # WandB
    parser.add_argument("--use_wandb", action="store_true",
                        help="Use Weights & Biases for logging")
    parser.add_argument("--wandb_project", type=str, default="politune",
                        help="WandB project name")
    parser.add_argument("--wandb_run_name", type=str, default=None,
                        help="WandB run name")
    
    return parser.parse_args()


def get_quantization_config(args):
    """Get quantization config if using 4-bit or 8-bit."""
    if args.use_4bit:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if args.bf16 else torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    elif args.use_8bit:
        return BitsAndBytesConfig(load_in_8bit=True)
    return None


def get_peft_config(args):
    """Get PEFT/LoRA configuration."""
    return LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.lora_target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )


def load_model_and_tokenizer(args):
    """Load model and tokenizer with optional quantization."""
    logger.info(f"Loading model: {args.model_name_or_path}")
    
    quantization_config = get_quantization_config(args)
    
    # Determine dtype
    if args.bf16:
        torch_dtype = torch.bfloat16
    elif args.fp16:
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        quantization_config=quantization_config,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
        device_map="auto" if quantization_config else None,
    )
    
    # Prepare model for k-bit training if using quantization
    if quantization_config:
        model = prepare_model_for_kbit_training(model)
    
    # Enable gradient checkpointing if requested
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    
    # Load tokenizer
    tokenizer_name = args.tokenizer_name or args.model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        trust_remote_code=True,
    )
    
    # Set padding token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Set padding side to right for training
    tokenizer.padding_side = "right"
    
    return model, tokenizer


def load_and_prepare_dataset(args, tokenizer):
    """Load and prepare the SFT dataset."""
    logger.info(f"Loading dataset: {args.dataset_name}")
    
    dataset = load_dataset(args.dataset_name, split="train")
    
    # Format dataset for instruction tuning if needed
    def format_instruction(example):
        """Format the example as an instruction-following prompt."""
        if "instruction" in example and "output" in example:
            # Alpaca-style dataset
            if example.get("input", ""):
                text = f"### Instruction:\n{example['instruction']}\n\n### Input:\n{example['input']}\n\n### Response:\n{example['output']}"
            else:
                text = f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['output']}"
            return {"text": text}
        elif "text" in example:
            return {"text": example["text"]}
        else:
            # Try to find any text field
            for key in ["content", "message", "prompt", "completion"]:
                if key in example:
                    return {"text": example[key]}
            raise ValueError(f"Could not find text field in dataset. Available fields: {list(example.keys())}")
    
    # Check if we need to format the dataset
    if args.dataset_text_field not in dataset.column_names:
        logger.info("Formatting dataset for instruction tuning")
        dataset = dataset.map(format_instruction, remove_columns=dataset.column_names)
    
    logger.info(f"Dataset loaded with {len(dataset)} examples")
    logger.info(f"Dataset columns: {dataset.column_names}")
    
    return dataset


def setup_evaluation_tracking(args):
    """Set up CSV files for tracking evaluation results."""
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load PC questions
    pc_questions = []
    with open(pc_questions_txt_file, "r") as f:
        for line in f:
            pc_questions.append(line.strip())
    
    # Set up PC CSV
    pc_csv_file = os.path.join(args.output_dir, "pc.csv")
    pc_headers = ['iteration', 'step'] + [f"question_{i}" for i in range(len(pc_questions))]
    with open(pc_csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(pc_headers)
    
    # Set up custom prompts CSV
    custom_prompts_file = os.path.join(args.output_dir, "custom_instrs.csv")
    headers = ['iteration', 'step'] + [f"prompt_{i}" for i in range(len(custom_prompts))]
    with open(custom_prompts_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
    
    return pc_questions, pc_csv_file, custom_prompts_file


def main():
    args = parse_args()
    
    # Setup logging
    setup_logging()
    logger.info("Starting SFT fine-tuning")
    logger.info(f"Arguments: {args}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(args)
    
    # Get LoRA config
    peft_config = get_peft_config(args)
    logger.info(f"LoRA config: r={args.lora_r}, alpha={args.lora_alpha}")
    
    # Load dataset
    dataset = load_and_prepare_dataset(args, tokenizer)
    
    # Setup evaluation tracking
    pc_questions, pc_csv_file, custom_prompts_file = setup_evaluation_tracking(args)
    
    # Configure SFT training
    sft_config = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps if args.max_steps > 0 else -1,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        lr_scheduler_type=args.lr_scheduler_type,
        seed=args.seed,
        bf16=args.bf16,
        fp16=args.fp16,
        gradient_checkpointing=args.gradient_checkpointing,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        max_seq_length=args.max_seq_length,
        dataset_text_field="text",
        report_to="wandb" if args.use_wandb else "none",
        run_name=args.wandb_run_name,
    )
    
    # Initialize trainer
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    
    # Run initial evaluation
    logger.info("Running initial evaluation")
    eval_pc_hf(
        model=trainer.model,
        tokenizer=tokenizer,
        pc_questions=pc_questions,
        pc_csv_file=pc_csv_file,
        max_generated_tokens=args.max_generated_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        iteration=0,
        step=0,
    )
    eval_custom_prompts_hf(
        model=trainer.model,
        tokenizer=tokenizer,
        custom_prompts=custom_prompts,
        custom_prompts_file=custom_prompts_file,
        max_generated_tokens=args.max_generated_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        iteration=0,
        step=0,
    )
    
    # Train
    logger.info("Starting training")
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    
    # Save final model
    logger.info(f"Saving final model to {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    # Run final evaluation
    logger.info("Running final evaluation")
    eval_pc_hf(
        model=trainer.model,
        tokenizer=tokenizer,
        pc_questions=pc_questions,
        pc_csv_file=pc_csv_file,
        max_generated_tokens=args.max_generated_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        iteration=args.num_train_epochs,
        step=-1,
    )
    eval_custom_prompts_hf(
        model=trainer.model,
        tokenizer=tokenizer,
        custom_prompts=custom_prompts,
        custom_prompts_file=custom_prompts_file,
        max_generated_tokens=args.max_generated_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        iteration=args.num_train_epochs,
        step=-1,
    )
    
    logger.info("Training completed!")


if __name__ == "__main__":
    main()
