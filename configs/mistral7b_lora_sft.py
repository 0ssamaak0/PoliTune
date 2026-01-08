# Config for LoRA SFT fine-tuning with Mistral 7B
#
# Usage:
#   python -m finetune.base_finetune \
#       --model_name_or_path mistralai/Mistral-7B-Instruct-v0.2 \
#       --dataset_name yahma/alpaca-cleaned \
#       --output_dir outputs/mistral-7b-sft \
#       --bf16 \
#       --gradient_checkpointing
#
# Or import this config in your script:
#   from configs.mistral7b_lora_sft import CONFIG

CONFIG = {
    # Model
    "model_name_or_path": "mistralai/Mistral-7B-Instruct-v0.2",
    
    # Dataset
    "dataset_name": "yahma/alpaca-cleaned",
    "dataset_text_field": "text",
    "max_seq_length": 1024,
    
    # LoRA
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "lora_target_modules": ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    
    # Training
    "output_dir": "outputs/mistral-7b-sft",
    "num_train_epochs": 2,
    "per_device_train_batch_size": 8,
    "gradient_accumulation_steps": 8,
    "learning_rate": 3e-4,
    "weight_decay": 0.01,
    "warmup_steps": 100,
    "lr_scheduler_type": "cosine",
    "seed": 42,
    
    # Precision
    "bf16": True,
    "fp16": False,
    "gradient_checkpointing": True,
    
    # Logging
    "logging_steps": 1,
    "save_steps": 500,
    "save_total_limit": 3,
    
    # Evaluation
    "eval_freq": 64,
    "max_generated_tokens": 300,
    "temperature": 0.3,
    "top_k": 200,
}


def get_training_args():
    """Get training arguments as a list for command line."""
    args = []
    for key, value in CONFIG.items():
        if isinstance(value, bool):
            if value:
                args.append(f"--{key}")
        elif isinstance(value, list):
            args.append(f"--{key}")
            args.extend([str(v) for v in value])
        else:
            args.extend([f"--{key}", str(value)])
    return args


if __name__ == "__main__":
    # Print command line arguments
    print("Training command:")
    print("python -m finetune.base_finetune \\")
    for key, value in CONFIG.items():
        if isinstance(value, bool):
            if value:
                print(f"    --{key} \\")
        elif isinstance(value, list):
            print(f"    --{key} {' '.join(str(v) for v in value)} \\")
        else:
            print(f"    --{key} {value} \\")

