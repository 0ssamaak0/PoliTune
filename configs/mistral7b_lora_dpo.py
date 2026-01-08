# Config for LoRA DPO fine-tuning with Mistral 7B
#
# Usage:
#   python -m finetune.dpo_finetune \
#       --model_name_or_path mistralai/Mistral-7B-Instruct-v0.2 \
#       --dataset_name scale-lab/politune-left \
#       --output_dir outputs/mistral-7b-dpo-left \
#       --bf16 \
#       --gradient_checkpointing
#
# Or import this config in your script:
#   from configs.mistral7b_lora_dpo import CONFIG

CONFIG = {
    # Model
    "model_name_or_path": "mistralai/Mistral-7B-Instruct-v0.2",
    
    # Dataset
    "dataset_name": "scale-lab/politune-left",  # or "scale-lab/politune-right"
    "max_seq_length": 1024,
    "max_prompt_length": 512,
    
    # LoRA
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "lora_target_modules": ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    
    # DPO
    "beta": 0.1,
    "loss_type": "sigmoid",
    "label_smoothing": 0.0,
    
    # Training
    "output_dir": "outputs/mistral-7b-dpo",
    "num_train_epochs": 4,
    "per_device_train_batch_size": 4,
    "gradient_accumulation_steps": 16,
    "learning_rate": 5e-4,
    "weight_decay": 0.05,
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
    print("python -m finetune.dpo_finetune \\")
    for key, value in CONFIG.items():
        if isinstance(value, bool):
            if value:
                print(f"    --{key} \\")
        elif isinstance(value, list):
            print(f"    --{key} {' '.join(str(v) for v in value)} \\")
        else:
            print(f"    --{key} {value} \\")

