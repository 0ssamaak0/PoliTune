# PoliTune: Analyzing the Impact of Data Selection and Fine-Tuning on Economic and Political Biases in Large Language Models

This repository provides training scripts for fine-tuning LLMs using our preference datasets as described in the [paper](https://arxiv.org/abs/2404.08699).

## Dataset

The datasets are hosted on Hugging Face Hub. There are two preference datasets:
- [Left-leaning preference dataset](https://huggingface.co/datasets/scale-lab/politune-left)
- [Right-leaning preference dataset](https://huggingface.co/datasets/scale-lab/politune-right)

## Repository Structure

- `configs/` - Contains Python configuration files for different models and training methods
- `data/` - Contains dataset loading utilities
- `finetune/` - Contains fine-tuning scripts (DPO and SFT)

## Installation

```bash
pip install -r requirements.txt
```

### Dependencies

The codebase uses the Hugging Face ecosystem:
- [transformers](https://huggingface.co/docs/transformers) - For model loading and tokenization
- [trl](https://huggingface.co/docs/trl) - For DPO and SFT training
- [peft](https://huggingface.co/docs/peft) - For LoRA adapters
- [datasets](https://huggingface.co/docs/datasets) - For dataset loading

## Fine-Tuning the Model

### DPO Fine-Tuning (Preference Learning)

To fine-tune a model using Direct Preference Optimization (DPO) on the PoliTune datasets:

#### Llama 3 8B

```bash
# Left-leaning fine-tuning
python -m finetune.dpo_finetune \
    --model_name_or_path meta-llama/Meta-Llama-3-8B-Instruct \
    --dataset_name scale-lab/politune-left \
    --output_dir outputs/llama3-8b-dpo-left \
    --num_train_epochs 4 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 16 \
    --learning_rate 5e-4 \
    --lora_r 16 \
    --lora_alpha 32 \
    --bf16 \
    --gradient_checkpointing

# Right-leaning fine-tuning
python -m finetune.dpo_finetune \
    --model_name_or_path meta-llama/Meta-Llama-3-8B-Instruct \
    --dataset_name scale-lab/politune-right \
    --output_dir outputs/llama3-8b-dpo-right \
    --num_train_epochs 4 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 16 \
    --learning_rate 5e-4 \
    --lora_r 16 \
    --lora_alpha 32 \
    --bf16 \
    --gradient_checkpointing
```

#### Mistral 7B

```bash
# Left-leaning fine-tuning
python -m finetune.dpo_finetune \
    --model_name_or_path mistralai/Mistral-7B-Instruct-v0.2 \
    --dataset_name scale-lab/politune-left \
    --output_dir outputs/mistral-7b-dpo-left \
    --num_train_epochs 4 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 16 \
    --learning_rate 5e-4 \
    --lora_r 16 \
    --lora_alpha 32 \
    --bf16 \
    --gradient_checkpointing

# Right-leaning fine-tuning
python -m finetune.dpo_finetune \
    --model_name_or_path mistralai/Mistral-7B-Instruct-v0.2 \
    --dataset_name scale-lab/politune-right \
    --output_dir outputs/mistral-7b-dpo-right \
    --num_train_epochs 4 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 16 \
    --learning_rate 5e-4 \
    --lora_r 16 \
    --lora_alpha 32 \
    --bf16 \
    --gradient_checkpointing
```

### SFT Fine-Tuning (Supervised Fine-Tuning)

For standard supervised fine-tuning:

```bash
python -m finetune.base_finetune \
    --model_name_or_path meta-llama/Meta-Llama-3-8B-Instruct \
    --dataset_name yahma/alpaca-cleaned \
    --output_dir outputs/llama3-8b-sft \
    --num_train_epochs 2 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 8 \
    --learning_rate 3e-4 \
    --lora_r 16 \
    --lora_alpha 32 \
    --bf16 \
    --gradient_checkpointing
```

### QLoRA (4-bit Quantization)

To train with reduced memory using 4-bit quantization:

```bash
python -m finetune.dpo_finetune \
    --model_name_or_path meta-llama/Meta-Llama-3-8B-Instruct \
    --dataset_name scale-lab/politune-left \
    --output_dir outputs/llama3-8b-dpo-left-qlora \
    --use_4bit \
    --bf16 \
    --gradient_checkpointing
```

### Key Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--model_name_or_path` | HuggingFace model ID or path | Required |
| `--dataset_name` | Dataset to use for training | `scale-lab/politune-left` |
| `--output_dir` | Directory to save outputs | `./outputs` |
| `--num_train_epochs` | Number of training epochs | 4 (DPO), 2 (SFT) |
| `--per_device_train_batch_size` | Batch size per GPU | 4 |
| `--gradient_accumulation_steps` | Gradient accumulation steps | 16 |
| `--learning_rate` | Learning rate | 5e-4 |
| `--lora_r` | LoRA rank | 16 |
| `--lora_alpha` | LoRA alpha | 32 |
| `--beta` | DPO beta parameter | 0.1 |
| `--bf16` | Use bfloat16 precision | False |
| `--use_4bit` | Use 4-bit quantization (QLoRA) | False |
| `--gradient_checkpointing` | Enable gradient checkpointing | False |
| `--use_wandb` | Enable Weights & Biases logging | False |

### Logging with Weights & Biases

To enable W&B logging:

```bash
python -m finetune.dpo_finetune \
    --model_name_or_path meta-llama/Meta-Llama-3-8B-Instruct \
    --dataset_name scale-lab/politune-left \
    --output_dir outputs/llama3-8b-dpo-left \
    --use_wandb \
    --wandb_project politune \
    --wandb_run_name llama3-dpo-left \
    --bf16 \
    --gradient_checkpointing
```

## Configuration Files

Pre-configured settings are available in `configs/`:

- `llama8b_lora_dpo.py` - Llama 3 8B DPO configuration
- `llama8b_lora_sft.py` - Llama 3 8B SFT configuration  
- `mistral7b_lora_dpo.py` - Mistral 7B DPO configuration
- `mistral7b_lora_sft.py` - Mistral 7B SFT configuration

You can view the full command for a config by running:

```bash
python configs/llama8b_lora_dpo.py
```

## Citation

If you use this codebase or the datasets in your work, please cite our paper:

```bibtex
@inproceedings{agiza2024politune,
  title={PoliTune: Analyzing the Impact of Data Selection and Fine-Tuning on Economic and Political Biases in Large Language Models},
  author={Agiza, Ahmed and Mostagir, Mohamed and Reda, Sherief},
  booktitle={Proceedings of the 2024 AAAI/ACM Conference on AI, Ethics, and Society},
  pages={},
  year={2024}
}
```

## License

This repository contains code under two licenses:
- Original code written by SCALE Lab is licensed under the MIT License (see [LICENSE](LICENSE)).
- Files derived from Meta Platforms' code are licensed under the BSD-3-Clause License (see [LICENSE-BSD](LICENSE-BSD)).
