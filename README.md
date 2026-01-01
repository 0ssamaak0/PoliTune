# PoliTune: Analyzing the Impact of Data Selection and Fine-Tuning on Economic and Political Biases in Large Language Models

This repository provides training scripts for fine-tuning LLMs using our preference datasets as described in the [paper](https://arxiv.org/abs/2404.08699).

## Dataset

The datasets are hosted on Hugging Face Hub. There are two preference datasets:
- [Left-leaning preference dataset](https://huggingface.co/datasets/scale-lab/politune-left)
- [Right-leaning preference dataset](https://huggingface.co/datasets/scale-lab/politune-right)

## Repository Structure

- `configs/` - Contains JSON configuration files for training.
- `data/` - Contains dataset loading utilities.
- `finetune/` - Contains the fine-tuning scripts.

## Installation

```bash
pip install -r requirements.txt
```

For GPU support, ensure you have CUDA installed and install PyTorch with CUDA support:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

## Fine-Tuning the Model

### DPO (Direct Preference Optimization) Training

This is the main training approach used in the paper. DPO fine-tunes the model to prefer responses aligned with either left-leaning or right-leaning perspectives.

```bash
python -m finetune.dpo_finetune --config configs/llama8b_lora_dpo.json --dataset politune_left --output_dir outputs/llama8b_left
```

#### Available configurations:
- `configs/llama8b_lora_dpo.json` - Llama 3 8B with LoRA for DPO
- `configs/mistral7b_lora_dpo.json` - Mistral 7B with LoRA for DPO

#### Dataset options:
- `politune_left` - Left-leaning preference dataset
- `politune_right` - Right-leaning preference dataset

### Examples

**Fine-tune Llama 3 8B with left-leaning preferences:**
```bash
python -m finetune.dpo_finetune \
    --config configs/llama8b_lora_dpo.json \
    --dataset politune_left \
    --output_dir outputs/llama8b_left
```

**Fine-tune Llama 3 8B with right-leaning preferences:**
```bash
python -m finetune.dpo_finetune \
    --config configs/llama8b_lora_dpo.json \
    --dataset politune_right \
    --output_dir outputs/llama8b_right
```

**Fine-tune Mistral 7B with left-leaning preferences:**
```bash
python -m finetune.dpo_finetune \
    --config configs/mistral7b_lora_dpo.json \
    --dataset politune_left \
    --output_dir outputs/mistral7b_left
```

### SFT (Supervised Fine-Tuning) Training

For standard supervised fine-tuning:

```bash
python -m finetune.base_finetune --config configs/llama8b_lora_sft.json --output_dir outputs/llama8b_sft
```

#### Available configurations:
- `configs/llama8b_lora_sft.json` - Llama 3 8B with LoRA for SFT
- `configs/mistral7b_lora_sft.json` - Mistral 7B with LoRA for SFT

## Configuration

Configuration files are in JSON format. Key parameters:

```json
{
    "model": {
        "name_or_path": "meta-llama/Meta-Llama-3-8B-Instruct",
        "torch_dtype": "bfloat16",
        "use_flash_attention_2": false
    },
    "lora": {
        "r": 16,
        "alpha": 32,
        "dropout": 0.05,
        "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"]
    },
    "training": {
        "epochs": 4,
        "batch_size": 4,
        "gradient_accumulation_steps": 16,
        "learning_rate": 5e-4
    },
    "dpo": {
        "beta": 0.1,
        "loss_type": "sigmoid"
    }
}
```

### Key Parameters

| Parameter | Description |
|-----------|-------------|
| `model.name_or_path` | HuggingFace model identifier or local path |
| `lora.r` | LoRA rank (higher = more parameters) |
| `lora.alpha` | LoRA scaling factor |
| `training.epochs` | Number of training epochs |
| `training.batch_size` | Per-device batch size |
| `dpo.beta` | DPO beta parameter (temperature for preference learning) |

## Outputs

During training, the following outputs are generated in the output directory:
- `pc.csv` - Political compass evaluation results at each checkpoint
- `custom_instrs.csv` - Custom prompt evaluation results
- `final_model/` - The final fine-tuned model (LoRA adapters)
- Checkpoint directories with intermediate saves

## Loading the Fine-tuned Model

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3-8B-Instruct",
    torch_dtype="auto",
    device_map="auto"
)

# Load fine-tuned LoRA adapters
model = PeftModel.from_pretrained(base_model, "outputs/llama8b_left/final_model")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("outputs/llama8b_left/final_model")

# Generate
inputs = tokenizer("What is your opinion on economic policy?", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=200)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Citation

If you use this codebase or the datasets in your work, please cite our paper:

```bibtex
@inproceedings{agiza2024politune,
  title={PoliTune: Analyzing the Impact of Data Selection and Fine-Tuning on Economic and Political Biases in Large Language Models},
  author={Agiza, Ahmed and Mostagir, Mohamed and Reda, Sherief},
  booktitle={Proceedings of the 2024 AAAI/ACM Conference on AI, Ethics, and Society},
  year={2024}
}
```

## License

This repository contains code under two licenses:
- Original code written by SCALE Lab is licensed under the MIT License (see [LICENSE](LICENSE)).
- Files derived from Meta Platforms' code are licensed under the BSD-3-Clause License (see [LICENSE-BSD](LICENSE-BSD)).
