# Copyright (c) 2024 SCALE Lab, Brown University
# Licensed under the MIT License (see LICENSE for details).

from datasets import load_dataset
from typing import Optional


def format_alpaca_prompt(example: dict, input_key: str = "prompt") -> str:
    """Format example using Alpaca-style template."""
    instruction = example.get(input_key, example.get("instruction", ""))
    input_text = example.get("input", "")
    
    if input_text:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input_text}

### Response:
"""
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
"""


def load_politune_dataset(
    source: str,
    split: str = "train",
    max_samples: Optional[int] = None,
):
    """
    Load a PoliTune preference dataset from HuggingFace Hub.
    
    Args:
        source: HuggingFace dataset identifier (e.g., "scale-lab/politune-left")
        split: Dataset split to load
        max_samples: Optional maximum number of samples to load
    
    Returns:
        HuggingFace Dataset object
    """
    dataset = load_dataset(source, split=split)
    
    if max_samples is not None:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    
    return dataset


def prepare_dpo_dataset(
    source: str,
    tokenizer=None,
    max_seq_len: int = 1024,
    split: str = "train",
    max_samples: Optional[int] = None,
):
    """
    Prepare dataset for DPO training.
    
    The dataset should have 'prompt', 'chosen', and 'rejected' columns.
    This function formats them appropriately for TRL's DPOTrainer.
    
    Args:
        source: HuggingFace dataset identifier
        tokenizer: Tokenizer (optional, for chat template formatting)
        max_seq_len: Maximum sequence length
        split: Dataset split
        max_samples: Optional maximum samples
    
    Returns:
        Formatted dataset ready for DPOTrainer
    """
    dataset = load_politune_dataset(source, split=split, max_samples=max_samples)
    
    def format_example(example):
        # Format the prompt using Alpaca template
        prompt = format_alpaca_prompt(example, input_key="prompt")
        
        return {
            "prompt": prompt,
            "chosen": example["chosen"],
            "rejected": example["rejected"],
        }
    
    formatted_dataset = dataset.map(
        format_example,
        remove_columns=[col for col in dataset.column_names if col not in ["prompt", "chosen", "rejected"]]
    )
    
    return formatted_dataset


def politune_right(
    tokenizer=None,
    source: str = "scale-lab/politune-right",
    max_seq_len: int = 1024,
    split: str = "train",
    max_samples: Optional[int] = None,
):
    """Load right-leaning preference dataset for DPO training."""
    return prepare_dpo_dataset(
        source=source,
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
        split=split,
        max_samples=max_samples,
    )


def politune_left(
    tokenizer=None,
    source: str = "scale-lab/politune-left",
    max_seq_len: int = 1024,
    split: str = "train",
    max_samples: Optional[int] = None,
):
    """Load left-leaning preference dataset for DPO training."""
    return prepare_dpo_dataset(
        source=source,
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
        split=split,
        max_samples=max_samples,
    )


def prepare_sft_dataset(
    source: str,
    tokenizer=None,
    max_seq_len: int = 1024,
    split: str = "train",
    max_samples: Optional[int] = None,
):
    """
    Prepare dataset for SFT training.
    
    Args:
        source: HuggingFace dataset identifier
        tokenizer: Tokenizer for formatting
        max_seq_len: Maximum sequence length
        split: Dataset split
        max_samples: Optional maximum samples
    
    Returns:
        Formatted dataset ready for SFTTrainer
    """
    dataset = load_dataset(source, split=split)
    
    if max_samples is not None:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    
    def format_example(example):
        # Get instruction and response
        instruction = example.get("instruction", example.get("prompt", ""))
        input_text = example.get("input", "")
        output = example.get("output", example.get("response", example.get("chosen", "")))
        
        # Format prompt
        prompt = format_alpaca_prompt({"prompt": instruction, "input": input_text})
        
        # Combine prompt and response
        text = prompt + output
        
        return {"text": text}
    
    formatted_dataset = dataset.map(format_example)
    
    return formatted_dataset
