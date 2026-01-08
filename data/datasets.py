# Copyright (c) 2024 SCALE Lab, Brown University
# Licensed under the MIT License (see LICENSE for details).

"""
Dataset utilities for PoliTune preference datasets.

These datasets are hosted on Hugging Face Hub:
- scale-lab/politune-left: Left-leaning preference dataset
- scale-lab/politune-right: Right-leaning preference dataset
"""

from datasets import load_dataset, Dataset
from typing import Optional, Dict, Any


def load_politune_dataset(
    dataset_name: str = "scale-lab/politune-left",
    split: str = "train",
) -> Dataset:
    """
    Load a PoliTune preference dataset from Hugging Face Hub.
    
    Args:
        dataset_name: Name of the dataset on HuggingFace Hub.
                     Either "scale-lab/politune-left" or "scale-lab/politune-right"
        split: Dataset split to load (default: "train")
    
    Returns:
        HuggingFace Dataset object with columns: prompt, chosen, rejected
    """
    dataset = load_dataset(dataset_name, split=split)
    return dataset


def load_politune_left(split: str = "train") -> Dataset:
    """Load the left-leaning PoliTune preference dataset."""
    return load_politune_dataset("scale-lab/politune-left", split=split)


def load_politune_right(split: str = "train") -> Dataset:
    """Load the right-leaning PoliTune preference dataset."""
    return load_politune_dataset("scale-lab/politune-right", split=split)


def format_dpo_dataset(
    dataset: Dataset,
    prompt_column: str = "prompt",
    chosen_column: str = "chosen",
    rejected_column: str = "rejected",
) -> Dataset:
    """
    Format a dataset for DPO training.
    
    Ensures the dataset has the required columns: prompt, chosen, rejected
    
    Args:
        dataset: Input dataset
        prompt_column: Name of the prompt column in input dataset
        chosen_column: Name of the chosen response column
        rejected_column: Name of the rejected response column
    
    Returns:
        Formatted dataset ready for DPOTrainer
    """
    # Check if columns need to be renamed
    column_mapping = {}
    
    if prompt_column != "prompt" and prompt_column in dataset.column_names:
        column_mapping[prompt_column] = "prompt"
    
    if chosen_column != "chosen" and chosen_column in dataset.column_names:
        column_mapping[chosen_column] = "chosen"
    
    if rejected_column != "rejected" and rejected_column in dataset.column_names:
        column_mapping[rejected_column] = "rejected"
    
    # Apply column renaming if needed
    if column_mapping:
        for old_name, new_name in column_mapping.items():
            dataset = dataset.rename_column(old_name, new_name)
    
    # Verify required columns exist
    required_columns = ["prompt", "chosen", "rejected"]
    missing_columns = [col for col in required_columns if col not in dataset.column_names]
    
    if missing_columns:
        raise ValueError(
            f"Dataset is missing required columns: {missing_columns}. "
            f"Available columns: {dataset.column_names}"
        )
    
    return dataset


def format_sft_dataset(
    dataset: Dataset,
    instruction_column: str = "instruction",
    input_column: str = "input",
    output_column: str = "output",
    text_column: Optional[str] = None,
) -> Dataset:
    """
    Format a dataset for SFT training.
    
    If the dataset has instruction/input/output columns, formats them into a 
    single text column. If text_column is specified and exists, uses that directly.
    
    Args:
        dataset: Input dataset
        instruction_column: Name of the instruction column
        input_column: Name of the input column (optional in the data)
        output_column: Name of the output column
        text_column: Name of existing text column to use directly
    
    Returns:
        Formatted dataset with a 'text' column ready for SFTTrainer
    """
    # If text column already exists and is specified, return as is
    if text_column and text_column in dataset.column_names:
        if text_column != "text":
            dataset = dataset.rename_column(text_column, "text")
        return dataset
    
    # Format from instruction/input/output columns
    def format_example(example: Dict[str, Any]) -> Dict[str, str]:
        instruction = example.get(instruction_column, "")
        input_text = example.get(input_column, "")
        output = example.get(output_column, "")
        
        if input_text:
            text = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output}"
        else:
            text = f"### Instruction:\n{instruction}\n\n### Response:\n{output}"
        
        return {"text": text}
    
    # Apply formatting
    dataset = dataset.map(
        format_example,
        remove_columns=dataset.column_names,
    )
    
    return dataset
