# Copyright (c) 2024 SCALE Lab, Brown University
# Licensed under the MIT License (see LICENSE for details).

"""
PoliTune Fine-tuning Module

This module provides scripts for fine-tuning LLMs using DPO and SFT
with the PoliTune preference datasets.
"""

from finetune.utils import (
    pc_instruction,
    pc_questions_txt_file,
    custom_prompts,
    format_instruction,
    eval_pc_hf,
    eval_custom_prompts_hf,
)

__all__ = [
    "pc_instruction",
    "pc_questions_txt_file", 
    "custom_prompts",
    "format_instruction",
    "eval_pc_hf",
    "eval_custom_prompts_hf",
]

