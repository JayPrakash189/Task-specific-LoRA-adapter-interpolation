# Task-Specific Fine-Tuning with LoRA and Adapter Interpolation for Language Models

This project explores parameter-efficient fine-tuning of large language models using LoRA and investigates adapter merging using task arithmetic.

## Objective

To study how multiple LoRA adapters trained on different tasks can be merged using weight interpolation to enable skill composition while mitigating catastrophic interference.

## Methods

- LoRA fine-tuning on Alpaca-style instruction data
- Adapter merging using linear interpolation
- Alpha-weighted task arithmetic
- Comparative inference across adapters

## Experiments

Adapters trained:
- Adapter A → Instruction explanation
- Adapter B → Summarization

Merged using:
- alpha = 0.2
- alpha = 0.5
- alpha = 0.8

## Results

Observed behavioral shift:
- Low alpha → Task A dominant
- Mid alpha → Balanced response
- High alpha → Task B dominant

## Tech Stack

- HuggingFace Transformers
- PEFT
- BitsAndBytes (QLoRA)
- PyTorch
- Google Colab

## Research Inspiration

- LoRA (Hu et al., 2022)
- Task Arithmetic (Ilharco et al., 2023)
- Model Soups (Wortsman et al., 2022)
- TIES-Merging (Yadav et al., 2023)

## Future Work

- Spherical interpolation
