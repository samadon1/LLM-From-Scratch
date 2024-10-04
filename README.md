# Medical LLM Project

A Language Model fine-tuned for medical applications, progressing from pretraining to instruction fine-tuning and Direct Preference Optimization (DPO).

Inspired by [Sebastian Raschka]([https://github.com/rasbt/LLMs-from-scratch](https://github.com/rasbt)).

## Datasets

1. Pretraining: [Medical Text Dataset](https://www.kaggle.com/datasets/chaitanyakck/medical-text) (Kaggle)
2. Fine-tuning: [PMC LLaMA Instructions](https://huggingface.co/datasets/axiong/pmc_llama_instructions) (Hugging Face)

## Project Stages

1. **Pretraining**
   - Custom GPT model on medical texts
   - 2 epochs, AdamW optimizer

2. **Instruction Fine-tuning**
   - Used LitGPT for LoRA fine-tuning
   - 3 epochs on instruction dataset

3. **Direct Preference Optimization (DPO)**
   - Generated variants using fine-tuned model
   - Created preference pairs based on Levenshtein distance

## Key Features

- Customized for medical domain
- Progression from general language model to instruction-following
- Experiment with preference optimization

## Usage

```bash
# Pretraining
python pretrain.py

# Fine-tuning
python finetune.py

# DPO
python dpo_train.py
```

## Future Work

- Larger medical datasets
- Advanced DPO techniques
- Multi-task learning in medical domain

[Add license information here]
