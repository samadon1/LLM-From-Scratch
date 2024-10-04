# Medical LLM Project

A Language Model fine-tuned for medical applications, progressing from pretraining to instruction fine-tuning and Direct Preference Optimization (DPO).

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

## Future Work

- Larger medical datasets
- Advanced DPO techniques
- Multi-task learning in medical domain

