---
language:
- ta
- en
license: apache-2.0
base_model: Qwen/Qwen3.5-2B
base_model_relation: finetune
pipeline_tag: text-generation
tags:
- math
- gsm8k
- tamil
- lora
- reinforcement-learning
- tapo
- transformers
library_name: transformers
---

# Qwen3.5-2B Tamil TAPO

Standalone BF16 Qwen3.5-2B model with the trained Tamil SFT + TAPO LoRA merged into the base weights. Uploaded from **RL checkpoint 1525**. The trainable LoRA adapter is also provided under `adapter/`.

## Training

Base: `Qwen/Qwen3.5-2B`. A fresh text-only LoRA (rank 16, alpha 32) was supervised on 2,000 Tamil GSM8K TRAIN questions paired with faithful original English questions and official worked solutions, for one epoch / 500 optimizer steps. Completion-only loss masks prompt tokens. A disjoint 128-question TRAIN validation subset selected the SFT checkpoint and was excluded from the subsequent RL pool.

RL initializes the same adapter from SFT and uses a local TAPO adaptation on TRL GRPO: separate group-normalized translation and correctness rewards, token-dependent translation/reasoning advantages, alpha 0.25, beta 0, DAPO loss, and 8 completions per prompt. Translation reward is ChrF++ against the original English question; correctness requires the correct boxed answer after a valid translation block. This is the project's adaptation, not an official TAPO implementation. Configuration is included in `training/tapo_config.yaml`.

The RL dataset contains 7,345 GSM8K TRAIN questions translated into Tamil. Official GSM8K test questions were not used for SFT, RL, or SFT checkpoint selection. Original worked solutions originate from `openai/gsm8k`; Tamil training questions come from the project's translated GSM8K training file. `Vishal0407/GSM8K_TAMIL` was used for evaluation only.

## Prompt and usage

Use the official Qwen chat template with hidden thinking disabled (`enable_thinking=False`). The expected response translates the question into English inside `<english_translation>...</english_translation>`, then reasons in English and ends with `\boxed{...}`.

```python
import torch
from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

model_id = "GL3MON/Qwen3.5-2B-Tamil-TAPO"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = Qwen3_5ForConditionalGeneration.from_pretrained(
    model_id, dtype=torch.bfloat16, device_map="auto"
)
messages = [
    {"role": "system", "content": 'Solve the mathematical word problem. First translate the complete question faithfully into English inside <english_translation>...</english_translation>. Preserve all quantities, units, and relationships. Then reason in English and give the final answer in \\boxed{...}. Do not put reasoning inside the translation tags.'},
    {"role": "user", "content": "ஒரு பெட்டியில் 12 பந்துகள் உள்ளன. மேலும் 8 பந்துகள் சேர்த்தால் மொத்தம் எத்தனை?"},
]
inputs = tokenizer.apply_chat_template(
    messages, tokenize=True, return_dict=True, return_tensors="pt",
    add_generation_prompt=True, enable_thinking=False,
).to(model.device)
with torch.inference_mode():
    output = model.generate(**inputs, do_sample=False, max_new_tokens=1536)
print(tokenizer.decode(output[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True))
```

Tested training/inference environment: PyTorch 2.11, Transformers 5.6.2, PEFT 0.20, TRL 1.9.2, and vLLM 0.22.0. Text inference is the intended use; only language-model LoRA targets were trained.

## Evaluation

**This uploaded checkpoint (1525) has not yet been evaluated.** The following results describe earlier checkpoints, not this upload:

| Checkpoint | Correct on fixed 500-question Tamil subset | Accuracy |
|---|---:|---:|
| 2,000-example SFT | 180/500 | 36.0% |
| RL step 1125 | 184/500 | 36.8% |
| RL step 1375 | 186/500 | 37.2% |

The subset uses seed 42 and 500 questions from the inferred official-test portion of `Vishal0407/GSM8K_TAMIL`, whose combined train/test ordering was checked against official final answers and calculation signatures. Evaluation uses one greedy answer per question and a 1,536-token completion cap. These are not full English GSM8K scores, Coverage@8, or combined retry scores. Translation quality can affect results, and the observed gains are small. Earlier checkpoint evaluation details are included in `earlier_checkpoint_evaluation.json`.

## License

Apache-2.0, following the base model's license. See the original [Qwen3.5-2B model card](https://huggingface.co/Qwen/Qwen3.5-2B).
