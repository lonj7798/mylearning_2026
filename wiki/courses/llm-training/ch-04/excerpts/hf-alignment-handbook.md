---
chapter: ch-04
course: llm-training
phase: read
excerpt_of: huggingface/alignment-handbook — Zephyr-7B-β SFT recipe config and scripts/sft.py
source_url: https://github.com/huggingface/alignment-handbook/tree/1de1fc996972aa76b7d40c64c07b66dec8b6976a
created_at: "2026-04-23"
revised_at: "2026-09-15 (generality revision; rewritten from the repository at commit 1de1fc9, 2026-04-08)"
---

# Excerpt: Alignment Handbook — Zephyr SFT config and training script

**Artifact:** github.com/huggingface/alignment-handbook at commit 1de1fc996972aa76b7d40c64c07b66dec8b6976a. Source type: released config/code. The files below may differ from the configuration that produced the 2023 Zephyr-7B-β release; older commits were not read.

**Correction to the library card.** `blogs/hf-alignment-handbook.md` (2026-07 version) shows `SFTConfig(train_on_response_only=True, ...)` and lists "packing: true" and "Train on response only: true" for Zephyr. TRL has no `train_on_response_only` parameter (its options are `completion_only_loss` and `assistant_only_loss`; see [[trl-sft-packing-masking]]), and the Zephyr config at this commit has neither a `packing` key nor a loss-mask key. The card's "#1 silent bug" lesson and its DPO table were not found in the files read.

## `recipes/zephyr-7b-beta/sft/config_full.yaml`

```yaml
# L2-L5
model_name_or_path: mistralai/Mistral-7B-v0.1
torch_dtype: bfloat16
attn_implementation: flash_attention_2
# L8 (chat template, shown with line breaks expanded)
chat_template: "{% for message in messages %}
{% if message['role'] == 'user' %}{{ '<|user|>\n' + message['content'] + eos_token }}
{% elif message['role'] == 'system' %}{{ '<|system|>\n' + message['content'] + eos_token }}
{% elif message['role'] == 'assistant' %}{{ '<|assistant|>\n'  + message['content'] + eos_token }}
{% endif %}
{% if loop.last and add_generation_prompt %}{{ '<|assistant|>' }}{% endif %}
{% endfor %}"
# L9-L22: datasets HuggingFaceH4/ultrachat_200k, splits train_sft and test_sft, columns [messages]
# L37-L57 (selected)
learning_rate: 2.0e-05
lr_scheduler_type: cosine
max_seq_length: 2048
num_train_epochs: 1
per_device_train_batch_size: 16
gradient_accumulation_steps: 1
warmup_ratio: 0.1
seed: 42
```

Keys absent from the file: `packing`, `assistant_only_loss`, `completion_only_loss`. Under TRL defaults at aa89588 (`packing=False`, `assistant_only_loss=False`), a conversational `messages` dataset is trained on the full sequence without packing (derived for ch-04; the handbook does not state it). `setup.py` requires `trl>=0.19.1` and `transformers>=4.53.3` without an upper pin (L68-69), so the effective defaults depend on the installed TRL version.

## `scripts/sft.py`

- The usage docstring example passes `--packing` with `--max_seq_length 4096` for Qwen2.5-1.5B-Instruct on `trl-lib/Capybara` (L21-35).
- When the tokenizer has no chat template, the script logs "No chat template provided, using ChatML." and calls `setup_chat_format(model, tokenizer, format="chatml")`, which adds ChatML tokens and resizes the model (L98-100).
- The trainer is `SFTTrainer(model, args, train_dataset, eval_dataset, processing_class=tokenizer, peft_config)` (L105-112).
