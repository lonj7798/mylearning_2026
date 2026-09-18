---
chapter: ch-08
course: llm-training
phase: read
excerpt_of: https://github.com/huggingface/alignment-handbook
source_url: https://github.com/huggingface/alignment-handbook
created_at: "2026-04-23"
revised: 2026-09 (generality revision)
---

# Excerpt: the Zephyr-7B SFT recipe as it is written in the alignment-handbook repository

**Artifact:** `huggingface/alignment-handbook` at commit `1de1fc9`; files
`recipes/zephyr-7b-beta/sft/config_full.yaml` and `scripts/sft.py`.
**Read on:** 2026-09-17 from the text cached at `scratchpad/sources/handbook-zephyr-sft-config-1de1fc9.txt`
and `scratchpad/sources/handbook-sft-py-1de1fc9.txt`.
**Source type:** released config/code.

> **Correction to the 2026-04 version of this excerpt and to the library card [[hf-alignment-handbook]].**
> Both reproduced a `SFTConfig(packing=True, max_seq_length=2048, train_on_response_only=True, …)` block as
> "the attested `SFTConfig`" and a lesson "template mismatch is the #1 silent bug". Neither is in the
> repository: `train_on_response_only` is not a TRL argument ([[trl-sft-trainer]]), the YAML has no `packing`
> key, the field is `max_length` in current TRL, and the "#1 silent bug" sentence appears in no file of this
> repository. The FSDP wiring attributed to this recipe is also absent; `scripts/sft.py` launches with a
> DeepSpeed ZeRO-3 accelerate config.

---

## `recipes/zephyr-7b-beta/sft/config_full.yaml` — the keys, as printed

| Key | Value |
|---|---|
| `model_name_or_path` | `mistralai/Mistral-7B-v0.1` |
| `torch_dtype` | `bfloat16` |
| `attn_implementation` | `flash_attention_2` |
| `chat_template` | inline Jinja template emitting `<|user|>`, `<|system|>`, `<|assistant|>` with `eos_token` after each message |
| `dataset_mixture` | `HuggingFaceH4/ultrachat_200k`, splits `train_sft` and `test_sft`, weight 1.0 each, `test_split_size: 1000`, `seed: 0` |
| `bf16` | `true` |
| `learning_rate` | `2.0e-05` |
| `lr_scheduler_type` | `cosine` |
| `warmup_ratio` | `0.1` |
| `num_train_epochs` | `1` |
| `max_seq_length` | `2048` |
| `per_device_train_batch_size` | `16` |
| `gradient_accumulation_steps` | `1` |
| `gradient_checkpointing` | `true` (`use_reentrant: False`) |
| `eval_strategy` | `epoch`; `per_device_eval_batch_size: 8` |
| `save_strategy` | `steps`, `save_steps: 100`, `save_total_limit: 1` |
| `seed` | `42` |

Keys **not** in the file: `packing`, `train_on_response_only`, `max_grad_norm`, `fsdp`, `fsdp_config`,
`weight_decay`, `optim`, any explicit loss-masking flag.

## `scripts/sft.py` — the launch line in the module docstring (L20-35)

```
# One 1 node of 8 x H100s
accelerate launch --config_file recipes/accelerate_configs/zero3.yaml scripts/sft.py \
    --model_name_or_path Qwen/Qwen2.5-1.5B-Instruct \
    --dataset_name trl-lib/Capybara \
    --learning_rate 2.0e-5 \
    --num_train_epochs 1 \
    --packing \
    --max_seq_length 4096 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --gradient_checkpointing \
    --bf16 true \
    --logging_steps 5 \
    --eval_strategy steps \
    --eval_steps 100 \
    --output_dir data/Qwen2.5-1.5B-SFT
```

Two facts for the lab's recipe table:

- Packing is passed on the command line in the example, not set in the Zephyr YAML.
- The distributed configuration is DeepSpeed ZeRO-3 (`recipes/accelerate_configs/zero3.yaml`), on one node of
  8 H100s. The global batch of the Zephyr YAML is **derived**, not printed: 16 per device × 1 accumulation
  step × 8 devices = 128 sequences.

## What this recipe does not report

No evaluation numbers for the `zephyr-7b-sft-full` checkpoint are in these two files, and no ablation selects
any value in the YAML. The recipe is a reproducible configuration, not evidence that a value is optimal.

## Connections

- [[trl-sft-trainer]] — the trainer this script calls, and the current names of these settings.
- [[hf-trainer-loop]] — where `max_grad_norm`, unset here, would be applied.
