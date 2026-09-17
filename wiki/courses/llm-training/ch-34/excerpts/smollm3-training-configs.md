---
chapter: ch-34
course: llm-training
phase: read
excerpt_of: "huggingface/smollm text/pretraining/smollm3 nanotron configs, with the SmolLM3 blog post-training statements"
source_url: https://github.com/huggingface/smollm/tree/main/text/pretraining/smollm3
created_at: "2026-09-15"
---

# Excerpt: SmolLM3 released context-extension configs and blog post-training settings

This excerpt stands in for the library card `smollm3-training-configs`, which did not exist when ch-34 was written.
The card [[smollm-3]] has not been verified and does not give the SFT settings below.
Source types: released config (GitHub, main branch, read 2026-09-15) and official blog (huggingface.co/blog/smollm3).

## Released configs (directory lists stage1_8T.yaml, stage2_8T_9T.yaml, stage3_9T_11T.yaml, long_context_4k_to_32k.yaml,
## long_context_32k_to_64.yaml)

`text/pretraining/smollm3/long_context_4k_to_32k.yaml` (main):
```yaml
    max_position_embeddings: 32768      # L228
    rope_theta: 2000000.0               # L238
    learning_rate: 0.00002              # L250
  dp: 12                                # L269
  batch_accumulation_per_replica: 6     # L293
  micro_batch_size: 1                   # L296
  sequence_length: 32768                # L297
  train_steps: 20000                    # L298
```
`long_context_32k_to_64.yaml` (same settings at L231, L241, L253, L272, L296, L299-301): `max_position_embeddings:
65536`, `rope_theta: 5000000.0`, `learning_rate: 0.00002`, `dp: 12`, `batch_accumulation_per_replica: 3`,
`sequence_length: 65536`, `train_steps: 22000`. Both use cosine decay to `min_decay_lr: 0` after 1,000 linear warmup steps.

Derived tokens (dp × accumulation × micro batch × sequence length × steps):
- 4k→32k: 12 × 6 × 1 × 32,768 = 2,359,296 tokens per step; × 20,000 = 47.2B tokens.
- 32k→64k: 12 × 3 × 1 × 65,536 = 2,359,296 tokens per step; × 22,000 = 51.9B tokens.

## Blog statements (huggingface.co/blog/smollm3)
- Long context ("Long Context extension"): two stages of 50B tokens each, "first transitioning from 4k to 32k context
  with RoPE theta increased to 1.5M, then from 32k to 64k context with RoPE theta increased to 5M"; YaRN at inference
  to 128k ("2x extension beyond the 64k training length").
- Conflict: the blog's 1.5M differs from `rope_theta: 2000000.0` in the released 4k→32k config.
- Reasoning mid-training: 35B tokens (OpenThoughts3-1.2M and a Llama-Nemotron subset), ChatML template, "wrapped
  packing", 4 epochs (~140B tokens) ("Reasoning Mid-training").
- SFT ("Supervised Finetuning"): "1.8B tokens: 1B in non-reasoning mode and 0.8B in reasoning mode, comprising 12
  non-reasoning datasets and 10 datasets with reasoning traces. We trained for 4 epochs (~8B tokens) using BFD
  (best-fit decreasing) packing with the loss masked on user turns and the results from tool calls." Reasoning traces
  for sparse domains were generated with Qwen3-32B in reasoning mode.
- Mode control: `/think` and `/no_think` flags in the system prompt; non-reasoning responses are pre-filled with empty
  think blocks "similar to Qwen3" ("Chat Template").
- Alignment and merge ("APO", "Model Merging"): APO training data limited to 24k tokens; RULER degraded after
  reasoning mid-training and APO; a soup of APO checkpoints was linearly merged with a mid-training checkpoint at
  weights 0.9 and 0.1, which "recover[ed] the base model's RULER score on contexts up to 128k tokens".
- Not in the blog: SFT learning rate, batch size, maximum SFT sequence length, SFT checkpoint-selection rule.
