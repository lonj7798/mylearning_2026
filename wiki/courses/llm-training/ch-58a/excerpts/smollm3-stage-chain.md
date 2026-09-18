---
chapter: ch-58a
course: llm-training
phase: read
excerpt_of: "SmolLM3: smol, multilingual, long-context reasoner (Hugging Face blog, July 8, 2025) and the released SmolLM3 training configs"
source_url: https://huggingface.co/blog/smollm3
created_at: "2026-09-17"
note: "The library card model-reports/smollm-3.md has no Verification section and gives none of the values below. Read on 2026-09-15 in the cached Markdown of the blog (scratchpad sources/smollm3-blog-md.txt) and the cached configs for the ch-14a, ch-45a and ch-30c excerpts of the same passages."
---

# Excerpt: SmolLM3 — the full stage chain of a 3B model, ending in a merge

Source type: official blog (the organization that trained SmolLM3), plus its released training configs.

## Stage chain
pre-training in three stages (11.2T tokens) → long-context extension (two 50B-token stages) → reasoning mid-training
(35B tokens × 4 epochs) → SFT → APO (off-policy preference optimization) → model soup of APO checkpoints → linear merge
with a mid-training checkpoint. The merged model is the released checkpoint.

## Pre-training (blog "Training Configuration" and "Data mixture and training stages")
- "We use a global batch size of 2.36M tokens with 4096 sequence length, a learning rate of 2e-4, and the AdamW optimizer
  (beta1: 0.9, beta2: 0.95) with weight decay of 0.1 and gradient clipping of 1. We use the WSD (Warmup-Stable-Decay)
  scheduler, with 2000 warmup steps, and a linear decay to 0 in the final 10% training steps."
- "The model was trained on 384 H100 GPUs for 24 days."
- "we train SmolLM3 on 11.2T tokens using a three-stage training strategy"; mixtures were chosen from "ablations on 3B
  models trained on 50B to 100B tokens". Stage 1 (0T → 8T): Web 85%, Code 12%, Math 3%. Stage 2 (8T → 10T): Web 75%,
  Code 15%, Math 10%. Stage 3, decay (10T → 11.1T): Web 63%, Code 24%, Math 13%.
- The headline says "11T tokens", the stages paragraph "11.2T", and stage 3 ends at "11.1T".

## Long context and reasoning mid-training
- Long-context extension: 100B tokens in two 50B stages, 4k → 32k with RoPE theta 1.5M and 32k → 64k with theta 5M. The
  released config for the 4k → 32k stage sets `rope_theta: 2000000.0`.
- Reasoning mid-training: 35B tokens of reasoning data for 4 epochs (about 140B tokens).
- SFT: 1.8B tokens (1B non-reasoning across 12 datasets, 0.8B reasoning across 10 datasets) for 4 epochs, with
  best-fit-decreasing packing and loss masked on user turns and tool results.

## APO and the merge (blog, "Off-policy model alignment with APO" and "Model Merging")
- Pairs: "we performed a round of model alignment using a combination of the Tulu3 preference dataset for non-reasoning
  mode and new synthetic preference pairs for reasoning mode, that we generated from Qwen3-32B and Qwen3-0.6B... We
  selected generations from Qwen3-32B as 'chosen' and responses from Qwen3-0.6B as 'rejected' for alignment with Anchored
  Preference Optimization." Both sides are off-policy.
- "The APO objective has been shown to be more stable, and we also observed higher downstream performance in our internal
  ablations." No β, learning rate, batch size, epoch count, or pair count is printed, and the internal ablation has no
  numbers.
- Regression: "While downstream evaluations showed improvements across mathematics, science, instruction following,
  coding, chat, and multilingual tasks, we observed performance degradation on long context benchmarks like RULER. We
  traced this degradation back to the reasoning mid-training stage, where the focus on reasoning capabilities impacted
  long context performance. Additionally, the APO training data was limited to 24k tokens since the vast majority of our
  reasoning dataset fell below this length."
- Merge: "1. Take each APO checkpoint and create a model 'soup'. 2. Combine the model soup with a mid-training checkpoint
  that has strong long-content performance. A linear merge with weights of 0.9 and 0.1 for the APO model soup and
  mid-training checkpoint, respectively, achieved the best performance. We were able to recover the base model's RULER
  score on contexts up to 128k tokens." The merge was performed with MergeKit, and "The resulting model is the checkpoint
  we are releasing today."

## Config-derived stage boundaries (released configs)
- Stage-1 config: dp 192, tp 2, pp 1, accumulation 1, micro-batch 3, sequence length 4,096, so the global batch is
  192 × 1 × 3 × 4,096 = 2,359,296 tokens on 384 GPUs, matching the blog's "2.36M" and "384 H100 GPUs".
- The configs place "stable stage 2" at step 3,450,001, the "decay stage" at step 4,198,001, and the end at step
  4,720,000, which at 2,359,296 tokens per step is 8.14T, 9.90T and 11.14T tokens; the blog places the stages at 8T, 10T
  and 11.1T. The config decay covers 522,000 / 4,720,000 = 11.06% of steps against the blog's "final 10%".
- The SmolLM3 configs were edited on 2025-08-03 by a commit titled "fix nb of steps in early configs", after the
  July 8, 2025 release.

## Verification
- Read on 2026-09-15 in the cached Markdown of the blog and in the cached config files; reassembled for ch-58a on
  2026-09-17 without changing a value or locus.
- Not reported by the blog: APO β, learning rate, batch size, epochs, pair count; RULER numbers before and after the
  merge; the number of APO checkpoints in the soup; which mid-training checkpoint entered the merge; the effect of the
  merge on non-long-context benchmarks.
