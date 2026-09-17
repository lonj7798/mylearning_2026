---
chapter: ch-04
course: llm-training
phase: read
excerpt_of: Jain et al. 2023 — "NEFTune: Noisy Embeddings Improve Instruction Finetuning"
source_url: https://arxiv.org/abs/2310.05914
created_at: "2026-04-23"
revised_at: "2026-09-15 (generality revision; aligned with the verified library card and arXiv v2)"
---

# Excerpt: NEFTune

**Artifact:** arXiv:2310.05914 (v2, 2023-10-10). Authors: Neel Jain, Ping-yeh Chiang, Yuxin Wen, John Kirchenbauer, Hong-Min Chu, Gowthami Somepalli, et al. (13 authors). Library card: `papers/neftune.md` (verified 2026-09-14).

## Method (§2, Algorithm 1)

```
X′ = X + (α / √(L·d)) · ε,   ε ~ Uniform(−1, 1)^{B×L×d}
```

`X` is the embedding output (batch B, sequence length L, embedding dimension d) and `α` the base noise scale. "If sequence lengths in a batch are not equivalent, then L is a vector … and the scaling factor (α/√(Ld)) is computed independently for each sequence in batch" (Algorithm 1 footnote). The scaling rule comes from adversarial-training work and gives an expected Euclidean norm of about α/√3 (§2). Noise is applied only during training.

## Results used in ch-04

- AlpacaEval (v1; win rate against Text-Davinci-003, GPT-4 judge), LLaMA-2 7B (Table 1): Alpaca 29.79 → 64.69; Evol-Instruct 70.34 → 79.60; ShareGPT 68.74 → 76.28; OpenPlatypus 62.00 → 70.61; average 57.71 → 72.80.
- ARC, HellaSwag, MMLU, and TruthfulQA remain stable (§4, Fig. 3).
- Training loss (measured without noise) is higher with NEFTune and held-out loss on Evol-Instruct slightly lower (§5.1, Fig. 4); ROUGE-L and BLEU against training responses are lower (Fig. 5).
- Output length, LLaMA-2 7B on Alpaca: 375.22 → 1,061.89 characters (Table 4). LLaMA-1 Alpaca-7B, GPT-4 judge: baseline 32.36, prompting for long answers 48.01, forcing 250 minimum new tokens 38.58, NEFTune 61.99 (Table 5).
- α per dataset for LLaMA-2 7B: Alpaca 5, Evol-Instruct 5, ShareGPT 10, OpenPlatypus 15, chosen as the best of {5, 10, 15} on AlpacaEval with a ChatGPT judge (Table 7; App. A.1).
- 7B training settings: LR 5e-5, Adam, 3 epochs, effective batch 128, maximum length 512 (App. A.1). Loss masking and packing are not reported.

## Limits stated by the authors (§6, §7)

Single-judge evaluation; 70B tested on one dataset; mostly fixed hyperparameters; no conclusive explanation of why the method works; toxicity and refusal behaviour not evaluated.

## Implementation note (Transformers, not the paper)

`src/transformers/integrations/neftune.py` at commit 05e078a (L47-50) computes `dims = output.size(1) * output.size(2)` and `mag_norm = alpha / sqrt(dims)`. `output.size(1)` is the row length. In a padding-free packed row, L is the packed length, not the per-sequence length of Algorithm 1. See [[transformers-chat-templates]].

## Removed from the previous excerpt as unsupported

"one line of code / free accuracy lift"; "α = 5 on LLaMA" as a universal default; "+~10 AlpacaEval on Llama-2-7B"; "token-identity overfitting" as the mechanism; "positional embeddings left untouched" as a design finding; "no gain at pre-training or Tülu-3 scale".
