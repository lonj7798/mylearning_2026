---
chapter: ch-22
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/less.md
source_url: https://arxiv.org/abs/2402.04333
created_at: "2026-04-23"
revised: "2026-09-15 (rewritten against the primary source; the library card was unverified and disagreed with the paper on warmup size, projection, Adam form, and final training)"
---

# Excerpt: LESS — Selecting Influential Data for Targeted Instruction Tuning

**Paper:** Mengzhou Xia, Sadhika Malladi, Suchin Gururangan, Sanjeev Arora, Danqi Chen (Princeton). arXiv:2402.04333 (v1 2024-02; ICML 2024). Checked 2026-09-15 against the arXiv PDF.

## Setting (§3, §5.1)
- Goal: select a subset D_train of a large instruction pool D so that training on it lowers loss on a small target validation set D_val with subtasks D_val^(1..m) (for example 1 example per language in TydiQA).
- Pool: Flan V2, CoT, Dolly, Open Assistant 1, about 270K data points, with no obvious in-domain data for the targets (§5.1).
- Targets: MMLU, TydiQA, BBH.

## Formula (§3)
- Adam update direction: Γ(z, θ) = m / (√v + ε), with m and v the bias-corrected moment estimates (§3.1).
- Definition 3.1: Inf_Adam(z, z′) = Σ_{i=1..N} η̄_i · cos(∇ℓ(z′; θ_i), Γ(z, θ_i)), with N warmup epochs, η̄_i the average learning rate in epoch i, θ_i the checkpoint after epoch i.
- Eq. 2: validation gradients are averaged within each subtask; a sample's score is the maximum over subtasks (§4.2).
- Cosine instead of dot product: the sequence gradient is the mean of token gradients, so its norm falls with completion length; the dot product upweights short sequences and performs worse (§3.2, Table 13).

## Steps (§4.1, §5.1)
1. LoRA warmup on a random 5% subset of D for N = 4 epochs, saving each epoch (LoRA gradients are under 2% of LLaMA-2-7B's parameters).
2. Compute Adam LoRA gradient features for all of D and project them with Π whose entries are Rademacher (uniform on {−1, 1}); d = 8192.
3. For each target, score with Eq. 2 and keep the top 5%.
4. Train the target model (LoRA) on the kept set; results are means over 3 seeds.

## Table 2 (5% budget unless "Full")
| Model | MMLU Full / Rand / LESS-T / LESS | TydiQA Full / Rand / LESS-T / LESS | BBH Full / Rand / LESS-T / LESS |
|---|---|---|---|
| LLaMA-2-7B | 51.6 / 46.5 / – / 50.2 | 54.0 / 52.7 / – / 56.2 | 43.2 / 38.9 / – / 41.5 |
| LLaMA-2-13B | 54.5 / 53.4 / 54.6 / 54.0 | 54.3 / 53.0 / 57.5 / 54.6 | 50.8 / 47.0 / 49.9 / 50.6 |
| Mistral-7B | 60.4 / 60.0 / 60.6 / 61.8 | 57.7 / 56.9 / 61.7 / 60.3 | 53.0 / 54.5 / 56.0 / 56.0 |
LESS-T selects with the LLaMA-2-7B datastore.

## Other tables
- Table 3 (LLaMA-2-7B): Random 46.5 / 52.7 / 38.9; BM25 47.6 / 52.7 / 39.8; DSIR 46.1 / 44.5 / 36.8; RDS 45.0 / 46.8 / 36.7; LESS 50.2 / 56.2 / 41.5 (MMLU / TydiQA / BBH).
- Table 4: warmup 6 A100 GPU hours; gradient features 48 hours; datastore 17.7 GB; selection under 1 minute.
- Table 5: selection model gradients from base LLaMA-2-7B (average 46.2) or LLaMA-2-7B-Chat (46.2) vs LoRA warmup on 5% (49.3), 25% (49.9), 100% (50.5).
- Table 6: N = 1 checkpoint: MMLU 48.2, TydiQA 54.9; N = 4: 50.2, 56.2.
- Table 9 (warmup on all data): average Random 46.0, SGD 49.7, SignGD 47.8, Adam 50.5.
- Table 10: MMLU base / LESS 5%: LLaMA-2-13B 55.3 / 54.0; Mistral-7B 62.4 / 61.8.

## Limitations stated by the authors (Limitations)
1. Warmup training is required. 2. Completion-token gradients are averaged, which may be less effective for long sequences. 3. Lower validation loss does not always raise task accuracy. 4. Influences are treated as additive, so duplicates are double-counted.
Footnote 8: useful data may remain in the discarded 95%; the optimal threshold was not explored.
