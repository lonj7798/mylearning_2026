---
chapter: ch-45
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/self-rewarding-lm.md (card not re-verified; see Corrections)
source_url: https://arxiv.org/abs/2401.10020
version: arXiv v3, 2025-03-28
verified: 2026-09-15 (rewritten from the primary text; replaces the 2026-04 excerpt)
---

# Excerpt: Self-Rewarding Language Models (Yuan et al., Meta / NYU)

**Used by:** [[read]] §1, §2.1, §6, Recipe

## Model sequence (§2.4)

- M0: Llama 2 70B, no fine-tuning.
- M1: SFT of M0 on the IFT + EFT seed data.
- M2: DPO from M1 on AIFT(M1), 3,964 preference pairs.
- M3: DPO from M2 on AIFT(M2), 6,942 preference pairs.

"Three iterations" in the abstract means M1, M2, M3 — one SFT model and two DPO rounds. No fourth iteration is run or reported.

## Loop (§2.2–2.3, §3.1.3)

1. New prompts come from a fixed model, Llama 2-Chat 70B, 8-shot, T = 0.6, p = 0.9, with the Self-Instruct filters (ROUGE-L, keyword, length).
2. N = 4 candidate responses are sampled from the current model at T = 0.7, p = 0.9.
3. The same model scores each candidate under a fixed 5-point additive rubric (Figure 2); the judgment is sampled 3 times and averaged.
4. Highest-scored candidate = chosen, lowest = rejected; the pair is discarded when the two scores are equal.
5. DPO, β = 0.1, learning rate 1e−6 decaying to 1e−7, batch 16, dropout 0.1. Checkpoints every 200 steps, selected by Claude 2 pairwise judgments on 253 validation examples.

Seed data: 3,200 Open Assistant first turns for IFT; 1,630 train / 541 evaluation examples for EFT, built by having the SFT baseline write judgments and keeping those whose score ranking agrees with the human ranking (§3.1.1).

## Results as printed

AlpacaEval 2.0 win rate against GPT-4 Turbo (Table 1): M1 9.94%, M2 15.38%, M3 20.44%. Reference rows on the same table: GPT-4 0314 22.07%, Claude 2 17.19%, GPT-4 0613 15.76%.

MT-Bench /10 (Table 2): SFT baseline 6.85 overall (3.93 math/code/reasoning), M1 6.78 (3.83), M2 7.01 (4.05), M3 7.25 (4.17).

NLP benchmarks (Table 3): ARC-Challenge 55.97 / 57.51 / 54.51 / 53.13; HellaSwag 85.17 / 84.99 / 84.27 / 83.29; GSM8K 50.72 / 60.27 / 59.29 / 57.70; MMLU 69.76 / 69.34 / 69.31 / 69.37; NQ 34.35 / 35.48 / 33.07 / 31.86 (order: SFT baseline, M1, M2, M3). Llama 2 base values are 57.40 / 85.30 / 56.80 / 68.90 / 25.30.

Reward modeling against held-out Open Assistant human rankings (Table 4), order SFT baseline / M1 / M2 / M3: pairwise accuracy 65.1% / 78.7% / 80.4% / 81.7%; 5-best 39.6 / 41.5 / 44.3 / 43.2; exact match 10.1 / 13.1 / 14.3 / 14.3; Spearman 0.253 / 0.279 / 0.331 / 0.349; Kendall τ 0.233 / 0.253 / 0.315 / 0.324.

Length: average AlpacaEval generation length 1,092 (M1), 1,552 (M2), 2,552 (M3) (§3.2.1).

Limitations stated by the authors (§6): only three iterations in a single setting; the scaling of the effect over more iterations is open; generations grow longer and length correlates with estimated quality; whether reward hacking occurs inside the framework is not measured, and both the training reward and part of the evaluation come from language models; no safety evaluation.

## Corrections to the library card `papers/self-rewarding-lm.md`

1. "judge's Spearman correlation ... from 0.62 (iter 0) to 0.71 (iter 3)" → Table 4 gives 0.253 (SFT baseline) to 0.349 (M3); pairwise accuracy goes 65.1% to 81.7%.
2. "the paper notes iter 4 regresses on reward bench (likely reward hacking)" → no fourth iteration exists in the paper; §6 lists reward hacking as unmeasured.
3. "Figure 3 (AlpacaEval win-rate vs iteration) ... plateau after" → AlpacaEval 2.0 results are Table 1; Figure 3 is the head-to-head win-rate comparison; no plateau is reported.
4. "Table 2 (judge agreement with Open-Assistant humans)" → Table 2 is MT-Bench; judge agreement is Table 4.
5. "DPO with β = 0.1 for 1 epoch, AdamW lr=5e-7" → learning rate 1e−6 decaying to 1e−7 with batch 16 and step-based checkpoint selection (§3.1.3); the epoch count is not stated.
