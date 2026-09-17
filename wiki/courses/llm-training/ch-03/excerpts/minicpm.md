---
chapter: ch-03
course: llm-training
phase: read
excerpt_of: primary source arXiv:2404.06395v3 (no library card as of 2026-09-15)
source_url: https://arxiv.org/abs/2404.06395
created_at: "2026-09-15"
---

# Excerpt: MiniCPM: Unveiling the Potential of Small Language Models with Scalable Training Strategies

**Report:** Shengding Hu, Yuge Tu, Xu Han, Chaoqun He, Ganqu Cui, Xiang Long, et al. (Tsinghua University;
Modelbest Inc.). arXiv v1 2024-04; v3 2024-06-03 read. Source type: official technical report. ch-03 uses §3–§6
and App. A.

## Hyperparameter search at small scale ("model wind tunnel", §3)
- μP width and depth scaling from Tensor Programs, operations listed in Table 7 (see [[weight-init]] excerpt);
  attention softmax scaling not applied (§3.1).
- Batch size: 0.009B, 0.03B, 0.17B models, 6 batch sizes each, global LR 0.01, cosine schedule; optimal batch size
  fit against C4 loss L as `bs = 1.21 × 10^9 / L^6.24` (§3.2, Fig. 1–2). App. A.2: with a fixed number of GPUs,
  doubling batch "almost equals doubling the single-step time", so the authors minimize tokens to reach a loss
  rather than steps.
- LR: optimal base LR "remains around 0.01" from 0.04B to 0.5B; a 2.1B check confirmed 0.01 (§3.3, Fig. 3).
- QK-norm on the 0.009B search model gave "a significant decrease in the learning rate sensitivity", but was not
  adopted because the project did not require low LR sensitivity once μP located the best LR (App. A.1, Fig. 14).

## Cosine horizon and WSD (§4)
- Cosine(T) vs CosineLoop(T) on a 0.036B model: at S = 20N, 40N, 60N, 80N tokens the lowest loss is "always
  achieved by the Cosine(T) where T = S. Both T < S and T > S are not optimal" (§4.1, Fig. 4).
- WSD definition: Eq. 1 (§4.2). Loss "experiences a significant rapid decline" in the decay stage and reaches or
  passes cosine at T = S; checkpoints before decay can continue at high LR (§4.3, Fig. 5).
- Decay length: from stable checkpoints at 40N, 60N, 80N tokens, decaying over 10% of total tokens "is sufficient to
  achieve the best results, while a decay of 2.5% of total tokens falls short" (§4.3, Fig. 5).
- Decay-stage analysis on MiniCPM-2.4B and a 0.2B model: maximum weight updates track LR magnitude; during decay the
  gradient norm falls and consecutive gradient cosine becomes mostly positive (§4.4, Fig. 7–8).
- Scaling fit with WSD(D, 0.1D) runs: compute-optimal data size "should be 192 times larger than the model size on
  average, as opposed to 20 times in Hoffmann et al. (2022)" (§4.5).

## High-quality data in the decay stage (§5, Table 1)
- Strategy: coarse-quality pre-training data during the stable stage; "diverse and high-quality knowledge and
  ability-oriented SFT data, mixed into the pre-training data" during annealing (§5).
- Table 1 (C-Eval / CMMLU / MMLU / GSM8K / MATH / HumanEval / MBPP):

| Run | Setting | C-Eval | CMMLU | MMLU | GSM8K | MATH | HumanEval | MBPP |
|---|---|---|---|---|---|---|---|---|
| A-1 | 2.4B, decay on pre-training data only, then 4B-token SFT | 40.0 | 41.5 | 44.6 | 27.7 | 5.1 | 27.7 | 24.4 |
| A-2 | 2.4B, decay with high-quality + SFT data mixed in, then 4B-token SFT | 52.6 | 51.1 | 50.9 | 42.3 | 5.4 | 30.4 | 30.3 |
| B-1 | 1.2B, decay on pre-training data only, then 6B-token SFT | 40.9 | 41.5 | 47.9 | 34.2 | 7.9 | 43.9 | 30.5 |
| B-2 | 1.2B, decay on pre-training data only, then 12B-token SFT | 41.2 | 42.0 | 47.9 | 34.4 | 7.3 | 43.9 | 29.8 |
| B-3 | 1.2B, decay with high-quality + SFT data mixed in, then 6B-token SFT | 49.1 | 46.8 | 49.6 | 31.8 | 10.5 | 44.5 | 32.8 |

- Authors' conclusion: benefits of introducing high-quality data at the start of decay "are much higher than simply
  adding it during the SFT phase"; B-2 vs B-3 shows the SFT-only deficit "is not due to the insufficient training
  tokens in SFT stage" (§5). No seeds or variance are reported; the evaluated benchmarks overlap in domain with the
  added knowledge, math, and code SFT data (Interpretation for ch-03).

## Released models' schedule (§6.1–§6.4, Table 2)
- MiniCPM-2.4B: 2,442,057,984 non-embedding parameters, batch 4M, 1.1T tokens; MiniCPM-1.2B: batch 2M → 4M, 1.1T
  tokens (Table 2).
- Stable stage: about 1T tokens, "WSD LRS, with a batch size of 3.93 million and a max learning rate of 0.01" (§6.2).
- Decay stage: pre-training data plus high-quality SFT data; exponential annealing `f(s − T) = 0.5^((s−S)/T)`, "in
  which T is set to be 5000 steps (20B tokens)" (§6.2, notation as printed).
- SFT stage: about 6B tokens; SFT LR "aligned with the one at the end of annealing", WSD with exponential decay (§6.2).
- "The first drop in MiniCPM-1.2B is the result of enlarging batch size, which might have a similar effect as
  decreasing learning rate" (§6.4).

## Verification
- Read on 2026-09-15 against arXiv:2404.06395v3 PDF text (Abstract, §3–§6, App. A.1–A.3).
- Not reported: seeds or run-to-run variance for Table 1; token counts of each decay run in Table 1; checkpoint
  averaging (none described in these sections).
