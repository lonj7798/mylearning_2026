---
chapter: ch-43
course: llm-training
phase: read
excerpt_of: "no library card exists for this source as of 2026-09-15; extracted directly from the primary text"
source_url: https://arxiv.org/abs/2508.10751
primary_version: arXiv:2508.10751v1 (14 Aug 2025)
created_at: "2026-09-15"
---

# Excerpt: Pass@k Training for Adaptively Balancing Exploration and Exploitation of Large Reasoning Models

Chen, Qin, Wu, Ling, Ye, Zhao, Shi (Renmin University of China; ByteDance Seed), 2025.

## Method
- RLVR usually rewards each sample individually (Pass@1 training). Pass@k training groups the `N_rollout`
  samples of a prompt into groups of `k` and rewards a group with 1 if at least one member is correct
  (Eq. 5, §2.2).
- Full sampling is replaced by bootstrap sampling of groups (§2.3) and then by an analytical advantage that
  removes the sampling variance (§2.4). With `N_pos` correct and `N_neg = N_rollout − N_pos` incorrect
  samples:
  - group-level mean reward `R̄ = 1 − C(N_neg, k) / C(N_rollout, k)` (Eq. 11);
  - `σ = sqrt(R̄ (1 − R̄))` (Eq. 12);
  - advantage of a correct response `Â_pos = (1 − R̄) / σ` (Eq. 14);
  - advantage of an incorrect response `Â_neg = (1 − R̄ − C(N_neg − 1, k − 1)/C(N_rollout − 1, k − 1)) / σ`
    (Eq. 15).
  The advantage depends only on `N_rollout`, `N_pos`, `N_neg` and `k`, so it is computed after the rollout with
  no extra sampling (§2.4).
- `k = 1` recovers the Pass@1 (DAPO-style) advantage.

## Reported effects
- Entropy (§3.2, Fig. 7b): "Pass@k Training keeps the entropy of policy distribution at a relatively high
  level, while Pass@1 Training induces entropy to converge to a low value", with entropy rising from about
  step 200 of the Pass@k run. No numeric entropy values are printed.
- Table 1, Qwen2.5-7B-Instruct, Pass@1/Pass@k scores (evaluation samples 8 responses per question for
  non-Maze tasks, temperature 1.0, top-p 0.95, App. A):
  | Run | ARC-AGI 1 (in-domain) | Enigmata (in-domain) | KORBench (OOD) | AIME 2025 (OOD) |
  |---|---|---|---|---|
  | Qwen2.5-7B-Instruct | 2.4 / 4.8 | 4.8 / 10.1 | 36.5 / 45.9 | 4.2 / 15.8 |
  | + Pass@1 training | 3.3 / 3.8 | 12.9 / 21.3 | 37.7 / 45.6 | 5.4 / 19.1 |
  | + Pass@k training | 4.0 / 5.3 | 17.9 / 29.8 | 47.7 / 63.5 | 7.1 / 22.4 |
- Table 2 (Enigmata overall, Pass@1/Pass@k): Qwen2.5-7B-Instruct baseline 4.7/10.1, Pass@1 training
  12.9/21.3, Pass@k training 17.9/29.8, Pass@k followed by Pass@1 30.8/40.6. On Qwen2.5-32B-Instruct the
  Pass@1 and Pass@k runs are close (45.2/56.0 versus 44.5/57.4) and the two-stage run reaches 46.8/57.9.
- Robustness (§3.4): `k ∈ {4, 8, 16}` all reach a high training reward; larger `k` gives smaller advantages
  and therefore slower optimization, which the authors compensate with a larger learning rate
  (1e-6, 2e-6, 4e-6 tested at `N = 32`, `k = 8`).
- Entropy-regularization comparison (§3.1): entropy coefficients {0.001, 0.003, 0.005} were tested; 0.005
  "might cause the collapse of the model", and the smaller coefficients did not outperform Pass@k training.
- Setup (App. A): rewards are 1 for a verified response and 0 otherwise; "we do not employ any regularization
  methods, such as KL or Entropy regularization".

## Verification
- Read on 2026-09-15 from the cached primary text of arXiv:2508.10751v1 (scratchpad `sources/pass-at-k-training.txt`).
- Not reported in Table 1: the value of `k` used for the printed Pass@k column (the evaluation protocol in
  App. A samples 8 responses per question outside the Maze task); no entropy numbers; no forgetting or
  retention measurement outside the listed benchmarks.
