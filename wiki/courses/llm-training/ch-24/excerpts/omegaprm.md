---
chapter: ch-24
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/omegaprm.md
source_url: https://arxiv.org/abs/2406.06592
created_at: "2026-04-23"
revised_at: "2026-09-15"
---

# Excerpt: OmegaPRM — binary-search Monte-Carlo step labels for process reward models

**Source library:** `wiki/raw-data/llm-training/papers/omegaprm.md` (verified 2026-09-14 against arXiv:2406.06592v2).
Revised for ch-24 §6 and the negative-samples section. The earlier excerpt (git 4a72e54) had the wrong loss, selector, data size, and cost arithmetic.

## Mechanism

- Monte-Carlo estimate (Eq. 1): `c_t = (correct rollouts from step t) / (total rollouts from step t)`; a rollout is correct if its final answer equals the golden answer (§3.2).
- Binary search (§3.2): roll out from the midpoint m; if `c_m > 0` the first half is taken as correct and the error is in the second half; if `c_m = 0` the error is likely in the first half; repeat until one step. Cost O(k log M) policy calls vs O(kM) for M steps.
- Tree search (§3.3): `Q(s, r) = α^(1 − MC(s)) · β^(len(r)/L)` and `U(s) = c_puct · sqrt(Σ_i N(s_i)) / (1 + N(s))`; rollouts with `0 < MC(s) < 1` form the pool.
- PRM training (§3.4, Eq. 4): classification loss; main results use the pointwise soft label `ŷ = MC(s)`; the hard label is `1[MC(s) > 0]`.

## Settings and results

- 12K MATH training questions; search limit 100 per question; 1.5M per-step annotations; k = 8 rollouts; α = 0.5, β = 0.9, L = 500, c_puct = 0.125 (§4).
- Solutions split into 16 pieces (§4.2). Worked example in ch-24: 8 × 16 = 128 rollouts for per-step estimation vs 8 × log₂16 = 32 for binary search (derived).
- Question filter: 32 rollouts per question; drop questions with no correct answer (false-negative risk) or no wrong answer (false-positive risk) (App. A).
- Table 1, MATH500 at k = 64 (Gemini Pro / Gemma2 27B): majority vote 67.2 / 54.7; PRM800K 67.6 / 57.2; Math-Shepherd 67.2 / 57.4; OmegaPRM 69.4 / 58.2. Metric: PRM-weighted majority voting with the product of step scores.
- Objectives (Table 2): step accuracy soft 70.1%, pairwise 64.2%, hard 63.3%.
- Stated limit: "the precise impact of noise on PRM performance remains uncertain"; open-ended tasks need adaptation (§5).

## Removed from the earlier excerpt (not in the source)

- "PRM regressed via MSE"; "weighted best-of-N (PRM × policy log-prob)"; "~80K problems"; "K = 16, L = 10, 4× saving"; "~100K TPU-hours".
