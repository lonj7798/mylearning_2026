---
chapter: ch-44
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/lets-verify.md
source_url: https://arxiv.org/abs/2305.20050
created_at: "2026-04-23"
---

# Excerpt: Let's-Verify — process vs outcome head-to-head

**Source library:** `wiki/raw-data/llm-training/papers/lets-verify.md`, `wiki/raw-data/llm-training/papers/let-verify.md`
**Anchor paper:** Lightman et al. 2023 — "Let's Verify Step by Step"

---

## Why this source anchors ch-44

The PRM-vs-ORM comparison at matched compute is the empirical claim every later process-supervision paper cites. If you strip PRM800K away and keep only the comparison, you still have the justification for the entire process-supervision research line. This excerpt extracts the numbers and the shape of the PRM-vs-ORM curve.

---

## The headline table — verbatim

From `let-verify.md` §Key Contributions:

> Showed the process RM outperforms outcome RM at best-of-N selection: **78.2% vs 72.4% on a MATH test subset at N=1860**.
> **Table 1 (MATH subset):** 78.2% PRM vs 72.4% ORM vs 69.6% majority-vote — key headline.

Written as a ch-44 table:

| Selector | MATH-500 subset acc | vs majority |
|----------|----------------------|-------------|
| Majority vote | 69.6 | — |
| ORM Best-of-1860 | 72.4 | +2.8 |
| PRM Best-of-1860 | 78.2 | +8.6 |

The ORM beats majority by 2.8 pp; the PRM beats ORM by 5.8 pp. Those two gaps are the ones to remember.

---

## The curve shape

From `let-verify.md` §Key Figures/Tables to Study:

> **Figure 1 (MATH test-set accuracy vs N):** PRM curve dominates ORM at every N; gap grows as N grows.

And from `prm800k.md`:

> With matched compute and matched number of samples, the PRM Pareto-dominates the ORM on MATH across N from 1 to 1860.

The "at every N" line matters. Two misreadings to avoid:

1. "PRM only helps when you can afford Best-of-1860." False. PRM > ORM at N=16 already, and the gap is already a meaningful fraction of the N=1860 gap.
2. "ORM is fine for small tasks, PRM for big tasks." Also false. The difference is the task's *chain length*, not N. Short-chain tasks (GSM8K) gain less from PRMs than long-chain ones (MATH) because there are fewer steps that could independently fail.

---

## Why PRM dominates — calibration

From `let-verify.md` §Key Figures:

> **Figure 3 (calibration):** PRM is better calibrated per-step than ORM on full-solution.

ORM asks one question ("is this whole solution correct?") from one forward pass. PRM asks `L` questions ("is this step correct?") from `L` forward passes, and each is a local, bounded decision. The local decisions calibrate; the global decision does not. When you aggregate the calibrated per-step probabilities with `min` or `prod`, you get a well-ordered statistic. The ORM score is not that.

---

## The inference-cost asymmetry

From `let-verify.md` §Technical Details:

> **Inference cost:** PRM scores every step (1 forward per step); ORM scores final answer (1 forward per solution) — PRM is ~L× more expensive at inference where L is step count.

On MATH-500 median `L` is ~10. Best-of-N at N=1860 with PRM is therefore ~18,600 scorer forwards, vs ~1,860 for ORM. This is the one axis where ORM wins — PRM's +5.8 pp at fixed N costs 10x more scorer compute.

Practical consequence for ch-46: if the lab budget caps scorer compute, compare PRM at N=64 vs ORM at N=640 at matched scorer forwards, not just at matched N.

---

## What the paper deliberately does not do

From `let-verify.md` §Technical Details:

> The paper intentionally fixes the generator and does **not** run RL; it studies verifier quality in isolation.

This is the gap Math-Shepherd fills: once you have a good PRM, feeding it as a dense reward inside PPO is the natural next move. Let's-Verify is a ranker-quality paper. Math-Shepherd is the RL paper.

---

## Carry into ch-44

- Headline numbers for §3 table: `69.6 / 72.4 / 78.2` on MATH-500.
- "PRM dominates at every N" is a stronger claim than "PRM dominates at large N"; the chapter quotes the stronger form.
- Calibration argument is the *why*, not the *what* — the chapter uses it to justify why `min`/`prod` aggregation is well-ordered only for the PRM case.
- Inference-cost asymmetry is what makes RLVR attractive when a ground-truth verifier exists: one verifier call replaces `L` PRM forwards.
