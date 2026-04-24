---
chapter: ch-31
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/rest-em.md
source_url: https://arxiv.org/abs/2312.06585
created_at: "2026-04-23"
---

# Excerpt: ReST-EM — STaR as expectation-maximization and the 2-iter saturation curve

**Source library:** `wiki/raw-data/llm-training/papers/rest-em.md`
**Artifact:** E-step / M-step / diversity-cap recipe and the MATH saturation numbers

---

## Why this source anchors ch-31

ReST-EM (Singh 2023) gives ch-31 two things nothing else in the literature provides with the same clarity:

1. **A clean recipe with concrete hyperparameters.** K=32 samples per problem at T=1.0, top-p=0.95, verifier-filter by exact-match or unit-test, SFT at lr=1e-5 for 1 epoch, batch 128, diversity cap 4 distinct accepted solutions per problem.
2. **The 2-iter saturation curve.** MATH iter-1 +8%, iter-2 +6%, iter-3 flat. This is the number ch-31's decision tree (node 3) uses to decide "switch to on-policy RL." Without a concrete saturation number, the decision tree is hand-waving.

---

## The attested recipe — quoted

From the source (lines 30–37):

> - **Base model:** PaLM-2-L (~340B active params), also ablated on -S and -XS.
> - **E-step:** sample K=32 solutions per problem at T=1.0, top-p=0.95.
> - **Verifier:** exact-match on ground-truth final answer (MATH) or unit-test pass (APPS).
> - **M-step:** SFT on (problem, correct-solution) pairs; 1 epoch; lr=1e-5; batch 128.
> - **Diversity cap:** keep at most 4 distinct correct solutions per problem (prevents memorization of one solution path).
> - **Iterations:** 2 for MATH; gains saturate.
> - **Compute split:** inference for E-step dominates (~100 H100-hrs per iter at K=32, N=10K problems).

Ch-31 §4 reproduces these numbers as the reference configuration for the "verifier-is-exact-match" branch of the decision tree.

---

## The EM formalism — why framing matters

ReST-EM is STaR cast as expectation-maximization over a latent rationale variable `z`:

- **E-step:** given the current policy `\pi_t`, sample K rationales per problem; filter by the verifier's indicator `1[answer correct]`; this is a Monte-Carlo approximation of the posterior `p(z | x, y_correct)`.
- **M-step:** maximize `E_{z ~ posterior}[log \pi_\theta(z | x)]` — standard SFT on the filtered rationales.

The gain from this framing is not mathematical; it is pedagogical. Once you see the loop as EM, three things become obvious:

1. **You are maximizing a lower bound on the log-likelihood of correct answers** under the rationale prior. This is why the loop has a well-defined local optimum: the EM fixed-point.
2. **The E-step's K is a Monte-Carlo budget knob**, not a quality knob. Larger K reduces E-step variance; it does not change the fixed point.
3. **The M-step's LR and epochs are the regularizer.** Overfitting in the M-step pulls the policy away from the filtered posterior; 1-epoch / lr=1e-5 keeps it close.

Ch-31 does not lean heavily on the EM formalism in the prose, but the decision tree's "do another round" nodes are EM iterations and the "switch to on-policy RL" node is "EM has converged; the policy-gradient objective is different."

---

## The 2-iter saturation number, read carefully

From the source (line 26):

> **Figure 2 (MATH accuracy vs iteration):** iter-1 +8%, iter-2 +6%, iter-3 flat; the canonical saturation curve.

Three things hide inside this number:

1. **Saturation is not over.** Iter-3 is flat *on MATH*. On APPS, the curve saturates similarly. On harder distributions with more rationale diversity, the saturation point likely extends — but no open run has published a clean >3-iter saturation curve, so ch-31 defaults to "2-3 rounds" as the operational ceiling.
2. **The diversity cap is load-bearing.** Without the cap (keep all correct solutions), iter-3 *regresses*. The paper documents this (Figure 6 in the source). The cap is not a tuning knob; it is a pre-condition for iter-2 being usable at all.
3. **Transfer gains outlive single-task saturation.** Training on MATH improves Big-Bench-Hard tasks that were not in the training distribution. This is the evidence for ch-31's claim that iterative-SFT on verifier-filtered reasoning is pushing the base model's reasoning prior rather than narrowly distilling one skill.

---

## What ReST-EM adds to STaR

STaR uses K=1 (one sample per problem) with rationalization on failure. ReST-EM uses K=32 with *no* rationalization. The trade-off:

- **STaR (K=1 + rationalize):** gets a correct trace on nearly every problem (via the backward-rationalization branch). Cost: rationalized traces are off-distribution and can inject failure modes.
- **ReST-EM (K=32, filter only):** gets correct traces only on problems where pass@32 > 0. Cost: zero signal from problems where pass@32 = 0.

Ch-31's decision tree node 2 ("is pass@1 between 0.1 and 0.8?") is the ReST-EM constraint made explicit. If pass@1 < 0.1, K=32 may still not be enough; you need either STaR-style rationalization or a harder curriculum. If pass@1 > 0.8, there is no signal left to extract.

---

## Ch-31's default borrowed from ReST-EM

- **Default K for verifier-exact-match tasks:** 32. (Llama-2's K=10 is for RM-scored chat; exact-match is cheaper to compute, so K can grow.)
- **Default iterations:** 2-3. Beyond 3 on the same data, stop.
- **Default diversity cap:** 4 distinct accepted per problem.
- **Default M-step:** 1 epoch, lr=1e-5, batch 128 (for MATH-scale problems).

These are the numbers the HTML companion's slider defaults are chosen to bracket.

---

## Connections

- [[star]] — the predecessor with K=1 + rationalize.
- [[v-star]] — the successor with a learned verifier over partial rationales.
- [[rlvr-tulu3]] / [[deepseek-r1]] — replace the M-step SFT with an RL objective on the same verifier filter.
- [[rejection-sampling-finetuning]] — the pattern ReST-EM is a specific instance of.
- **ch-31 §4** — reference configuration for the "verifier-is-exact-match" branch of the decision tree.
