---
chapter: ch-51
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/model-reports/olmo-2.md
source_url: https://arxiv.org/abs/2501.00656
created_at: "2026-04-23"
---

# Excerpt: OLMo 2 — single-digit-pp per-stage gains are exactly the CI regime ch-51 targets

**Source library:** `wiki/raw-data/llm-training/model-reports/olmo-2.md`
**Artifact:** SFT → DPO → RLVR per-stage lifts as single-digit percentage points; the regime where CI halfwidth determines whether you ship.

---

## Why this source is the ch-51 "headline number" case study

Source §Technical Details / Reported post-training gains:

> RLVR stage lifts GSM8K and MATH consistently for 7B and 13B (single-digit pp gains). DPO stage contributes most of the chat-quality / IFEval lift.

Single-digit pp gains are exactly the regime where ch-51 matters. At N=500 (MATH500) a 95% percentile-bootstrap CI has halfwidth ≈ 4.4 pp at p=0.5 — *wider than a 3 pp lift*. To claim the RLVR gain is real, OLMo 2 must either (a) report paired evaluation against the pre-RLVR checkpoint (paired CI is ~2× tighter), or (b) run ≥2 seeds and show consistent sign — ch-51's "two-run minimum" rule applied to the public process.

---

## The three-stage pipeline as three CI decisions

Source §Technical Details / Post-training:

> SFT: OLMo-specific variant of Tulu 3 SFT mix (~939K prompts).
> DPO: on-policy preferences generated from the SFT checkpoint + Tulu 3 preference mix.
> RLVR: PPO with verifiable rewards (GSM8K/MATH exact-match, IFEval constraint checks, code unit tests). LR 3e-7, beta KL 0.05, clip eps 0.2, GAE lambda 0.95.

Each arrow is a go/no-go. The ch-51 memo template in §6 is designed for exactly this: SFT → DPO is one memo, DPO → RLVR is another. The claim field is scoped to one transition; the evidence field is a paired bundle across reasoning + chat + safety; the reviewer's check list contains the specific commands to rerun.

---

## Verifiable-reward evals have zero judge variance — but not zero noise

Source §Technical Details / RLVR:

> RLVR: PPO with verifiable rewards (GSM8K/MATH exact-match, IFEval constraint checks, code unit tests).

The reward is a verifier (string match, unit-test pass), so σ_judge ≈ 0. But decode variance at training-time temperature T=1.0 is non-trivial; eval at T=0 is the convention. Ch-51 §1's per-run variance row for decode reads: "T=0 → 0; T=0.7 → 0.5–1.5 pp; T=1.0 → 1–3 pp." OLMo 2's eval protocol of T=0 at eval is exactly the variance-minimizing choice; ch-51's noise budget template adopts it as the default.

---

## 32B claim requires the harshest CI

Source §Key Contributions:

> 32B variant is the first fully-open model to beat GPT-3.5 and GPT-4o-mini on average benchmarks.

"Beats by average" across benchmarks is a *bundle* claim. If individual per-benchmark deltas are mostly within their CIs but the bundle mean is not, you need to compute a CI for the bundle mean — which is tighter than any individual CI because N_bundle is the sum. Ch-51 §6 memo treats this as the default: the claim is on a 4-task bundle, and the paired CI is on the bundle-averaged delta, not on any single task.

---

## What ch-51 borrows wholesale

- Per-stage reporting discipline (SFT / DPO / RLVR as three separate memos).
- T=0 at eval as default to zero out decode variance.
- Publishing the bundle composition alongside per-task deltas.

## What ch-51 adds

- Explicit per-task and bundle CI numbers.
- Paired-bootstrap + sign-test concordance as a double-check.
- The variance-source table with labeled per-source σ.

---

## Connections

- **[[tulu-3]]** — inherited post-training recipe; the concrete setting of OLMo 2's eval.
- **[[rlvr-tulu3]]** — RLVR-verifier detail that zeroes out σ_judge.
- **[[llama-3]]** — contemporaneous per-round gains with the same CI regime.
- **ch-51 §6 memo** — template derived from the OLMo-2 per-stage workflow.
- **[[olmo-3]]** — successor that adds 1000+ intermediate checkpoints and raises the rolling-average question from ch-51 §5.
