---
chapter: ch-51
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/model-reports/llama-3.md
source_url: https://arxiv.org/abs/2407.21783
created_at: "2026-04-23"
---

# Excerpt: Llama 3 — six iterative rounds as the setting where paired CIs matter every round

**Source library:** `wiki/raw-data/llm-training/model-reports/llama-3.md`
**Artifact:** SFT → Rejection Sampling → DPO loop; per-round preference data regenerated from current-best checkpoint.

---

## Why this source is ch-51's canonical go/no-go setting

Source §Core Insight:

> Six iterative rounds of SFT -> Rejection Sampling -> DPO on increasingly high-quality synthetic data beats a single-pass RLHF pipeline.

Each of those six rounds is a go/no-go decision. The checkpoint chosen as "best" at round N becomes the generator for round N+1's synthetic data and rejection-sampling pool. A single-seed delta that happens to favour a noisy variant *propagates forward* for all remaining rounds. Ch-51's "two-run minimum" rule and paired-bootstrap requirement are direct responses to this failure mode.

---

## The rejection-sampling inner loop amplifies eval noise

Source §Technical Details / SFT:

> Rejection sampling: for each prompt, sample K=10–30 completions from the best round-(N-1) chat model at temperature T=0.6–1.0, then keep the top by RM score.

The RM here is itself an estimator with noise. The "best" completion is chosen by an RM that has its own variance; the SFT data then inherits that variance. Ch-51's §1 variance budget must include *RM variance* when the eval is downstream of RM-gated synthetic data — not only decode variance on the eval set itself.

---

## DPO hyperparameters as the ceiling on detectable effect

Source §Technical Details / DPO:

> Learning rate: 1e-5; Beta (KL coefficient): 0.1; Auxiliary NLL loss coefficient: 0.2; Single epoch per round.

Per-round DPO gains are typically sub-2 pp on any single benchmark. Ch-51 §3 shows that N=500 (MATH500) has a 95% CI halfwidth of ~4.4 pp — *wider than the expected effect*. So Llama-3-style rounds require either (a) paired comparisons (halfwidth typically 2× tighter than unpaired) or (b) composite bundles (averaging across 4 tasks with total N ~ 2500 brings HW to ~2 pp). Ch-51 §6's memo template uses a 4-task bundle explicitly for this reason.

---

## The "most-recent-batch only" rule implies a per-round eval

Source §Technical Details / DPO:

> Most-recent-batch preference data only (older batches cause format drift).

This is the eval-side dual of ch-51 §5's "never smooth across a trend reversal." Format drift between rounds is a real shift — averaging rounds together masks it. Each round must be evaluated in isolation; the memo's "claim" is round-over-round, not cumulative.

---

## Llama-3 does NOT publish CIs — ch-51's gap

The Llama 3 technical report does not publish item-level bootstrap CIs alongside its benchmark tables. This is a documented gap in the 2024–2025 frontier-model-report genre; ch-51 exists partly to close it in internal process even when external reports omit it. The guideline: regardless of what the public report prints, the internal go/no-go memo has CIs. A "ship" decision is never made on the public-table granularity.

---

## Connections

- **[[dpo]]** — the per-round objective whose gains need CI bracketing.
- **[[rejection-sampling-finetuning]]** — the upstream step whose RM variance propagates into the next round.
- **[[olmo-2]]** — contemporaneous, explicitly per-stage gain reporting; same CI regime.
- **ch-51 §4** — paired bootstrap and two-run minimum tuned for this iterative setting.
- **[[reward-model-overoptimization]]** — what happens if you do *not* run gold-reward on a held-out slice each round.
