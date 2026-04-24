---
chapter: ch-50
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/model-reports/olmo-2.md
source_url: https://arxiv.org/abs/2501.00656
created_at: "2026-04-23"
---

# Excerpt: OLMo 2 — stage-attribution as the cleanest per-slice ledger

**Source library:** `wiki/raw-data/llm-training/model-reports/olmo-2.md`
**Artifact:** Stage-by-stage (SFT -> DPO -> RLVR) per-benchmark gain table; architectural-stability ablation table.

---

## Why this source is ch-50's canonical stage-attribution example

OLMo 2 is the open-data counterpart to Llama 3 for the ch-50 teaching: its release table reports a separate column per stage, so a reader can answer "which stage produced which gain on which benchmark?" with no guesswork. Ch-50 §6 lists "which stage regressed?" as one of the five decisions; OLMo 2's table is the answer format.

---

## The per-stage per-benchmark ledger

Source §Technical Details — Reported post-training gains:

> RLVR stage lifts GSM8K and MATH consistently for 7B and 13B (single-digit pp gains).
> DPO stage contributes most of the chat-quality / IFEval lift.

This is the entire ch-50 §5 failure-ledger concept in two sentences. Three cells attested:

- `(RLVR, GSM8K)` and `(RLVR, MATH)` are positive with single-digit pp gains.
- `(DPO, IFEval)` is the positive cell for chat-quality.
- `(SFT, *)` is the baseline the two above are measured against.

The dual of each positive cell is an implicit negative: `(DPO, GSM8K)` is flat (the reader infers this because if DPO had lifted GSM8K, RLVR would not be the attributed stage). Ch-50 §4's "effect-size threshold" rule is why: DPO's GSM8K move is below threshold, so it is rolled into the SFT baseline.

---

## Architectural-stability ablation — slicing at the pretraining side

Source §Key Contributions:

> Architectural stability recipe: RMSNorm + reordered norm + QK-Norm + RoPE + Z-loss. Prevents the training-spike phenotype that plagued OLMo 1.

Source §Key Figures/Tables to Study:

> **Architecture ablation table:** which stability trick (QK-Norm, Z-loss, reorder) contributes which fraction of the spike-free runs.

"Fraction of spike-free runs" is a per-configuration slice, with `spike` as the named failure bucket. The ablation table is a ch-50 §5-style ledger: rows = tricks, columns = {attempted_runs, spike_free_count, spike_free_rate}. Turning it off removes a stability trick; the spike-rate bucket swells; the signed delta per-trick is the ablation's punchline.

Ch-50's rule "cluster by *reason* not surface" applies at the pretraining stage too: "spike" is a reason-bucket, "loss_spike_magnitude > 2x" is a surface signal. OLMo 2's ablation names the spike bucket and measures its rate per configuration — the exact pipeline ch-50 §3 codifies.

---

## SFT -> DPO -> RLVR on a non-Llama base — the transportability evidence

Source §Key Contributions:

> Confirms the Tulu 3 recipe generalizes: SFT -> DPO -> RLVR works on a non-Llama base without modification.

"Generalizes" is a per-slice claim. To confirm it, you run the same three-stage recipe on OLMo and measure the per-stage per-benchmark gain pattern; if the pattern (DPO lifts IFEval, RLVR lifts GSM8K/MATH) matches Tulu 3's, the recipe generalizes. If you only looked at the final aggregate, "generalizes" would mean "similar final number," which is weaker — two different stage-attribution patterns can produce the same aggregate.

Ch-50 §1's argument against aggregation is exactly this. The per-stage table is what lets OLMo 2 claim transportability honestly.

---

## Two-stage pretraining cooldown — a slicing decision masquerading as a curriculum

Source §Technical Details — Pretraining:

> **Stage 1 data:** OLMo-Mix-1124 — ~3.9T tokens [...].
> **Stage 2 cooldown data:** Dolmino mix — curated higher-quality subset, ~50B tokens.

The two-stage split exists because per-slice evaluation on the Stage 1 checkpoint showed specific slices (math, reasoning, code) underperforming. Stage 2 is a *targeted* correction: ~50B tokens of higher-quality data aimed at the underperforming slices. The decision to do a cooldown at all is a ch-50 §6 "which slice is under the threshold?" triage outcome.

---

## Spike-free fraction as a ledger across runs

Source §Key Figures/Tables to Study:

> **Architecture ablation table:** which stability trick (QK-Norm, Z-loss, reorder) contributes which fraction of the spike-free runs.

Read this ablation as a time-series ledger: OLMo 1 runs had a non-trivial spike-rate; adding QK-Norm reduced it; adding Z-loss reduced it further; reordered norm closed it out. Each row is a ledger entry, each column a stage. Ch-50 §5's "ledger row across runs" framing turns this one table into a historical artifact — three independent interventions retiring a named failure bucket.

---

## Size-by-size re-tuning — when per-slice thresholds differ by config

Source §Technical Details — Post-training (Tulu 3 recipe):

> **DPO:** on-policy preferences generated from the SFT checkpoint + Tulu 3 preference mix. Beta/LR per-size (following Tulu 3 defaults; LRs re-tuned lightly).

"LRs re-tuned lightly" means the per-size per-slice sweep produced a slightly different optimum at 7B vs 13B vs 32B. If you aggregated by averaging win-rate across sizes, you would pick a single LR that is wrong for at least one size. Slicing by size *inside* the DPO tuning phase is what lets "Beta/LR per-size" happen — a pre-eval slicing decision.

---

## Connections to ch-50

- **§1 aggregate-hides-story** — OLMo 2's per-stage per-benchmark gain table is the direct rebuttal to "one post-training number."
- **§2 per-slice-beats-per-task** — stage x benchmark is the minimum viable slicing for post-training.
- **§3 cluster-by-reason** — `spike` is a named reason-bucket at the pretraining stability layer.
- **§5 failure-ledger** — architecture ablation table *is* a failure-ledger across intervention runs.
- **§6 three-line-vs-50-slice** — OLMo 2-Instruct release headline uses the three-line format; the technical report uses the stage x benchmark table.
- **[[tulu-3]]** — recipe source; OLMo 2's per-slice gain pattern matches Tulu 3's, which is the transportability claim.
