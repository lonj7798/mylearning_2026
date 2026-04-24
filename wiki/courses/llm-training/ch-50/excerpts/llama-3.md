---
chapter: ch-50
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/model-reports/llama-3.md
source_url: https://arxiv.org/abs/2407.21783
created_at: "2026-04-23"
---

# Excerpt: Llama 3 Herd — the per-capability iterative ledger

**Source library:** `wiki/raw-data/llm-training/model-reports/llama-3.md`
**Artifact:** Six-round SFT -> Rejection Sampling -> DPO loop with per-capability synthetic pipelines and an explicit "failure modes" disclosure.

---

## Why this source grounds ch-50

Llama 3's post-training is the canonical example of slice-level discipline at scale. The report does not report one post-training number — it reports gains *per capability* across *six rounds*, with each round re-measuring the per-capability slices against a fresh reward model. Ch-50 §1 "aggregate hides the story" and §5 "failure-ledger as row-across-runs" both have their empirical shape here.

---

## Per-capability synthetic pipelines — slicing at the data level

Source §Key Contributions:

> Heavy synthetic-data generation for coding, math, multilingual, reasoning, long-context, tool use, and factuality — each capability gets a dedicated synthetic pipeline.

Seven named capabilities, each with its own pipeline, which is the pre-slicing of the training-data side. The eval side must match: the team measures per-capability scores per round, otherwise "Round 3 improved" is meaningless — improved where? The per-capability synthetic pipelines implicitly define the per-slice eval axis ch-50 §2 names.

Ch-50's guideline "slice by the axis your next decision depends on" is why Llama 3 slices by *capability* (the pipeline being re-generated) rather than by dataset source. Each round's decision is "which capability pipeline needs another iteration?" — a per-capability question.

---

## Six-round loop = six snapshots of the same ledger

Source §Technical Details — Overall structure:

> Six rounds of: (a) Reward Model (RM) training, (b) Rejection Sampling to build the round's SFT pool, (c) SFT on curated pool, (d) DPO on preference data collected with the latest RM-ranked generations. Each round uses a fresh batch of ~human preference annotations plus synthetic data resampled from the round-N-1 best model.

Six rounds gives five signed-delta measurements per slice (Round 2 - Round 1, ..., Round 6 - Round 5). That is exactly the ch-50 §5 ledger format: `run_id = round_i`, one row per capability-slice, `count` = failure-bucket count from that round's spot-checks. Because the RM is retrained each round, round-over-round deltas wash out single-round RM noise — an empirical answer to ch-50 §4's "two-run minimum" worry at production scale.

---

## Rejection sampling with K=10-30 — the selection lever

Source §Technical Details — SFT:

> **Rejection sampling:** for each prompt, sample K=10–30 completions from the best round-(N-1) chat model at temperature T=0.6–1.0, then keep the top by RM score.

K and T are the knobs; the *per-capability pass-rate distribution* is the measurement. A capability whose K=10 rejection-sampling pass-rate stays flat over rounds is a ledger row that is not shrinking — a signal that SFT data alone is not enough for that slice and a different lever (DPO preferences, new synthetic pipeline) is needed. This is precisely the "ledger row across runs" read ch-50 §5 prescribes.

---

## DPO with NLL stabilization — a bug that would have been invisible without slicing

Source §Key Contributions:

> DPO with auxiliary NLL loss (coeff 0.2) on chosen sequences to prevent chosen-logprob collapse.

Chosen-logprob collapse is a slice-level failure. Aggregate DPO win-rate can rise while the logprob on chosen sequences collapses — the policy becomes good at *beating rejected* while becoming *bad at producing chosen*. Ch-50's unsigned-delta discussion (§4 "Unsigned deltas — a different question") is the cleanest frame for this: signed `chosen-vs-rejected` margin improves, unsigned `|chosen-logprob shift|` is huge, and only slicing the evidence exposes the shape change.

The Llama 3 team's fix (NLL coef 0.2) was motivated by slice-level evidence; a team reading only the aggregate margin would not have proposed it.

---

## Preference-data granularity — margin labels as a slice axis

Source §Technical Details — Reward Model:

> Preference data: human annotators rank two responses from different Llama 3 variants with margin labels ("significantly better", "better", "slightly better", "negligibly better").

Four margin labels = four slices of the preference set. A DPO trained only on the "significantly better" slice generalizes differently than one trained on "slightly better"; slicing the held-out win-rate by margin label reveals which preference stratum is driving the gain. Ch-50's point that "per-slice beats per-task" applies *inside* the preference set, not only on the eval.

---

## The published failure-ledger

Source §Key Contributions:

> Full disclosure of failure modes (preference-data noise, multi-turn dialog drift) in the data section.

Two named failure buckets persist across rounds: `preference-data-noise` and `multi-turn-dialog-drift`. Ch-50 §5's "canonical buckets" list echoes these two literally. The Llama 3 team publishing them is the standard ch-50 is teaching: a bucket that persists is not a shameful secret, it is the most valuable row of the ledger.

---

## Why most-recent-batch DPO is a slicing decision

Source §Technical Details — DPO:

> Most-recent-batch preference data only (older batches cause format drift).

"Older batches cause format drift" is a per-slice observation — format drift is a bucket name. The fix (drop older batches) is an eval-triage decision: the ledger row for `format-violation` showed older batches produced preferences that, once trained on, re-introduced the bucket. Without slicing by batch age on the preference set, this decision is invisible.

---

## Connections to ch-50

- **§1 aggregate-hides-story** — Llama 3's report never shows one aggregate post-training number; per-capability per-round is the native format.
- **§2 per-slice-beats-per-task** — capability slicing (code / math / multilingual / long-context / tool / factuality) is the operative axis.
- **§4 when-is-regression-real** — six rounds is a built-in replicate budget that makes signed per-slice deltas interpretable.
- **§5 failure-ledger** — Llama 3's §Failure modes is a published ledger; preference-data-noise and multi-turn-drift are canonical bucket names.
- **§6 three-line-vs-50-slice** — the release blog shows a three-line chart; the technical report has the per-capability per-round table. Same team, two artifacts, two audiences.
