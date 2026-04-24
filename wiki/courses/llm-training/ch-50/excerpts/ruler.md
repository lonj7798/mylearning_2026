---
chapter: ch-50
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/ruler.md
source_url: https://arxiv.org/abs/2404.06654
created_at: "2026-04-23"
---

# Excerpt: RULER — the 13-task suite as a purpose-built slice report

**Source library:** `wiki/raw-data/llm-training/papers/ruler.md`
**Artifact:** Synthetic long-context benchmark with configurable task generators across retrieval / tracing / aggregation / QA categories.

---

## Why this source is ch-50's most elegant slice-report example

RULER is already a per-slice report; its authors performed the correlation study and selected the 13 tasks *so that they cover distinct failure modes, not redundant variants*. That selection — "find the minimum set of slices that each reveals a different failure mode" — is ch-50 §2 / §3's job, done once and published. A learner who does not know how to design an eval slice set can read RULER Table 5 and copy the template.

---

## The "effective context size" gap — ch-50 §1 made unavoidable

Source §Key Figures/Tables to Study:

> **Table 3** - the practical headline: claimed context length versus effective context length across models.

Effective context size is the per-slice analogue of the per-task context-window claim. Table 3 reports, per model, the longest length at which the 13-task *average* stays above the Llama2-7B@4K baseline. That is a slice-aware definition: it fails the model once any of the 13 slices crashes past a threshold, not only when the aggregate crashes.

Ch-50's opening claim ("One aggregate number cannot tell you whether a checkpoint is better") has no cleaner attestation: models claiming 128K windows measure effective windows of 8K to 32K, and the gap is caused by slices that saturate at different lengths.

---

## The correlation study — how to pick slices that are not redundant

Source §Key Contributions:

> Defines **13 representative task settings** selected from a larger configuration space after a task-correlation study, so the benchmark covers distinct failure modes instead of redundant variants.

This is the only raw-data source that describes a slice-selection procedure. The logic: enumerate a configuration space (here: task type x needle format x haystack type x distractor density x output cardinality x chain depth), score a pilot set of models on all of them, correlate pairs of slices, and prune any slice whose score correlates >0.95 with another slice's — the redundant one adds no information.

Ch-50 §2's rule "slice by the axis your next decision depends on" is the principle; RULER's correlation study is the algorithm. A team building a new eval should run the same procedure before committing to a slice set — otherwise half the slices are telling you the same thing.

---

## Four failure modes RULER explicitly separates

Source §Technical Details — Task families:

> - **Retrieval** (S-NIAH / MK-NIAH / MV-NIAH / MQ-NIAH): single vs multi needle, plus distractor and recall variants.
> - **Multi-hop tracing (VT)**: variable-binding chains.
> - **Aggregation (CWE / FWE)**: count / rank across the full context.
> - **QA (SQuAD-long / HotpotQA-long)**: QA with gold-document-plus-distractor padding.

Four categories = four ch-50 §5 ledger rows on the long-context side. Each has a canonical bucket name:

- **`long-context-needle-format`** — S-NIAH holds, MK-NIAH-UUID crashes → the model can retrieve by word-key but not by UUID-key.
- **`long-context-distractor-density`** — MK-NIAH-4-keys holds, MK-NIAH-full-haystack crashes → the model ignores low-density distractors but drowns under high-density.
- **`long-context-multi-target`** — S-NIAH holds, MV/MQ-NIAH crashes → single retrieval works, multi-item recall fails.
- **`long-context-tracing`** — retrieval holds, VT crashes → chain-of-bindings is a different skill from lookup.
- **`long-context-aggregation`** — retrieval holds, CWE/FWE crashes → counting/ranking across the full context collapses.

Ch-50 §5's canonical-buckets list pulls two of these (`long-context-distractor-density`, `long-context-aggregation`) directly; the other three are available to any long-context-focused ledger.

---

## Independent knobs — the slice axis design principle

Source §Technical Details:

> The key design goal is to hold the evaluation domain narrow and controlled so that **input length** and **task complexity** can be varied independently.

Independent knobs is the cleanest statement of what a slice axis must be. If you cannot vary one axis while holding the others fixed, you cannot attribute a regression to that axis. Ch-50 §4's signed-delta test assumes axis independence — a signed Δ on `VT-4-hops` means something only if length, distractor density, and needle format are held fixed across the A vs B comparison.

The contrapositive is ch-50's warning against confounded slices: a benchmark that entangles length with task type (e.g., NarrativeQA is always ~10K, Musique is always ~20K) cannot cleanly separate "length failure" from "task-type failure." RULER avoids this by design.

---

## The 13-task suite — a designed slice set

Source §Technical Details — Representative 13-task suite used in the paper:

> - `S-NIAH`: word->number with repeated-noise haystack, roughly passkey retrieval.
> - `S-NIAH`: word->number with essay haystack, roughly vanilla NIAH.
> - `S-NIAH`: word->UUID with essay haystack.
> - `MK-NIAH`: `4` keys [...] `MK-NIAH`: full-haystack [...]
> - `MV-NIAH`: `4` values for one key.
> - `MQ-NIAH`: `4` queried keys.
> - `VT`: `1` chain and `4` hops.
> - `CWE`: `10` common words [...] `FWE`: `alpha = 2.0` [...]
> - `QA`: SQuAD long-context / HotpotQA long-context.

Thirteen slices, each tested at `4K, 8K, 16K, 32K, 64K, 128K`. Total eval cells = 13 x 6 = 78. A per-cell report is overwhelming; RULER's aggregate-per-model-per-length (Table 3) is the triage view. The per-cell raw data is the regression-triage view. Same data, two views, ch-50 §6's decision matrix again.

---

## Recall-based accuracy + weighted averages

Source §Technical Details — Metrics and evaluation protocol:

> Accuracy is computed with **recall-based matching** of the target outputs.
> [...]
> Two weighted averages are reported: `wAvg. (inc)` and `wAvg. (dec)`, where the weights increase or decrease linearly with context length.

`wAvg. (inc)` weights long-length cells more; `wAvg. (dec)` weights short-length cells more. A single model has two headline numbers depending on which question you are asking: "does it work at the long end?" vs "does it work overall?" Ch-50 §6's three-line-vs-50-slice rule — "pick by decision" — is RULER's authors' rationale. They refused to commit to one average; they publish both.

---

## Connections to ch-50

- **§1 aggregate-hides-story** — effective-context-size gap vs claimed window is the direct attestation.
- **§2 per-slice-beats-per-task** — the 13-task suite *is* per-slice-beats-per-task for long-context.
- **§3 cluster-by-reason** — five reason-buckets (needle-format, distractor-density, multi-target, tracing, aggregation) all named from RULER categories.
- **§4 when-is-regression-real** — independent knobs is the precondition for axis-attributable signed deltas.
- **§5 failure-ledger** — `long-context-distractor-density` and `long-context-aggregation` are RULER-sourced bucket names.
- **§6 three-line-vs-50-slice** — Table 3 (aggregate-per-length) vs full 78-cell table is the exact two-view design.
- **[[longbench]]** — realistic-task counterpart; RULER is the controlled-knob version of the same argument.
