---
chapter: ch-50
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/bfcl.md
source_url: https://gorilla.cs.berkeley.edu/leaderboard.html
created_at: "2026-04-23"
---

# Excerpt: BFCL — the canonical confusion-matrix decomposition

**Source library:** `wiki/raw-data/llm-training/papers/bfcl.md`
**Artifact:** Multi-category function-calling leaderboard with AST-based call matching and relevance-detection category.

---

## Why this source grounds ch-50 §3's confusion-matrix clustering

BFCL is the sharpest published example of "cluster-by-confusion-matrix." Every function-calling failure is assigned to one of a small set of orthogonal buckets (wrong function name, right function but wrong arguments, called when irrelevant, didn't call when needed, unparseable output). The buckets are the confusion matrix's off-diagonal cells, and each bucket has its own fix path. Ch-50 §3 uses this decomposition as the template any structured-output eval should copy.

---

## The seven scoring categories — per-slice by call type

Source §Evaluation methodology — Scoring categories:

> - **Simple:** 1 call to 1 function.
> - **Multiple:** 1 call to 1 function chosen from ≥2 candidates.
> - **Parallel:** ≥2 calls to same function in same turn.
> - **Parallel-Multiple:** ≥2 calls across multiple functions.
> - **Relevance-Detection:** user query is irrelevant to offered tools → model must refuse / not call.
> - **Live (V2+):** real user data in above categories.
> - **Multi-Turn (V3+):** sequence of turns with state mutation.

Seven ch-50 §5 ledger rows, each with a distinct failure mechanism. A model can be perfect on `simple` (pick one function, fill args) and catastrophic on `relevance-detection` (refuse when no tool applies) — the attested 2025 pattern. Aggregating them into "BFCL score" hides the gap; ch-50's point lands literally.

---

## The AST matcher — where per-failure decomposition becomes automatic

Source §Evaluation methodology — AST matcher:

> Call matching uses an AST comparator:
> 1. Parse predicted call and gold call into (name, kwargs).
> 2. Normalize kwargs: sort by key, strip whitespace, canonicalize literals (e.g., `1.0` ≡ `1`, `"red"` ≡ `'red'`).
> 3. Name must match exactly; kwargs must be equivalent; possible args may be absent if default.

The AST matcher's decomposition is the confusion matrix. Each predicted call produces one of:

- **`correct-call`** (diagonal) — name matches, kwargs equivalent.
- **`name-mismatch`** — wrong function selected; kwargs irrelevant. Fix: retrieval / function-selector training.
- **`kwargs-mismatch`** — right function, wrong arguments. Fix: argument-extraction SFT or instruction-tuning on the arg schema.
- **`call-when-irrelevant`** (hallucinated call) — no tool should have been called. Fix: relevance-detection training, constrained decoding.
- **`missing-call`** — a tool was needed, none emitted. Fix: recall-side training on multi-turn task completion.
- **`unparseable-output`** — model emitted prose or broken JSON. Fix: output-format SFT.

Six buckets, each a named row in ch-50 §5's ledger. The AST matcher emits them automatically, so this is the rare case where cluster-by-confusion-matrix runs without an LLM-judge.

---

## Relevance-detection as the standalone hallucination bucket

Source §Current leaderboard snapshot (2025):

> Relevance-detection gap: even frontier models still call tools on ~10% of irrelevant queries.

One named bucket, one attested number, one persistent ledger row. Ch-50's argument that "a bucket can shrink, grow, or oscillate across runs" is demonstrated by BFCL's leaderboard history: the relevance-detection gap has not closed from V2 (2024) to the 2025 snapshot, meaning this bucket is stuck at ~10% across many models' runs. That is a ledger-shaped insight — the bucket name exists, the count persists, the fix has not arrived.

This is also why the confusion-matrix panel in ch-50's mock HTML report isolates `hallucinated-call` as a distinct off-diagonal cell: it is the single most reliable, cross-model, cross-run failure bucket in modern function-calling eval.

---

## V2 Live — why benchmark-specific fine-tuning is a slice-visible bug

Source §Risks + gotchas:

> **Benchmark-specific fine-tuning:** some labs train directly on BFCL-style data → inflated scores. V2 Live mitigates by using unseen real queries.

Per-slice deltas catch this. A model's V1 score lifts while its V2-Live score stays flat → the signed delta on `V1-simple` is big and positive, on `V2-Live-simple` is near zero → the bucket `benchmark-overfitting` lights up. Aggregate "BFCL score" blurs V1 and V2-Live together and hides the overfit.

Ch-50 §4's "effect-size threshold" extends here: set a different threshold on V1 (higher, because overfit risk) than V2-Live (lower, because it is the unseen-query slice that matters). A V1 lift without a matching V2-Live lift is not a real capability gain.

---

## Pass^k — the variance-adjusted agentic slice

Source §Modality-specific technical details:

> **Pass^k metric:** from V3 onward, key agentic metric — model must succeed on all k independent trials of the same task.

Pass^k is per-slice variance made into a scalar. A model with pass@1 = 0.8 and pass^k=5 = 0.2 has high per-task variance — 80% success once, but 20% success on all of five consecutive tries. Ch-50 §4's "two-run minimum" rule is the pass^k principle applied at the training level; BFCL applies it at the eval level. Both are the same insight — a single-shot number conceals variance.

A failure-bucket ledger row `multi-turn-state-drift` (ch-50 §5) is measured by pass^k explicitly: tasks where pass@1 succeeds and pass^k fails are the pure variance bucket.

---

## The AST matcher's edge cases — a named sub-bucket

Source §Risks + gotchas:

> **AST matcher is lenient on argument order but strict on value canonicalization** — edge cases (list-vs-tuple) cause spurious failures.

Strict on value canonicalization means a `kwargs-mismatch` attribution can be *spurious* — the call is semantically correct but the matcher's literal normalization flagged it wrong. Ch-50 §3's LLM-judge reason-tagger applied to the kwargs-mismatch bucket will often split it into `true-kwargs-error` and `matcher-canonicalization-artifact`. The ledger should carry this sub-split explicitly; otherwise the bucket count is inflated by a known grader bias.

This is ch-49's judge-calibration-meets-ch-50's-bucketing intersection: even a deterministic grader (AST matcher) has calibration failures that surface only under per-bucket audit.

---

## Leaderboard snapshot — the three-line chart view

Source §Current leaderboard snapshot (2025):

> - Top proprietary: GPT-4o-class, Claude 3.7 Sonnet.
> - Top open < 13B: ToolACE-8B, xLAM-2-8B, Hammer 2.1.
> - Top open overall: xLAM-2-70B-fc-r, Llama-4-class derivatives.

Three-line chart for leaderboard communication. The full per-category breakdown is the 50-slice view the leaderboard page also serves. BFCL publishes both — ch-50 §6's decision matrix is again the design principle, not a critique.

---

## Connections to ch-50

- **§3 cluster-by-confusion-matrix** — AST-matcher produces the canonical six-bucket decomposition.
- **§5 failure-ledger** — seven category rows + within-category confusion sub-rows; relevance-detection is the persistent cross-run bucket.
- **§4 when-is-regression-real** — differential thresholds on V1 vs V2-Live catch benchmark-specific fine-tuning.
- **§6 three-line-vs-50-slice** — leaderboard top-3 list vs full category table.
- **ch-49** — AST-matcher canonicalization errors are a judge-calibration artifact that contaminates bucket counts.
- **[[ruler]]** — same "slice-set designed for orthogonality" discipline, applied to long-context.
