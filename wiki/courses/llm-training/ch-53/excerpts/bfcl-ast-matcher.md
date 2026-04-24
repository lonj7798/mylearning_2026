---
chapter: ch-53
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/bfcl.md
source_url: https://gorilla.cs.berkeley.edu/leaderboard.html
created_at: "2026-04-23"
---

# Excerpt: BFCL — AST matching, not substring equality

**Source library:** `wiki/raw-data/llm-training/papers/bfcl.md`
**Anchor page:** Berkeley Gorilla team 2024-2025 — "Berkeley Function-Calling Leaderboard"

---

## Why this source anchors the tool-use extension of ch-53

The default ch-53 harness does not include tool-use because the resource-constrained path prioritizes reasoning and safety. The moment the learner adds a BFCL slice, a new class of scoring bug appears: "was the tool called correctly?" is not a string comparison. This excerpt pins down the correct answer.

From `bfcl.md` §Abstract:

> BFCL is both a benchmark (data) and a methodology (scoring categories + executors).

Methodology is what the ch-53 harness inherits. Data is a file you can swap.

---

## The seven scoring categories — slice keys out of the box

From `bfcl.md` §Scoring categories:

> - **Simple:** 1 call to 1 function.
> - **Multiple:** 1 call to 1 function chosen from >=2 candidates.
> - **Parallel:** >=2 calls to same function in same turn.
> - **Parallel-Multiple:** >=2 calls across multiple functions.
> - **Relevance-Detection:** user query is irrelevant to offered tools -> model must refuse / not call.
> - **Live (V2+):** real user data in above categories.
> - **Multi-Turn (V3+):** sequence of turns with state mutation.

Each becomes a `Sample.slices["category"]`. Reporting a single `function_calling_acc` hides the dimension that matters most: a model can be perfect on `Simple` and catastrophic on `Relevance-Detection` (false-positive tool calls). That slice is where RLVR-tuned models often regress because verifiable rewards on tool-use prompts do not penalize calling an unnecessary tool.

---

## The AST matcher — the core metric contract

From `bfcl.md` §AST matcher:

> Call matching uses an AST comparator:
> 1. Parse predicted call and gold call into (name, kwargs).
> 2. Normalize kwargs: sort by key, strip whitespace, canonicalize literals (e.g., `1.0` == `1`, `"red"` == `'red'`).
> 3. Name must match exactly; kwargs must be equivalent; possible args may be absent if default.

The `metric_fn` for a BFCL slice in ch-53 is not `prediction == gold`; it is:

```python
def bfcl_metric(pred: str, gold: str) -> float:
    # Reference: bfcl.md AST matcher contract.
    pred_calls = parse_function_calls(pred)
    gold_calls = parse_function_calls(gold)
    if len(pred_calls) != len(gold_calls): return 0.0
    # Match by name, then by canonicalized kwargs.
    for p, g in zip(pred_calls, gold_calls):
        if p.name != g.name: return 0.0
        if not kwargs_equivalent(p.kwargs, g.kwargs): return 0.0
    return 1.0
```

The critical canonicalization step — `1.0 == 1`, `"red" == 'red'` — is where ad-hoc scorers routinely fail. An RL checkpoint that learned to emit `temperature=0.7` instead of the gold's `temperature=0.70` must not be penalized. Substring equality over-penalizes formatting drift; an over-lenient matcher under-penalizes wrong arguments. AST matching splits the difference correctly.

---

## Relevance detection — the false-positive lane

From `bfcl.md` §Relevance detection and §Current leaderboard snapshot:

> Model is penalized for calling any tool when query is unrelated; only "no call" or text response is correct.
> Relevance-detection gap: even frontier models still call tools on ~10% of irrelevant queries.

This slice is the tool-use analogue of WildGuard refusal: a model that calls tools too often is not "better at tool use", it is worse in exactly the way that causes production incidents. The ch-53 harness treats `relevance_detection` as a non-aggregatable slice — it is always reported separately, never rolled into a headline function-calling number.

---

## The pass^k metric — stability, not just correctness

From `bfcl.md` §Modality-specific technical details:

> **Pass^k metric:** from V3 onward, key agentic metric - model must succeed on all k independent trials of the same task.

Pass^k is the inverse of pass@k. A model that succeeds 1 in 10 times gets 0 on pass^10 — exactly right for agentic settings where one failed tool call derails the whole task. The ch-53 harness records per-sample outcomes from `k` resampled trials (usually `k=3` in resource-constrained path, `k=5` in full path); the metric module can emit pass@k or pass^k from the same underlying data.

```python
def pass_at_k(outcomes: list[int], k: int) -> float:
    # outcomes are 0/1 per trial.
    return 1.0 if any(outcomes[:k]) else 0.0

def pass_hat_k(outcomes: list[int], k: int) -> float:
    return 1.0 if all(outcomes[:k]) else 0.0
```

---

## The Live-vs-V1 distinction — why the contamination gate still matters

From `bfcl.md` §Risks + gotchas:

> **Benchmark-specific fine-tuning:** some labs train directly on BFCL-style data -> inflated scores. V2 Live mitigates by using unseen real queries.
> **V1 overfit risk:** V1 ceiling has saturated.

V1 data is old enough that many open instruction-tuning mixes contain it. The ch-53 contamination gate must run against BFCL-V1 with a low threshold even if the V1 samples are "public" — public does not mean uncontaminated in the training-set sense. V2 Live is the safer default; V3 Multi-Turn and V4 Agentic are the ones where gains actually track capability rather than memorization.

---

## What this source does not tell you

BFCL is not a safety eval. From `bfcl.md` §Risks + gotchas:

> **Not a safety eval:** BFCL does not score harmful-tool refusal.

A tool-use regression on BFCL tells you nothing about whether the RL checkpoint is now happy to call a `delete_user()` tool on request. The ch-53 safety slice (HarmBench + WildGuard) remains the only safety signal; BFCL sits purely on the capability axis. The memo keeps the two lanes separate because a `GO` on BFCL with a `NO-GO` on HarmBench is a specific, common, and dangerous regression pattern — exactly the kind of pattern an aggregate score buries.
