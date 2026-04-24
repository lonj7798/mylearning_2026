---
chapter: ch-53
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/ruler.md
source_url: https://arxiv.org/abs/2404.06654
created_at: "2026-04-23"
---

# Excerpt: RULER — slice generators, not slice files

**Source library:** `wiki/raw-data/llm-training/papers/ruler.md`
**Anchor paper:** Hsieh et al. 2024 — "RULER: What's the Real Context Size of Your Long-Context Language Models?"

---

## Why this source anchors the optional long-context slice of ch-53

The ch-53 harness accepts static `Sample` lists as its default input. The moment the learner extends it to long-context evaluation, static files stop being adequate: length and complexity need to vary independently, and a fixed corpus bakes them together.

From `ruler.md` §Abstract:

> The durable contribution is the controlled generation protocol: labs can vary sequence length and task complexity independently and inspect which long-context behaviors break first.

Treat RULER as a `TaskSpec` factory, not a benchmark file. The harness extension is: pass a `generator_fn(length, complexity) -> list[Sample]` where the static `samples` would normally be.

---

## The four task families — what slices to emit

From `ruler.md` §Task families and generation protocol:

> - **Retrieval:** RULER extends needle-in-a-haystack into four retrieval families. `S-NIAH`, `MK-NIAH`, `MV-NIAH`, `MQ-NIAH`.
> - **Multi-hop tracing (`VT`):** variable-binding chains.
> - **Aggregation:** `CWE` (Common Words Extraction), `FWE` (Frequent Words Extraction).
> - **QA:** SQuAD and HotpotQA converted into long-context settings.

Each family becomes a slice key. A RULER run in the ch-53 harness emits samples with `slices = {"family": "MK-NIAH", "length": "32k", "difficulty": "full_haystack"}`. The comparator then asks: did the RL checkpoint regress on `family=VT` at `length=64k`? This is the one question a single "long-context accuracy" number cannot answer.

---

## The generator knobs — concrete bucketing for slices

From `ruler.md` §Concrete generation knobs:

> **Context length:** examples are generated at `4K`, `8K`, `16K`, `32K`, `64K`, and `128K` tokens.
> **Needle type:** keys and values can be **words**, **7-digit numbers**, or **32-digit UUIDs**.
> **Haystack type:** distractor background can be repeated noise sentences or natural long text such as **Paul Graham essays**.
> **Distractor density:** `MK-NIAH` can scale from `4` keys to a haystack filled entirely with distractor needles.
> **Output cardinality:** `MV-NIAH` and `MQ-NIAH` turn retrieval into multi-item recall rather than single-span lookup.
> **Tracing difficulty:** `VT` increases complexity by increasing the number of chains or the number of hops per chain.

Each knob is a slice dimension. The harness's slice table has columns for `length`, `needle_type`, `haystack_type`, `distractors`, `cardinality` — each generated sample is tagged with its exact knob configuration. A regression at `length=64k & haystack_type=essay & cardinality=4` tells you a very specific thing; at `length=64k` in aggregate it tells you little.

---

## The metric — why recall-based, not exact-match

From `ruler.md` §Metrics and evaluation protocol:

> Accuracy is computed with **recall-based matching** of the target outputs.
> **Effective context size** is defined as the maximum length whose average score stays above the `Llama2-7B @ 4K` baseline of `85.6`.

On multi-value and multi-query tasks, partial recall matters: returning 3 of 4 UUIDs is genuinely more useful than returning 0. The ch-53 harness metric for RULER-style tasks returns a recall fraction in `[0, 1]` per sample, not a hard 0/1. The bootstrap CI operation is identical — `metric_fn` returns a float either way.

---

## The effective-context-size metric — a harness gate, not a headline

The paper defines `effective context` as the longest input length whose average accuracy stays above a baseline (Llama2-7B at 4K = 85.6). The ch-53 harness exposes this as a derived metric rather than a raw number:

```python
def effective_context(task_results, baseline_score=0.856):
    # task_results is {length -> mean_accuracy} at a fixed task config
    for length in sorted(task_results.keys(), reverse=True):
        if task_results[length] >= baseline_score:
            return length
    return 0
```

Reporting `effective_context=32k` for ch-44-RL vs `effective_context=64k` for ch-34-SFT is a cleaner regression signal than a row-by-row accuracy table. The memo carries both.

---

## The 500-per-task-per-length convention

From `ruler.md` §Metrics and evaluation protocol:

> The paper evaluates **500 generated examples per task per length**.

500 is the ch-53 default sample count for generator-based tasks. Low enough to fit the resource-constrained path (13 task configs x 6 lengths x 500 = 39,000 rollouts at 32k average context = infeasible without caching), high enough to keep the bootstrap CI meaningful. The caching discipline from OLMES applies: generate once per `(task_config, length)` against each checkpoint, cache, then vary the scorer offline.

---

## What this source does not tell you

RULER does not define a judge protocol (the tasks are self-verifying). For harness design, this is a feature — no judge bias to probe. But for learners who extend ch-53 to include long-context subjective evals (long-form writing, long-document summarization), the RULER framework does not generalize; the judge-bias probe from §5 re-enters and runs on top of generator output. Treat RULER as the template for objective long-context slices; treat MT-Bench-style judging as the complementary subjective lane.

---

## What carries forward

Later chapters (if any future course extends to long-context RL) will want a RULER-shaped generator to stress-test post-training effects on long-context retention. The ch-53 harness is built to accept that extension without a rewrite: `TaskSpec.samples` is a `Sequence`, not a `list`, so a lazy generator is a drop-in replacement. Karpathy's "generalize a special case" rule ([[karpathy-training-neural-net-recipe]]) applies: build the harness against fixed data first, then lift to generators.
