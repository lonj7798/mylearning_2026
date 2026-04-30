---
chapter: ch-50
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/longbench.md
source_url: https://arxiv.org/abs/2308.14508
created_at: "2026-04-23"
---

# Excerpt: LongBench — why "one long-context score" is a lie

**Source library:** `wiki/raw-data/llm-training/papers/longbench.md`
**Artifact:** 21-task, 6-category bilingual long-context benchmark with explicit per-task-quality disclosure.

---

## Why this source grounds ch-50 §1 and §2

LongBench is the real-task complement to synthetic RULER/NIAH. Its authors explicitly warn that individual tasks within the suite have different long-context sensitivities — some are solvable at 4K context, others require the full 32K. That makes the aggregate "LongBench score" a dangerous number on its own. Ch-50's "per-task vs per-slice" distinction needs an attested failure story inside a single benchmark; LongBench provides it on the realistic-task side.

---

## The 21-task, 6-category decomposition

Source §Task categories:

> ### Single-Document QA: NarrativeQA, Qasper, MultiFieldQA-en/zh.
> ### Multi-Document QA: HotpotQA, 2WikiMQA, Musique, DuReader.
> ### Summarization: GovReport, QMSum, MultiNews, VCSum (zh).
> ### Few-shot learning: TriviaQA, SAMSum, TREC (long-context few-shot).
> ### Synthetic tasks: PassageRetrieval-en/zh, PassageCount.
> ### Code completion: LCC, RepoBench-P.

Six categories, 21 sub-tasks. The minimum informative slicing is *category*, not task — a category-level delta catches "the model got better at QA but worse at summarization," which a per-task report of 21 rows makes harder to see. The maximum informative slicing is *task*, where a specific gold-versus-distractor failure mode gets isolated. Ch-50 §2's composition rule (per-task and per-slice *compose*) is why reports give both: category roll-ups for go/no-go, per-task detail for triage.

---

## The gotcha that justifies per-slice rigour

Source §Risks + gotchas:

> **Task quality varies:** some tasks (NarrativeQA) are solvable with ~4K context; real long-context signal is strongest on multi-doc QA and synthesis.

This is the single most important sentence for ch-50's motivation. NarrativeQA is in the "long-context" benchmark but does not require long context. A checkpoint that gets better on NarrativeQA can raise the LongBench aggregate without improving long-context capability. A checkpoint that gets better on multi-doc QA (HotpotQA, 2WikiMQA, Musique) is improving long-context capability. The aggregate cannot tell these apart; per-slice can.

Ch-50's mock slice-report HTML shows this effect in a row: `LongBench-en / NarrativeQA (4K-ok)` can be positive while `LongBench-en / HotpotQA multi-doc` and `LongBench-en / 2WikiMQA multi-doc` are both negative. Aggregate is slightly positive. The aggregate verdict is wrong.

---

## Contamination as a per-task bias, not a per-benchmark bias

Source §Risks + gotchas:

> **Contamination:** NarrativeQA, HotpotQA, TriviaQA are older benchmarks likely seen during pretraining.

Contamination attacks specific tasks, not the benchmark as a whole. A per-slice ledger has a row `contamination-suspect` flag per task. Ch-50's ledger (§5) extends naturally: `first_seen_run` per task lets a reader see that `TriviaQA` score saturated in round 1 while `Musique` kept improving. Saturation + contamination-suspect = slice is no longer a useful signal, drop it or weight it down.

The per-benchmark contamination narrative loses this: "LongBench is partially contaminated" is un-actionable. "TriviaQA and HotpotQA are the contaminated slices, weight them to zero in the next report" is actionable.

---

## Input-length distribution — a within-task slice axis

Source §Dataset size:

> Input length distribution: median ~10K, tail to 31K.

A single task can be further sliced by input length. A model's accuracy on HotpotQA-long at 8K vs 16K vs 31K is three different signals. Ch-50's `long-context-aggregation` bucket and RULER's length-axis slicing both apply *within* LongBench tasks. The report does not always do this slicing; the authors left it available via raw data.

---

## Few-shot learning category — a different failure mode

Source §Task categories:

> ### Few-shot learning: TriviaQA, SAMSum, TREC (long-context few-shot).

Few-shot-in-long-context is a distinct failure axis from QA. A model can retrieve one needle well (S-NIAH) and still fail to use 10 examples in its context effectively (few-shot). The bucket `few-shot-ignored-examples` is a named slice the LongBench category indirectly measures. Ch-50 §3's "cluster-by-reason" ontology can split this from pure-retrieval or pure-summarization failures.

---

## LongBench-v2 — the ledger row as a time series

Source §Key Contributions:

> Updated LongBench-v2 (2024) with more Chinese coverage and longer contexts.

Benchmarks themselves evolve. A ledger that tracks `benchmark_version` lets a reader compare "v1 score at launch" vs "v2 score in a later run" honestly. Mixing versions is a classic aggregate-hides-story failure: v2 is harder, so an apparent regression may be a version change, not a capability change. Ch-50 §5's ledger rows include `run_id` specifically to guard against this kind of confound.

---

## Per-benchmark correlation to human judgement — a slice axis at the judge layer

Source §Quality / diversity evaluation (of benchmark):

> Correlation with human-judged long-context quality: strong for QA and summarization; weaker for few-shot.

Correlation-to-human varies per category. That is itself a slice axis: a per-category "judge-trust" weight that ch-49's judge-calibration work would set. Ch-50's failure bucketing downstream of LongBench should weight QA failures more heavily than few-shot failures when estimating "real long-context regression." Uniform weighting of all 21 tasks is the default and it is wrong.

---

## English / Chinese bilingual coverage — why "language-drift" is a named bucket

Source §Key Contributions:

> **Bilingual coverage** — English and Chinese each.

A model trained mostly on English data can score fine on LongBench-en and crash on LongBench-zh. Aggregate LongBench score weighted 50/50 would under-report the crash; per-language slicing names the bucket `language-drift` (ch-50 §5) and lets the regression memo say "zh is the broken slice, en is stable."

---

## Connections to ch-50

- **§1 aggregate-hides-story** — NarrativeQA vs multi-doc QA is the inside-one-benchmark version of Tulu 3's inside-one-pipeline argument.
- **§2 per-slice-beats-per-task** — 6-category roll-up is the minimum; 21-task detail is the maximum.
- **§3 cluster-by-reason** — few-shot-ignored-examples, language-drift, aggregation-failure are named reason-buckets LongBench surfaces.
- **§5 failure-ledger** — version-tracking + contamination-suspect flag are ledger columns LongBench justifies.
- **§6 three-line-vs-50-slice** — a category roll-up is the three-line view; the per-task report is the 50-slice view. Never substitute one for the other.
- **[[ruler]]** — synthetic counterpart; category-level mapping (retrieval / tracing / aggregation) is sharper in RULER but shares the same slicing logic.
