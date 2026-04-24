<!-- chapter: ch-50
     track: eval
     kind: content
     title: Slice Analysis and Failure Bucketing
     deps: [ch-49]
     sources: [[llama-3]], [[olmo-2]], [[olmo-3]], [[tulu-3]], [[longbench]], [[ruler]], [[bfcl]], [[physics-of-lm-3]], [[interplay-pretraining-midtraining-rl]], [[karpathy-training-neural-net-recipe]]
     figures: figures/slice-report.html
-->

# Chapter 50 — Slice Analysis and Failure Bucketing

> **Core insight.** One aggregate number cannot tell you whether a checkpoint is better. Every modern post-training report — [[llama-3]]'s six-round loop, [[tulu-3]]'s SFT -> DPO -> RLVR ledger, [[olmo-2]]'s stage-by-stage contribution table, [[ruler]]'s 13-task suite, [[bfcl]]'s seven categories — is built on the same move: decompose the eval into slices, measure per-slice deltas between checkpoints, cluster the failed rollouts into named buckets, and only then decide "ship" or "regress." The aggregate exists to fit on a slide; the per-slice report exists to make the decision. Knowing which one to produce is the skill this chapter names.
>
> **Guideline.** When you evaluate a checkpoint, always compute per-slice scores first and aggregate second — aggregation throws signal away, so throw it away *after* you have seen it. Slice by the axis that your next decision depends on (capability for "does this ship?", difficulty bucket for "is the RL stage doing real work?", judge-category for "is the judge biased?"). For two-checkpoint comparisons, use a paired bootstrap on item-level scores and flag a slice as *regressed* only if the signed delta's 95% CI excludes zero AND the effect size exceeds your pre-registered threshold. Cluster failures by *reason*, not by surface — the canonical reasons are [[reward-hacking-taxonomy]]-style. Maintain a failure-ledger of named buckets that persists across runs; reading a row across runs is more informative than reading a column within a run. Write the three-line chart for go/no-go communication; write the 50-slice report for regression triage. Never invert the two.

---

## §1 Why aggregate hides the story

[[tulu-3]] Figure 1 reports *+0.0 to +2.0 pp on the aggregate average* for the RLVR stage over the DPO-only checkpoint. The aggregate is the punchline of the release post. The *report* tells a different story:

- **GSM8K:** +5 to +10 pp.
- **MATH:** +5 to +10 pp.
- **IFEval:** +~4 pp.
- **MMLU / TruthfulQA / AlpacaEval:** neutral to slightly negative.

The aggregate-average mean obscures a ~10 pp gain on the RL-verifiable slice and a small regression on the knowledge slice. If you only ship when aggregate > 0, you ship RLVR; if your user cares about MMLU, you just regressed. The three-line chart (aggregate per stage) is correct for the blog post; the 50-slice report is correct for the release decision. Neither is "better."

[[olmo-2]]'s own release table makes the same point per stage: DPO does the chat-quality and IFEval lift; RLVR does the GSM8K/MATH lift. Merge the stages into one row and you lose the attribution. Keep them as separate rows and you can answer "which stage should we re-run?" — the actual downstream question.

## §2 Per-slice vs per-task: two decompositions, one rule

Two different axes both get called "slicing." They compose, they do not substitute.

- **Per-task** — split the eval by dataset: MMLU, GSM8K, HumanEval, IFEval, BFCL-V3, RULER-64K, LongBench-en, AlpacaEval-LC. This is the Llama 3 / OLMo 2 / Tulu 3 release-table format. One row per task, one column per checkpoint.
- **Per-slice** — split *within* a task by an attribute of the prompt: [[longbench]]'s 6 task categories or 21 sub-tasks; [[ruler]]'s 13 generator configurations (S-NIAH / MK-NIAH / MV-NIAH / MQ-NIAH / VT / CWE / FWE / QA x length); [[bfcl]]'s 7 scoring categories (simple, multiple, parallel, parallel-multiple, relevance-detection, Java, JS); GSM8K by difficulty bucket; MMLU by domain (STEM, humanities, social sciences, other).

**Rule.** If you have capacity for only one, go per-slice: a single-task per-slice report catches failures per-task cannot. [[ruler]] Table 3 attests this — a model with 32K claimed context scores ~99% on S-NIAH (per-task view: looks fine) while crashing to ~40% on MK-NIAH-full-haystack and ~20% on VT-4-hops (per-slice view: effective window is 8K). The per-task number would have shipped a 32K-marketing model with 8K-real context. [[longbench]] makes the converse observation: NarrativeQA is solvable at 4K context, so "LongBench score" can rise purely from better QA on the easy slice while the long-range slice regresses.

**MMLU-by-domain example.** Checkpoint B beats Checkpoint A by +0.5 pp on MMLU aggregate. Per-domain (inferred representative split; not exact Hendrycks categories):

| Slice | A | B | Δ |
|---|---|---|---|
| STEM | 58.2 | 60.8 | +2.6 |
| Humanities | 62.1 | 61.4 | -0.7 |
| Social Sci. | 65.3 | 65.9 | +0.6 |
| Other | 59.0 | 58.1 | -0.9 |
| **Aggregate** | 61.1 | 61.6 | +0.5 |

The aggregate shows a win. The per-slice report shows that Humanities and Other regressed — and if your RL stage introduced a math-specific SFT mix ([[tulu-3]] §SFT mixture), that regression is the signature of [[interplay-pretraining-midtraining-rl]]'s "edge-of-competence" story: RL only helped the slice where the base model had headroom. Ship depends on whether the Humanities regression is within noise (§4).

**GSM8K-by-difficulty example.** [[interplay-pretraining-midtraining-rl]]'s headroom-and-boundary experiments are built on slicing by difficulty. Bucket GSM8K by step-count of gold solution: easy (1-2 steps), medium (3-4), hard (5+). A post-RL checkpoint that gains +8 pp on easy, +6 pp on medium, and -1 pp on hard is *not* "getting better at math" — it is shifting mass toward the edge it was already close to, and the hard slice is where the next RL round has to do work. Aggregate GSM8K would report +5 pp and hide the failure to extrapolate.

## §3 Failure bucketing — cluster by reason, not by surface

Aggregate accuracy tells you *how many* failed; bucketing tells you *why they failed in the same way*. Two clustering axes, both needed.

**Cluster-by-reason.** [[karpathy-training-neural-net-recipe]]'s "review the 10 worst validation examples — they reveal systematic errors" is the literal origin. Read 30-50 failed rollouts, tag each with one label from a fixed ontology, iterate the ontology until coverage >90%, then count. A serviceable starter ontology for a post-trained chat model: `format-violation`, `refusal-when-answerable`, `hallucination-fact`, `hallucination-tool`, `arithmetic-error`, `reasoning-skip-step`, `length-hack`, `sycophancy`, `language-drift`, `stale-knowledge`, `other`. [[reward-hacking-taxonomy]] gives the RL-relevant subset; [[bfcl]]'s relevance-detection column measures `call-when-irrelevant` directly.

**Cluster-by-confusion-matrix.** When the task has enumerable outputs, build the confusion matrix. [[bfcl]] does this implicitly — the AST matcher decomposes every failure into one of: name-mismatch (wrong function), kwargs-mismatch (right function, wrong params), hallucinated-call (called when irrelevant), missing-call (didn't call when needed), or format-unparseable. These are orthogonal buckets: a function-calling model can improve name-mismatch while regressing on hallucinated-call, which is the attested 2025 pattern (frontier models still hallucinate ~10% on irrelevant queries).

### Pipeline pseudocode

```python
# eval/bucket_failures.py — one pass over rollouts, two cluster outputs.
from collections import Counter, defaultdict

REASON_ONTOLOGY = [
    "format-violation", "refusal-when-answerable", "hallucination-fact",
    "hallucination-tool", "arithmetic-error", "reasoning-skip-step",
    "length-hack", "sycophancy", "language-drift", "stale-knowledge", "other",
]

def bucket_failures(rollouts, grader, reason_tagger, confusion_tagger=None):
    # rollouts: [{prompt, response, gold, slice_tags: {task, domain, difficulty, length}}]
    by_reason = defaultdict(Counter)        # slice -> Counter(reason)
    by_confusion = defaultdict(Counter)     # slice -> Counter(confusion_cell)
    correct = defaultdict(int); total = defaultdict(int)
    for r in rollouts:
        slice_key = (r["slice_tags"]["task"], r["slice_tags"].get("domain", "_"))
        total[slice_key] += 1
        if grader(r["response"], r["gold"]):
            correct[slice_key] += 1
            continue
        reason = reason_tagger(r)           # LLM-judge or regex ontology
        assert reason in REASON_ONTOLOGY
        by_reason[slice_key][reason] += 1
        if confusion_tagger:
            by_confusion[slice_key][confusion_tagger(r)] += 1
    return {"accuracy": {k: correct[k]/total[k] for k in total},
            "by_reason": dict(by_reason), "by_confusion": dict(by_confusion)}
```

Two outputs, two decisions. `by_reason` answers "which training-data fix addresses the biggest failure bucket?" — a data question. `by_confusion` answers "which metric move predicts the biggest win?" — an objective/reward question. Most teams build only one; the report table in §5 demands both.

### LLM-judge reason-tagger — the 2025 default

A capable LLM with a strict rubric is the attested reason-tagger: [[tulu-3]]'s safety-specific DPO slice was constructed this way. The rubric is the ontology; each failure gets one label and a one-sentence justification. Run judge bias checks from [[ch-49]] — position swap, length control — on a 100-rollout calibration set before trusting the bucket counts. Cluster labels are only as reliable as the judge's agreement rate with a human spot-check (target: kappa >= 0.7).

## §4 Comparing checkpoints — when is a regression real?

The signed per-slice delta is the unit of the report, but a single-run delta is a noisy measurement. [[karpathy-training-neural-net-recipe]]'s predict-before-run discipline applies: write down the expected direction of every slice before scoring, then test whether the observed delta is both (a) statistically distinguishable from zero and (b) large enough to matter.

### Paired bootstrap — the cheap-and-correct test

For two checkpoints scored on the *same* N items of a slice:

```python
# eval/paired_bootstrap.py
import numpy as np
def paired_bootstrap_ci(score_a, score_b, n_boot=10_000, seed=0):
    # score_a, score_b: per-item scalar scores for checkpoint A, B (same items, same order).
    rng = np.random.default_rng(seed)
    diff = np.asarray(score_b) - np.asarray(score_a)
    idx = rng.integers(0, len(diff), size=(n_boot, len(diff)))
    boot_means = diff[idx].mean(axis=1)
    lo, hi = np.percentile(boot_means, [2.5, 97.5])
    return float(diff.mean()), float(lo), float(hi)
```

Report `(mean_delta, CI_low, CI_high)` per slice. The "signed regression" rule:

- `CI_high < 0` — real regression; investigate.
- `CI_low > 0` — real gain; ship if size matters.
- `CI` crosses 0 — inconclusive on this slice; aggregate with siblings or collect more data.

### Effect-size threshold — keep you from chasing noise

A 0.3 pp gain on a 500-item slice can be statistically significant and still operationally meaningless. Pre-register an effect-size threshold per slice: e.g., `|Δ| >= 1.0 pp` for knowledge tasks (MMLU-domain), `|Δ| >= 2.0 pp` for reasoning (GSM8K/MATH), `|Δ| >= 5.0 pp` for narrow-verifier slices (IFEval constraint-category). [[tulu-3]]'s attested RLVR gain (+5-10 pp GSM8K) clears the 2.0 threshold with room; its neutral MMLU move (~+0.2 pp) does not clear the 1.0 threshold — the honest report says "GSM8K improved, MMLU held."

### Two-run minimum — a seed is not a sample

A single training run's per-slice deltas can flip under seed changes. [[llama-3]] mitigates this implicitly: the six-round iterative loop means every slice gets re-measured against a fresh RM per round, so a one-round flicker washes out. For one-shot runs, the cheap defense is two training seeds with identical data + LR; take the per-slice delta that is consistent in *sign* across seeds as "real," the ones that flip as "unresolved." Ch-51 quantifies the seed-variance budget.

### Unsigned deltas — a different question

Signed delta answers "did the slice improve?" Unsigned `|Δ|` answers "did the checkpoint's per-slice profile *change shape*?" Large unsigned delta with small signed delta means the model solves different items than before — behaviour shift without capability shift. Worth flagging: [[interplay-pretraining-midtraining-rl]]'s "reshape vs expand" question is literally this distinction. An RL stage that only reshapes is not the same as one that expands, even at equal aggregate.

## §5 The failure-ledger — what a bucket looks like across runs

One chapter's report is a snapshot. A *ledger* is the row across runs. Maintain `eval/failure-ledger.jsonl` where each row is `{run_id, checkpoint, slice, bucket, count, share, example_ids[], first_seen_run, fix_attempted}`. Reading a ledger row across six months shows whether a bucket is shrinking (fix working), growing (regression), or oscillating (unresolved). [[llama-3]]'s "failure modes" section is a published snapshot of this ledger — preference-data noise and multi-turn dialog drift are named buckets that *persisted* across rounds, not one-off anomalies.

Canonical buckets from the raw-data sources, each a row in the ledger:

- **`length-hack`** ([[reward-hacking-taxonomy]], [[dr-grpo]]) — wins correlate with `len(response) > len(gold) * 1.3`. Diagnostic: `reward-vs-length` scatter.
- **`verifier-loophole`** ([[rlvr-tulu3]]) — `\boxed{42}` inside prose accepted by lax grader. Diagnostic: strong-verifier scoring on held-out slice.
- **`format-violation-IFEval`** ([[tulu-3]]) — constraint-category confusion matrix. Diagnostic: per-constraint-type pass-rate.
- **`long-context-distractor-density`** ([[ruler]] MK-NIAH-full-haystack) — accuracy drops >30 pp when distractor needles saturate haystack. Diagnostic: S-NIAH vs MK-NIAH ratio.
- **`long-context-aggregation`** ([[ruler]] CWE, FWE) — retrieval holds, aggregation collapses past 32K. Diagnostic: CWE-score vs S-NIAH-score ratio.
- **`call-when-irrelevant`** ([[bfcl]] relevance-detection) — tool-hallucination bucket. Diagnostic: relevance-detection sub-score.
- **`multi-turn-state-drift`** ([[bfcl]] V3) — single-turn fine, multi-turn collapses. Diagnostic: pass@1 single vs pass^k multi-turn.
- **`language-drift`** ([[llama-3]] multilingual) — English prompts answered in English, Chinese prompts drift to English. Diagnostic: language-match rate per prompt-language.
- **`stale-knowledge`** ([[physics-of-lm-3]]) — factual-tuple retrieval fails on recently-updated facts. Diagnostic: split eval by fact-freshness.
- **`reasoning-skip-step`** ([[interplay-pretraining-midtraining-rl]]) — correct answer, broken trace. Diagnostic: process-reward score.

Each bucket has a *diagnostic signal* already logged in a modern pipeline. The ledger simply extracts and persists it.

## §6 The three-line chart vs the 50-slice report — the decision matrix

Both artifacts exist; pick by decision.

| Decision | Audience | Format | Example |
|---|---|---|---|
| Ship this checkpoint? (go/no-go) | leadership, release manager | three-line chart: aggregate per stage (pretrain / SFT / DPO / RLVR) | [[tulu-3]] Figure 1 |
| Which stage regressed? | training lead | per-stage x per-task table | [[olmo-2]] post-training gain table |
| Which slice within a task regressed? | eval lead | 50-row per-slice report with paired-bootstrap CI | [[ruler]] Table 3 style |
| Why did these items fail? | data lead | failure-bucket table (reason-clusters + confusion matrix) | [[bfcl]] category breakdown |
| Is this bucket shrinking over runs? | program manager | ledger row across runs | [[llama-3]] §Failure modes |
| Is this capability at the edge of competence? | research lead | per-difficulty histogram of pass-rate | [[interplay-pretraining-midtraining-rl]] headroom |

**The rule.** Don't use a three-line chart for regression triage (you'll miss the slice). Don't use a 50-slice report for go/no-go (leadership will aggregate in their head, badly, and optimize the wrong slice). The chart and the report are answers to different questions. A good eval harness produces both from the same evidence table — the chart is a `GROUP BY stage` aggregate of the report, and regenerating either takes one function call on `failures.parquet`.

**Decision-first design, not format-first.** The common anti-pattern is "we run benchmark X, report its score, done." It fails because "its score" already baked in a slicing choice — [[longbench]]'s equal-weighted mean over 21 tasks, [[ruler]]'s `wAvg. (inc)` vs `wAvg. (dec)`, BFCL's micro vs macro average. None of these was chosen with *your* decision in mind. Ch-50 reverses the order: start from the decision, pick the slicing and aggregation that answers it, then produce the report. [[ruler]]'s two weighted averages (`inc` vs `dec`) are the cleanest public example — the authors refused to commit to one headline because the two questions ("does it work at the long end?" vs "does it work overall?") deserve two answers.

## §7 Worked example — staging the report for a single checkpoint release

Concretize. You have finished a post-training run on a 7B base using the Tulu 3 recipe. Build the artifacts in order:

1. **Evidence table.** `failures.parquet` — one row per `(prompt_id, checkpoint_id, score, slice_tags, response, gold, judge_reason)`. Every downstream artifact is a query against this.
2. **Three-line chart.** `GROUP BY stage` on the evidence table yields three rows: SFT checkpoint, DPO checkpoint, RLVR checkpoint. Plot aggregate accuracy per task category. This is your ship-decision summary.
3. **Per-task table.** `GROUP BY stage, task` with paired-bootstrap CI per cell. This answers "which stage lifted which task" — [[olmo-2]]'s attested pattern (`DPO -> IFEval`, `RLVR -> GSM8K/MATH`) is your comparison template.
4. **Per-slice report for any task that regressed.** `GROUP BY stage, task, slice` with signed deltas. Apply the effect-size threshold from §4; flag rows where `CI_high < 0 and |Δ| >= threshold`.
5. **Bucket table for any flagged slice.** Run `bucket_failures()` from §3 on the failed rollouts of that slice. Report top five buckets and their cross-run ledger row (shrinking / growing / stuck).
6. **Go/no-go memo.** One page. Chart + one-paragraph narrative tying each regressed bucket to a proposed fix. [[karpathy-training-neural-net-recipe]]'s predict-before-run discipline sits here — the memo cites which regressions were predicted (acceptable) vs unpredicted (require investigation).

Six artifacts, same evidence table, six different audiences. A harness that cannot produce all six from one query layer is under-built — [[olmo-3]]'s model-flow philosophy (stages as auditable artifacts) applied to the eval side.

## §8 Anti-patterns to name

A short list of mistakes this chapter's guideline forbids:

- **Weighted-average trick.** Reweighting slices so the aggregate tells the story you want. [[longbench]]'s NarrativeQA case would let you drag the mean up by up-weighting the easy slice. The fix is pre-registration: commit to slice weights before running.
- **Compare-different-slicings.** Run A was reported on 6 MMLU domains; Run B is reported on 2 (STEM vs humanities). The deltas are not comparable. Fix: one fixed slicing spec per benchmark, versioned.
- **Judge-as-grader without calibration.** LLM-judge reason-tagger labels drift as the judge model changes. Ch-49's calibration discipline must be reapplied every time the judge changes; otherwise ledger counts across runs are comparing apples to retrained apples.
- **Cherry-picked worst examples.** "Review the 10 worst" does not mean "reviewer reads 10 worst, picks the 3 most embarrassing, ignores the rest." Fix: tag all 10, report the tag histogram, not a narrative.
- **Aggregating across sizes.** Tulu 3 re-tunes β per size; mixing 7B and 70B eval scores under a single aggregate erases the size-specific per-slice optimum. Fix: size is a slice axis, never an aggregation axis.
- **Reporting only improvements.** Post-hoc filtering of regressions out of the report. The ledger exists specifically to prevent this — a failing bucket that disappears from the report without a shipped fix is a bug in the reporting process, not the model.

---

## Companion visualization

**[figures/slice-report.html](figures/slice-report.html)** — an interactive mock eval report. Toggle between "aggregate-only" view (three lines) and "per-slice" view (50 rows); watch regressions that are invisible in the aggregate appear as signed bars under the per-slice toggle. Second panel: a failure-bucket table with reason-counts per slice, plus a confusion-matrix heatmap for a BFCL-style function-calling slice. Numbers are illustrative; the *shape* of the effect is attested by [[tulu-3]], [[ruler]] Table 3, and [[bfcl]] category breakdowns. Use the toggle before reading §6's decision matrix — the rule clicks once you have seen the slice-bars reverse the aggregate verdict.

## Connections

- **ch-47 (eval harness)** — produces the evidence table this chapter slices. Slicing requires item-level scores, not per-task means.
- **ch-48 (benchmark zoology)** — taxonomy of per-task datasets; this chapter is the per-slice layer on top.
- **ch-49 (judge calibration)** — reason-tagger judge must itself be calibrated or the bucket counts are laundered bias.
- **ch-51 (variance & go/no-go memo)** — formalizes the two-run minimum, the effect-size threshold, and the memo structure the three-line chart lives inside.
- **ch-42 ([[reward-hacking-taxonomy]])** — supplies named buckets for the ledger.
- **ch-44 ([[rlvr-tulu3]])** — verifier-loophole bucket originates here.

## Further reading

- [[tulu-3]] Figure 1 + §RLVR — per-stage per-task ledger.
- [[olmo-2]] post-training gain table — stage-attribution discipline.
- [[olmo-3]] model-flow diagram — stage-as-artifact framing.
- [[llama-3]] §Failure modes — published failure-ledger snapshot.
- [[longbench]] §Risks + gotchas — why per-task can mislead (NarrativeQA solvable at 4K).
- [[ruler]] Table 3 + Figure 3 — effective-context via per-slice, not per-task.
- [[bfcl]] scoring categories — canonical confusion-matrix decomposition.
- [[interplay-pretraining-midtraining-rl]] — edge-of-competence slicing; reshape-vs-expand.
- [[physics-of-lm-3]] — knowledge-capacity framing motivating per-capability slicing.
- [[karpathy-training-neural-net-recipe]] — "review the 10 worst" as the literal root of failure bucketing.
