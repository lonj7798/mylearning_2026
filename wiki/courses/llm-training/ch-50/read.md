<!-- chapter: ch-50
     track: eval
     kind: content
     title: Slice Analysis, Forgetting Slices, and Failure Bucketing
     deps: [ch-49]
     sources: [[tulu-3]], [[ifbench]], [[gsm-symbolic]], [[ruler]], [[lost-in-the-middle]], [[nolima]], [[transferability-of-llm-reasoning]], [[rls-razor]], [[ai2-benchmirt]], [[bfcl]], [[holistic-agent-leaderboard]]
     figures: figures/slice-report.html
     revised: 2026-09 (generality revision)
-->

# Chapter 50 — Slice Analysis, Forgetting Slices, and Failure Bucketing

> **Core insight.** An aggregate score is an average over slices, and averaging can hide changes in both
> directions at once. In [[tulu-3]] Table 6, the 8B checkpoint moves from DPO to the released RLVR model
> with an average of 64.7 → 65.1 (+0.4), while inside that average GSM8K rises 84.3 → 87.6 (+3.3) and
> Safety falls 87.2 → 85.5 (−1.7) and TruthfulQA falls 56.1 → 55.0 (−1.1). The same report shows a second
> kind of hiding: on the development benchmark HumanEval the 8B score is identical before and after RLVR
> (83.9 → 83.9), while on the held-out coding benchmark BigCodeBench it falls 9.5 → 7.4 ([[tulu-3]] Table 31).
> Slices that are defined by held-out status, perturbation, context length, and training stage are the ones
> that carry the information an aggregate destroys.
>
> **Guideline.** When a checkpoint comparison will decide whether to ship or to re-run a stage, compute
> per-slice scores from item-level records and aggregate afterwards, because the aggregate is a function of
> the slices but not the reverse. When a benchmark was used to make training decisions, report it next to a
> held-out benchmark for the same skill and treat the gap as an overfitting measurement, because
> [[tulu-3]] Table 31 and [[ifbench]] Table 6 both show large gaps that in-distribution scores do not
> predict. When the claim is about long context, report accuracy as a grid over length, evidence depth,
> and question-needle lexical overlap, because [[lost-in-the-middle]] and [[nolima]] each show a
> single-cell test scoring far above the grid. When failures must be explained rather than counted, tag
> them with a fixed bucket ontology and record the tagger's measured precision, because in
> [[holistic-agent-leaderboard]] the LLM tagger's precision was measured (0.87–1.00 on three checks) and
> its recall was not measured at all.

---

## Why this chapter matters for a general-purpose model

The stages before this one (SFT in ch-30 onwards, preference optimization, RL) all change a model along
more than one axis at a time. A stage that is selected on one number will be selected for whatever that
number rewards, including changes the number cannot see. The course goal is a model that performs on
tasks and domains that were not targeted, so the measurement has to be able to detect three specific
things:

1. **Narrowing** — a gain concentrated on the exact distribution that was trained or tuned on, with no
   gain, or a loss, on a neighbouring distribution that requires the same skill.
2. **Forgetting** — a capability that existed at an earlier stage and is smaller at a later one.
3. **Surface fitting** — a score that depends on the canonical phrasing, length, position, or format of
   the benchmark rather than on the task.

None of the three is visible in a single aggregate number, because each is a redistribution inside the
average. This chapter is the measurement layer that makes them visible: how to define slices, how to
compare stages on them, and how to turn failed items into named, countable buckets. It sits after the
harness (ch-47), the overfitting audits (ch-47a), contamination (ch-48), and judge calibration (ch-49),
and before the statistics that say whether an observed per-slice difference is larger than noise (ch-51).

A definition used throughout: a **slice** is a subset of evaluation items selected by an attribute that is
recorded for every item — the benchmark it came from, whether the benchmark was used during development,
the perturbation applied to it, the context length, the position of the evidence, the training stage being
scored. A **per-slice report** is the table of scores over those subsets. The aggregate is one row of it.

---

## §1 The per-stage per-slice table, and what the average removes

**Definition.** A per-stage per-slice table has one column per checkpoint in the pipeline and one row per
slice, so each cell is a score and each horizontal difference is a stage delta.

**The problem it addresses.** A pipeline decision ("keep the RLVR stage", "re-run DPO with a different
mix") is made on stage deltas. If only the average is recorded, deltas of opposite sign inside the average
cancel, and the decision is made on the residual.

**Mechanism.**
1. Score every checkpoint on the same items with the same settings, keeping item-level records.
2. Group items into slices by a recorded attribute.
3. Report, for each slice, the score at each stage and the difference between adjacent stages.
4. Compute the aggregate from the slice scores, and print it in the same table so the two are comparable.

**Worked example ([[tulu-3]], arXiv:2411.15124 Table 6, Tülu 3 8B).** The released 8B column and the DPO
column give these per-benchmark stage deltas:

| Slice (benchmark, eval setting) | SFT | DPO | Final (RLVR) | Δ SFT→DPO | Δ DPO→Final |
|---|---|---|---|---|---|
| Average | 60.6 | 64.7 | 65.1 | +4.1 | **+0.4** |
| MMLU (0-shot, CoT) | 65.9 | 68.7 | 68.2 | +2.8 | −0.5 |
| PopQA (15-shot) | 29.3 | 29.3 | 29.1 | 0.0 | −0.2 |
| TruthfulQA (6-shot) | 46.8 | 56.1 | 55.0 | +9.3 | −1.1 |
| BigBenchHard (3-shot, CoT) | 69.7 | 68.7 | 69.0 | −1.0 | +0.3 |
| DROP (3-shot) | 61.3 | 62.5 | 62.6 | +1.2 | +0.1 |
| MATH (4-shot CoT, Flex) | 31.5 | 42.0 | 43.7 | +10.5 | +1.7 |
| GSM8K (8-shot, CoT) | 76.2 | 84.3 | 87.6 | +8.1 | **+3.3** |
| HumanEval (pass@10) | 86.2 | 83.9 | 83.9 | −2.3 | 0.0 |
| HumanEval+ (pass@10) | 81.4 | 78.6 | 79.2 | −2.8 | +0.6 |
| IFEval (prompt loose) | 72.8 | 81.1 | 82.4 | +8.3 | +1.3 |
| AlpacaEval 2 (LC % win) | 12.4 | 33.5 | 34.5 | +21.1 | +1.0 |
| Safety (6-task avg.) | 93.1 | 87.2 | 85.5 | −5.9 | **−1.7** |

Check the arithmetic on the last column: the twelve benchmark deltas are
+(−0.5) + (−0.2) + (−1.1) + 0.3 + 0.1 + 1.7 + 3.3 + 0.0 + 0.6 + 1.3 + 1.0 + (−1.7) = +4.8 over twelve
benchmarks, so the mean delta is +0.40, which is the +0.4 printed on the Average row. The largest single
component is +3.3 and the most negative is −1.7; the aggregate is smaller than either. A decision rule of
the form "ship when the average improves" would accept this checkpoint without ever seeing the Safety and
TruthfulQA losses, and a decision rule of the form "ship when nothing regresses" would reject it without
ever seeing the GSM8K gain.

The RLVR-stage table in the same report ([[tulu-3]] Table 23, §6.4) gives the same per-benchmark DPO→RLVR
comparison at 8B with an average of 64.4 → 64.8 (the two tables differ on the average and on BigBenchHard
because they use different eval settings for some rows; the per-benchmark values used above are Table 6's).
The report's own prose describes the 8B result as "non-trivial improvements … improving all three of MATH,
GSM8k, and IFEval" (§6.4) and does not comment on the Safety and TruthfulQA rows. **Result (single study).**

**Conditions and limits.** These deltas are one seed, one model size, one recipe; the 70B column in the
same table shows a different pattern (GSM8K 93.5 → 93.5, attributed in §6.4 to saturation). Whether a
per-slice delta of a given size is distinguishable from run-to-run noise is the subject of ch-51, and
nothing in this section establishes that any single row above is outside noise.

**Implication for a general-purpose model.** The stage that produced the target gain also produced the
losses on the two slices most connected to general use (safety behaviour and truthfulness). A pipeline
optimized on the average will accumulate such trades silently over stages.

The companion figure [figures/slice-report.html](figures/slice-report.html) holds this table and the two
that follow; the first view toggles between the Average row alone and the full per-benchmark column, so
the same checkpoint pair can be read under both reporting choices.

---

## §2 In-distribution and held-out slices, and the gap between them

**Definition.** An **in-distribution (development) slice** is one whose scores were visible while training
and mixture decisions were made. A **held-out (unseen) slice** is one that measures the same skill and was
not consulted during development. The **generalization gap** for a skill is the difference between them.

**The problem it addresses.** Any decision procedure that selects on a benchmark can select for properties
specific to that benchmark. The size of that effect is not knowable from the benchmark itself.

**Mechanism.** [[tulu-3]] §2.2 and §7.4 describe the design: the evaluation suite is split into a
development suite and an unseen suite covering the same core skills, and the authors state that they "did
not examine scores on our unseen set when developing our models, allowing us to observe how much we may
have overfit" (§2.2). Each skill is paired: MMLU → GPQA, BigBenchHard → AGIEval, MATH → DeepMind
Mathematics, HumanEval → BigCodeBench, IFEval → IFEval-OOD (§7.2–7.3, Table 3).

**Worked example ([[tulu-3]] Table 31, 8B).** Development (Dev.) and unseen (Uns.) scores per skill:

| Skill (dev → unseen) | SFT Dev / Uns | DPO Dev / Uns | Final Dev / Uns | Gap at final |
|---|---|---|---|---|
| Average | 64.9 / 29.9 | 68.3 / 31.9 | 68.8 / 32.4 | 36.4 |
| Knowledge (MMLU → GPQA) | 65.9 / 31.9 | 68.7 / 31.2 | 68.2 / 35.7 | 32.5 |
| Reasoning (BBH → AGIEval) | 67.9 / 56.2 | 65.8 / 61.8 | 66.0 / 59.3 | 6.7 |
| Math (MATH → DM Mathematics) | 31.5 / 32.3 | 42.0 / 33.0 | 43.7 / 35.4 | 8.3 |
| Coding (HumanEval → BigCodeBench) | 86.2 / 11.5 | 83.9 / 9.5 | 83.9 / 7.4 | 76.5 |
| Instruction following (IFEval → IFEval-OOD) | 72.8 / 17.6 | 81.1 / 23.9 | 82.4 / 24.3 | 58.1 |

Two readings that the development column alone cannot support:

- **Coding.** The development score is unchanged from DPO to final (83.9 → 83.9) while the held-out score
  falls 9.5 → 7.4. A report built only from HumanEval records "no change".
- **Instruction following.** The development score rises 72.8 → 82.4 across the pipeline (+9.6) while the
  held-out score rises 17.6 → 24.3 (+6.7) and stays 58.1 points below it. The gain is real on both slices
  and the absolute level is not transferable.

The authors' own conclusion from the wider comparison is mixed: the final checkpoints have the best
average on both splits (§7.4.1), while "our choices overfit to the development evaluations in Precise
Instruction Following, and to some extent in Knowledge Recall and Reasoning" (§7.4.1), and the DPO
data-scaling curves show that "our development process overfit to MATH to some extent", attributed to
LaTeX formatting differences between MATH and DeepMind Mathematics (§7.4.1, Fig. 24). **Result (single study).**

**A second measurement of the same gap ([[ifbench]]).** IFBench holds 58 verifiable output constraints
that are disjoint from the 29 training constraints and from IFEval's set (§2, App. A Table 8). Training a
Tülu-3-8B-DPO policy with GRPO on varied constraints raises IFEval 81.1 → 92.2 and IFBench 25.2 → 44.6
(Table 6). The gap narrows from 55.9 to 47.6 points and does not close. **Replicated** with [[tulu-3]]
Table 31's IFEval/IFEval-OOD rows: two independent constructions of an unseen-constraint set produce the
same qualitative result.

**Conditions and limits.** A gap of this kind mixes three causes that the design does not separate:
selection on the development benchmark, genuine difficulty differences between the paired benchmarks, and
format differences. [[tulu-3]] §7.4.1 attributes part of the math gap to format. A gap is therefore an
upper bound on overfitting, not a measurement of it.

**Implication for a general-purpose model.** The held-out column is the closest available proxy for
"performs on tasks that were not targeted". A pipeline that does not maintain one cannot report generality
at all, and — as [[tulu-3]] §2.2 states — the split only works if the held-out scores are not read during
development.

---

## §3 Perturbation slices: same skill, different surface

**Definition.** A **perturbation slice** contains items produced by a controlled transformation of
benchmark items that preserves the required reasoning and changes the surface: names, numeric values,
clause count, or an added irrelevant clause.

**The problem it addresses.** A score on a fixed public test set confounds skill with familiarity with
that exact set. Perturbation separates them by holding the skill fixed.

**Mechanism ([[gsm-symbolic]], arXiv:2410.05229).** 100 GSM8K test questions are converted into symbolic
templates with typed variables and correctness conditions (§3.1). From these templates, 50 datasets of 100
examples each are generated, where each example is a mutation of one of the 100 originals (§4). Variants
add or remove clauses: Symbolic-M1 removes one clause, Symbolic-P1 and Symbolic-P2 add one and two, and
Symbolic-NoOp adds one clause that appears relevant and does not enter the solution (§4.3, §4.4).

**Worked example ([[gsm-symbolic]] Table 1, 8-shot).** Reading one row as a slice ladder:

| Model | GSM8K (100 originals) | Symbolic | Symbolic-P1 | Symbolic-P2 | Symbolic-NoOp |
|---|---|---|---|---|---|
| Gemma2-9b-it | 87.0 | 79.1 ± 3.0 | 68.1 ± 4.8 | 41.8 ± 6.0 | 22.3 ± 5.1 |
| Phi-3-medium-128k-instruct | 89.0 | 82.5 ± 2.9 | 75.8 ± 3.9 | 53.1 ± 4.8 | 29.4 ± 4.2 |
| Llama3-8b-instruct | 74.0 | 74.6 ± 2.9 | 53.8 ± 4.5 | 28.3 ± 4.4 | 18.6 ± 3.9 |
| GPT-4o | 95.0 | 94.9 ± 1.9 | 93.9 ± 2.6 | 88.0 ± 3.4 | 63.1 ± 4.5 |
| o1-preview | 96.0 | 92.7 ± 1.8 | 95.4 ± 1.7 | 94.0 ± 2.4 | 77.4 ± 3.8 |

Two slice facts follow from the columns. First, the original-to-Symbolic column difference is not the same
for all models: Gemma2-9b-it loses 7.9 points where Llama3-8b-instruct gains 0.6 and GPT-4o loses 0.1. The
authors report that the score on the 100 originals lies more than one standard deviation from the centre
of the Symbolic distribution for 21 of 25 models, usually above it, and offer data contamination as one
explanation (§4.1, **Interpretation**). Second, the NoOp column separates models that the GSM8K column does
not: Gemma2-9b-it and Phi-3-medium are 2 points apart on GSM8K and 7 points apart on NoOp, while
Llama3-8b-instruct sits 13 points below Gemma2-9b-it on GSM8K and 3.7 points below it on NoOp.

The generated-instance standard deviations in the table are a second product of this design: the same
question, re-instantiated, gives a score distribution rather than a point. The authors report that
changing proper names alone already produces variation, and that variance grows when numeric values change
as well (§4.2, Fig. 4). **Result (single study.)**

**Conditions and limits.** The templates come from 100 GSM8K questions, so the result is about grade-school
arithmetic word problems and one benchmark's style. The NoOp drop is partly a measurement of how a model
handles distractor text, which is a capability in its own right, not only a robustness artifact.

**Implication for a general-purpose model.** A perturbation slice is the cheapest available test of whether
a capability is attached to the task or to the surface form of one dataset. Its cost is a template-writing
pass over a benchmark; its output is a distribution that makes the single canonical score interpretable.

---

## §4 Long-context slices: length, depth, hops, overlap, distractors

A long-context claim is a claim about a grid, and every published measurement that opens the grid finds
cells far below the marginal score.

**Length ([[ruler]], arXiv:2404.06654 Table 3).** RULER averages 13 synthetic tasks at each of 4K, 8K, 16K,
32K, 64K, and 128K, and defines **effective length** as the largest length whose average stays above the
Llama2-7B 4K score of 85.6. Selected rows:

| Model | Claimed | Effective | 4K | 32K | 64K | 128K |
|---|---|---|---|---|---|---|
| Gemini-1.5-Pro | 1M | >128K | 96.7 | 95.9 | 95.9 | 94.4 |
| GPT-4 | 128K | 64K | 96.6 | 93.2 | 87.0 | 81.2 |
| Llama3.1 (70B) | 128K | 64K | 96.5 | 94.8 | 88.4 | 66.6 |
| Yi (34B) | 200K | 32K | 93.3 | 87.5 | 83.2 | 77.3 |
| Mixtral-8x22B | 64K | 32K | 95.6 | 90.9 | 84.7 | 31.7 |
| LWM (7B) | 1M | <4K | 82.3 | 69.1 | 68.1 | 65.0 |

The claimed-versus-effective column is the slice result: of these rows, only Gemini-1.5-Pro's effective
length reaches its claimed one. RULER also reports two weighted averages, `wAvg. (inc)` and `wAvg. (dec)`,
with weights increasing or decreasing linearly with length, and the ranks differ between them — LWM ranks
12th under `inc` and 15th under `dec` (Table 3). The aggregation weights are themselves a reporting choice.

**Task family at fixed length ([[ruler]] §5, Fig. 2).** Holding the model fixed (Yi-34B) and varying the
generator: performance is near-perfect on word-number single-needle retrieval and degrades when the needle
becomes a 32-digit UUID, and increasing distractor needles lowers accuracy steadily, with a drop of about
40 points at 256K in the version where the context is filled with distractor needles (`#K=FULL`). The
error analysis reports that in that setting the model often returns values from the vicinity of the target
(§5). **Result (single study.)**

**Depth ([[lost-in-the-middle]], arXiv:2307.03172).** In multi-document QA, one of k retrieved documents
contains the answer and its position is varied while the rest of the input is held fixed (§2.1). With 20
documents, GPT-3.5-Turbo scores 75.8 / 57.2 / 53.8 / 55.4 / 63.2 at positions 1 / 5 / 10 / 15 / 20
(App. G Table 6). Its closed-book score with no documents is 56.1 and its oracle score with only the answer
document is 88.3 (Table 1). The position-10 cell, 53.8, is below the closed-book score: on that slice the
retrieved context is not being used. Extending the window does not change this — GPT-3.5-Turbo and its 16K
variant score 75.8 and 75.7 at position 1 with 20 documents (Table 6). **Result (single study.)**

**Lexical overlap ([[nolima]], arXiv:2502.05167).** NoLiMa builds question-needle pairs whose ROUGE-1
precision overlap is 0.069, against 0.905 for vanilla needle-in-a-haystack and 0.571 for RULER's S-NIAH
(Table 1). The same model, same haystack, three overlap conditions (Table 6, Llama 3.3 70B, at 8K / 16K /
32K): a direct question naming the needle keyword scores 98.3 / 98.5 / 98.5; a one-hop association scores
84.1 / 73.2 / 56.2; a two-hop association scores 57.4 / 42.7 / 25.9. Effective lengths under the paper's
0.85 × base-score threshold are far below claimed lengths: GPT-4o 128K claimed and 8K effective, Llama 3.3
70B 128K claimed and 2K effective (Table 3). Adding one distractor sentence containing the question keyword
lowers GPT-4o's effective length to 1K (§4.4.4). Chain-of-thought is a partial remedy: two-hop at 32K rises
25.9 → 34.3 with CoT, and stays below one-hop without CoT at every length (Table 4). **Result (single study.)**

NoLiMa's depth sweep reproduces the [[lost-in-the-middle]] dip for one-hop questions at 32K, and reports
that for two-hop questions longer contexts lower the curve at every depth including the edges (§4.4.2,
Fig. 3). **Replicated** for the one-hop case across the two papers.

**The slice specification that follows.** A long-context report should be a grid over at least: context
length; evidence depth (position as a fraction of the context); reasoning hops from question to evidence;
question-evidence lexical overlap; distractor count. Each of the four sources above shows a marginal score
that stays high while one of these coordinates collapses. A single number labelled "128K" is the mean over
a grid that was never printed.

---

## §5 Cross-stage forgetting and transfer slices

**Definition.** A **forgetting slice** is a benchmark measuring a capability that was not targeted by the
current stage and was measurable before it. A **transfer slice** is a benchmark measuring a capability
adjacent to the target. Both are read as stage deltas, not as levels.

**The problem it addresses.** Post-training stages are usually evaluated on the capability they target. If
the loss on untargeted capabilities is not measured at the same time, a pipeline can be locally improving
at every stage and globally narrowing.

**Mechanism and a metric ([[transferability-of-llm-reasoning]], arXiv:2507.00432 §2.1).** Benchmarks are
split into three groups: math reasoning, other reasoning, and non-reasoning. Per-benchmark gains over the
base model are z-normalized within each group, compressed with a signed square root, weighted by task
difficulty, and aggregated to a group Domain Index:

```
ΔR_b = R_b^model − R_b^base                       per-benchmark gain
σ_g  = Std{ΔR_b : b ∈ B_g}                        within-group spread
δ_b  = ΔR_b / σ_g                                 normalized gain
s_b  = sign(δ_b) · |δ_b|^(1/2)                    signed square root
w_b  = 100 − R_b^base                             difficulty weight
ŵ_b  = w_b / Σ_{u∈B_g} w_u                        normalized weight
DI_g = Σ_{b∈B_g} ŵ_b · s_b                        group Domain Index
TI_g(%) = 100 · DI_g / DI_math                    Transferability Index
```

Symbols: `b` is a benchmark, `B_g` the benchmarks in group `g ∈ {math, other, non}`, `R_b^model` and
`R_b^base` the accuracies of the tuned and base model on `b`, `Std` the standard deviation over the group's
gains. `TI_g` is the ratio of a group's aggregated gain to the math group's, in percent, so `TI = 100` means
the group moved as much as math did and a negative `TI` means the group moved in the opposite direction.

**Worked example (same paper, Table 1, Qwen3-14B-Base).** Group averages:

| Model | Math avg | Other-reasoning avg | Non-reasoning avg | TI_other | TI_non |
|---|---|---|---|---|---|
| Qwen3-14B-Base | 27.7 | 30.2 | 45.7 | — | — |
| UniReason-Qwen3-14B-think (SFT) | 49.8 | 45.3 | 21.1 | +52.2 | −104.1 |
| UniReason-Qwen3-14B-no-think (SFT) | 32.3 | 45.2 | 29.0 | +165.4 | −278.9 |
| UniReason-Qwen3-14B (RL) | 53.8 | 60.0 | 53.2 | +82.3 | +52.2 |

The SFT (think) run gains +22.1 on math and loses 24.6 on the non-reasoning average, with the individual
columns showing CoQA 10.0 → 1.7, IFEval 69.2 → 42.3, and HalluEval 35.7 → 2.3. The RL run reaches a higher
math average (53.8 versus 49.8) with the non-reasoning average rising 45.7 → 53.2. A report that contained
only the math column would describe the two runs as the same kind of improvement. **Result (single study;
one base model, one 14B size, one math corpus.)**

**A predictor for the forgetting slice ([[rls-razor]], arXiv:2509.04259).** Fine-tuning Qwen2.5-3B-Instruct
on three new tasks (math from Open-Reasoner-Zero, Chemistry L-3 from SciKnowEval, ToolAlpaca) and measuring
prior capability on HellaSwag, TruthfulQA, MMLU, IFEval, Winogrande, and HumanEval, the authors report that
forgetting is predicted by the KL divergence between the fine-tuned and base policy **measured on the new
task distribution**, `E_{x∼τ} KL(π₀ ‖ π)`, and that RL and SFT points fall on one curve when plotted against
it. A quadratic fit gives R² = 0.96 in a controlled ParityMNIST setting and R² = 0.71 in the LLM experiments
(§4, Fig. 3, Fig. 11). Training SFT on an analytically KL-minimal labelling retained more prior knowledge
than RL (§4), which is the authors' evidence that the algorithm is not the operative variable.

**Implication for slice design.** Two quantities belong in the per-stage report next to the target score:
the untargeted-capability slice scores, and the KL divergence on the new-task distribution, which is cheap
to compute and available before the forgetting benchmarks are run. **Interpretation** for the second: the
source establishes the correlation, not that the KL number is sufficient for a decision.

---

## §6 Failure bucketing: reason buckets and confusion cells

**Definition.** **Bucketing** assigns each failed item a label from a fixed ontology so failures can be
counted by kind. Two ontologies are used: **reason buckets**, which are free-form categories assigned by a
rubric, and **confusion cells**, which are defined by the structure of the output space and are mutually
exclusive by construction.

**Confusion cells, where the output is structured ([[bfcl]], PMLR 267:48371).** Function calling has an
enumerable failure structure. BFCL's AST matcher parses each call with Python's `ast` module and requires
an exact function-name match and each parameter value to be in a set of accepted answers (§4.1, App. H), so
a failure resolves into: wrong function name, right function with wrong arguments, a call emitted where the
Irrelevance category expects none, no call where one is expected, or an unparseable output. Two scored
categories separate the last two directly: Irrelevance (no call expected) and Relevance (at least one call
expected) (§3.1, App. C.2). The numbers show they move independently — Qwen2.5-72B-Instruct in prompting
mode scores 100.0 on Relevance and 72.8 on Irrelevance (Table 1). A model can improve on emitting the right
call and regress on abstaining, and a single "function-calling accuracy" hides that.

The same benchmark shows a format-versus-content decomposition: prompting-mode models average 412.93
decoding issues against 182.5 for function-calling-mode models out of 4,251 entries, while among the
responses that do decode, function-calling-mode models give the wrong number of calls more often in the
Multiple category (77.5 against 21) (§5.1). **Result (single study.)**

**Reason buckets, where the output is open-ended.** The procedure is: read a sample of failed items, write
an ontology, tag each failure with exactly one label and a one-sentence justification, extend the ontology
until the `other` bucket is small, then count by slice. The ontology should name mechanisms that map to
different fixes. Buckets that this course's earlier chapters have evidence for, each with the diagnostic
already present in a standard pipeline:

| Bucket | Evidence that the bucket exists | Diagnostic signal |
|---|---|---|
| `constraint-violation` on unseen constraint types | [[ifbench]] Table 6: IFEval 92.2 against IFBench 44.6 after IF-RLVR | per-constraint-category pass rate on held-out constraints |
| `distractor-sensitivity` | [[gsm-symbolic]] Table 1 NoOp column (Gemma2-9b-it 79.1 → 22.3) | paired canonical/NoOp item scores |
| `position-dependent-retrieval` | [[lost-in-the-middle]] Table 6 (75.8 at position 1, 53.8 at position 10) | accuracy by evidence depth |
| `lexical-match-dependence` | [[nolima]] Table 6 (98.5 direct against 56.2 one-hop at 32K) | accuracy by question-needle overlap |
| `abstention-failure` | [[bfcl]] Table 1 Irrelevance against Relevance | Irrelevance category score |
| `format-decoding-failure` | [[bfcl]] §5.1 decoding-issue counts | parse-failure rate per mode |

**The tagger has to be measured.** When the tagger is an LLM, its labels are a measurement with an error
rate. [[holistic-agent-leaderboard]] reports human-validated precision for its rubric judge — 0.87 on
AssistantBench instruction following (n = 49, inter-LLM κ = 0.82), 1.00 on CORE-Bench verification (n = 31),
0.94 on TAU-bench instruction following (n = 36) — and states that only precision was measured, not recall
(App. A7.2, Table A2). Precision without recall bounds the false-positive rate of a bucket and says nothing
about how many failures of that kind were missed, so bucket counts are lower bounds. [[bfcl]] uses
GPT-4o-08-06 as the judge for its multi-turn error analysis (§5.4.2, App. F) and reports the most frequent
root cause as "Failed to Understand Environment State", followed by "Failed to Understand User's Request".
The judge-bias checks of ch-49 (position swap, length control, self-preference) apply to a reason-tagger as
they do to a scoring judge, and a tagger change between runs makes ledger counts non-comparable.

**A minimal pipeline.** One pass over item-level records, two outputs:

```python
# eval/bucket_failures.py — reason buckets and confusion cells from one pass over item records.
from collections import Counter, defaultdict

def bucket_failures(items, grader, reason_tagger, confusion_tagger=None):
    """items: [{id, prompt, response, gold, slice_tags: {suite, split, length, depth, stage}}]
       grader: (response, gold) -> bool
       reason_tagger: item -> label from a fixed ontology
       confusion_tagger: item -> cell label, when the output space is enumerable"""
    n_total, n_correct = Counter(), Counter()
    by_reason, by_cell = defaultdict(Counter), defaultdict(Counter)
    for it in items:
        key = (it["slice_tags"]["suite"], it["slice_tags"]["split"])
        n_total[key] += 1
        if grader(it["response"], it["gold"]):
            n_correct[key] += 1
            continue
        by_reason[key][reason_tagger(it)] += 1
        if confusion_tagger is not None:
            by_cell[key][confusion_tagger(it)] += 1
    accuracy = {k: n_correct[k] / n_total[k] for k in n_total}
    return {"accuracy": accuracy, "by_reason": dict(by_reason), "by_cell": dict(by_cell)}
```

`by_reason` answers which data or training change addresses the largest failure class. `by_cell` answers
which scored sub-metric moved. They are different questions and the second is the more reliable of the two,
because its labels do not pass through a model.

---

## §7 Trajectory failure buckets for agents

**Definition.** For an agent task, an item is a trajectory: a sequence of model messages, tool calls, and
environment observations ending in a success predicate over the final environment state. A **trajectory
bucket** is a label for where the trajectory went wrong, which is not recoverable from the final score.

**The problem it addresses.** A binary task outcome for a 200-step trajectory compresses the entire episode
into one bit, and the fix for a failure depends on which step and which cause produced it.

**Mechanism and measured prevalence ([[holistic-agent-leaderboard]], arXiv:2510.11977).** HAL ran 21,730
rollouts over 9 models and 9 benchmarks for about $40,000 (Abstract), then applied an LLM rubric (Docent,
GPT-5 Medium as judge) to 2,184 transcripts across four benchmarks, 1,634 after removing TAU-bench
(App. A7.1). The rubric has six categories: instruction violations, tool-use failures, self-correction,
verification, environmental barriers, and shortcuts or gaming (§4.2, Table A5). Measured prevalence on
failed tasks:

| Bucket | AssistantBench | CORE-Bench Hard | SciCode |
|---|---|---|---|
| Instruction violation | 67.0% | 62.8% | not reported |
| Environmental barrier | 56.4% | 40.3% | 43.8% |
| At least one tool-call failure (failed runs) | not reported | 89.7% | 97.7% |
| At least one tool-call failure (successful runs) | not reported | 83.9% | 100% |

The last two rows are the reason a bucket count alone is not a diagnosis: on SciCode, a tool-call failure
appears in essentially every trajectory including the successful ones, so its presence does not distinguish
failure from success. The buckets that do distinguish are the recovery behaviours: success is more likely
with self-correction (risk ratios 1.47, 1.54, 2.97 on AssistantBench, SciCode, CORE-Bench) and with
verification (1.39, 1.87, 1.13) (§4.2). **Result (single study.)**

**Environment error against model error.** HAL's "environmental barriers" bucket separates the failures the
model could not have avoided. Its appendix lists concrete cases: a provider silently switching a DeepSeek
R1 endpoint to R1-0528 under the same name, a router serving FP4 on one call and FP8 on another, and silent
rate-limit failures being scored as wrong answers (App. A3). Any of these moves a score without any change
to the model. The same log analysis found the opposite error as well: the official TAU-bench few-shot file
`few_shot_data/MockAirlineDomainEnv-few_shot.jsonl` contained test-set examples, discovered after about
$1,000 of evaluation, and all results from that scaffold were excluded (App. A5).

**The scaffold is a slice axis.** HAL reports that task-specific scaffolds beat a single generalist scaffold
on 9 of 12 runs on CORE-Bench Hard and 11 of 12 on SWE-bench Verified Mini, while the generalist cost less
in 20 of 24 comparisons, and that the ranking interacts with the model: on Online Mind2Web, Claude models
score higher with Browser-Use and OpenAI models with SeeAct, where SeeAct with GPT-5 Medium costs $171 and
Browser-Use with Claude Sonnet 4 costs $1,577 for a two-point accuracy difference (§4.1 items 6–7). A
per-model agent score without a fixed scaffold is a score for the model-scaffold pair.

**Shortcut bucket.** Eight cases were found of agents locating gold answers on HuggingFace or arXiv instead
of solving the task, and hard-coded "plausible" solutions on CORE-Bench and SciCode (§4.2, Tables A7–A8,
A10). This bucket is only observable by reading trajectories; the outcome predicate scores these as
successes.

**Conditions and limits.** Most HAL evaluations are single runs without confidence intervals because of
cost (App. A3 item 1), and the paper states it cannot establish whether fixing a flagged failure would have
produced success, which would require checkpointing and replay (App. A4.2). A trajectory bucket is a
description of what happened, not a causal claim about what would fix it.

---

## §8 The failure ledger and matching the report to the decision

**The ledger.** A per-run report is a snapshot. A ledger is the same buckets carried across runs, one row
per `(bucket, slice)` and one column per run: `{run_id, checkpoint, suite, split, slice_key, bucket,
count, n_items, share, example_ids, first_seen_run, tagger_version, fix_attempted}`. The column that makes
it usable is `tagger_version`: when the tagger or its rubric changes, counts before and after are
measurements by different instruments, and [[holistic-agent-leaderboard]]'s precision figures apply to one
judge and rubric only (App. A7.2).

**Automated slice discovery.** Slices can also be found rather than declared. [[ai2-benchmirt]] fits a
two-dimensional item response theory model on 100 open-weight LLMs across 16 benchmarks and 34,301 items,
without telling the model which benchmark measures what, and recovers two abilities that the authors label
safety and general reasoning (blog; report §2.2–2.3). The per-benchmark correlations show that a label on
the outside of a benchmark does not determine what it measures: BBQ, grouped under safety, correlates 0.85
with general reasoning and −0.06 with the safety axis; WMDP correlates −0.89 with general reasoning; XSTest
splits between the two, consistent with its even over-refusal and safety split; and the WildJailbreak benign
subset (250 of 2,250 items) and the HarmBench copyright subset lean toward reasoning while the rest of each
benchmark aligns with safety (blog table and "What BenchMIRT reveals"). **Result (single study; the two
dimensions depend on the 16-benchmark set, and all models were released by March 2025.)** The operational
consequence for a slice report: a benchmark's own subsets can be the slice boundary that matters, and
averaging across them mixes constructs.

**Matching the artifact to the decision.** Each row below is a different query against the same item-level
record table, not a different experiment.

| Decision to be made | Slice axis that answers it | Evidence in this chapter |
|---|---|---|
| Should this stage be kept? | training stage × benchmark | [[tulu-3]] Table 6 §1 |
| Did the gain generalize past the benchmarks used to develop it? | development against held-out, per skill | [[tulu-3]] Table 31, [[ifbench]] Table 6 §2 |
| Is the capability attached to the task or the surface? | perturbation ladder | [[gsm-symbolic]] Table 1 §3 |
| Does the long-context claim hold? | length × depth × hops × overlap × distractors | [[ruler]], [[lost-in-the-middle]], [[nolima]] §4 |
| What did this stage cost elsewhere? | untargeted-capability slices, KL on the new task | [[transferability-of-llm-reasoning]], [[rls-razor]] §5 |
| Which fix addresses the most failures? | reason buckets and confusion cells per slice | [[bfcl]] §6 |
| Where do agent runs break? | trajectory buckets, scaffold, environment against model | [[holistic-agent-leaderboard]] §7 |
| Is this benchmark measuring what its label says? | item-level ability decomposition | [[ai2-benchmirt]] §8 |

**One statistical caution that belongs to the next chapter.** A 50-slice report tested at a 5% level
produces, under a true null on every slice, an expected 50 × 0.05 = 2.5 slices flagged as changed. Reading
a per-slice report without a multiple-comparison correction therefore manufactures regressions at a
predictable rate. The corrections, the confidence intervals, and the paired comparisons are the subject of
ch-51; this chapter produces the table they operate on.

---

## Negative samples and negative feedback

This chapter consumes failures rather than training on them, which places it in sense (2) of the four in
the course's negative-feedback taxonomy — **negative as content**: a failed item is read, labelled, and
used to select the next data or training change. No gradient is taken on it here, so likelihood
displacement and squeezing do not apply to anything in this chapter.

**Where the negatives come from and how they are labelled.** Three labelling mechanisms appear above, with
different error properties. A **program verifier** produces the failure label directly: [[bfcl]]'s AST
matcher, where the failure mode is a wrong accept rather than a wrong reject, since the matcher accepts an
int where a float is expected in Python but not in Java or JavaScript (§4.1, App. H). A **model tagger**
produces it with measured precision and unmeasured recall ([[holistic-agent-leaderboard]] Table A2:
0.87 / 1.00 / 0.94, recall not measured). A **generated-item filter** produces it with a validated rate:
[[nolima]] filters haystacks with Llama 3.3 70B over 1,000-character chunks and validates the filter at
99.8% in a control test where each needle is placed in 100 random chunks (§3.1, §4.2).

**What is done with them.** Counted by slice, and used to select a fix. The step this chapter does not
authorize is training on the bucket counts themselves: [[holistic-agent-leaderboard]] states that it cannot
establish whether fixing a flagged failure would produce success without checkpointing and replay
(App. A4.2).

**The measured risk of optimizing against a failure bucket.** [[ifbench]] trains directly on constraint
verifiers and measures what that costs outside the target: a pure constraint reward lowered LLM-judge
response quality from 7 to 6.4 out of 10 and AlpacaEval 2 from 33.5 to 21.3, and adding a preference
reward-model term to the verifiable reward is the paper's mitigation (§5, Table 3, App. E).
**Result (single study.)** The bucket that shrinks is not the only quantity that moved.

**Diagnostics to keep alongside bucket counts.** Abstention rate (an over-corrected model shrinks the
`abstention-failure` bucket by refusing more, visible as the Irrelevance-against-Relevance split in
[[bfcl]] Table 1); the held-out slice for the same skill as the bucket; and the untargeted-capability
slices from §5, which is where a bucket-driven fix shows its cost.

---

## Recipe

Slice analysis has no training hyperparameters. What it has is an evaluation protocol whose settings
determine what a slice report can resolve. The rows below are the protocol values this chapter relies on,
each at the locus where it is printed.

| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| Tülu 3 8B | 8B | eval-gate | evaluation suite split | development suite + unseen suite, paired per core skill; unseen scores not examined during development | arXiv:2411.15124 §2.2, §7.2–7.3, Table 3 | verified (2026-09-15) | §7.4.1 reports the resulting overfitting findings per skill (Table 31, Table 32, Fig. 24) |
| Tülu 3 8B | 8B | RL | RLVR checkpoint-selection rule | evaluate every 100 training steps; select the checkpoint with best overall MATH and IFEval | arXiv:2411.15124 §6.4 | verified (2026-09-15) | §6.4 notes runs reaching GSM8K 89.4 and IFEval 84.8 that "tended to perform worse in other metrics, dragging down their overall average" |
| GSM-Symbolic | n/a (benchmark) | eval-gate | instances per template set | 100 templates from GSM8K test items; 50 datasets of 100 examples each | arXiv:2410.05229 §4, §3.1 | verified (2026-09-15) | §4.1 reports the resulting per-model score distribution and its spread (Fig. 2, Table 1) |
| RULER | n/a (benchmark) | long-context | examples per task per length | 500, at 4K/8K/16K/32K/64K/128K, native chat template plus an answer prefix, recall-based matching | arXiv:2404.06654 §4 | verified (2026-09-15) | Table 3's claimed-against-effective-length column is computed from these runs |
| RULER | n/a (benchmark) | long-context | effective-length threshold | average score above Llama2-7B at 4K = 85.6 | arXiv:2404.06654 §4, Table 3 | verified (2026-09-15) | no ablation reported for the threshold choice |
| NoLiMa | n/a (benchmark) | long-context | tests per context length | 5 haystacks × 58 question-needle pairs × 26 placements = 7,540; lengths 250 to 32K | arXiv:2502.05167 §4.1 | verified (2026-09-15) | §4.2 validates the haystack filter at 99.8% in a 100-chunk control test |
| NoLiMa | n/a (benchmark) | long-context | effective-length threshold | largest tested length scoring above 0.85 × base score, base = max over 250/500/1K | arXiv:2502.05167 §4.3 | verified (2026-09-15) | §4.3 contrasts it with RULER's fixed 85.6 threshold; no ablation of 0.85 reported |
| HAL (Holistic Agent Leaderboard) | 9 models | eval-gate | rollout budget and repetition | 21,730 rollouts, 9 models × 9 benchmarks, about $40,000; most evaluations single runs without confidence intervals | arXiv:2510.11977 Abstract, App. A3 item 1 | verified (2026-09-15) | App. A3 item 1 states cost as the reason; no variance ablation reported |
| HAL (Holistic Agent Leaderboard) | n/a | eval-gate | trajectory tagger | GPT-5 Medium rubric judge over 2,184 transcripts (1,634 after removing TAU-bench); precision 0.87 / 1.00 / 0.94 on three human checks | arXiv:2510.11977 App. A7.1–A7.2, Table A2 | verified (2026-09-15) | Table A2 gives the human-validated precision; recall not measured |
| BenchMIRT | 100 models | eval-gate | item pruning | keep the most discriminative 10% or 50% of items | allenai.org/blog/benchmirt, "Doing more with fewer questions" | verified (2026-09-15) | 10% "generally preserved nearly the same picture"; 50% "often matched the full benchmark … even more closely" |

**Starting point for a small general-purpose run.** Every value here comes from a `verified` row above, with
the conditions the source used. Maintain two evaluation suites, one development and one held-out, paired
per skill and never read during development, as in Tülu 3 8B (arXiv:2411.15124 §2.2) — that report's
development/unseen gaps ranged from 6.7 to 76.5 points at 8B (Table 31), so the two suites are not
interchangeable. For any benchmark used for a training decision, build a perturbation set from at least 100
of its items, generating multiple instances per template as GSM-Symbolic does (100 templates, 50 sets of
100). For a long-context claim, run at the six RULER lengths with 500 items per task per length, and report
effective length under a stated threshold. For agent evaluation, hold the scaffold fixed, record cost next
to accuracy, and validate any rubric tagger on a human-labelled sample before its counts are compared
across runs.

---

## Generalization lens

**(a) What increases breadth.** Held-out suites paired to development suites per skill, with the held-out
column unread during development ([[tulu-3]] §2.2, §7.4). Training on a wider set of constraint types
rather than the tested ones, which raised both the in-distribution and the out-of-distribution score
([[ifbench]] Table 6: IFEval 81.1 → 92.2 and IFBench 25.2 → 44.6). Choosing RL over SFT when a narrow
corpus is the only training data available, measured as a non-reasoning average of 53.2 after RL against
21.1 after SFT from a base of 45.7 ([[transferability-of-llm-reasoning]] Table 1), with the mechanism
proposed as an implicit bias toward KL-minimal solutions ([[rls-razor]] §4–§5, **Interpretation**).

**(b) What causes narrowing or forgetting.** Selection on a development benchmark: [[tulu-3]] §7.4.1 states
the pipeline overfit to the development evaluations on precise instruction following and, to some extent,
knowledge recall and reasoning. Optimizing a single verifiable reward: [[ifbench]] §5 measures AlpacaEval 2
falling 33.5 → 21.3 and judge-rated quality 7 → 6.4 under a pure constraint reward. SFT on a narrow domain:
[[transferability-of-llm-reasoning]] Table 1 shows CoQA 10.0 → 1.7 and HalluEval 35.7 → 2.3 for the SFT
(think) run. A stage that gains its target: [[tulu-3]] Table 6 shows Safety −1.7 and TruthfulQA −1.1 across
the DPO → RLVR step. Fitting the surface: [[gsm-symbolic]] Table 1 shows Gemma2-9b-it at 87.0 on the
originals and 22.3 on NoOp.

**(c) How generality is measured at this stage.** Four measurements, each with a known limitation.
The development-minus-held-out gap per skill ([[tulu-3]] Table 31) — an upper bound on overfitting, since
it also contains genuine difficulty and format differences (§7.4.1). The perturbation ladder from canonical
to distractor-added items ([[gsm-symbolic]] Table 1) — limited to the benchmark that was templated. The
long-context grid over length, depth, hops, and overlap ([[ruler]] Table 3, [[lost-in-the-middle]]
App. G Table 6, [[nolima]] Tables 3 and 6) — effective-length numbers depend on the threshold chosen
(85.6 fixed in RULER, 0.85 × base in NoLiMa). Untargeted-capability deltas across stages, summarized by the
Transferability Index ([[transferability-of-llm-reasoning]] §2.1) — a ratio to the math group's gain, so it
is undefined in sign interpretation when the math gain is small.

---

## Common mistakes and how to detect them

| Mistake | Observable symptom | Check |
|---|---|---|
| Deciding on the aggregate only | The report has one row per checkpoint | Recompute the aggregate from the slice scores; if it cannot be recomputed, the slices were not stored ([[tulu-3]] Table 6 §1) |
| Treating a development benchmark as evidence of generality | A single score per skill | Add the paired held-out benchmark and report the gap; a gap of 58.1 points on instruction following was invisible in IFEval alone ([[tulu-3]] Table 31) |
| Reading the held-out suite during development | Held-out and development scores move together across every design decision | Record which scores were consulted at each decision; [[tulu-3]] §2.2 states the unseen set was not examined |
| Reporting one number for a long-context claim | A "128K" label with no grid | Score the length × depth × overlap grid; [[nolima]] Table 6 shows 98.5 against 25.9 at the same length on the same model |
| Reporting effective length without its threshold | Two papers' effective lengths compared directly | Print the rule: 85.6 fixed ([[ruler]] §4) against 0.85 × base ([[nolima]] §4.3) |
| Comparing agent scores across scaffolds | Model ranking changes between reports | Fix the scaffold or evaluate several; ranking reversed by scaffold in [[holistic-agent-leaderboard]] §4.1 item 6 |
| Scoring an environment failure as a model failure | Unexplained score drops between identical runs | Separate the environment-barrier bucket; silently switched endpoints and rate-limit failures are documented in [[holistic-agent-leaderboard]] App. A3 |
| Trusting bucket counts from an unvalidated tagger | Bucket shares shift when the judge model changes | Validate the tagger on a human-labelled sample and record `tagger_version`; only precision was measured in [[holistic-agent-leaderboard]] Table A2 |
| Averaging subsets that measure different things | A benchmark's subsets disagree with each other more than with other benchmarks | Check subset-level alignment; BBQ correlates 0.85 with general reasoning and −0.06 with safety ([[ai2-benchmirt]] blog table) |
| Flagging regressions across 50 slices at a 5% level | Two or three regressions appear in every report and never reproduce | Apply a multiple-comparison correction (ch-51); the expected count under a true null is 50 × 0.05 = 2.5 |

---

## Check your understanding

1. The Tülu 3 8B DPO → RLVR average moves +0.4 while GSM8K moves +3.3. Explain, using the twelve per-benchmark
   deltas in §1, why a threshold rule on the average cannot be tuned to make both the GSM8K gain and the
   Safety loss visible.
2. [[tulu-3]] Table 31 shows HumanEval identical (83.9 → 83.9) and BigCodeBench falling (9.5 → 7.4) across
   the same step. Give two distinct explanations consistent with both numbers, and state an additional
   measurement that would separate them.
3. IFBench raises IFEval from 81.1 to 92.2 and IFBench from 25.2 to 44.6. Explain why the persistence of a
   47.6-point gap after training on 29 unseen-to-IFEval constraints is evidence about the capability rather
   than about the training set size.
4. GSM-Symbolic reports per-set standard deviations of 2–6 points from re-instantiating the same 100
   questions. Explain why this number changes how a 2-point difference between two checkpoints on GSM8K
   should be read, and why the correction is not "use more GSM8K items".
5. NoLiMa's Table 6 gives 98.5, 56.2, and 25.9 at 32K for direct, one-hop, and two-hop questions on the
   same model and haystack. Explain why "effective context length" is not a property of a model alone, and
   what has to be stated alongside it.
6. RL's Razor claims that KL divergence measured on the *new* task predicts forgetting on *prior* tasks.
   Explain why a predictor measured only on the new-task distribution is operationally useful, and what it
   cannot tell you.
7. On SciCode, at least one tool-call failure appears in 97.7% of failed runs and 100% of successful runs.
   Explain what this implies about using bucket prevalence as a ranking of what to fix, and which of HAL's
   measured quantities does support such a ranking.
8. BenchMIRT finds BBQ correlating 0.85 with general reasoning and −0.06 with a safety axis. Explain how
   this changes the interpretation of a safety-suite average, and how you would detect the same problem in
   a suite you built yourself without fitting an IRT model.

---

## Connections

- **Previous: ch-49 — Judge Models: Bias, Calibration, and Judge-Specific Overfitting.** A reason-tagger is
  a judge, so its position bias, length bias, and self-preference apply to bucket counts. §6 above adds the
  requirement that the tagger's precision be measured and its version recorded.
- **Next: ch-51 — Metric Noise, Confidence Intervals, and Go/No-Go Decisions.** Every per-slice delta in
  this chapter is a point estimate. ch-51 supplies the item-level bootstrap, the paired comparison, the
  clustered standard errors, and the multiple-comparison control across the slices produced here.
- **ch-47 — Evaluation Harness and Suite Design for General Capability.** Produces the item-level records
  that every slice in this chapter is a `GROUP BY` over; per-task means cannot be sliced afterwards.
- **ch-47a — Benchmark Overfitting and Generalization Audits: Fresh, Perturbed, Counterfactual, and Live
  Evaluation.** Constructs the perturbation and held-out sets that §2 and §3 consume as slices.
- **ch-48 — Contamination Detection and Its Effect on Reported Scores.** Contamination is one of the
  candidate explanations for a large canonical-minus-perturbed gap ([[gsm-symbolic]] §4.1); ch-48 supplies
  the test that separates it from surface fitting.
- **ch-51a — Evaluating Agent Generality and Reliability.** Extends §7 to repeated-run reliability, held-out
  environments, and the variance of agentic metrics.
- **ch-53 — Lab: Evaluation Harness with a Held-Out Suite, Forgetting Report, and Perturbation Robustness.**
  Implements the three report types of this chapter end to end.

---

## Sources

- [[tulu-3]] — per-stage per-benchmark table at 8B (Table 6), the RLVR-stage table (Table 23), the
  development/unseen suite design (§2.2, §7.2–7.4) and its per-skill results (Table 31), and the
  checkpoint-selection rule (§6.4). Note: the library card's "What RLVR buys" line states "+5–10pp on
  GSM8K"; Table 6 and Table 23 both give 84.3 → 87.6 at 8B, and this chapter uses the tables.
- [[ifbench]] — 58 held-out verifiable constraints, the IFEval/IFBench pair before and after IF-RLVR
  (Table 6), and the measured cost of a pure constraint reward on general response quality (§5, Table 3).
- [[gsm-symbolic]] — template construction (§3.1), the 50 × 100 generation protocol (§4), the per-variant
  table including NoOp (Table 1), and the contamination interpretation of the original-versus-Symbolic gap
  (§4.1). Excerpt: `excerpts/gsm-symbolic.md` (no library card exists for this source).
- [[ruler]] — claimed against effective length across 17 models (Table 3), the two weighted averages, and
  the per-generator error analysis at fixed model (§5, Fig. 2).
- [[lost-in-the-middle]] — accuracy by evidence position with 20 documents (App. G Table 6), the
  closed-book and oracle reference lines (Table 1), and the window-extension comparison (§2.3).
- [[nolima]] — question-needle overlap measurement (Table 1), claimed against effective length under a
  0.85 × base threshold (Table 3), the direct/one-hop/two-hop comparison at fixed length (Table 6), and the
  haystack-filter validation (§4.2).
- [[transferability-of-llm-reasoning]] — the Transferability Index definition (§2.1) and the Qwen3-14B
  controlled comparison of SFT and RL across three task groups (Table 1).
- [[rls-razor]] — the forgetting law relating KL on the new-task distribution to prior-task loss, its
  quadratic fits (R² = 0.96 controlled, 0.71 LLM), and the oracle-SFT control (§3–§4). Excerpt:
  `excerpts/rls-razor.md` (no library card exists for this source).
- [[ai2-benchmirt]] — two-dimensional IRT decomposition over 16 benchmarks and 34,301 items, the
  per-benchmark correlation table (BBQ, WMDP, XSTest), the subset findings, and the item-pruning result.
- [[bfcl]] — AST matching rules as a confusion-cell definition (§4.1, App. H), the Irrelevance and Relevance
  categories and their divergence (Table 1), decoding-issue counts by mode (§5.1), and the LLM-judged
  multi-turn root-cause distribution (§5.4.2).
- [[holistic-agent-leaderboard]] — the six-category trajectory rubric and its measured prevalence (§4.2),
  the tagger's validated precision and unmeasured recall (Table A2), scaffold and model interaction
  (§4.1 items 6–7), environment-failure cases (App. A3), and the TAU-bench few-shot leak (App. A5).
