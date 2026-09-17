<!-- chapter: ch-47a
     track: eval
     kind: content
     title: Benchmark Overfitting and Generalization Audits: Fresh, Perturbed, Counterfactual, and Live Evaluation
     deps: [ch-47]
     sources: [[gsm1k]], [[gsm-symbolic]], [[reasoning-or-reciting]], [[embers-of-autoregression]], [[ifbench]],
              [[livebench]], [[livecodebench]], [[matharena]], [[leaderboard-illusion]], [[training-on-the-test-task]],
              [[swe-bench-illusion]], [[adding-error-bars-evals]]
     figures: figures/benchmark-audit-gap.html
     revised: 2026-09 (generality revision)
-->

# Chapter 47a — Benchmark Overfitting and Generalization Audits: Fresh, Perturbed, Counterfactual, and Live Evaluation

> **Core insight.** A benchmark score is an estimate of capability on the exact distribution that benchmark
> samples, and the gap between that estimate and general capability has been measured several times. On a
> human-written parallel set built to match GSM8K in difficulty, the worst-affected models score up to 8 points
> lower than on GSM8K while gpt-4o moves 0.2 points and gpt-4-turbo 0.3 ([[gsm1k]], App. F). On GSM8K questions
> rewritten as templates with one irrelevant clause added, GPT-4o falls from 94.9% to 63.1% and Phi-3-medium-128k
> from 82.5% to 29.4% ([[gsm-symbolic]], Table 1). Fine-tuning a 7B model on 70% Chatbot Arena data raises its
> ArenaHard win rate from 23.5% to 49.9% and leaves MMLU at 65.9% against 66.5% for the 0% mixture
> ([[leaderboard-illusion]], Fig. 10, Table 9). None of these gaps require the test data to have leaked; the first
> two are measured on data the model never saw, and the third is measured on a model trained with no test data at all.
>
> **Guideline.** When a benchmark number is used as evidence about general capability, pair it with one audit
> counterpart of the same task and report the paired difference with an interval, because a comparison on a few
> hundred items cannot resolve differences smaller than about 5 points, and the unpaired form of the same comparison
> is weaker still ([[adding-error-bars-evals]] §4.2, §5, and the worked example in §7 below). When the claim concerns a skill, use a perturbed or counterfactual variant that holds the skill fixed and
> changes an incidental property, because default-versus-counterfactual gaps of 30 to 60 points appear on tasks the
> models describe correctly when asked ([[reasoning-or-reciting]] Table 18). When the claim concerns a domain with a
> stream of new problems, use a release-date filter, because performance on a fixed set can fall by tens of points on
> problems published after a model's cutoff ([[livecodebench]] §5.1). When the claim comes from a leaderboard, check
> what the leaderboard allows to be withheld, because two identical checkpoints submitted to Chatbot Arena scored
> 1052 and 1069 with four other models ranked between them ([[leaderboard-illusion]] §3.3). When no audit counterpart
> exists, state the benchmark's name in the claim instead of the capability's name.

## Why this chapter matters for a general-purpose model

The pipeline for a general-purpose model runs pre-training → mid-training → SFT → preference optimization → RL →
evaluation. Every stage before this one used benchmark numbers to make decisions: which data mixture to keep, which
checkpoint to release, whether an RL run helped. Those decisions are only as good as the relationship between the
number and the capability it stands for.

That relationship fails in two distinct ways, and the distinction is what this chapter is built around.

1. **Contamination**: the evaluation items, or close paraphrases of them, were in the training data. Detection and
   the size of the score effect are the subject of [[ch-48]].
2. **Benchmark overfitting**: the training procedure was shaped by the benchmark without the benchmark's test items
   ever entering training. Selecting a data mixture by ablations on a benchmark, generating synthetic data from a
   benchmark's taxonomy, choosing a checkpoint by a benchmark's score, and training on a distribution the benchmark
   samples from all produce this. [[training-on-the-test-task]] groups these practices under one name and shows they
   are large enough to reverse conclusions.

The measurable problem of this chapter: **for a given headline claim, how large is the gap between the benchmark
score and performance on an equally difficult set that the training procedure could not have been shaped by, and is
that gap larger than the measurement error of the comparison.**

A chapter later in this phase, [[ch-50]], takes the same numbers apart by slice. This chapter asks the prior
question of whether the aggregate number means what its name says.

## §1 Four ways a score overstates capability, and what each one predicts

**Definition.** An *audit* here is a second measurement, of the same claimed capability, on a set that differs from
the headline benchmark in one controlled respect: freshness, surface form, world assumptions, constraint set, or
release date. The audit's purpose is to produce a difference that discriminates among the causes below.

| Cause | What entered training | What an audit sees | Primary evidence |
|---|---|---|---|
| Verbatim or near-verbatim contamination | the test items or paraphrases of them | large drop on a fresh parallel set; high sequence log-likelihood on the test set | [[gsm1k]] §5.4; [[ch-48]] |
| Training on look-alike data and benchmark-driven selection | data resembling the benchmark; mixtures and checkpoints chosen by its score | drop on a fresh set, but also equalization once every model is fine-tuned on the same task data | [[training-on-the-test-task]] §2.2 |
| Narrow procedures learned from the default form of a task | nothing improper; the task is common in text in exactly one form | large drop on perturbed, counterfactual, or rarer variants, with the comprehension check still passed | [[reasoning-or-reciting]] Table 18; [[embers-of-autoregression]] §5 |
| Distribution overfitting through leaderboard feedback | data drawn from the evaluation's own distribution, plus selective submission | in-distribution gain with no out-of-distribution gain; rank movement under repeated submission | [[leaderboard-illusion]] §3.3, §4.2 |

These causes are not exclusive, and the point of separating them is that they call for different responses. The
first is a data-pipeline defect and is repaired by decontamination. The second is a legitimate engineering choice
that has to be disclosed and adjusted for. The third is a property of the model that no amount of data hygiene
removes. The fourth is a reporting problem in the benchmark's own governance.

[[gsm1k]] §5.4 gives a measurement that separates the first from the rest. For each model, the authors compute the
per-character sequence log-likelihood of the GSM8K test set,

```
(1/c) Σ_i log p(x_i | x_<i)
```

where `x_i` is the i-th token of a test example, `x_<i` the tokens before it, and `c` the number of characters in the
example, used to make models with different tokenizers comparable. Across the evaluated models, this quantity
correlates with the GSM8K-minus-GSM1k gap at Spearman 0.36 (p = 0.03), with Pearson r² = 0.26 and Kendall τ = 0.29
(§5.4, Fig. 5). **Result (single study).** A rank correlation of 0.36 leaves most of the variation in the gap
unexplained, and the authors name the outliers: Math-Shepherd-Mistral-7B-RL has one of the largest gaps with a low
log-likelihood, which they attribute to its process reward model having been trained on GSM8K reasoning chains, and
Llemma has a high log-likelihood with a minimal gap. (The abstract of that paper writes "Spearman's r² = 0.36" while
§5.4 writes "Spearman's rank correlation of 0.36" and reports Pearson r² separately as 0.26; the rank correlation is
the figure to quote.)

**Implication for a general-purpose model.** Decontaminating the training set is necessary and does not by itself
make a benchmark number a capability estimate. The other three causes remain, and the third is the one that a
general-purpose model is being built to avoid.

## §2 Fresh parallel test sets

**Definition.** A *fresh parallel test set* is a new set of items, written after the original benchmark and matched
to it in difficulty and format, used to re-measure the same claimed capability.

**The problem it addresses.** A drop on a new set is uninterpretable unless the new set is as hard as the old one.
GSM1k spends most of its construction effort on that comparison, which is what makes it usable as an audit rather
than as a second benchmark.

**Mechanism** ([[gsm1k]] §3):

1. Human annotators write problems, prompted with three GSM8K examples each; no language model is used at any stage.
2. Difficulty is requested by resolution-step count, where the step count of a GSM8K problem is estimated from its
   calculator tags, and the requested distribution matches GSM8K's.
3. Three review layers: trusted-annotator review, an independent second solve with mismatches discarded, and a
   general quality audit. Of 2,108 initial problems, 1,419 survive the second solve and 1,375 the audit.
4. Problems are discarded until the answer-magnitude distribution matches GSM8K's, leaving 1,205 items.
5. The set is withheld from publication, with a pre-committed release condition.

**Evidence that the two sets are comparable.** 19 annotators asked to find the single GSM1k item among four GSM8K
items succeeded 21.83% of the time over 1,205 attempts, against 20% for chance (§3.2.1). 14 annotators solving under
time pressure completed 4.07 ± 0.93 GSM8K problems and 4.36 ± 1.11 GSM1k problems in 15 minutes (§3.2.2). Two models
predating GSM8K's release show minimal difference between the sets (§3.2.3).

**Result.** Under one prompt for both sets — 5 examples drawn from the GSM8K *train* split, an LM Evaluation Harness
fork, max 1000 generated tokens, temperature 0 for open models — the largest drop is 8 points, and the direction of
the drop varies by family (App. F). Yi-6B-Chat falls 0.437 → 0.357, math-shepherd-mistral-7b-rl 0.826 → 0.754,
Meta-Llama-3-8B-Instruct 0.752 → 0.690, gpt-4o 0.931 → 0.929, and gemini-1.5-flash-preview-0514 rises 0.797 → 0.835.
The Phi and Mistral families drop across nearly every release and size (§5.1); frontier models do not (§5.2).

The interactive scatter in [figures/benchmark-audit-gap.html](figures/benchmark-audit-gap.html) plots all 63 models
against the line of equal accuracy and shows each model's z-score on hover; use it to see that the drops concentrate
in specific families rather than scaling with the headline score.

**Worked numeric example — is one model's drop larger than sampling noise?** Take
Phi-3-medium-128k-instruct: 0.869 on GSM8K, 0.825 on GSM1k. Treat the two as independent samples of 1,205 items each
(GSM1k's size). The pooled rate is

```
p̂ = (0.869 + 0.825)/2 = 0.847
SE = sqrt( p̂(1 − p̂) (1/n₁ + 1/n₂) ) = sqrt( 0.847 × 0.153 × (2/1205) ) = 0.0146
z  = (0.869 − 0.825) / 0.0146 = 3.01
```

The paper reports z = 3.103 for this row. The residual difference comes from the GSM8K test split having a different
size than 1,205, which the paper does not state (checked §4 and App. F). Two points follow. First, a 4.4-point gap at
this sample size is outside noise. Second, the same 4.4-point gap measured on 300 items would give SE = 0.029 and
z = 1.5, which is not. Sample size decides what an audit can conclude, and §7 makes that quantitative.

**Conditions and limits.** GSM1k is grade-school arithmetic, a task the same paper shows the most overfit models
still perform: Phi-2 drops 6 points and still solves over half of GSM1k (§5.3). A drop on a fresh set bounds how much
of the headline number is inflated; it does not show the model cannot do the task. The authors also state that the
two sets are "highly similar, but not identically distributed" despite the matching effort (§3).

**Implication for a general-purpose model.** A fresh parallel set is the cleanest audit available, and the most
expensive: 2,108 human-written problems, three review layers, and a decision not to publish. Reserve it for the one
or two capabilities a release claims most loudly.

## §3 Perturbed and counterfactual variants

A fresh set changes everything at once. Perturbation changes one property and holds the rest fixed, which makes the
result diagnostic: it names which property the model depends on.

### 3.1 Symbolic templates: measuring the distribution instead of the point

[[gsm-symbolic]] rewrites 100 GSM8K test questions as templates with named variables, sampling ranges, and
consistency conditions, then draws 50 instances per template, giving 50 datasets of 100 items each (§3, §3.2).
Evaluation is 8-shot chain-of-thought with greedy decoding on more than 20 open models from 2B to 27B plus four
OpenAI models.

Two results matter for auditing.

1. **A single benchmark score hides a distribution.** Across the 50 sets, Gemma2-9B's best-to-worst gap exceeds 12%
   and Phi-3.5-mini's is about 15%, with only names and values changed (§4.1).
2. **The original items sit high in that distribution.** For 21 of 25 models, the score on the 100 original GSM8K
   questions is more than one standard deviation from the centre of that model's GSM-Symbolic distribution, usually
   on the high side (§4.1). **Result (single study).**

*Worked numeric example.* Phi-3-medium-128k-instruct scores 82.5% ± 2.86 across the 50 symbolic sets and 89.0% on
the 100 original questions. The original score sits (89.0 − 82.5)/2.86 = 2.27 standard deviations above the mean of
the perturbed distribution. Under a normal approximation, a draw at least that far above the mean occurs about 1.2% of the time. The
authors name contamination as one explanation and do not claim it is the only one.

### 3.2 No-op clauses: GSM-NoOp

GSM-NoOp adds one clause that is topically related and carries no operation — in the paper's example, that five of
the kiwis picked on one day were smaller than average in a problem that asks for the total count (§4.4, Fig. 7). The
arithmetic is unchanged. Models subtract the five.

| Model | GSM-Symbolic | GSM-NoOp | Relative drop (derived) |
|---|---|---|---|
| GPT-4o | 94.9 | 63.1 | −33.5% |
| o1-mini | 94.5 | 66.0 | −30.2% |
| o1-preview | 92.7 | 77.4 | −16.5% |
| GPT-4o-mini | 91.7 | 54.1 | −41.0% |
| Gemma2-27b-it | 88.3 | 30.0 | −66.0% |
| Phi-3-medium-128k-instruct | 82.5 | 29.4 | −64.4% |
| Phi-3-mini-128k-instruct | 80.7 | 18.0 | −77.7% |
| Mathstral-7b-v0.1 | 74.0 | 20.4 | −72.4% |

Absolute values from [[gsm-symbolic]] Table 1; the relative column is computed here. §4.4 states a drop "over 65%"
for Phi-3-mini and the abstract states drops "up to 65%"; Fig. 8a prints per-model bars that agree with Table 1 to
about 1.5 points for GPT-4o and o1-preview and differ by 6.6 points for Phi-3-medium-128k, and the paper does not
state which baseline each bar uses. Quote Table 1.

The shot-source ablation rules out the obvious remedy. For Phi-3-medium-128k (Fig. 8b): 87.3 on GSM8K questions with
GSM8K shots, 82.5 on symbolic instances, 29.4 on NoOp with GSM8K shots, **30.2 on NoOp when the eight in-context
examples are symbolic instances of the same question with the full reasoning chain**, and 22.6 with NoOp shots of
other questions. Showing the model the correct procedure for the identical problem does not restore the loss.

### 3.3 Counterfactual worlds and the comprehension check

[[reasoning-or-reciting]] separates "did not understand the instruction" from "cannot execute under it". Each of its
11 tasks has a default world and a counterfactual world fully specified in the prompt: base-10 versus base-9
arithmetic, 0-based versus 1-based Python indexing, standard versus transposed musical keys, standard versus rotated
spatial axes. Each task also has a **counterfactual comprehension check (CCC)**, a separate question that tests only
whether the model grasped the stated world.

Two-digit addition, 0-shot chain of thought, 1,000 test instances and 200 CCC instances (Table 18):

| | base 8 | base 9 | base 10 | base 11 | base 16 |
|---|---|---|---|---|---|
| GPT-4 accuracy | 60.2 | 38.6 | 98.2 | 56.5 | 74.0 |
| GPT-4 CCC | 98.0 | 90.0 | 100.0 | 91.0 | 100.0 |
| GPT-3.5 accuracy | 12.6 | 9.8 | 99.0 | 2.7 | 17.7 |
| Claude accuracy | 1.4 | 0.9 | 98.7 | 4.0 | 6.6 |

GPT-4 answers 90% of base-9 comprehension questions and 38.6% of base-9 additions. The deficit is in execution, not
in reading. In-context examples narrow the gap without closing it: at 16 shots, base-9 two-digit addition reaches
88.4% against 99.9% for base-10, and at 0 shots the gap widens with digit count — 4-digit base-9 addition is 14.6%
against 83.4% for base-10 (Table 19).

§5.1 of the same paper adds the pattern that predicts where counterfactual performance survives: the more common the
counterfactual world is in text, the smaller the drop. Swapping north and south in spatial reasoning produces the
smallest drop and for PaLM-2 exceeds the default, which the authors attribute to plotting libraries with an inverted
y-axis. Drop-D guitar tuning gives the highest counterfactual chord-fingering score.

### 3.4 The probability confound in perturbation design

[[embers-of-autoregression]] makes that pattern the object of study, on gpt-3.5-turbo-0613 and gpt-4-0613 at
temperature 0. Three properties of a test item predict accuracy on deterministic tasks:

| Property | Comparison | GPT-4 |
|---|---|---|
| Task probability | rot-1 / rot-3 / rot-2 shift-cipher decoding | 0.82 / 0.76 / 0.02 |
| Task probability | Pig Latin versus an invented system of equal complexity | 0.39 versus 0.13 |
| Task probability | linear function (9/5)x + 32 versus (7/5)x + 31 | 0.33 versus 0.00 |
| Output probability | word-sequence reversal, high- versus low-probability answer | 97% versus 53% |
| Input probability | rot-13 encoding, high- versus low-probability input | 21% versus 11% |

The sorting result (Table 3) is the cleanest control, because it varies frequency while holding the operation fixed:

| Task | Count in C4 | GPT-3.5 | GPT-4 |
|---|---|---|---|
| Alphabetical order | 95,942 | 0.76 | 0.80 |
| Reverse alphabetical order | 629 | 0.15 | 0.32 |
| Ascending order | 21,562 | 0.66 | 0.82 |
| Descending order | 31,378 | 0.58 | 0.80 |

Two orderings with a 150× frequency difference differ by 48 points for GPT-4; two orderings with similar frequencies
differ by 2 points. **Result (single study),** on two models from one provider, in 2023, with corpus counts taken
from C4 rather than from either model's training set.

**Consequence for audit design.** A perturbation changes task probability, output probability, and input probability
at the same time unless it is designed not to. A perturbed item that is also a rarer *phrasing* measures both the
skill and the phrasing frequency. State which property the perturbation intends to change, and hold the others as
fixed as the task allows — the GSM-Symbolic choice to keep numeric ranges close to GSM8K's ([[gsm-symbolic]] §3.1) is
an example of that discipline.

## §4 Constraint generalization: when the audit is a new set of rules

The same pattern appears in instruction following, where the "task" is a verifiable output constraint.
[[ifbench]] builds 58 constraints in 7 categories over 300 prompts, disjoint from the 25 constraint templates of
IFEval, and 29 further training constraints with verifiers.

- Models at 80+% on IFEval, at sizes as small as 2B, score below 50% on IFBench; GPT-4.1 and Claude 3.7 Sonnet are
  among them (§1, Fig. 1).
- Training closes the gap partially and not fully: GRPO on varied constraints moves a Tülu-3-8B-DPO policy from
  81.1 to 92.2 on IFEval and from 25.2 to 44.6 on IFBench (Table 6).
- The training that closes it costs breadth: on the same policy, AlpacaEval 2 falls 33.5 → 21.3 and MMLU 68.7 → 66.4,
  and GPT-4.1 judge scores of the same responses with the constraint removed fall from 7 to 6.4 out of 10 (Table 3, §5).

The authors attribute the IFEval-to-IFBench gap to overfitting a small constraint set, with task and evaluation
setup held fixed (§2). The card notes one claim that the paper does not make: it does not say that RLVR training
caused the overfitting, and it names targeted synthetic data built from the IFEval taxonomy as the common practice.

**Implication.** An audit does not have to be a new dataset. Enumerating the *rule set* a benchmark covers, then
building rules outside it with the same verification machinery, measures generalization within a capability at much
lower cost than writing new problems.

## §5 Live and dynamic benchmarks

**Definition.** A *live benchmark* replaces or adds items on a schedule from a stream of newly published material. A
*date-filtered benchmark* keeps all items but records each item's publication date, so a model can be scored only on
items published after its training cutoff.

### 5.1 Date filtering as a contamination probe

[[livecodebench]] collects competition problems with release dates — 511 problems from May 2023 to May 2024 in the
version evaluated — and scores pass@1 in four scenarios (§3, §4). Filtering by date turns the benchmark into a
detector:

- DS-Ins-33B drops sharply on LeetCode problems released after August 2023, immediately before its release date, and
  the same shape appears in the repair and execution scenarios (§5.1, Figs. 1, 10).
- DS-Base-33B falls from pass@1 ≈ 60 on May problems to ≈ 0 on September problems, which the authors read as
  competition problems entering DeepSeek pre-training and propagating to every instruct model built on it.
- GPT-4-O drops on problems released after November 2023, its stated cutoff.
- Codestral scores 36.5 on problems released May 2023 – January 2024 and 28.3 on problems since February 2024, a
  relative drop of (36.5 − 28.3)/36.5 = 22.5%.
- The drop appears for LeetCode and not for AtCoder problems from the same months (Fig. 11), and GPT-4-Turbo,
  Gemini-Pro, Mistral-L and the Claude-3 models show no comparable structure (Fig. 12).

Because of this, all headline comparisons in that paper use only problems released after September 2023 (§5.2).

The same paper gives a direct measurement of benchmark-specific ability: pass@1 on HumanEval+ correlates with pass@1
on the easy split of LiveCodeBench generation at 0.72, and the residual is structured — base and closed models lie
near the identity line, fine-tuned open-access variants above it. DS-Ins-1.3B scores 59.8% on HumanEval+ and 26.3% on
LCB-Easy (§5.3, Fig. 5).

### 5.2 Scheduled replacement with objective scoring

[[livebench]] combines three commitments: questions from recent sources updated monthly, automatic scoring against
objective ground truth with no LLM judge and no preference vote, and six categories (math, coding, reasoning,
language comprehension, instruction following, data analysis) over 18 tasks (§1, §2). Each task holds 40–100
questions and targets a 30–70% success rate for the top models. The first update added 90 questions to reach 1,000;
the second replaced 132 questions to keep the total at 1,000 (§2.7).

The category correlations are the part worth carrying forward: math, coding and reasoning correlate with each other,
while instruction following correlates weakly with every other category (§3.2, Fig. 2). A single averaged score over
correlated categories is not six independent measurements.

The title of the v2 paper is "contamination-limited", not contamination-free. Once published, a live benchmark's
questions can enter the next training run; freshness is relative to a stated cutoff, not permanent.

### 5.3 Same-day evaluation and the scoring format

[[matharena]] evaluates six USAMO 2025 proof problems within hours of their release, so contamination is excluded by
construction. Four expert graders, two per problem, score each of four runs per model out of 7 points, 42 per run
(§2.1–2.3).

Gemini-2.5-Pro averages 10.1/42 = 24.4%; every other model scores below 5%, with R1 and Grok 3 at 2.0/42 and o3-mini
at 0.9/42 (Table 1). The same models rank at the top of AIME and HMMT leaderboards, which grade the final numeric
answer only. The audit here is not freshness alone: it changes the *output being scored* from an answer to a proof.

The paper also measures what happens when the graders are models. Automated grading with the schemes and a reference
solution supplied inflates the totals — QwQ 1.2 → 23.8, R1 2.0 → 19.3, Claude 3.7 1.5 → 19.0 — by up to about a
factor of 20 (Table 2, §3.2). This is the same result [[ch-49]] develops: a judge model is a measuring instrument
with its own failure modes, and it cannot be assumed to transfer to a new scoring format. (The Table 2 caption in
that paper names a different second grader than its column headers do; the excerpt records this.)

## §6 Distortions that live in the reporting, not in the data

### 6.1 Selective disclosure on a leaderboard

[[leaderboard-illusion]] audits Chatbot Arena's own process. Providers may test private variants and retract scores,
which turns a submission into a best-of-N selection. Meta tested 27 private variants before the Llama-4 launch
(§3.1). In simulation, testing 20 variants raises the expected maximum discovered Arena Score by about 50 points
(§3.2, Fig. 7).

The controlled version is the one to remember: the authors submitted **two identical checkpoints** of Aya-Vision-8B,
without disclosing that they were identical. Final scores were 1052 (±21/22) and 1069 (±19/23), with four other
models ranked between them (§3.3, Fig. 9). Any difference here is selection, since the models were the same file.

### 6.2 Training on the evaluation's distribution

The same paper isolates the effect of training on data from the evaluation's distribution. A 7B base model from the
Cohere Command family is fine-tuned three times with identical settings — 1.3K steps, batch size 128, no
hyperparameter sweep — varying only the share of Arena battle data against a general instruction mixture (§4.2):

| Arena share | ArenaHard win rate vs Llama-3.1-8B-Instruct | MMLU |
|---|---|---|
| 0% | 23.5% | 66.5% |
| 30% | 42.7% | 64.4% |
| 70% | 49.9% | 65.9% |

Relative win-rate gains are (42.7 − 23.5)/23.5 = 81.7% and (49.9 − 23.5)/23.5 = 112.3%. MMLU does not rise. **Result
(single study)**, one model, one mixture family, one judge (gpt-4o-2024-11-20), described by the authors as a lower
bound because they did not optimize the variants.

This is the cleanest available demonstration that a large benchmark gain can carry no general gain, with no test data
involved. It also shows why the audit set has to be *outside the headline benchmark's distribution*: ArenaHard is
reported to correlate with Arena human rankings at 98.6%, so it is a second reading of the same distribution, not an
audit of it.

Two further asymmetries in that paper change how a leaderboard number should be read: estimated data shares of 20.4%
(OpenAI) and 19.2% (Google) against 29.7% for 83 open-weight models combined, and 205 silently deprecated models
against 47 officially listed (§4.1, §5).

### 6.3 Training on the test task, and why fresh sets do not remove it

[[training-on-the-test-task]] takes 56 base models from 70M to 70B and asks whether newer models are better or
better-prepared. Its adjustment is to remove the difference by force: fine-tune *every* model on the same
task-relevant data before evaluation — for MMLU, the HF auxiliary training set of about 100,000 examples and 30M
tokens, which is not an i.i.d. split of MMLU; for GSM8K, MetaMathQA plus Orca-Math, about 600,000 examples and 200M
tokens; three epochs in both cases (§2.1).

Accuracy is regressed on pre-training compute with an indicator for release after November 2023 (Eq. 1):

```
A = α max(0, log C − c_e) + θ N + r + ε
```

where `A` is benchmark accuracy, `C` pre-training compute in FLOPs, `c_e` the compute at which the task emerges,
`r` random-chance accuracy, `N` the indicator for a post-November-2023 release, `θ` the average difference
attributed to being newer, and `ε` noise. Fits reach R² > 0.9, with standard errors clustered by model family.

- Before adjustment: `θ̂` is over 7 accuracy points on MMLU and over 19 on GSM8K, and statistically significant.
- After adjustment: `θ̂` is small and not statistically significant on either benchmark (§2.2, Fig. 1).
- Older models gain far more from the adjustment fine-tuning than newer ones (Fig. 2), which the authors read as
  newer models already having seen task-relevant data.
- Rankings move: average shift 7.7 ranks on GSM8K with a maximum of 21, and 4.8 ranks on MMLU with a maximum of 16
  (§4.2).

The authors state the consequence for this chapter directly (App. D): measures aimed at contamination, dynamic
benchmarks included, do not remove this confound, because no test data needs to leak for it to occur. A fresh
GSM1k-style set is still a multiple-step arithmetic word-problem set, and a model trained on more arithmetic
word-problem data will score higher on it for reasons that have nothing to do with the capability the number is
quoted for.

### 6.4 Agentic benchmarks carry the same problem

[[swe-bench-illusion]] probes SWE-Bench Verified with subtasks rather than end-to-end resolution. Given only the
repository name and the issue text, with no file tree and no code, ten OpenAI and Anthropic models name a file
touched by the gold patch in 60–76% of Verified instances, and in under 53% of 245 issues from seven popular
repositories outside SWE-Bench (§4.1.1–4.1.2, Fig. 8). The ordering Verified > Full > later issues from the same
repositories > outside repositories holds for every vendor tested (§4.1.3) and survives removing issues whose text
mentions a path or import (§4.2, Fig. 9). Function-reproduction 5-gram overlap maxima follow the same ordering: 34.9%
on Verified against 13.9% outside (§4.3).

The confounds the authors state matter for audit design: RefactorBench instructions average 14.6 tokens against 451.2
for Verified issues, so part of the external drop is task difference rather than memorization, and n-gram similarity
is noisy because correct solutions legitimately resemble the gold patch (§4.1.2, §5.2). Agent-specific evaluation is
developed in [[ch-51a]].

## §7 An audit protocol

The protocol below is what the rest of this chapter supports. It is stated as steps because the order matters: the
audit set and the margin are chosen before the headline number is known.

1. **Write the claim as a sentence with a capability in it.** "The model follows precise output constraints." If the
   sentence can only be written with a benchmark name in it, the benchmark is the claim and no audit is needed.
2. **Name the incidental property the headline benchmark holds fixed.** Item identity (fresh set), surface form
   (perturbation), world assumptions (counterfactual), rule set (IFBench-style), publication date (live), or
   distribution (leaderboard). Pick one.
3. **Build or select one counterpart that changes only that property.** Record how difficulty was matched, since an
   unmatched counterpart produces an uninterpretable difference ([[gsm1k]] §3.2).
4. **Fix the inference settings once and use them for both sides.** Prompt, shot count and shot source, temperature,
   maximum tokens, answer extraction. GSM1k's decision to draw both sets' shots from the GSM8K train split is the
   pattern (§4). Changing decoding between the two sides makes the comparison uninterpretable, and lowering
   temperature to reduce variance changes the estimand ([[adding-error-bars-evals]] §3.3).
5. **Score both sides on the same items where possible and compute a paired difference.** Report the difference, its
   standard error, its 95% interval, and the per-item score correlation ([[adding-error-bars-evals]] §4.2, Table 5).
6. **Check that the audit could have detected the effect you care about.** Compute the minimum detectable effect
   before running.
7. **Report the interval, not the verdict.** An audit that yields an interval containing zero has measured nothing
   about the gap; it has not shown the gap is absent.

**Formulas** ([[adding-error-bars-evals]] §4.1, §4.2, §5). With `s_{A,i}` the score of side A on item `i`,
`s_{A−B,i} = s_{A,i} − s_{B,i}`, and `s̄_{A−B}` the mean difference over `n` items:

```
SE_unpaired = sqrt( SE_A² + SE_B² )
SE_paired   = sqrt( Var(s_{A−B}) / n )
CI_95%      = s̄_{A−B} ± 1.96 × SE
δ           = (z_{α/2} + z_β) × sqrt( ω² / n )          (one sample per item, so ω² = Var(s_{A−B}))
```

`δ` is the minimum detectable effect: the smallest true difference the audit detects with probability `1 − β` at
false-positive rate `α`. For α = 0.05 and 80% power, `z_{α/2} = 1.96` and `z_β = 0.8416`.

**Worked numeric example — a 300-item audit.** The headline benchmark scores 80% and the audit counterpart 76%, so
`s̄ = 0.04`. The two sides disagree on 12% of items: `a = 8%` where the headline side is right alone and `b = 4%`
where the audit side is. For binary paired scores, `Var(s_{A−B}) = (a + b) − s̄² = 0.12 − 0.0016 = 0.1184`.

```
SE_paired = sqrt(0.1184 / 300) = 0.0199
CI_95%    = 0.040 ± 1.96 × 0.0199 = (0.001, 0.079)
z         = 0.040 / 0.0199 = 2.01
```

The unpaired comparison on the same numbers uses only the marginals:

```
SE_unpaired = sqrt( 0.80×0.20/300 + 0.76×0.24/300 ) = sqrt(0.000533 + 0.000608) = 0.0338
z           = 0.040 / 0.0338 = 1.18
```

The same data supports a conclusion when paired and does not when unpaired. Pairing is available whenever the two
sides share items, which is the normal case for a perturbation audit and never the case for a fresh parallel set.

Minimum detectable effect for this audit:

```
δ = (1.96 + 0.8416) × sqrt(0.1184 / 300) = 2.8016 × 0.0199 = 0.056
```

A 300-item audit resolves differences of about 5.6 points and nothing smaller. Inverting for δ = 0.03:

```
n = (1.96 + 0.8416)² × 0.1184 / 0.03² = 7.849 × 0.1184 / 0.0009 = 1,033 items
```

which is close to the 969 items the source's own example gives for ω² = 1/9 and the same α, β, δ, and is the basis
for its recommendation that a new evaluation hold at least 1,000 questions (§5). The second panel of
[figures/benchmark-audit-gap.html](figures/benchmark-audit-gap.html) computes all of these for any `n`, disagreement
rate, and gap, so the audit size can be chosen before any generation is run.

**What each audit licenses you to say.**

| Audit | Result | Claim it supports |
|---|---|---|
| Fresh parallel set | gap outside the interval | the headline number overstates performance on new items of this type by that much |
| Fresh parallel set | gap inside the interval | nothing about the gap; the audit was underpowered or the gap is smaller than δ |
| Perturbation, comprehension check passed | large gap | the procedure is tied to the default surface form, not to the abstract task |
| Date filter | drop after the cutoff | items before the cutoff may have been exposed; it does not separate exposure from look-alike training |
| Rule-set extension | drop on unseen rules | the capability is specific to the trained rule set |
| Same model, two submissions | rank difference | the leaderboard's variance, and the size of the selection advantage |
| Equal-fine-tuning adjustment | difference vanishes | the original difference is attributable to task-specific training, not to the models |

## Recipe

Evaluation settings of the audits in this chapter. These are eval-gate rows: they define measurements, not training
runs. A value that disagrees with the source card or excerpt is an error in this table.

| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| GSM1k audit, all 63 models | 0.1B–70B+ and API models | eval-gate | prompt; shot source | 5-shot, examples drawn from the GSM8K **train** split, identical for both sets | arXiv:2405.00332v4 §4 | verified 2026-09-15 | App. E, K, L: alternative prompt and shot-count ablations preserve the ordering |
| GSM1k audit | same | eval-gate | harness; max generated tokens | LM Evaluation Harness fork; 1000 (raised from the default 256) | §4 | verified 2026-09-15 | raised because the default truncated some chains of thought (§4) |
| GSM1k audit | same | eval-gate | decoding; serving | temperature 0 for open models, vLLM where compatible; API models at provider defaults via LiteLLM | §4 | verified 2026-09-15 | no ablation reported |
| GSM1k audit | same | eval-gate | answer extraction; API window | last numeric answer in the response; API queries 2024-04-16 to 2024-07-10 | §4 | verified 2026-09-15 | App. J: manual extraction for a subset does not change the findings |
| GSM1k set | n/a | eval-gate | audit set size; construction | 1,205 items from 2,108 written (1,419 after second solve, 1,375 after audit); no LLM involvement | §3, §3.1, §3.2 | verified 2026-09-15 | §3.2.1–3.2.3 distinguishability, solve-rate, and pre-2021-model checks |
| GSM-Symbolic audit, 25 models | 2B–27B plus GPT-4o-mini/4o/o1-mini/o1-preview | eval-gate | prompt; decoding | 8-shot chain of thought; greedy | arXiv:2410.05229v2 §3.2 | verified 2026-09-15 | §3.2: shot count did not change conclusions in preliminary experiments |
| GSM-Symbolic audit | same | eval-gate | templates; instances; sets | 100 templates × 50 instances = 5,000 items, reported as 50 sets of 100 | §3.2 | verified 2026-09-15 | §4.1: the 50-set spread is the reported quantity, not a single score |
| GSM-Symbolic audit | same | eval-gate | numeric ranges | kept close to the GSM8K test ranges | §3.1, App. A.6 | verified 2026-09-15 | App. A.6: models retain arithmetic accuracy over the expanded ranges |
| Counterfactual audit (GPT-4 gpt-4-0314, GPT-3.5, claude-v1.3, PaLM-2 text-bison-001) | not stated for PaLM-2, which is not the largest version | eval-gate | prompting; instances | with and without 0-shot chain of thought; 1,000 test and 200 CCC instances for arithmetic | arXiv:2307.02477v3 §4, Table 18 | verified 2026-09-15 | Table 19: shot-count and digit-count ablations |
| Embers-of-autoregression audit | gpt-3.5-turbo-0613, gpt-4-0613 | eval-gate | decoding; item sources | temperature 0.0; high-probability sentences from GlobalVoices, frequencies counted in C4 | arXiv:2309.13638v1 §4.2, §4.3 | verified 2026-09-15 | §5.5: matched-frequency pairs as the control condition |
| LiveCodeBench | 34 instruction-tuned and 18 base models | eval-gate | decoding; metric; date filter | nucleus sampling, temperature 0.2, top-p 0.95, vLLM; pass@1; only problems released after September 2023 for headline comparisons | arXiv:2403.07974v2 §4.3, §5.2 | verified 2026-09-15 | §5.1: DeepSeek and GPT-4-O month-by-month drops motivate the filter |
| LiveBench | 40 models, 0.5B–405B | eval-gate | prompting; scoring; size | zero-shot chain of thought with a parseable answer format; objective ground truth, no LLM judge; 1,000 questions over 18 tasks in 6 categories | arXiv:2406.19314v2 §2, §2.7, §3 | verified 2026-09-15 | §2: tasks targeted to a 30–70% success rate for top models |
| MathArena USAMO 2025 | 8 reasoning models | eval-gate | runs; scoring; graders | 4 runs per problem; 0–7 per problem, 42 per run; two of four expert graders per problem | arXiv:2503.21934v5 §2.1–2.3 | verified 2026-09-15 | §3.2: automated graders inflate by up to about 20× |
| MathArena USAMO 2025 | R1, QwQ | eval-gate | decoding | temperature 0.6, top-p 0.95, as recommended by the model authors | App. A.2 | verified 2026-09-15 | no ablation reported |
| Arena-overfitting experiment | 7B Cohere Command base | SFT | steps; batch; mixture shares | 1.3K steps; batch 128; 0% / 30% / 70% arena-mix against other-sft-mix | arXiv:2504.20879v2 §4.2 | verified 2026-09-15 | Fig. 10 and Table 9: ArenaHard win rate rises, MMLU does not |
| Arena-overfitting experiment | same | eval-gate | in-distribution metric; judge | win rate on the 500 English ArenaHard prompts against Llama-3.1-8B-Instruct; judge gpt-4o-2024-11-20 | §4.2 | verified 2026-09-15 | ArenaHard's reported 98.6% correlation with Arena rankings |
| Training-on-the-test-task adjustment | 56 base models, 70M–70B | eval-gate | adjustment data; epochs | MMLU: HF auxiliary set, ~100k examples / 30M tokens; GSM8K: MetaMathQA + Orca-Math, ~600k examples / 200M tokens; 3 epochs | arXiv:2407.07890v3 §2.1 | verified 2026-09-15 | App. A.3: hyperparameter robustness check |
| Any audit comparison | n/a | eval-gate | statistic; minimum size | paired difference with 95% interval and per-item correlation; ≥ 1,000 items for a 3-point effect at 80% power | arXiv:2411.00640v1 §4.2, §5 | verified 2026-09-15 | §5 worked example: n ≈ 969 at ω² = 1/9, δ = 0.03 |

**Starting point for a small general-purpose run.** With one headline capability to defend and no budget for a
human-written parallel set, the configuration supported by verified rows above is: one perturbation audit of 1,000
items or more, generated from templates over the benchmark's own items with numeric ranges held close to the
original ([[gsm-symbolic]] §3.1–3.2, built at 100 templates × 50 instances for models from 2B to 27B); the same
prompt, shot source, decoding, and answer extraction on both sides ([[gsm1k]] §4, used across 63 models from
0.1B to API scale); a paired difference with a 95% interval ([[adding-error-bars-evals]] §4.2); and the minimum
detectable effect computed before the run. Every setting here was fixed by authors evaluating released checkpoints,
not by anyone tuning a model against these audits, so treat the audits as measurements, never as development
targets — §6.2 is what happens when an evaluation distribution becomes a training distribution.

## Generalization lens

**(a) What increases breadth.** Training on varied instances of a rule rather than a fixed set of rules:
[[ifbench]] Table 6 moves IFBench from 25.2 to 44.6 by training GRPO on 29 constraints disjoint from the test
constraints, which no amount of IFEval-shaped data had produced. Auditing before shipping changes which checkpoint
is shipped, and the effect is measurable in the other direction: the equal-fine-tuning adjustment of
[[training-on-the-test-task]] moves GSM8K rankings by 7.7 ranks on average, so a selection rule that ignores it is
selecting partly for task-specific training.

**(b) What causes narrowing.** Optimizing against a fixed evaluation distribution: at 70% Arena data the ArenaHard
win rate rises 112.3% in relative terms and MMLU moves from 66.5% to 65.9% ([[leaderboard-illusion]] §4.2, Table 9).
Optimizing against a fixed constraint set: IF-RLVR raises IFEval to 92.2 and lowers AlpacaEval 2 from 33.5 to 21.3
([[ifbench]] Table 3). Learning a procedure that holds only in the default world: GPT-4 at 98.2% base-10 and 38.6%
base-9 two-digit addition with a 90.0% comprehension check ([[reasoning-or-reciting]] Table 18). Learning a
procedure attached to a surface pattern: GSM-NoOp costs Phi-3-medium-128k 53.1 points against its own symbolic
score, and eight in-context examples of the same question with full reasoning recover 0.8 points
([[gsm-symbolic]] Fig. 8b).

**(c) How to measure it at this stage.** The audit protocol of §7. Concretely, for each headline claim: one
counterpart that changes one incidental property; identical inference settings on both sides; a paired difference
with an interval; a minimum detectable effect computed in advance; and, when the claim is about a capability rather
than a benchmark, a second audit of a different type, because a fresh set does not detect training on the test task
([[training-on-the-test-task]] App. D) and a date filter does not detect look-alike training ([[livecodebench]] §5.1).
Known measurement errors: audits of a few hundred items resolve nothing below about 5 points (§7); relative drops
computed from figure bars may not match the source's own tables ([[gsm-symbolic]] Fig. 8a against Table 1); n-gram
similarity between generated and gold code is confounded by correct solutions resembling the gold patch
([[swe-bench-illusion]] §5.2); and model graders inflate scores when the output format changes
([[matharena]] Table 2).

## Common mistakes and how to detect them

| Mistake | Observable symptom | Check |
|---|---|---|
| Treating a drop on a fresh set as proof of contamination | the fresh-set gap is reported as a "leak" with no likelihood or overlap evidence | compute the per-character log-likelihood of the test set and correlate it with the gap; expect a weak rank correlation, 0.36 in [[gsm1k]] §5.4, and name the outliers |
| Using an in-distribution set as the audit | the audit correlates with the headline benchmark above 0.95 | check the audit's reported correlation with the headline; ArenaHard's 98.6% with Arena makes it a second reading, not an audit ([[leaderboard-illusion]] §4.2) |
| Comparing unpaired scores on a few hundred items | "the model improved by 3 points" with no interval | recompute with the paired formula; a 300-item audit has a minimum detectable effect near 5.6 points (§7) |
| Changing decoding or prompt between headline and audit | audit run uses a different temperature, shot count, or answer parser | require one settings file for both sides; GSM1k draws both sets' shots from the GSM8K train split ([[gsm1k]] §4) |
| Perturbing surface form and frequency at once | the perturbed items are also phrased more rarely | hold numeric ranges and phrasing frequency fixed ([[gsm-symbolic]] §3.1); check against a matched-frequency control pair ([[embers-of-autoregression]] Table 3) |
| Reading a low counterfactual score as a comprehension failure | no comprehension check was run | run the comprehension check separately; GPT-4 scores 90.0% on the base-9 CCC and 38.6% on the task ([[reasoning-or-reciting]] Table 18) |
| Treating a live benchmark as permanently clean | scores on a live set rise across months with no model change | filter by item release date against the model cutoff; the LeetCode-versus-AtCoder contrast in [[livecodebench]] Figs. 11–12 shows what a real exposure signal looks like |
| Reading a leaderboard rank as a model property | a rank changes with no checkpoint change | check the submission policy for private testing and retraction; two identical Aya-Vision-8B checkpoints scored 1052 and 1069 ([[leaderboard-illusion]] §3.3) |
| Comparing model families without adjusting for task-specific training | a newer family "beats" an older one at equal compute | fine-tune both on the same task data and re-measure; the difference fell from over 7 MMLU points to non-significant ([[training-on-the-test-task]] §2.2) |
| Scoring an agentic benchmark without a repository control | SWE-Bench Verified is quoted as general coding ability | run the file-path probe on repositories outside the benchmark; 60–76% inside against under 53% outside ([[swe-bench-illusion]] §4.1) |
| Letting a model grade a format it was not validated on | judge totals far above human totals | grade a sample by hand; automated graders inflated USAMO totals by up to about 20× ([[matharena]] Table 2) |

## Check your understanding

1. A model drops 6 points from a benchmark to its fresh parallel set, and a second model drops 0 points. Both were
   trained with the same decontamination pipeline. Give two mechanisms other than contamination that produce this
   difference, and name a measurement that distinguishes them.
2. GSM-NoOp performance does not recover when the in-context examples are symbolic instances of the same question
   with the full reasoning chain ([[gsm-symbolic]] Fig. 8b). Explain why that observation rules out "the model did
   not know the procedure" as the explanation, and what it leaves.
3. [[training-on-the-test-task]] shows that a fresh benchmark does not remove its confound. Construct the causal
   chain: which training decision produces a higher score on a set of items the model has never seen, and why does
   changing the items not interrupt it?
4. An audit of 400 items shows the headline set 3 points above the audit set, with the two disagreeing on 20% of
   items. Compute the paired standard error, the 95% interval, and the minimum detectable effect. State what you can
   and cannot conclude.
5. The counterfactual comprehension check is asked in a separate query from the task ([[reasoning-or-reciting]]
   §2.1). Explain what would be lost if it were asked in the same prompt as the task.
6. Two identical checkpoints scored 1052 and 1069 on Chatbot Arena. Explain why this is evidence about the
   leaderboard's variance and about selection, and why it is not evidence about either checkpoint.
7. Your team is choosing between a 300-item human-written fresh set and a 2,000-item template-perturbed set at the
   same cost. State the conditions under which each is the better audit, using the minimum-detectable-effect formula
   and the causes in §1.
8. [[leaderboard-illusion]] §4.2 reports a 112.3% relative win-rate gain with no MMLU gain. Explain why the MMLU
   result, and not the ArenaHard result, is the one that bears on general capability, and what a third measurement
   would have to look like to change that reading.

## Connections

- **Previous:** [[ch-47]] — Evaluation Harness and Suite Design for General Capability. That chapter fixes the
  harness, the splits, and the reproducibility record; this chapter asks whether the resulting numbers correspond to
  the capabilities they are named after.
- **Next:** [[ch-48]] — Contamination Detection and Its Effect on Reported Scores. That chapter takes the first cause
  in §1 and measures it directly, through n-gram overlap, LSH, clean-subset comparison, and black-box detection.
- [[ch-00]] — What General Capability Means and How It Is Measured. Supplies the capability coverage map and the
  development-versus-held-out split that this chapter's audits sit on top of.
- [[ch-49]] — Judge Models: Bias, Calibration, and Judge-Specific Overfitting. Develops the judge failure the
  MathArena automated grading exposes.
- [[ch-50]] — Slice Analysis, Forgetting Slices, and Failure Bucketing. Takes the aggregate number apart by slice
  once the aggregate has been audited.
- [[ch-51a]] — Evaluating Agent Generality and Reliability. Continues the agentic case of §6.4.
- [[ch-36]] — Lab: SFT Run with Masking Tests, a Forgetting Report, and a Held-Out Evaluation Split, and [[ch-46]] —
  Lab: DPO or RLVR Experiment with Negative-Signal Ablation and Held-Out Capability Retention. Both labs gate a
  training run on held-out capability; the paired statistics of §7 are the ones those gates use.

## Sources

- [[gsm1k]] — construction and quality checks of a human-written parallel set for GSM8K, the 63-model results table,
  the evaluation settings, and the log-likelihood-versus-gap analysis.
- [[gsm-symbolic]] — symbolic templates, the 50-set performance distribution, the GSM8K-original score's position in
  it, the GSM-P1/P2 difficulty ladder, GSM-NoOp, and the shot-source ablation.
- [[reasoning-or-reciting]] — counterfactual task variants, the comprehension check, the base-8/9/10/11/16 arithmetic
  table, and the shot-count recovery table.
- [[embers-of-autoregression]] — task, output, and input probability as confounds in any perturbation design, with
  the shift-cipher and sorting-frequency controls.
- [[ifbench]] — the IFEval-to-IFBench constraint generalization gap, the IF-RLVR training result, and the breadth
  costs measured alongside it.
- [[livebench]] — monthly-updated questions with objective ground truth, the task and category structure, and the
  category correlations.
- [[livecodebench]] — release-date filtering as a contamination probe, the DeepSeek and GPT-4-O month-by-month drops,
  and the HumanEval+-versus-LiveCodeBench overfitting split.
- [[matharena]] — same-day evaluation on USAMO 2025, proof grading against answer-only grading, and the automated-grader
  inflation.
- [[leaderboard-illusion]] — private testing and retraction, the two-identical-checkpoints result, the data-access
  asymmetries, and the arena-mix fine-tuning experiment with its MMLU control.
- [[training-on-the-test-task]] — the definition of training on the test task, the equal-fine-tuning adjustment, the
  regression result before and after adjustment, and the ranking shifts.
- [[swe-bench-illusion]] — file-path and function-reproduction probes showing SWE-Bench-specific advantage, with the
  confounds the authors state.
- [[adding-error-bars-evals]] — paired standard errors, confidence intervals, the power formula, and the argument
  against changing sampling temperature to reduce variance.
