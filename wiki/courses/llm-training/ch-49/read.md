<!-- chapter: ch-49
     track: eval
     kind: content
     title: Judge Models: Bias, Calibration, and Judge-Specific Overfitting
     deps: [ch-48]
     sources: [[judge-llm-bias]], [[chatbot-arena]], [[arena-hard-benchbuilder]], [[length-controlled-alpacaeval]], [[lmsys-style-control]], [[null-model-cheating-benchmarks]], [[leaderboard-illusion]], [[self-preference-recognition]], [[rewardbench]], [[rm-bench]], [[ppe-reward-model-eval]], [[self-taught-evaluators]], [[direct-judgement-preference]], [[generative-reward-models]], [[meta-rewarding-lm]]
     figures: figures/judge-bias.html
     revised: 2026-09 (generality revision)
-->

# Chapter 49 — Judge Models: Bias, Calibration, and Judge-Specific Overfitting

> **Core insight.** A judge score is an estimate produced by a model with measurable, reproducible biases, and the size of those biases is large enough to change a ranking. On the MT-Bench position-bias probe, GPT-4 gave the same verdict after swapping the two answers in 65.0% of cases, GPT-3.5 in 46.2%, and Claude-v1 in 23.8% ([[judge-llm-bias]] Table 2). On AlpacaEval, changing only the verbosity instruction moved one model's win rate from 22.9% to 64.3% ([[length-controlled-alpacaeval]] §4.1). A constant string that ignores the instruction reached an 86.5% length-controlled win rate on AlpacaEval 2.0 and a score of 9.55 on MT-Bench ([[null-model-cheating-benchmarks]] Table 2). Judge benchmark accuracy also does not automatically predict downstream policy quality: [[ppe-reward-model-eval]] measured a negative correlation between RewardBench scores of top reward models and post-RLHF Arena scores (§1, Fig. 4).
>
> **Guideline.** When a judge decides between two checkpoints, run every pair in both orders and report the position-consistency rate alongside the win rate, because consistency is judge-specific and ranged from 23.8% to 65.0% across three judges in [[judge-llm-bias]] Table 2. When responses differ in length or markdown, report a style-controlled score (the AlpacaEval GLM of [[length-controlled-alpacaeval]] Eq. 1 or the Bradley–Terry style features of [[lmsys-style-control]]), because style control reduced the normalized standard deviation across verbosity prompts from 25% to 10% and raised the Spearman correlation with Chatbot Arena from 0.94 to 0.98. When the judge family also supplied preference labels for training, add a second judge from a different family and report both numbers, because Arena-Hard-Auto's agreement with human rankings moved from 90.9% (GPT-4-Turbo judge) to 66.7% (Claude-3-Opus judge) and 91.5% (ensemble) on the same prompts ([[arena-hard-benchbuilder]] Table 4). When the decision is a release gate rather than an internal comparison, do not rely on a judge-scored leaderboard alone: use a held-out human-labeled set and a verifiable-correctness set, because both null-model exploits and selective private testing raise leaderboard scores without raising capability ([[null-model-cheating-benchmarks]], [[leaderboard-illusion]]).

---

## Why this chapter matters for a general-purpose model

Breadth of capability is measured on tasks with no reference answer: open-ended writing, multi-turn assistance, explanation quality, style-sensitive instruction following. For those tasks the measuring instrument is another language model. Every claim in this course of the form "the model kept its general ability after RL" rests, in part, on a judge number.

That creates two distinct risks for a general-purpose model.

1. **Measurement risk.** If the judge has a bias that correlates with the difference between two checkpoints — one is more verbose, one is from the judge's own family, one produces more markdown — the measured difference is partly the bias. This is a property of the instrument and is present even when no one is trying to game it.
2. **Optimization risk.** If a training signal is derived from the same judge family that scores the evaluation, the training loop can raise the judge score without raising capability. Chapter 42 covered this for training-time reward models; here the same mechanism appears at evaluation time, where it is harder to notice because there is no reward curve to inspect.

The chapter sits at the evaluation stage, after [[ch-48]] established that a benchmark score can be inflated by contamination. Contamination inflates a score because the model saw the test item. Judge-specific overfitting inflates a score because the model matched the grader's preferences. Both produce the same symptom — a number that does not transfer — and they need different detections.

---

## §1 What a judge score is, and how judge-scored benchmarks are built

**Definition.** An LLM judge is a language model prompted with a task input and one or two candidate responses, which outputs either a score for one response (single-answer grading) or a verdict naming the better of two (pairwise comparison). A judge-scored benchmark fixes a prompt set, a judge model, a judge prompt, a baseline or reference model, and an aggregation rule that turns individual verdicts into one number per candidate model.

**The problem it addresses.** Human evaluation of open-ended responses is slow. In [[chatbot-arena]] §6.3, expert labeling of one pairwise comparison took 3 to 5 minutes, with fact-checking against external resources. A judge produces the same verdict in seconds; Arena-Hard-Auto reports an evaluation cost of $20 per model over 500 prompts, against "very high" for Chatbot Arena at 10,000+ prompts per model ([[arena-hard-benchbuilder]] Table 1).

**Three reference points.** The following are the constructions this chapter uses. Each row is the configuration as published; a judge number is not comparable across rows.

| Benchmark | Prompts | Judge and baseline | Aggregation | Source locus |
|---|---|---|---|---|
| MT-Bench | 80 questions across 8 categories, 2 turns each | GPT-4; pairwise and single-answer (1–10) grading | Mean score, or pairwise agreement against human votes | [[judge-llm-bias]] §3.1, §4.1 |
| Chatbot Arena | Live, crowdsourced user prompts; over 240K votes from about 90K users at the time of the paper | Human crowd votes, not a judge | Bradley–Terry coefficients with active sampling and confidence intervals | [[chatbot-arena]] §1, §4 |
| Arena-Hard-Auto | 500 prompts curated by BenchBuilder from Chatbot Arena data | gpt-4-1106-preview as judge; gpt-4-0314 as baseline | Bradley–Terry fit over pairwise comparisons against the baseline, with 100 bootstrap rounds for 95% CIs | [[arena-hard-benchbuilder]] §5, §6.1 |
| AlpacaEval 2.0 | 805 instructions | gpt-4-1106-preview as annotator and as baseline model | Win rate, and length-controlled win rate from a GLM | [[length-controlled-alpacaeval]] §3; [[null-model-cheating-benchmarks]] Table 1 |

**Construction criteria.** [[arena-hard-benchbuilder]] §3 states two requirements for a benchmark used to compare models — separability and agreement with human preference — and defines three metrics for them:

- **Separability with Confidence**: the percentage of model pairs whose benchmark scores have non-overlapping confidence intervals, computed by bootstrapping.
- **Agreement with Confidence Interval**: for two benchmarks A and B and a model pair, score +1 if both separate the pair confidently and agree on the order, −1 if both separate it and disagree, 0 if either cannot separate it.
- **Pair Rank Brier Score**: the squared error of the benchmark's predicted pairwise ordering probability.

*Worked example.* On the 20-model set of [[arena-hard-benchbuilder]] Table 1 there are 20 × 19 / 2 = 190 model pairs. A separability of 87.4% means 0.874 × 190 ≈ 166 pairs had non-overlapping 95% intervals; MT-Bench's 22.6% means about 43 pairs. On the remaining 147 pairs, an MT-Bench difference is not distinguishable from bootstrap noise. This is the quantitative reason a checkpoint comparison on MT-Bench alone is weak: two checkpoints from the same run are a near-tie pair, and near-tie pairs are the ones MT-Bench does not separate.

Reported values on that 20-model set (Table 1): Arena-Hard-Auto separability 87.4%, MT-Bench 22.6%, AlpacaEval 2.0 LC 83.2%, Chatbot Arena 85.8%; Brier score 0.069 / 0.09 / 0.11 / n.a. **Result (single study).**

**Conditions and limits.** These metrics are computed against a Chatbot Arena reference ranking from 2024-04-13, over 20 models that were on both leaderboards. They say how well a benchmark reproduces a human *ranking of released models*; they do not say whether it reproduces a ranking of checkpoints from one training run, which is the comparison an internal evaluation makes.

---

## §2 The measured bias inventory

Four biases have published measurements with an explicit measurement procedure. Read each as "a probe, a number, and a correction", not as a general property of judges.

### 2.1 Position bias

**Definition.** A judge's verdict depends on which candidate is presented first.

**Measurement procedure** ([[judge-llm-bias]] §3.3): construct two similar answers for each first-turn MT-Bench question by calling GPT-3.5 twice at temperature 0.7, then judge each pair in both orders. *Consistency* is the percentage of cases where the verdict is unchanged under the swap.

| Judge | Prompt | Consistency | Biased toward first | Biased toward second | Error |
|---|---|---|---|---|---|
| Claude-v1 | default | 23.8% | 75.0% | 0.0% | 1.2% |
| Claude-v1 | rename | 56.2% | 11.2% | 28.7% | 3.8% |
| GPT-3.5 | default | 46.2% | 50.0% | 1.2% | 2.5% |
| GPT-3.5 | rename | 51.2% | 38.8% | 6.2% | 3.8% |
| GPT-4 | default | 65.0% | 30.0% | 5.0% | 0.0% |
| GPT-4 | rename | 66.2% | 28.7% | 5.0% | 0.0% |

([[judge-llm-bias]] Table 2. The "rename" prompt renames the assistants, which separates position from name; Claude-v1's jump from 23.8% to 56.2% is evidence of a name preference for "Assistant A".)

**Correction.** The paper's conservative rule: call the judge twice with the order swapped, declare a win only if the same answer is preferred both times, and count an inconsistent pair as a tie (§3.4). An alternative is random assignment of positions, which is unbiased in expectation across many pairs but noisy per pair.

*Worked example.* Take 200 evaluation pairs and the GPT-4 default row. Under the conservative rule, 0.650 × 200 = 130 pairs yield a decision and 70 become ties. With GPT-3.5 as judge, 0.462 × 200 ≈ 92 pairs yield a decision and 108 become ties. The same evaluation therefore has roughly 40% fewer decided pairs under the weaker judge, which widens every confidence interval computed from it. Swapping is not free: it doubles the judge calls and it shrinks the effective sample size in a judge-dependent way, so the consistency rate has to be reported next to the win rate for the win rate to be interpretable.

**A second measurement of the same effect.** [[self-preference-recognition]] §2.5 measured, on summarization pairs, how often an evaluator reverses its pairwise preference when the option order is reversed: GPT-4 25%, GPT-3.5 58%, Llama 2 89%, averaged across tasks and datasets (Fig. 5). Their correction is to average logit-based confidence scores across the two orderings rather than to discard inconsistent pairs. **Replicated**: two independent studies, different data, both find order reversal at tens of percent, with GPT-4 the most consistent of the models tested.

**Reported mitigation that is not a full fix.** A few-shot judge prompt with three judgment examples raised GPT-4's consistency from 65.0% to 77.5% ([[judge-llm-bias]] Table 12), at roughly 4× the prompt cost, and the authors state that higher consistency does not imply higher accuracy and that few-shot examples may introduce new biases (§3.4).

### 2.2 Verbosity bias

**Measurement procedure** ([[judge-llm-bias]] §3.3): take 23 MT-Bench answers containing a numbered list; ask GPT-4 to rephrase the list without adding information and prepend the rephrased items, so a 5-item answer becomes a 10-item answer with the same content. The attack succeeds if the judge prefers the longer version.

| Judge | Claude-v1 | GPT-3.5 | GPT-4 |
|---|---|---|---|
| Failure rate under "repetitive list" | 91.3% | 91.3% | 8.7% |

([[judge-llm-bias]] Table 3.) The same judges return a tie for two identical answers, so the failure is specific to added redundant length, not to a broken comparison procedure.

**Conditions and limits.** N = 23 answers and one attack shape. The number quantifies susceptibility to this manipulation, not a general length coefficient. The general length coefficient is what §3 estimates by regression.

### 2.3 Self-preference

**The weak version, and its caveat.** [[judge-llm-bias]] §3.3 compared each judge's win rates against human win rates on the same pairs (Fig. 3b) and found GPT-4 gave itself a 10% higher win rate and Claude-v1 gave itself 25% higher. The authors then write that "due to limited data and small differences, our study cannot determine whether the models exhibit a self-enhancement bias", because a controlled study would require rephrasing a response into another model's style without changing its quality. Treat the two numbers as observations, not as an established effect size. **Open question** as posed in that paper.

**The controlled version.** [[self-preference-recognition]] separates two properties: *self-preference* (the evaluator scores its own output higher than others rate it) and *self-recognition* (the evaluator can tell which output it wrote). Findings on XSUM and CNN/DailyMail summarization with Llama-2-7b-chat, GPT-3.5, and GPT-4:

1. All three evaluators exceed 50% accuracy at distinguishing their own summaries from other sources without fine-tuning; GPT-4 reaches 73.5% against two other LLMs and humans (§1, finding 2).
2. Fine-tuning on 500 examples raises GPT-3.5 and Llama 2 above 90% self-recognition accuracy (finding 3).
3. Fine-tuning models to raise or lower self-recognition moves self-preference along a linear trend (Fig. 1, finding 4).
4. Models fine-tuned on unrelated properties — response length, vowel count, Flesch–Kincaid readability, always answering "1", answering randomly — fall on the same linear trend, which is the authors' control against a confound introduced by fine-tuning (§3.4).

**Implication for a general-purpose model.** The operational rule does not depend on resolving the causal question. If the judge family and the candidate family are the same, a component of the score is attributable to something other than response quality, and the size of that component is not known for your setting. The measurement fix is a second judge from another family, and the quantity to report is the disagreement between them — not a re-attributed score. A difference between two judges can come from different rubrics, different calibration, or genuine disagreement about quality, so it bounds the uncertainty rather than isolating a leakage term.

### 2.4 Grading of math and reasoning answers

**Measurement procedure** ([[judge-llm-bias]] §3.3–§3.4): 10 math questions, answers from LLaMA-13B and Vicuna-13B, each pair judged in both orders, giving 20 judgments. A failure is GPT-4 declaring an incorrect answer correct.

| Prompt | Default | Chain-of-thought | Reference-guided |
|---|---|---|---|
| Judge failures out of 20 | 14 | 6 | 3 |

([[judge-llm-bias]] Table 4; the text reports the same change as "from 70% to 15%".) Two details matter for reuse. First, the reference answer is **the judge's own answer, generated independently before it sees the candidates** (§3.4), not a gold solution — the procedure works by keeping the judge from being anchored by the candidates. Second, chain-of-thought alone does not remove the effect: the authors report that the judge often reproduces the same mistake as the given answers inside its own reasoning (§3.4).

**Conditions and limits.** N = 10 questions. The correction applies to tasks where the judge can produce an answer on its own; for open-ended writing there is nothing to generate as a reference.

### 2.5 The agreement number, stated precisely

The often-quoted "GPT-4 agrees with humans as much as humans agree with each other" is setup-dependent. [[judge-llm-bias]] Table 5 (MT-Bench, first turn) reports two setups: S1 includes non-tie, tie, and position-inconsistent votes and counts inconsistent votes as ties (random baseline 33%); S2 includes only non-tie votes (random baseline 50%).

| Pair | S1 (R = 33%) | S2 (R = 50%) |
|---|---|---|
| GPT-4 pairwise vs human | 66% | 85% |
| GPT-4 single-answer vs human | 60% | 85% |
| human vs human | 63% | 81% |

Second turn: GPT-4 pairwise vs human 66% / 85%, human vs human 67% / 82%. On Chatbot Arena data (Table 6), GPT-4 vs human is 64% (S1) and 87% (S2). Agreement also rises with the size of the gap between the two models: from 70% to nearly 100% as the win-rate difference grows (Fig. 2).

That last point is the one to carry into checkpoint comparison. Judge–human agreement is highest exactly where the answer is obvious and lowest where two candidates are close — which is the regime of comparing two checkpoints from one run.

The companion figure **[figures/judge-bias.html](figures/judge-bias.html)** lets you select one of the measured position-consistency rates above, set the number of evaluation pairs, and see how many pairs survive the conservative swap rule; the second panel evaluates the AlpacaEval length-controlled GLM of §3.1 so the raw and length-controlled predictions can be read against each other. Both panels use the numbers quoted in this section.

---

## §3 Style control: removing a known confounder by regression

### 3.1 Length-controlled AlpacaEval

**The problem, measured.** Prompting a model to "Answer with as much detail as possible" versus "Be as concise as possible…" moved gpt4_1106_preview's raw AlpacaEval win rate from 22.9% to 64.3% ([[length-controlled-alpacaeval]] §4.1). The metric therefore responds to an instruction that changes no capability.

**Mechanism.** Fit a logistic regression that predicts the annotator's preference from three terms — model identity, length difference, and instruction difficulty — then evaluate it with the length term set to zero. The fitted model (Eq. 1):

```
q(y = 1 | z_m, z_b, m, b, x) =
    logistic( θ_m − θ_b  +  φ_{m,b} · tanh( (len(z_m) − len(z_b)) / std(len(z_m) − len(z_b)) )  +  (ψ_m − ψ_b) · γ_x )
```

- `z_m`, `z_b`: the outputs of the evaluated model `m` and the baseline `b` on instruction `x`.
- `θ_m`, `θ_b`: per-model strength coefficients.
- `φ_{m,b}`: the model pair's sensitivity to length difference.
- `len(·)`: output length; the difference is standardized to unit variance, then passed through `tanh` so that length differences have diminishing effect on the log odds.
- `ψ_m`, `ψ_b`: per-model sensitivity to instruction difficulty; `γ_x`: the difficulty of instruction `x`.
- `logistic(u) = 1 / (1 + e^{−u})`.

The length-controlled win rate (Eq. 2) drops the length term:

```
winrate_LC(m, b) = 100 · E_x [ logistic( θ_m − θ_b + (ψ_m − ψ_b) γ_x ) ]
```

The featurization satisfies two identities the raw win rate has: `q = 0.5` when a model is compared against itself, and swapping `m` and `b` maps `q` to `1 − q`, because `tanh` is odd and the other terms flip sign (§3). With `M` models and `N` instructions the GLM has `3M + N` parameters fitted from `M·N` examples; AlpacaEval has `M > 128` and `N = 805`.

*Worked example.* Suppose for one instruction `θ_m − θ_b = 0.30`, `φ_{m,b} = 0.80`, the standardized length difference is `1.0` so `tanh(1.0) = 0.762`, and the instruction term is `0`. The raw prediction is `logistic(0.30 + 0.80 × 0.762) = logistic(0.910) = 0.713`. Setting the length term to zero gives `logistic(0.30) = 0.574`. On this instruction, 14 points of the 71% win rate are attributed to the model being longer than the baseline. Repeat over the 805 instructions and average to get the reported LC win rate.

**Evidence.** Length control lowered the normalized standard deviation of a model's win rate across the three verbosity prompts from 25% to 10%; gpt4_1106_preview's spread narrowed from 22.9–64.3% to 41.9–51.6%; the Spearman correlation with Chatbot Arena rose from 0.94 to 0.98, computed over the 38 models present on both (§4.1–§4.2, Table 1). Alternative controls scored lower: length-normalized 0.96 correlation / 15% gameability, length-balanced 0.95 / 15% but with a 40.8-point adversarial win-rate gain against 8.5 for the GLM (Table 1). **Result (single study).**

### 3.2 Style control in Chatbot Arena and in Arena-Hard-Auto

[[lmsys-style-control]] applies the same idea to human votes. The Arena score is a Bradley–Terry logistic regression of the battle outcome `Y_i` on a model-indicator vector `X_i`; style control adds style features `Z_i` with their own coefficients:

```
β̂, γ̂ = argmin_{β, γ}  (1/n) Σ_i BCELoss( sigmoid( X_iᵀ β + Z_iᵀ γ ), Y_i )
```

with each style feature defined as `normalize( (feature_A − feature_B) / (feature_A + feature_B) )`. The four features are answer token length, markdown header count, markdown bold count, and markdown list count. Fitted coefficients when controlling for both length and markdown: length 0.249, markdown list 0.031, header 0.024, bold 0.019; controlling length alone gives 0.267. The authors state that length is the dominant style factor and the markdown effects are second order. Reported ranking effects: GPT-4o-mini and Grok-2-mini fall below most frontier models, Claude 3.5 Sonnet, Opus, and Llama-3.1-405B rise, and on the Hard Prompt subset Claude 3.5 Sonnet ties for first.

[[arena-hard-benchbuilder]] §6.5 adds the same four style features to Arena-Hard-Auto's Bradley–Terry fit and compares against a style-controlled Arena ranking: confidence agreement 98.6% with style control against 94.4% without, Spearman 98.6% against 94.9%, Kendall 93.7% against 85.3% (Table 3). They also instructed GPT-3.5-Turbo, Llama-3.1-70B-Instruct, and Gemini-1.5-Flash to increase verbosity and markdown usage: the raw score rose, and style control removed the advantage (§6.5, App. Table 12).

**Conditions and limits.** Style control removes the variance explained by the features you name. It does not remove style effects you did not model, and it does not bound adversarial manipulation — §4 shows a null model beating the length-controlled metric.

---

## §4 Judge-specific overfitting

"Judge-specific overfitting" means an increase in a judge-scored metric that does not correspond to an increase in capability on tasks the judge was standing in for. Three published demonstrations bound how large it can be.

### 4.1 Null models

[[null-model-cheating-benchmarks]] submits a "null model" that returns one constant string for every instruction, with no relation to the input. Results with gpt-4-1106-preview as the annotator on all three benchmarks (Table 2):

| Target | AlpacaEval 2.0 LC win rate | AlpacaEval 2.0 raw | Arena-Hard-Auto | MT-Bench score |
|---|---|---|---|---|
| Verified SOTA model (before 2024-10-01) | 57.5 | 51.3 | 82.6 | 8.96 |
| Structured cheating response | 76.8 | 64.2 | 67.2 | 7.75 |
| Structured response + random-search prefix | 86.5 | 84.0 | 83.0 | 9.55 |

Two mechanisms are worth separating. The *structured* response exploits the annotator's template: it is written so that the judge parses it as the end of the evaluation instructions plus a new empty comparison, at which point the annotator returns the first identifier (§3, Fig. 2). That alone gives 76.8% LC. The *random-search prefix* is then optimized against a public instruction set — the benchmark's own 805 AlpacaEval instructions are treated as private — and transfers to all three benchmarks (§3, Table 2). Average token count of the winning response is 205, so the length control of §3 is active and does not prevent the result.

**Implication.** Length control and style control fix a known confounder under non-adversarial conditions. They are not a defense against optimization directed at the judge. A leaderboard number from a judge that any submitter can query is a number that can be optimized directly.

### 4.2 Selective testing on a human leaderboard

[[leaderboard-illusion]] audits Chatbot Arena itself. Reported findings: 27 private LLM variants tested by one provider before the Llama-4 release; Google and OpenAI estimated to have received 19.2% and 20.4% of all arena data respectively, against 29.7% for 83 open-weight models combined; 205 of 243 public models silently deprecated against 47 officially listed, with 64% of the silently deprecated models open-weight or open-source; and, in their experiments, access to additional arena-distribution data produced relative gains of up to 112% on Arena-Hard (Abstract, §1). The authors' reading is that selective disclosure of the best of several private variants violates the unbiased-sampling assumption of the Bradley–Terry fit, and that these dynamics produce overfitting to arena-specific behaviour rather than general quality.

This is the same failure as §4.1 without any adversarial string: choosing which of `k` measured variants to disclose is a maximum over noise, and the maximum is biased upward.

### 4.3 Why the judge choice itself is part of the result

[[arena-hard-benchbuilder]] Table 4 runs Arena-Hard-Auto's 500 prompts through four judges and compares each resulting ranking against the same human reference:

| Judge | Confidence agreement | Separability | Spearman | Brier |
|---|---|---|---|---|
| gpt-4-1106-preview | 90.9% | 87.4% | 93.2% | 0.069 |
| Claude-3-Opus | 66.7% | 83.68% | 77.0% | 0.170 |
| gemini-1.5-pro-0514 | 84.8% | 82.11% | 95.2% | 0.064 |
| llama-3-70b-instruct | 65.6% | 81.6% | 70.5% | 0.196 |
| Ensemble (GPT-4-Turbo + Gemini-1.5-Pro) | 91.5% | 89.5% | 96.5% | 0.065 |

The authors also checked their default judge for self-bias: GPT models receive slightly higher average rankings than human preference gives them and Claude models rank lower, and the two-judge ensemble reduces this (§6.6, App. Table 10). **Result (single study).** The practical consequence is that "score on Arena-Hard-Auto" is not a property of the model; it is a property of the model, the judge, and the style-control setting together.

### 4.4 What this chapter does not cover

The measurements above are for single-response and pairwise text judging. Judging an agent trajectory adds a different failure surface: the outcome predicate, the environment state, and the tool-call sequence are all separate objects to grade. That is the subject of [[ch-51a]]. Long-context judging, where the judge itself must read a long input before grading, is not measured by any source in this chapter's set.

---

## §5 Does a judge or reward model generalize?

A judge is useful for a general-purpose model only if its accuracy holds outside the distribution it was validated on, and if its score predicts something downstream. Three benchmarks measure different parts of that.

### 5.1 RewardBench: coverage across categories

[[rewardbench]] is a set of prompt–chosen–rejected trios in five sections: Chat, Chat Hard, Safety, Reasoning, and Prior Sets (test splits of earlier preference datasets). Scoring is accuracy — the model is correct when it assigns the chosen response a higher score — with per-prompt weighted averaging inside sections, an unweighted average inside Prior Sets, and Prior Sets down-weighted to 0.5 of the other sections in the final number (§4.2). Random performance is 50%. The authors report that some subsets approach 100% accuracy while Chat Hard and Reasoning subsets have low ceilings and high variance, with Reasoning spanning 35% (below random) to 97% across models (§5).

The section structure is what makes it a generality probe: a judge that is accurate on Chat and below random on Reasoning has a narrow competence, and an aggregate number hides that.

### 5.2 RM-Bench: separating substance from style

[[rm-bench]] builds, for each prompt, a chosen and a rejected response at three style levels: `y∅` short and plain, `y^L` detailed and plain, `y^{L,M}` detailed with markdown. Comparing every chosen style against every rejected style gives a 3×3 Style–Substance matrix, from which three metrics are derived (§3.3):

- **Easy accuracy** = mean of the lower triangle (the chosen response also has the more favourable style).
- **Normal accuracy** = mean of the diagonal (both responses share a style).
- **Hard accuracy** = mean of the upper triangle (the *rejected* response has the more favourable style).

*Worked example*, using the published matrix for `sfairXC/FsfairX-LLaMA3-RM-v0.1` in the chat domain (Fig. 2), rows = chosen style, columns = rejected style:

```
              y_r      y_r^L    y_r^{L,M}
y_c         83.61%    3.83%     2.19%
y_c^L       99.45%   66.12%    49.73%
y_c^{L,M}  100.00%   80.33%    66.67%
```

Easy = (99.45 + 100.00 + 80.33) / 3 = 93.26%. Normal = (83.61 + 66.12 + 66.67) / 3 = 72.13%. Hard = (3.83 + 2.19 + 49.73) / 3 = 18.58%. The same reward model is near-perfect when style agrees with substance and far below random when they conflict.

Across nearly 40 reward models, the best average accuracy is 70.1% (Skywork-Reward-Llama-3.1-8B) with 46.6% hard accuracy — below the 50% random baseline (Abstract, Table 3). **Result (single study).**

On downstream correlation, [[rm-bench]] §5 uses four Tülu-v2.5 reward models trained on different 60k preference sets with matched PPO hyperparameters, and correlates reward-model scores with policy performance on GSM8k, Big Bench Hard, HumanEval+, MBPP+, ToxiGen, and XSTest: Pearson r = 0.55 (p = 0.07), against r = 0.21 (p = 0.51) reported for RewardBench in the same comparison. Both are small-sample correlations; the paper's own limitations section says the downstream correlation is limited.

### 5.3 PPE: correlating reward-model metrics with post-RLHF outcomes

[[ppe-reward-model-eval]] is the only source here that runs the end-to-end check. Its construction: 16,038 human preference pairs from Chatbot Arena plus verifiable-correctness preference sets, 12 metrics over 12 domains (Abstract, §3). It then trains a policy per reward model — 8,000 DPO rows built from responses of Llama-3.1-8B-Instruct — deploys the resulting models to Chatbot Arena, and collects 12,190 human votes over six days, about 2,032 battles per model (§6.1–§6.2, Table 3).

Findings (§7):
1. Reward-model **accuracy on the human preference set is the best predictor** of the post-DPO Arena score. Row-wise Pearson correlation, confidence agreement, and separability have some predictive power but less.
2. **Spearman and Kendall rank correlations have nearly zero correlation** with the final Arena score. The authors' interpretation: accuracy is measured per preference pair, while rank correlations aggregate over models and lose granularity.
3. Aggregating category scores at a **low quantile** predicts better than the mean: accuracy peaks at 0.80 Pearson correlation with downstream Arena score at low-quantile aggregation (Fig. 5). Their reading is that a reward model must be robust on every input distribution, because any weak domain can be exploited during training.
4. Among the verifiable-correctness domains, math is the most predictive single domain, and ROC AUC correlates better than accuracy there.
5. For top reward models, they report a **negative correlation** between RewardBench score and downstream RLHF performance (§1, Fig. 4).

**Conditions and limits.** Nine reward models, DPO rather than PPO, one policy size, one six-day battle window. The paper states both the leakage risk and the DPO-versus-PPO caveat (§8).

**Implication for a general-purpose model.** The order of trust for a judge or reward model is: measured downstream effect on the policy > per-pair accuracy on a fresh human-labeled set > style-robust benchmark accuracy > aggregate leaderboard rank correlation. Reporting only the last is the weakest available evidence, and [[ppe-reward-model-eval]] found it to be the least predictive.

---

## §6 Training a judge, and what that changes

Owning the judge fixes two problems: the instrument stops changing when an API version changes, and the evaluation-time judge can be made different from the training-time reward model. Four constructions, with their verified results.

### 6.1 Self-Taught Evaluators — iterative fine-tuning with no human preference labels

Pipeline ([[self-taught-evaluators]] §3–§4.1):
1. Take unlabeled instructions. The run uses WildChat instructions categorized by Mixtral 22Bx8 Instruct; 20,582 "reasoning" examples are kept.
2. For instruction `x`, generate a good response `y_w`. Then write a **modified instruction** `x′` that is "highly relevant but not semantically identical" to `x`, and a high-quality response `y_l` to `x′`. Since `y_l` answers a different question, the pair `y_w ≻ y_l` has a construction label with no human annotation.
3. Sample `N = 15` judgments per example from the current model at temperature 0.7, top-p 0.9; discard judgments whose verdict disagrees with the construction label; keep one correct judgment at random; drop the example if none is correct; balance the A-better and B-better labels.
4. Fine-tune from the seed model with negative log-likelihood **on the judgment tokens only**. Each iteration restarts from Llama-3-70B-Instruct rather than continuing the previous checkpoint.

Results (Table 1): RewardBench overall 75.4 (seed) → 83.9, 86.0, 87.5, 87.7, 88.3 over five iterations; 88.7 with a 32-sample majority vote. GPT-4-0125 scores 84.3 and is first exceeded at **iteration 2** (86.0). The same seed trained on human-labeled HelpSteer2 reaches 85.6. Per-category, seed → iteration 5: Chat 97.6 → 96.6, Chat Hard 58.9 → 84.2, Safety 69.2 → 91.5, Reasoning 78.5 → 81.0.

Two ablation results matter for this chapter. First, the construction matters: the "rewrite the answer to be worse" prompt gives 80.7 against 83.8 for the modified-instruction construction (§6.2). Second, **the evaluation is order-sensitive even at the end of training**: at iteration 5, RewardBench accuracy is 85.5 with the winner always first and 91.1 with the loser always first, averaging 88.3 (App. A.3, Table 9). A judge trained specifically as a judge still has a 5.6-point position effect, which is why §2.1's swap protocol is not optional after training one.

**Conditions and limits stated by the authors**: 70B models only, pairwise judgments only, the seed must already produce reasonable evaluations, compute not studied.

### 6.2 SFR-Judge — DPO on judgment pairs

[[direct-judgement-preference]] trains judges at 8B, 12B, and 70B on 680K judgement preference pairs of three types: CoT critique 70%, verdict-only 15%, response deduction 15% (§4.1). The loss combines a length-normalized SFT term on the chosen judgement with the DPO term (§3.4):

```
L = − log M_s(y_w | x) / (|y_w| + |x|)
    − log σ( β log[ M_s(y_w|x) / M_ref(y_w|x) ] − β log[ M_s(y_l|x) / M_ref(y_l|x) ] )
```

- `M_s`: the judge being trained; `M_ref`: a frozen copy of the same instruction-tuned initialization.
- `x = (p, i, r)`: protocol (task description and rubric), task input, and one or two responses.
- `y = {c, j}`: a critique and a verdict; `y_w` has the ground-truth verdict, `y_l` does not.
- `|·|`: token length; `σ`: the logistic function; `β`: the DPO temperature (not reported for the main runs).

Results: pairwise average over 7 benchmarks 84.25 (70B), 81.49 (12B), 80.91 (8B) against GPT-4o 76.78 and Self-taught-evaluator-Llama-3.1-70B 82.26 (Table 1); RewardBench 92.7 / 90.3 / 88.7 against GPT-4o-2024-08-06 86.7 (Table 7). Removing the critique at inference lowers the 8B single-rating Pearson from 0.68 to 0.58 and the pairwise average from 80.97 to 80.05 (App. E.6).

**Measurement caveat to carry.** For the six non-RewardBench pairwise benchmarks, the paper runs each twice with the response order swapped and reports **the better of the two runs** (§4.3). Order-averaged accuracy is not reported. When comparing to a number from another paper, check which convention produced it.

### 6.3 Generative reward models — where the training signal comes from

[[generative-reward-models]] trains Llama-3.1-8B-Instruct judges on UltraFeedback (61k pairs) and UltraInteract and evaluates in-distribution and on RewardBench. Its result on generalization is the one to keep: explicit Bradley–Terry reward models reach about 94% in-distribution on UltraInteract but fall **below random on RewardBench Reasoning**, while the judge trained with DPO on its own correct-versus-incorrect reasoning chains (STaR-DPO) reaches 87.2% there (§5.2). On UltraFeedback: in-distribution, Bradley–Terry, PairRM, and GenRM are around 73–74% and STaR-DPO 73.9%; on RewardBench, STaR-DPO 81.9% against GenRM 78.9% (§5.1). Majority voting over 32 samples adds 1.6 to 4.9 points depending on the setting (§5.4).

The paper does not test the trained judge as a reward inside PPO or online preference optimization, and lists reward hacking of generative RMs as future work (§7). So this is evidence about judge accuracy, not about judge robustness under optimization pressure.

### 6.4 Judges drift when they are trained in a loop

[[meta-rewarding-lm]] trains one Llama-3-8B-Instruct as actor, judge, and meta-judge for four DPO iterations. The headline is AlpacaEval 2 LC 22.92% → 39.44% (Table 1) and Arena-Hard 20.6% → 29.1% (Table 2). The measurements this chapter needs are the failure diagnostics:

- The meta-judge's preference for the higher-scored judgment rose from 63.04% at iteration 1 to **97.68%** at iteration 2 (Table 5).
- Positional bias over all judgment pairs rose from 43.92% to 68.11% (Table 5); the authors report that positional bias limited further improvement at iteration 3 (§5).
- The mean judge score rose from about 4.1 to above 4.7 on the 5-point rubric after two iterations of judge training (Fig. 5) — the score distribution compresses toward the ceiling.
- Judge agreement with humans, measured by Spearman correlation on 190 held-out Open Assistant samples: seed 0.315, iteration 2 0.382, iteration 4 0.326 (Table 7). The gain is not sustained.

Length control in this pipeline is a **pair-selection rule, not a rubric term**: the chosen response is the shortest response whose score falls in the top tier `[(1−ρ)S_max + ρS_min, S_max]`, and the rejected is the longest in the bottom tier (§2.1). With `ρ = 0` at iteration 4 the responses reach 2212 characters against 2003 at `ρ = 0.4` (Table 4). The rubric itself contains no length term (App. A.1).

**Implication.** A judge that is retrained inside the loop it scores needs an external anchor with a fixed, human-labeled set, and the anchor statistic — agreement or Spearman with humans — has to be recomputed per judge version. In this study the anchor statistic peaked two iterations before the training stopped.

---

## Negative samples and negative feedback

Judge training uses negative examples in three of the four senses defined by the course standard (see [[ch-43a]] for the derivations). Naming which sense a method uses prevents transferring a control from one to another.

**Where negatives come from.** In [[self-taught-evaluators]], a negative is a sampled judgment whose verdict disagrees with the construction label, and the rejected *response* `y_l` is a good answer to a modified instruction. In [[direct-judgement-preference]], a negative judgement is a teacher critique whose verdict disagrees with the dataset annotation. In [[generative-reward-models]], a negative is a reasoning chain that ends in the wrong verdict. In [[meta-rewarding-lm]], the rejected judgment is the one with the lowest Elo in the meta-judge's battle matrix. **None of these papers reports a false-negative rate for its construction**; [[self-taught-evaluators]] §3 describes `y_l` only as "likely of lower quality".

**What current practice does with them.**

| Method | Sense of "negative" | Handling |
|---|---|---|
| [[self-taught-evaluators]] | (1) negative marginal value | Incorrect judgments are **discarded**; training is NLL on one kept correct judgment per example (§3.4, §4.1) |
| [[direct-judgement-preference]] | (4) negative as gradient | Rejected judgement enters the DPO term; an SFT term on the chosen judgement anchors it (§3.4) |
| [[generative-reward-models]] | (4) negative as gradient | STaR-DPO uses wrong-verdict chains as the rejected output (§4, Eq. 9) |
| [[meta-rewarding-lm]] | (4) negative as gradient | Lowest-Elo judgment is the rejected term of judge DPO (§2.2) |

**Mechanism.** For a softmax over the verdict token, the gradient of the log-probability of the correct verdict `y` with respect to logit `z_j` is `∂ log p_y / ∂ z_j = 1[j = y] − p_j`. A DPO-style term additionally pushes down `log π(y_l | x)`, and the mass removed from `y_l` moves to whatever the model currently ranks highest — which, for a judge, is often the opposite verdict rather than a better-reasoned version of the same verdict. This is why the verdict-only pair type exists: [[direct-judgement-preference]] §3.2 states that in a long CoT critique only a few tokens decide the verdict, so critique-length targets dilute the signal on exactly those tokens, and adds `D_Std` pairs with the critique removed.

**Evidence with numbers.**
- Discarding versus gradient: [[generative-reward-models]] §5.1 reports STaR-SFT, which discards wrong-verdict chains, at 67.4% on UltraFeedback with no gain over the base model, and STaR-DPO, which uses them as the rejected term, at 73.9% in-distribution and 81.9% on RewardBench. The paper does **not** run an ablation that separates the use of negatives from the change of loss, so the split between the two causes is not measured.
- Negative difficulty: [[direct-judgement-preference]] App. E.8 Table 11 compares negatives generated by a 70B teacher against an 8B teacher at 8B scale: pairwise accuracy 78.83 vs 77.56, pairwise consistency 85.94 vs 80.70, Pearson 0.68 vs 0.67, classification 85.48 vs 84.54.
- Construction of the rejected response: [[self-taught-evaluators]] §6.2 reports 83.8 for modified-instruction negatives against 80.7 for a "rewrite the answer to be worse" prompt.

**Controls that make negatives safe here.** Anchor with a positive likelihood term (the SFT term in the SFR-Judge loss). Restrict the negative to the tokens that carry the decision (verdict-only pairs). Filter negatives by length so the loss does not encode "shorter is wrong" (the judgment-length filters in [[meta-rewarding-lm]] App. A.3: drop chosen judgments above 1100 characters at iteration 1 and above 1000 at iteration 2). Keep the label balanced across A-better and B-better ([[self-taught-evaluators]] §3.4).

**Diagnostics.** Log position-consistency separately from accuracy — [[self-taught-evaluators]] Table 3 shows HelpSteer2 validation accuracy rising 65.5 → 71.0 while position-consistent accuracy rises only 56.5 → 60.6. Log the judge's score distribution, because compression toward the top of the scale ([[meta-rewarding-lm]] Fig. 5) removes separability without changing mean agreement. Log agreement with a fixed human set per judge version.

**Effect on generality.** Training a judge on one prompt distribution moves its category profile: [[self-taught-evaluators]] §5.1 reports Chat Hard 58.9 → 84.2 and Safety 69.2 → 91.5 while Chat falls 97.6 → 96.6, which the authors attribute to the harder reasoning training data (Interpretation). [[direct-judgement-preference]] §5.5 shows the specialization trade-off explicitly: continual fine-tuning the 8B judge at `β = 0.01` reaches 55.6% on ContextualJudgeBench with what the authors describe as a minimal drop on the seven general pairwise benchmarks, while a smaller `β` is what produces the specialization (Fig. 6).

---

## Recipe

Values below are quoted from the source cards at the loci given. Rows marked `not reported` list what was checked.

| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| Self-Taught Evaluator (Llama-3-70B-Instruct seed) | 70B | reward-model | prompts | 20,582 WildChat "reasoning" instructions | arXiv:2408.02666v2 §4.1 | verified 2026-09-14 | Table 4: reasoning 83.5 vs coding 79.4, GSM8K 79.3, hh_rlhf 79.6 |
| same | 70B | reward-model | judgments sampled per example; kept | N = 15; one correct, label-balanced | v2 §4.1 | verified 2026-09-14 | no ablation reported |
| same | 70B | reward-model | judgment sampling | temperature 0.7, top-p 0.9 | v2 App. A.2 Table 8 | verified 2026-09-14 | no ablation reported |
| same | 70B | reward-model | iterations; initialization | 5; re-initialized from seed each iteration | v2 §3.5, §4.1, Table 1 | verified 2026-09-14 | Table 1: 83.9 → 88.3 over iterations 1–5 |
| same | 70B | reward-model | loss; lr; warmup; epochs | NLL on judgment tokens only; 1.0e-06 with final_lr_ratio 0.2; 100 steps; max 2 | v2 §4.1, Table 7 | verified 2026-09-14 | no ablation reported |
| same | 70B | eval-gate | RewardBench decoding; majority vote | temperature 0.0, top-p 1.0; 32 samples at 0.7 / 0.9 | v2 Table 8, Table 1 | verified 2026-09-14 | Table 1: 88.3 → 88.7 with voting |
| SFR-Judge-8B / 12B / 70B | 8B, 12B, 70B | reward-model | initialization (and frozen reference) | Llama-3.1-8B-Instruct / NeMo-Instruct-12B / Llama-3.1-70B-Instruct | arXiv:2409.14664v3 §4.1, §3.4 | verified 2026-09-14 | no ablation reported |
| SFR-Judge-8B / 12B / 70B | 8B, 12B, 70B | reward-model | preference pairs; type mix | 680K; D_CoT 70% : D_Std 15% : D_Ded 15% | v3 §4.1 | verified 2026-09-14 | Fig. 4 (8B): removing D_Std or D_Ded changes results; the ratio itself is not ablated |
| SFR-Judge-8B / 12B / 70B | 8B, 12B, 70B | reward-model | D_CoT sampling | Llama-3.1-70B-Instruct; 20 samples per prompt; temperature 0.7 | v3 §4.1 | verified 2026-09-14 | App. E.8 Table 11: 70B negatives beat 8B negatives |
| SFR-Judge-8B / 12B / 70B | 8B, 12B, 70B | reward-model | DPO β; LR; epochs; batch; max length; compute | not reported | checked arXiv v1 and v3 body and appendices | not reported | — |
| CoT-GenRM STaR-DPO | 8B | reward-model | initialization; data; β; peak LR; iterations | Llama-3.1-8B-Instruct; UltraFeedback 61k pairs; β = 1.0; 1.0e-6; 3, one fresh data third per iteration | arXiv:2410.12832v1 §5, App. A.2.1 Table 2, App. A.3 | verified 2026-09-14 | App. B Table 3: Maj@32 71.28 → 72.98 → 73.58 |
| CoT-GenRM STaR-DPO | 8B | eval-gate | sampling; majority vote | temperature 1.0, top-p 0.95; 32 samples | v1 App. A.2.2, Fig. 2 caption | verified 2026-09-14 | Fig. 5 majority-vote curve |
| Meta-Rewarding LLM | 8B | preference | judgments per response; scoring | N = 11 samples, temperature 0.8, top-p 0.95; mean of valid 5-point scores | arXiv:2407.19594v2 §2.1, §3.1 fn. 2 | verified 2026-09-14 | fn. 2: larger N gave similar or worse correlation with human judgments (no table) |
| Meta-Rewarding LLM iteration 4 | 8B | preference | pair-selection length control ρ | 0.4 | v2 §3.1, App. A.3, Table 4 | verified 2026-09-14 | Table 4: ρ = 0.4 gives 39.44% LC, length 2003 characters; ρ = 0 gives length 2212 |
| Meta-Rewarding LLM, all iterations | 8B | preference | loss; β; epochs; LR; batch; schedule | DPO; 0.1; 10 with per-iteration checkpoint selection; 5×10⁻⁶; 32; cosine | v2 App. A.3 | verified 2026-09-14 | no ablation reported |
| Arena-Hard-Auto (v0.1) | — | eval-gate | prompts; judge; baseline; aggregation; cost | 500; gpt-4-1106-preview; gpt-4-0314; Bradley–Terry with 100 bootstrap rounds for 95% CIs; $20 per model | arXiv:2406.11939v2 §5, §6.1, Table 1 | verified 2026-09-15 | Table 1: separability 87.4% vs MT-Bench 22.6% |
| Arena-Hard-Auto with style control | — | eval-gate | style features | answer token length, markdown header count, markdown bold count, markdown list count, added to the BT fit | v2 §6.5, App. A.2 | verified 2026-09-15 | Table 3: confidence agreement 98.6% vs 94.4% without style control |
| AlpacaEval 2.0 LC | — | eval-gate | instructions; baseline; control | 805; gpt4_1106_preview; GLM of Eq. 1 with the length term set to zero, `3M + N` parameters | arXiv:2404.04475v2 §3, §4 | verified 2026-09-15 | Table 1: Arena correlation 0.98 vs 0.94, gameability 10% vs 26% |
| Chatbot Arena | — | eval-gate | aggregation; scale | Bradley–Terry with active sampling and multiplicity-corrected intervals; 240K votes from about 90K users | arXiv:2403.04132v1 §1, §4, §7.1 | verified 2026-09-15 | §6.3: crowd–expert agreement 72.8–83.1%, expert–expert 79.4% and 89.8% |
| RM-Bench | — | eval-gate | metric to report | Hard accuracy (upper triangle of the 3×3 style matrix) | arXiv:2410.16184v1 §3.3, §5 | verified 2026-09-15 | §5.2: r = 0.55 (p = 0.07) with downstream policy performance vs r = 0.21 (p = 0.51) for RewardBench |
| PPE | — | eval-gate | metric to report; aggregation | Per-pair accuracy on the human preference set; low-quantile aggregation across categories | arXiv:2410.14872v2 §7 | verified 2026-09-15 | Fig. 5: accuracy peaks at 0.80 Pearson with post-DPO Arena score at low quantile |

**Starting point for a small general-purpose judging setup.** Every number here comes from a `verified` row above, with its original conditions. For comparing checkpoints of a model in the 8B class on open-ended prompts: use 500 prompts drawn from your own traffic distribution rather than 80, because 500 curated prompts gave 87.4% separability against 22.6% for 160 MT-Bench turns on a 20-model set ([[arena-hard-benchbuilder]] Table 1); fix one baseline checkpoint and score pairwise against it; run both orders and report position consistency; aggregate with a Bradley–Terry fit and 100 bootstrap rounds for 95% intervals; add the four style features of [[lmsys-style-control]] to the fit. Use two judges from different families and report both, because the same 500 prompts produced 90.9% and 66.7% confidence agreement under two judges ([[arena-hard-benchbuilder]] Table 4). If you train your own judge, the published configuration closest to a small run is the 8B SFR-Judge recipe (680K pairs at 70/15/15, teacher sampling 20 per prompt at temperature 0.7); note that the DPO β, learning rate, and batch size for those runs are not reported, so they must be tuned locally.

---

## Generalization lens

**(a) What increases breadth.**
- Curating the prompt set for difficulty and topic coverage raises the benchmark's ability to distinguish models: BenchBuilder's curated 500 prompts gave 87.4% separability and 90.9% confidence agreement, against 36.4% confidence agreement and 75.6% separability for 250 randomly selected WildChat prompts ([[arena-hard-benchbuilder]] Tables 1–2).
- Judging with more than one model family raises agreement with human rankings: the GPT-4-Turbo + Gemini-1.5-Pro ensemble reached 91.5% confidence agreement and 89.5% separability against 90.9% / 87.4% for the single default judge, and reduced the measured self-bias ([[arena-hard-benchbuilder]] Table 4, §6.6).
- Training the judge on harder prompts broadens its category coverage: Chat Hard 58.9 → 84.2 and Safety 69.2 → 91.5 over five iterations ([[self-taught-evaluators]] Table 1).
- Training the judge to reason before deciding generalizes better out of distribution than a Bradley–Terry head: on RewardBench Reasoning, Bradley–Terry falls below random while STaR-DPO reaches 87.2% ([[generative-reward-models]] §5.2).

**(b) What causes narrowing.**
- Optimizing a policy for a judge-scored leaderboard. A constant response reached 86.5% LC on AlpacaEval 2.0 and 83.0 on Arena-Hard-Auto ([[null-model-cheating-benchmarks]] Table 2); this is the upper bound of what judge-directed optimization can produce with zero capability.
- Selecting the best of several privately tested variants. [[leaderboard-illusion]] reports 27 private variants before one release and relative gains up to 112% on Arena-Hard from additional arena-distribution data.
- Sharing a model family between the training-time labeler and the evaluation-time judge. The self-preference effect has a controlled measurement ([[self-preference-recognition]] Fig. 1) and the ranking effect of the judge choice is 24 points of confidence agreement ([[arena-hard-benchbuilder]] Table 4).
- Retraining the judge inside its own loop. Judge–human Spearman peaked at iteration 2 (0.382) and fell to 0.326 by iteration 4, while positional bias rose from 43.92% to 68.11% ([[meta-rewarding-lm]] Tables 5, 7).
- Specializing a judge with a small DPO β: `β = 0.01` produced the most specialized judge in [[direct-judgement-preference]] Fig. 6.

**(c) How to measure it at this stage.**
1. **Position consistency** per judge per benchmark, reported next to the win rate ([[judge-llm-bias]] Table 2; [[self-taught-evaluators]] Table 9 shows it persists after judge training).
2. **Style-controlled and raw scores together.** A large gap between them is the size of the style component ([[length-controlled-alpacaeval]] Table 1; [[lmsys-style-control]] coefficients).
3. **Cross-family judge disagreement**, reported as a range rather than re-attributed to one cause ([[arena-hard-benchbuilder]] Table 4).
4. **Style-stressed reward-model accuracy**: hard accuracy on [[rm-bench]], where the rejected response has the more favourable style.
5. **A downstream check**: per-pair accuracy on a fresh human-labeled set predicts post-training Arena score better than rank correlations do ([[ppe-reward-model-eval]] §7), and a negative correlation between a judge benchmark and downstream outcome is possible (§1, Fig. 4).
6. **A verifiable-correctness slice** wherever one exists, so part of the evaluation does not depend on a judge at all.

---

## Common mistakes and how to detect them

| Mistake | Observable symptom | Check |
|---|---|---|
| Reporting a win rate without the swap protocol | Two runs of the same evaluation disagree on close pairs; the winner correlates with which model was passed first | Run both orders on a 200-pair sample and compute consistency; compare with the 23.8–65.0% range in [[judge-llm-bias]] Table 2 |
| Treating "85% agreement with humans" as the accuracy of the judge in your setting | A checkpoint comparison that the judge calls decisive is not reproduced by humans | Check which setup the number came from: S2 excludes ties and has a 50% random baseline; S1 is 66% with a 33% baseline ([[judge-llm-bias]] Table 5) |
| Comparing checkpoints on 80-question MT-Bench | Score differences between checkpoints are within the bootstrap interval | Compute separability: MT-Bench separated 22.6% of pairs on a 20-model set ([[arena-hard-benchbuilder]] Table 1) |
| Assuming style control makes the metric safe | A submission with a short constant string scores near the top | Run the [[null-model-cheating-benchmarks]] structured response against your own annotator template before trusting the metric |
| Using the same model family as training-time labeler and evaluation-time judge | The gain is large on the shared-family judge and small on any other judge | Score the same checkpoints with a second judge family and report the range ([[arena-hard-benchbuilder]] Table 4) |
| Selecting the best of several evaluated variants and reporting only that one | Reported score does not reproduce on a fresh prompt set | Record how many variants were scored; report the distribution, not the maximum ([[leaderboard-illusion]] §3) |
| Ranking reward models by aggregate leaderboard score | The selected reward model does not produce the best policy | Use per-pair accuracy and low-quantile aggregation; check for the negative RewardBench correlation reported in [[ppe-reward-model-eval]] §1 |
| Judging with a rubric that names no style constraint, then interpreting a length increase as a quality increase | Response length grows across iterations while agreement with humans is flat | Track mean response length per iteration alongside the score ([[meta-rewarding-lm]] Table 4: 2212 vs 2003 characters at ρ = 0 vs ρ = 0.4) |
| Grading math or reasoning pairs with a bare judge | The judge confirms a confidently stated wrong answer | Add a reference generated by the judge before it reads the candidates: failures fell from 14/20 to 3/20 ([[judge-llm-bias]] Table 4) |
| Retraining a judge in a loop without an external anchor | Judge scores compress toward the top of the scale; benchmark score still rises | Recompute agreement with a fixed human-labeled set per judge version ([[meta-rewarding-lm]] Table 7) |

---

## Check your understanding

1. GPT-4's position consistency on the MT-Bench probe is 65.0% and GPT-3.5's is 46.2%. Explain why the conservative swap rule makes the confidence interval on a win rate depend on the judge, and compute the number of decided pairs each judge leaves out of 300.
2. The AlpacaEval GLM has an antisymmetric length term passed through `tanh`. Explain what would break if the length term were linear in the raw token difference instead, and why the identity `q(y = 1 | z_b, z_b, b, b, x) = 0.5` matters for a metric that compares a model against a baseline.
3. Length control lowered gameability from 26% to 10% ([[length-controlled-alpacaeval]] Table 1), yet a constant string reached 86.5% LC win rate ([[null-model-cheating-benchmarks]] Table 2). Explain why these two results are consistent, and state precisely what class of manipulation the GLM removes.
4. [[ppe-reward-model-eval]] found Spearman and Kendall correlations near zero as predictors of post-DPO Arena score, while per-pair accuracy was the best predictor. Give the causal reason the authors propose, and say what it implies about how to report a reward-model evaluation.
5. In [[rm-bench]], the same reward model scores 93.26% easy accuracy and 18.58% hard accuracy on the chat matrix quoted in §5.2. Explain what a policy trained against this reward model would learn, and which of the two numbers predicts that behaviour.
6. [[self-taught-evaluators]] trains a judge with NLL on kept-correct judgments, while [[direct-judgement-preference]] trains one with a DPO term on incorrect judgments plus an SFT term. Explain, in terms of where probability mass moves, why the second method also adds verdict-only pairs.
7. [[meta-rewarding-lm]] raises AlpacaEval 2 LC from 22.92% to 39.44% while judge–human Spearman peaks at iteration 2 and falls by iteration 4. Explain how both can be true, and what measurement you would add to decide whether the benchmark gain reflects capability.
8. A team reports +12 points on a GPT-4-judged benchmark and +4 points with a Claude judge on the same checkpoints. Explain why "8 points are judge leakage" is not a valid inference, and state what the 8-point gap does license you to say.

---

## Connections

- **Previous chapter** — [[ch-48]] — Contamination Detection and Its Effect on Reported Scores. Contamination inflates a score because the test item was seen in training; judge-specific overfitting inflates it because the response matched the grader. Both produce non-transferring numbers and need separate detections.
- **Next chapter** — [[ch-50]] — Slice Analysis, Forgetting Slices, and Failure Bucketing. A judge bias that is uniform across slices shifts every score equally; a bias that tracks a slice (code answers are longer, safety answers are refusals) changes the slice ranking, which is where it is detected.
- [[ch-47]] — Evaluation Harness and Suite Design for General Capability. The judge configuration in §1 is the per-benchmark specification the harness must record.
- [[ch-47a]] — Benchmark Overfitting and Generalization Audits: Fresh, Perturbed, Counterfactual, and Live Evaluation. §4 of this chapter is the judge-specific case of that audit.
- [[ch-41]] — Reward Modeling: Bradley–Terry, Over-Optimization, and Reward-Model Generalization. The Bradley–Terry fit used for Arena scores and for reward models is the same model; §5 here evaluates it as an instrument rather than as a training signal.
- [[ch-42]] — Reward Hacking and Judge Design. Training-time hacking of a judge; this chapter is the evaluation-time case.
- [[ch-43a]] — Negative Samples and Negative Gradients: Likelihood Displacement, Squeezing, and Negative Advantages. Holds the derivations that the negative-feedback section applies to judge training.
- [[ch-44]] — Process Supervision and Verifiable Rewards. On verifiable prompts a checker replaces the judge and removes this chapter's bias inventory from the measurement path.
- [[ch-51]] — Metric Noise, Confidence Intervals, and Go/No-Go Decisions. Judge sampling variance and the swap-induced loss of decided pairs are inputs to the noise budget there.
- [[ch-51a]] — Evaluating Agent Generality and Reliability. Judging of trajectories and environment states, which §4.4 excludes here.

---

## Sources

- [[judge-llm-bias]] — Zheng et al. 2023 (arXiv:2306.05685). Position, verbosity, self-enhancement, and math-grading measurements; the swap, few-shot, CoT, and reference-guided corrections; MT-Bench agreement tables with the S1/S2 distinction.
- [[chatbot-arena]] — Chiang et al. 2024 (arXiv:2403.04132). Human-vote platform, Bradley–Terry with active sampling, crowd–expert agreement rates, and the cost of expert labeling.
- [[arena-hard-benchbuilder]] — Li et al. 2024 (arXiv:2406.11939). Separability, confidence agreement, and Brier metrics; the 500-prompt pipeline; judge-choice comparison; style control applied to an automatic benchmark.
- [[length-controlled-alpacaeval]] — Dubois et al. 2024 (arXiv:2404.04475). The length-control GLM, its identity and symmetry properties, gameability measurements, and the correlation with Chatbot Arena.
- [[lmsys-style-control]] — LMSYS 2024 blog. Style features added to the Arena Bradley–Terry fit, the fitted coefficients, and the ranking changes.
- [[null-model-cheating-benchmarks]] — Zheng et al. 2024 (arXiv:2410.07137). Null-model win rates on three judge-scored benchmarks and the transferable random-search prefix.
- [[leaderboard-illusion]] — Singh et al. 2025 (arXiv:2504.20879). Private-variant testing, data-access asymmetry, silent deprecation, and the measured gain from arena-distribution data.
- [[self-preference-recognition]] — Panickssery et al. 2024 (arXiv:2404.13076). Self-recognition accuracy, the linear relation with self-preference, the confounder controls, and order-reversal rates.
- [[rewardbench]] — Lambert et al. 2024 (arXiv:2403.13787). Section structure, scoring and weighting rules, and the spread of accuracies within Reasoning.
- [[rm-bench]] — Liu et al. 2024 (arXiv:2410.16184). The style–substance matrix, easy/normal/hard accuracy, the 46.6% hard-accuracy result, and the policy-correlation comparison.
- [[ppe-reward-model-eval]] — Frick et al. 2024 (arXiv:2410.14872). End-to-end reward-model evaluation with a real post-training Arena deployment; which offline metrics predict downstream preference scores.
- [[self-taught-evaluators]] — Wang et al. 2024 (arXiv:2408.02666). Modified-instruction pair construction, rejection-sampled judgments, five iterations on RewardBench, and the persisting order effect.
- [[direct-judgement-preference]] — Wang, Xu et al. 2024 (arXiv:2409.14664). The DPO+SFT judge loss, the three pair types, the hard-negative ablation, and the better-of-two-orders reporting convention.
- [[generative-reward-models]] — Mahan et al. 2024 (arXiv:2410.12832). GenRM and CoT-GenRM training variants, in-distribution versus RewardBench generalization, and the majority-vote effect.
- [[meta-rewarding-lm]] — Wu et al. 2024 (arXiv:2407.19594). Judge and meta-judge training, length control at pair selection, and the drift diagnostics (meta-judge bias, positional bias, score compression, human Spearman by iteration).
