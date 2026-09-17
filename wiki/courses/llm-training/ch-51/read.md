<!-- chapter: ch-51
     track: eval
     kind: content
     title: Metric Noise, Confidence Intervals, and Go/No-Go Decisions
     deps: [ch-50]
     sources: [[benchmark-variance-quantified]], [[signal-and-noise-eval]], [[adding-error-bars-evals]], [[datadecide]], [[prompt-format-sensitivity-formatspread]], [[emergent-abilities-mirage]], [[ai2-benchmirt]], [[judge-llm-bias]], [[reward-model-overoptimization]], [[llama-3]]
     figures: figures/bootstrap-ci.html
     revised: 2026-09 (generality revision)
-->

# Chapter 51 — Metric Noise, Confidence Intervals, and Go/No-Go Decisions

> **Core insight.** The error on a reported benchmark score has at least four independent sources, and three of
> them do not shrink when the evaluation set grows: the training seed, the sampling of generations, and the prompt
> format. Measured sizes: pretraining-seed standard deviation of 0.21 to 2.15 accuracy points across 13 benchmarks
> on 10 identically trained 7B models ([[benchmark-variance-quantified]] Table 1); standard deviation up to 2
> points between reruns of one 1B recipe ([[datadecide]] §2.1); median spread of 7.5 accuracy points across 10
> meaning-equivalent prompt formats, with 14.1% of model pairs reversing rank under a different format and 76% of
> those reversals statistically significant in both directions ([[prompt-format-sensitivity-formatspread]] §4.2).
> A clustered evaluation can also carry a standard error 3.05× the naive one (DROP, [[adding-error-bars-evals]]
> Table 4).
>
> **Guideline.** When comparing two checkpoints on the same items, report the paired standard error
> `sqrt(SE²_A + SE²_B − 2·SE_A·SE_B·Corr(s_A, s_B))` together with the score correlation, because pairing removes
> shared per-item difficulty and, in the worked case of [[adding-error-bars-evals]] §4.2 at correlation 0.5, lowers
> the estimator variance by one third. When questions arrive in groups (one passage, one problem in many
> languages), use the clustered form of that estimator, because the naive one was 3.05× too small on DROP
> (Table 4). When the decision is whether to ship a general-purpose checkpoint, require two tests: a gain test on
> the pre-registered target and a non-inferiority test against a pre-registered margin on a held-out general suite,
> with multiplicity controlled on the gain side. When the decision is made from models far smaller than the target,
> first check whether the benchmark can support that decision at that scale ([[signal-and-noise-eval]],
> [[datadecide]]); when it cannot, no interval width rescues it.

---

## Why this chapter matters for a general-purpose model

Every stage of the pipeline (pre-training → mid-training → SFT → preference optimization → RL) ends with the same
question: keep this checkpoint, or the previous one. The answer is read off a table of benchmark numbers. For a
model targeted at one task the table has one row. For a general model it has tens — target capability, instruction
following, knowledge, code, safety, over-refusal, and the forgetting slices of [[ch-50]] — and the decision is a
conjunction: the target must improve *and* nothing else may fall by more than a stated amount.

That conjunction changes the statistics in two ways.

1. With many rows, the chance that at least one shows a spurious improvement grows quickly. Testing 20
   independent slices at α = 0.05 gives a probability of at least one false positive of 1 − 0.95²⁰ = 0.642
   (derived). Selecting the best-looking slice afterwards is how a narrow model comes to be described as a broad
   one.
2. The non-regression half is an *absence* claim, and absence claims need power, not only an interval. "The paired
   interval on the general suite contains 0" is compatible with a 3-point loss whenever the interval is 3 points
   wide.

The chapter builds the pieces in order: where the noise comes from and how large it is (§1), intervals that
respect the sampling scheme (§2), comparing two checkpoints (§3), sizing the run beforehand (§4), many slices at
once (§5), the two-part gate (§6), and which decisions a benchmark can support at all (§7).

---

## §1 Four noise sources, with measured sizes

A reported score estimates the model's expected score over three populations at once: questions, training seeds,
and generations. Four sources separate the estimate from that quantity, and only the first shrinks with more items.

### 1.1 Item sampling noise

The evaluation set is a finite sample of questions; the score would change under a different sample of the same
size. For binary per-item scores with mean s̄ over n items, `SE_Bernoulli = sqrt(s̄(1 − s̄)/n)`
([[adding-error-bars-evals]] Eq. 2) and the 95% interval is `s̄ ± 1.96 · SE`.

**Worked example (by hand).** Ten items scored 1,1,0,1,0,1,1,1,0,1 give s̄ = 0.7,
SE = sqrt(0.7 × 0.3 / 10) = sqrt(0.021) = 0.1449, halfwidth 1.96 × 0.1449 = 0.284, interval [0.416, 0.984]. Ten
items measure almost nothing.

**Halfwidths at real benchmark sizes** (derived from 1.96·sqrt(p(1−p)/n), one generation per item, independent
items):

| Benchmark | n | at p = 0.5 | at a realistic p |
|---|---|---|---|
| HumanEval | 164 | 7.65 pp | 6.87 pp at p = 0.72 |
| MATH500 | 500 | 4.38 pp | 4.24 pp at p = 0.60 |
| IFEval | 541 | 4.21 pp | 3.49 pp at p = 0.78 |
| GSM8K test | 1319 | 2.70 pp | 1.87 pp at p = 0.86 |
| MMLU | 14042 | 0.83 pp | 0.76 pp at p = 0.70 |

The spread across this column is why a fixed rule such as "gains under 2 points are noise" is wrong: on HumanEval
a 5-point gain can sit inside the interval, and on MMLU a 1-point gain can sit outside it.

[figures/bootstrap-ci.html](figures/bootstrap-ci.html) — Panel 1 moves n, p, the generations per question and the
run-to-run standard deviation, and shows which term dominates the combined halfwidth, with markers at the
benchmark sizes above.

### 1.2 Training-run noise (seed and data order)

Two runs of one recipe that differ only in initialization seed or data order give different checkpoints with
different scores.

[[benchmark-variance-quantified]] trained 10 Llama-2-7B-architecture models on 210B tokens each, identical except
the seed, and measured the standard deviation across seeds averaged over 21 checkpoints (Table 1): HellaSwag 0.21,
MMLU-Cloze 0.22, GSM8k 0.41, MMLU 0.57, ARC-C 0.80, HumanEval 1.11, COPA 2.15 accuracy points. Its summary:
"Benchmarks with few test examples (like COPA and HumanEval) exhibit high variance (both seed variance and 95%
CIs). Generally, the 7B seed variance is well below the 95% CI for the same benchmark, though the ratio of the two
is quite variable." Result (single study), pretraining seeds only; fine-tuning seeds are not measured.
[[datadecide]] reran each 1B configuration with three seeds and reports run-to-run standard deviation reaching
"2 percentage points of accuracy for some recipes on most tasks" (§2.1).

[[signal-and-noise-eval]] defines a cheaper proxy: noise as the relative standard deviation over the final n
checkpoints of one run, `Rel.Std(m) = sqrt((1/(n−1)) Σ_i (m_i − m̄)²)/m̄` (§3.1), where m_i is the score of
checkpoint i and m̄ their mean. The final 30 checkpoints of its 1B models span 1.7% accuracy on ARC Challenge, and
for 10 seed runs and 10 data-order runs at 1B-5xC, seed noise, data-order noise and whole-run total variation
correlate with this final-n quantity at R² of 0.82, 0.86 and 0.95 (§3.1, Fig. 7). One run can therefore estimate
what two runs would show. The implication for a gate: an item-level interval alone understates the error whenever
the comparison is between two separately trained checkpoints rather than two evaluations of the same weights.

### 1.3 Generation noise

For generative tasks the model is sampled, so the same question scores differently across generations.
[[adding-error-bars-evals]] §3.1 decomposes the per-item variance as `Var(s) = Var(x) + E[σ²_i]`, with x the
question's conditional mean score (a property of the question) and σ²_i the conditional variance of question i
under sampling. With K generations per question, `Var(µ̂ | K) = (Var(x) + E[σ²_i]/K)/n`. In the paper's worked case
(binary scores, uniform difficulty, Var(x) = 1/12, E[σ²_i] = 1/6), `Var(µ̂|K) = Var(µ̂|K=1) × (1 + 2/K)/3`: K = 2
removes a third of the variance, K = 4 a half, and the limit as K grows is two thirds of the K = 1 variance. The
floor is Var(x)/n — generations never remove the item-sampling term.

Lowering the temperature to zero removes σ²_i but changes the quantity measured, and can raise the variance of the
conditional means. In the single-token examples of §3.3, with x_{T=1} ~ U[0,1], Var(x_{T=1}) = 1/12 while
Var(x_{T=0}) = 1/4; with x_{T=1} ~ U[1/3, 1] the expected score moves from 2/3 to 3/4 and the variance from 1/27
to 3/16. The author's rule: "In neither case should the sampling temperature be adjusted for the sake of reducing
variance in the scores." Greedy decoding is a legitimate condition to measure; it is not a variance-reduction
technique. Treating the K·n generations as K·n independent observations is inconsistent (§3.1) — the unit is the
per-question mean over its K generations.

### 1.4 Prompt-format and harness noise

Two prompt formats built from the same rules and descriptors express the same task, so a score difference between
them is a property of the model rather than of the task.

[[prompt-format-sensitivity-formatspread]] evaluated 53 Super-NaturalInstructions tasks on LLaMA-2 7B/13B/70B,
Falcon-7B, Falcon-7B-Instruct and GPT-3.5-Turbo. With 10 randomly sampled formats per task the median spread (max
minus min accuracy over formats) is 7.5 points; 20% of tasks have a spread of at least 15 points in every LLaMA-2
setting; several tasks exceed 70 points; the reported maximum is 76 points for LLaMA-2-13B. These are lower bounds,
since only 10 of the possible formats were sampled (§4.2). The spread persists at 70B, after instruction tuning,
and with more few-shot examples.

The result that should change how a gate is written: taking model M better than M′ by at least 0.02 accuracy under
format p, the probability that M′ is better by at least 0.02 under another format p′ is 0.141 for LLaMA-2-13B vs
70B and 0.140 for LLaMA-2-7B vs Falcon-7B, and "often both experiments (first using p, and then p′) were
statistically significant (p-value < 0.05) on 1000 samples: 76% and 47% respectively for the two model
comparisons" (§4.2). A correctly paired, significant comparison can be reversed by a formatting choice that
carries no meaning, and the reversal can be significant too.

The same effect appears in benchmark formulation. [[benchmark-variance-quantified]] §3.3 compares two MMLU
formulations on one set of models: standard MMLU sits at 25.86 against a chance level of 25 after 210B tokens,
with monotonicity 0.09 and seed standard deviation 0.57; MMLU-Cloze reaches 37.47 with seed standard deviation
0.22 and monotonicity 0.95. The two correlate at Pearson 0.92 across 70 fully trained models, so the formats agree
on ranking large models while disagreeing entirely on what small models can do.

### 1.5 Judge noise, when the metric is a judge

An LLM judge adds a term that does not shrink with n. [[judge-llm-bias]] Table 2 measures consistency under an A/B
swap on deliberately similar answer pairs: GPT-4 65.0%, GPT-3.5 46.2%, Claude-v1 23.8% under the default prompt,
so the verdict changes for 35.0%, 53.8% and 76.2% of pairs. The paper's conservative rule is to count a win only
when the answer wins in both orders and call the rest ties (§3.4). ch-49 holds the full treatment; here the point
is that swap-and-average is part of the estimator, and the residual inconsistency belongs in the noise budget.

### 1.6 The budget, side by side

| Source | Shrinks with more items? | Shrinks with more runs or generations? | Measured size and locus |
|---|---|---|---|
| Item sampling | yes, as 1/sqrt(n) | no | 0.83 pp (MMLU, n=14042) to 7.65 pp (HumanEval, n=164) at p=0.5, derived |
| Training seed / data order | no | yes, as 1/sqrt(runs) | σ 0.21–2.15 pp over 13 benchmarks ([[benchmark-variance-quantified]] Table 1); up to 2 pp at 1B ([[datadecide]] §2.1) |
| Generation sampling | partly, to a Var(x)/n floor | yes, to a 2/3 floor | `Var(µ̂|K) = Var(µ̂|K=1)(1+2/K)/3` in the worked case ([[adding-error-bars-evals]] §3.1) |
| Prompt format | no | no | median spread 7.5 pp over 10 formats; 14.1% rank reversals ([[prompt-format-sensitivity-formatspread]] §4.2) |
| Judge inconsistency | no | partly, by swap-and-average | 35.0% swap-inconsistency for GPT-4 on similar pairs ([[judge-llm-bias]] Table 2) |

When the per-run sources exceed the item-sampling source, adding items is wasted compute and the budget belongs to
runs, generations or formats instead.

---

## §2 Intervals that respect the sampling scheme

### 2.1 The interval for one score

[[adding-error-bars-evals]] Eqs. 1-3, with s_i the score of question i and s̄ the mean over n questions:

```
SE_C.L.T.    = sqrt( Var(s)/n ) = sqrt( (1/(n−1)) Σ_i (s_i − s̄)² / n )      (1)
SE_Bernoulli = sqrt( s̄(1 − s̄)/n )                                          (2)   [binary scores only]
CI_95%       = s̄ ± 1.96 × SE_C.L.T.                                         (3)
```

Equation 2 is the special case of Equation 1 for s_i ∈ {0,1}. Applying it to fractional scores (F1, partial
credit, per-item pass rates) gives an interval that "will tend to be conservative (too wide)"; the paper names the
Llama 3 report ([[llama-3]]) as doing this (§2.1). Result (single study, one author's reanalysis).

### 2.2 Clustered items

Reading-comprehension suites draw several questions per passage and multilingual suites use one question in many
languages, so questions inside a group are correlated and the independence assumption behind Equations 1 and 3 —
and behind a naive item bootstrap — fails. With s_{i,c} the score of question i in cluster c and clusters assumed
independent ([[adding-error-bars-evals]] Eq. 4):

```
SE_clustered = [ SE²_C.L.T. + (1/n²) Σ_c Σ_i Σ_{j≠i} (s_{i,c} − s̄)(s_{j,c} − s̄) ]^(1/2)     (4)
```

Measured on Anthropic models (Table 4, described by the author as non-fictional): DROP 1.34 vs 0.44, ratio 3.05;
MGSM 1.62% vs 0.86%, ratio 1.88; RACE-H 0.51% vs 0.46%, ratio 1.10. Table 3 reports DROP as 9,622 questions in 588
clusters and recommends printing the cluster count beside the question count.

**Worked example (derived).** A standard-error ratio of 3.05 means 9.3× the variance, so DROP's 9,622 questions
carry about 9,622/9.3 ≈ 1,035 independent questions' worth of information. Sizing that evaluation by its question
count overstates its resolving power by an order of magnitude.

### 2.3 Bootstrap: when it earns its cost

The percentile bootstrap resamples n item indices with replacement B times, recomputes the statistic and takes the
2.5th and 97.5th percentiles as the interval. The BCa variant adjusts those two percentiles by a bias factor
z₀ = Φ⁻¹(#{θ̂_b < θ̂}/B) and a jackknife acceleration term, which matters when the statistic is skewed or sits near
0 or 1; θ̂ is the statistic on the full sample and θ̂_b its value on replicate b.

Two sources disagree about whether this is worth doing for a plain mean, and the disagreement is informative.
[[adding-error-bars-evals]] §2.1: "we regard bootstrapping as unnecessary unless a complicated sampling scheme or
estimator is being used". [[benchmark-variance-quantified]] §3.1 bootstrapped every checkpoint and found that
"bootstrapped and Analytic CIs converge when the number of bootstrap samples is large". Both point the same way:
for a mean over independent items the analytic interval suffices; use the bootstrap when the estimator is not a
simple mean (macro-average over unequal slices, win rate after swap-and-average, pass@k) or when the sampling is
clustered or stratified.

---

## §3 Comparing two checkpoints

### 3.1 Pair the items

Subtracting two independent intervals discards the fact that both checkpoints answered the same questions.
Question difficulty is shared, so it cancels in the difference ([[adding-error-bars-evals]] Eqs. 5-7, with
s_{A−B,i} = s_{A,i} − s_{B,i}):

```
unpaired: SE_{A−B}        = sqrt( SE²_A + SE²_B )                                  (5)
paired:   SE_{A−B,paired} = sqrt( Var(s_{A−B})/n )                                 (7)
                          = sqrt( SE²_A + SE²_B − 2·SE_A·SE_B·Corr(s_A, s_B) )
```

The gain from pairing is set by Corr(s_A, s_B) and nothing else: at correlation 0 Equation 7 equals Equation 5; at
correlation 0.5 with the paper's uniform-score example the estimator variance falls by one third (1/6 → 1/9,
§4.2); at correlation 1 the difference has no variance. There is no general multiplier, which is why the paper
recommends reporting the correlation next to the interval.

**Worked example (by hand).** Ten paired items.

```
A: 1 1 0 1 1 0 1 1 0 1      s̄_A = 0.7    SE_A = sqrt(0.7·0.3/10) = 0.1449
B: 1 0 0 1 1 0 1 0 0 1      s̄_B = 0.5    SE_B = sqrt(0.5·0.5/10) = 0.1581
d: 0 1 0 0 0 0 0 1 0 0      d̄   = 0.2
Var(d) = [2·(0.8)² + 8·(0.2)²]/9 = 1.6/9 = 0.1778
SE_paired   = sqrt(0.1778/10) = 0.1333          → 95% CI  0.2 ± 0.261 = [−0.061, +0.461]
SE_unpaired = sqrt(0.1449² + 0.1581²) = 0.2145  → 95% CI  0.2 ± 0.420 = [−0.220, +0.620]
implied Corr(s_A, s_B) = (0.2145² − 0.1333²) / (2·0.1449·0.1581) = 0.616
```

Pairing narrows the interval by a factor of 1.61 at this correlation. Neither interval excludes 0: at n = 10 a
20-point observed difference is not evidence.

When items are clustered, apply the clustered form directly to the differences ([[adding-error-bars-evals]]
Eq. 8), summing `(s_{A−B,i,c} − s̄_{A−B})(s_{A−B,j,c} − s̄_{A−B})` over pairs within clusters. In the paper's
running example, pairing plus clustering changes the conclusion: the difference is significant on MATH and
indistinguishable from noise on HumanEval and MGSM, reversing the informal reading of the same table (§4.2).

### 3.2 Paired bootstrap, and what its p-value is

```
# paired_bootstrap(s_A, s_B, B=10000, alpha=0.05)
#   s_A, s_B: arrays of n per-item scores for checkpoints A and B, in the same item order
d         = s_A - s_B                            # per-item paired differences
delta_hat = mean(d)
boot = []
for b in range(B):
    idx = sample_with_replacement(range(n), n)   # resample PAIRS: one index list, both arrays
    boot.append(mean(d[idx]))
boot.sort()
ci_lo, ci_hi = boot[int(0.025*B)], boot[int(0.975*B)]
# p-value by interval inversion (NOT the Efron-Tibshirani achieved significance level):
p_approx = 2 * min(frac(boot >= 0), frac(boot <= 0))
```

The last line needs its label. Efron and Tibshirani's achieved significance level is computed under a
**null-centred** bootstrap distribution: shift the replicates by −delta_hat so the null holds, then count how many
shifted replicates are at least as extreme as delta_hat. Doubling the tail mass of the uncentred distribution, as
written above, is an approximation obtained by inverting the percentile interval; it agrees with the centred
construction when the bootstrap distribution is close to symmetric. State which of the two a number came from.

### 3.3 Sign test

Let W = #{i : s_A,i > s_B,i} and L = #{i : s_A,i < s_B,i}, dropping ties. Under the null of no difference,
W ~ Binomial(W + L, 0.5). The exact two-sided p-value is `2 · P(X ≤ min(W, L))` with X ~ Binomial(W + L, 0.5),
capped at 1; for W + L ≥ 25 the normal approximation with continuity correction is
`z = (|W − L| − 1)/sqrt(W + L)`, `p = 2(1 − Φ(z))`.

**Worked example (by hand).** In §3.1's data W = 2, L = 0, so W + L = 2 and p = 2 · P(X ≤ 0) = 2 · (1/2)² = 0.5.
The test discards the eight tied items and is left with two observations. It is the right tool when per-item score
magnitudes are not comparable (rubric or judge scores on different scales) but their direction is; it is the wrong
tool when most items tie.

Panel 2 of [figures/bootstrap-ci.html](figures/bootstrap-ci.html) generates paired data at a chosen score
correlation and shows the paired interval, the unpaired interval and the sign-test p-value together, so the cost
of ignoring the pairing is visible at fixed n.

---

## §4 Sizing the evaluation before running it

"Is this difference significant?" is asked after the compute is spent. The version that can change a plan is: for
the smallest difference worth acting on, how many questions and generations are needed?

[[adding-error-bars-evals]] Eqs. 9-10, with ω² = Var(x_A) + Var(x_B) − 2Cov(x_A, x_B) the variance of the paired
conditional-mean difference, σ²_A and σ²_B the mean conditional variances under sampling, K_A and K_B the
generations per question, α the significance level, 1 − β the power and δ the minimum detectable effect:

```
n = (z_{α/2} + z_β)² (ω² + σ²_A/K_A + σ²_B/K_B) / δ²                (9)
δ = (z_{α/2} + z_β) sqrt( (ω² + σ²_A/K_A + σ²_B/K_B) / n )          (10)
```

**Worked example, as published.** With no generation variance (σ²_A = σ²_B = 0), ω² = 1/9, δ = 0.03, α = 0.05 and
β = 0.20: n = (1.96 + 0.8416)² × (1/9) / 0.0009 ≈ 969 questions. The author reads this as a case for new
evaluations containing at least 1,000 questions (§5).

**Worked example, generations instead of questions.** With σ²_A = σ²_B = 1/6, ω² = 1/9, Corr(x_A, x_B) = 0.5,
n = 198, α = 0.05 and β = 0.20, raising K from 1 to 10 moves the minimum detectable effect from 13.2% to 7.5%
(§5). When the evaluation set is fixed and small, generations are the remaining lever, and their return flattens
at the Var(x)/n floor of §1.3.

**Conditions and limits.** Equations 9 and 10 assume independent questions; with clusters, substitute the
cluster-adjusted variance (App. B of the same paper). They also assume ω², σ²_A and σ²_B are known, which in
practice means estimating them from a pilot run on a subset.

---

## §5 Many slices, many benchmarks

A general-purpose gate tests many things at once, and two different multiplicity problems appear in one memo.

**The gain side: selection inflates false positives.** Testing m independent slices at level α, the probability of
at least one spurious rejection is 1 − (1 − α)^m; for m = 20 and α = 0.05 that is 0.642 (derived). Two standard
corrections. **Holm–Bonferroni** controls the family-wise error rate: sort the m p-values ascending and reject
p_(k) while p_(k) ≤ α/(m − k + 1), so with m = 20 and α = 0.05 the smallest is compared against 0.0025, the next
against 0.00263. Use it when a single false claim is expensive, such as a ship decision or a public number.
**Benjamini–Hochberg** controls the false discovery rate: reject p_(k) for the largest k with p_(k) ≤ k·α/m. Use
it when the output is a ranked list of slices to investigate, where a known share of false leads is acceptable.

**The non-inferiority side: no correction, but a power problem.** "No slice regressed by more than δ" is a
conjunction, and an intersection-union test rejects the union null only when every component test rejects, so the
type-I error is already at most α without correction. The difficulty is the reverse: power is the product of the
per-slice powers, so a gate built from 20 under-powered non-inferiority tests fails even for a checkpoint that
regressed nowhere. Size the general suite as one pooled test plus a few pre-registered sub-suites rather than 50
slices at 200 items each.

**Formats are a comparison too.** §1.4's reversal result means a single-format significant difference is one draw
from a distribution over formats. Fixing the format set in advance, and reporting the spread across it as the
authors recommend, keeps that choice from becoming an uncounted selection over 10 tests.

---

## §6 The two-part go/no-go gate

**Claim 1 — target gain (superiority).** Pre-register the target metric or bundle before the run, decide with the
paired interval, and correct across the pre-registered slices as in §5. Reject the null when the paired 95%
interval excludes 0.

**Claim 2 — no regression on held-out general capability (non-inferiority).** Pre-register the general suite
(knowledge, instruction following, code, safety, over-refusal, plus the forgetting slices of [[ch-50]]) and the
margin δ_NI, which is a product decision rather than a statistical one. Declare non-inferiority when the lower
bound of the paired 95% interval for Δ = new − old exceeds −δ_NI.

**Worked example (by hand).** Held-out general suite, paired evaluation, Δ̂ = −0.6 pp with SE_paired = 0.5 pp and
margin δ_NI = 1.0 pp. The interval is −0.6 ± 1.96 × 0.5 = [−1.58, +0.38]. It contains 0, so no regression was
*detected*; its lower bound −1.58 is below −1.0, so non-inferiority at a 1-point margin is *not established*.
Those are different statements and only the second is the gate. Establishing it needs 1.96·SE < 0.4, so
SE < 0.204 pp, which at the same score correlation means (0.5/0.204)² ≈ 6.0× more items. Finding this out before
the run is §4's job.

**Memo template.** One page; the sections and the trigger line are the part to copy.

```
go-no-go-memo.md
----------------
## 1. Pre-registered claims (written before the evaluation ran)
C1 (gain): ckpt-round-3 improves over ckpt-round-2 on TARGET = {...}, paired mean
           difference, Holm-corrected across the pre-registered target slices.
C2 (no-regression): on HELD-OUT GENERAL SUITE (knowledge / instruction following /
           code / safety / over-refusal / ch-50 forgetting slices), the pooled paired
           difference is non-inferior at margin delta_NI = <value> pp.

## 2. Evidence for C1
| slice | n (clusters) | corr(s_A,s_B) | delta_hat | SE_paired | 95% CI | Holm threshold | reject? |
estimator stated per row: paired CLT, paired clustered, or paired bootstrap with B.

## 3. Evidence for C2
| suite (pooled) | n (clusters) | corr | delta_hat | SE_paired | 95% CI | lower bound vs -delta_NI |
a lower bound below -delta_NI fails C2 even when the interval contains 0.

## 4. Noise budget actually used
items and cluster counts; estimator; generations per question K and the sampling
temperature (the deployed one, per §1.3); number of training runs compared, and if one,
the seed sigma assumed with its source; number of prompt formats and observed spread;
judge swap-and-average and residual inconsistency rate.

## 5. Power statement
Minimum detectable effect at the chosen n, K, alpha = 0.05, power 0.80, from Eq. 10.
If MDE exceeds the claimed gain, the run cannot support C1.

## 6. What the reviewer re-runs
exact command, checkpoint hash, eval commit, item-id file for the paired sets, and the
reproduction tolerance derived from section 4.

## 7. Decision
GO only if C1 rejects at its Holm threshold AND C2's lower bound exceeds -delta_NI AND
section 5 covers the claimed effect. Otherwise NO-GO, with the missing piece named.
```

The trigger line carries the memo: three conditions, all required, each traceable to a number in the sections
above. A memo whose decision does not follow from its own tables is the failure this chapter exists to prevent.

**Curves during training.** When the decision is where to stop rather than which checkpoint to keep, the statistic
is a curve, and how it is summarized changes what is seen. [[signal-and-noise-eval]] §5.2 measures one option:
averaging the outputs of a run's final checkpoints instead of reading the last one raised 30-task decision
accuracy from 68.9% to 71.3% and lowered 13B scaling-law prediction error from 1.03 to 0.86 absolute percent, and
for early stopping an exponential moving average beat a single checkpoint at almost any training step (Fig. 5).
That is averaging checkpoints, which yields a model you can ship — a different operation from smoothing a plotted
curve. The separate risk of trusting a proxy curve is the shape in [[reward-model-overoptimization]]: optimizing a
policy against a learned proxy reward model makes the proxy score keep rising while the gold score peaks and falls
as √KL grows (Fig. 1). That result concerns proxy versus gold reward under optimization pressure, not benchmark
curves across checkpoints; carrying it over is an Interpretation, and the defensible version is narrow — when the
plotted quantity is a proxy for the thing you care about, monitor the held-out quantity itself rather than a
smoothed proxy.

---

## §7 Which decisions a benchmark can support

Intervals answer how precisely a score was measured. They do not answer whether that score, at that scale,
predicts the thing being decided. That question has its own measurements.

**Decision accuracy.** [[datadecide]] pretrained 25 corpora × 14 sizes × 3 seeds (1,050 models) and asked how
often small-model rankings predict 1B rankings. Ranking models at a single small size (for example 150M) picks the
corpus that wins at 1B in about 80% of pairwise comparisons, and none of 8 scaling-law baselines exceeds the
compute-versus-decision-accuracy frontier of single-scale ranking (§3.1-3.2). Per task the picture differs
sharply: ARC Easy is predictable with five orders of magnitude less compute, while BoolQ exceeds trivial decision
accuracy only with intermediate checkpoints of the target runs (§3.1, Fig. 2). With continuous likelihood metrics,
MBPP and HumanEval move from trivial to 80% decision accuracy, while Minerva and GSM8K stay near trivial unless
the target metric is continuous as well (§3.4, Fig. 6).

**Signal, noise and their ratio.** [[signal-and-noise-eval]] defines signal as relative dispersion across a
population of similarly trained models, noise as in §1.2, and SNR as their ratio (Eq. 2). SNR predicts decision
accuracy at R = 0.791, R² = 0.626, where neither component alone does (§4.1, Fig. 2). Its three interventions:
filter subtasks by SNR (top 16 MMLU subtasks, +2.6% decision accuracy; top 6 AutoBencher subtasks, +5%), average
final checkpoints (§5.2), and score with bits-per-byte — the negative log-likelihood of the correct answer divided
by its UTF-8 byte count — which raises GSM8K SNR from 1.2 to 7.0 and MBPP from 2.0 to 41.8 and improves decision
accuracy for 90.0% of benchmarks (§5.3). SNR does not transfer across scales: ARC Easy falls from 7.89 at
1.5B-4T to 5.10 at 32B-6T and SocialIQA from 8.73 to 1.95, while Minerva MATH rises from 0.91 at 1.5B-4T to 4.45
at 7B-4T (App. B.3, Table 4).

**Metric shape and apparent jumps.** A step in a checkpoint curve can be a property of the metric.
[[emergent-abilities-mirage]] shows that on fixed InstructGPT/GPT-3 outputs, arithmetic tasks look emergent under
Accuracy and improve smoothly under Token Edit Distance, and that more than 92% of claimed emergent abilities on
BIG-Bench appear under two metrics, Multiple Choice Grade and Exact String Match (§3-§4). Its second mechanism is
this chapter's subject: resolution is set by 1/(test-set size), so a 100-item slice cannot express any effect
smaller than one point, and models below that resolution appear to sit at zero (§2, App. A.1). Both lines agree
with the bits-per-byte result: continuous metrics carry higher SNR — in [[benchmark-variance-quantified]] Table 2
the continuous SNR exceeds the discrete SNR on all 10 benchmarks reported — and give more predictable curves.

**Smaller evaluation sets: two results that disagree.** Item response theory models each item with a difficulty
and a discrimination parameter and each model with an ability, which allows selecting the most informative items.
[[ai2-benchmirt]] fit a two-dimensional IRT model on 100 open-weight LLMs across 16 benchmarks and 34,301 items
and reports that keeping the most discriminative 10% of items "generally preserved nearly the same picture" while
50% "often matched the full benchmark's measure of those capabilities even more closely", with held-out item
prediction accuracy of 79% against 70% for a per-benchmark-average baseline. Against this,
[[benchmark-variance-quantified]] applied the 100-item tinyBenchmarks subsets to its 10 seed models and found seed
variance rising from 0.80 to 1.80/1.86 on ARC-C, 0.41 to 1.16/1.49 on GSM8k and 0.21 to 2.06/2.42 on HellaSwag,
with monotonicity falling and the mean drifting (ARC-C overestimated by 7%); its conclusion is that the method
"may have limited use during pretraining ablations" because it makes comparisons "more likely to be confounded by
randomness from the initialisation and data ordering seed" (§5). Open question. The two studies
measure different things — construct alignment and ranking fidelity across released models, versus seed-level
stability across intermediate checkpoints of one run — so the defensible position is that a pruned subset can
preserve rankings between well-separated models while losing the resolution needed for same-recipe ablations.
Validate a pruned subset on the decision it will be used for, with its seed variance measured rather than assumed.

**Construct validity.** [[ai2-benchmirt]] also reports that a benchmark can measure something other than its
label: BBQ correlates 0.85 with the recovered general-reasoning ability and −0.06 with the safety ability, and
WMDP runs opposite to reasoning at −0.89. A "safety did not regress" line in a gate that averages such benchmarks
is not measuring what its label says. This is slice analysis ([[ch-50]]) moved one level up: an aggregate can be
statistically sound while its components measure different constructs.

---

## Recipe

| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| Signal and Noise study (DataDecide small models) | 60M-750M | eval-gate | noise window for decision accuracy | final 5 checkpoints, averaged over the small models | arXiv:2508.13144v1 §4.1; App. A.3.2 | verified 2026-09-14 ([[signal-and-noise-eval]]) | Table 2: n = 5 lands within ±1σ of the true noise for almost all benchmarks |
| Signal and Noise study (OLMo 2 target) | 13B | eval-gate | noise window for scaling-law error | final 30 checkpoints, 1000 steps apart | §4.2, footnote 4 | verified 2026-09-14 | stated as a trade-off between sample size and compute cost |
| Signal and Noise study | 150M-1B | eval-gate | metric for small-scale decisions | bits-per-byte instead of the primary metric | §5.3, Fig. 6 | verified 2026-09-14 | 30-task average SNR 10.0 → 31.5; decision accuracy 77.0% → 83.7%; improved for 90.0% of benchmarks |
| Signal and Noise study | 60M-5xC → 1B-5xC | eval-gate | checkpoint averaging | average the final checkpoints on both sides of the comparison | §5.2, Table 1 | verified 2026-09-14 | 30-task decision accuracy 68.9% → 71.3%; 13B prediction error 1.03 → 0.86 |
| DataDecide | 150M predicting 1B | eval-gate | small-scale decision rule | rank recipes at one small size using a continuous likelihood metric (CORRECT PROB) | arXiv:2504.11393v2 §3.1-3.4 | verified 2026-09-14 ([[datadecide]]) | about 80% decision accuracy vs 1B; no scaling-law baseline of 8 exceeds the frontier (Fig. 3) |
| DataDecide | 1B | eval-gate | seed replication at the target scale | 3 full reruns per recipe | §2.1, Table 2 caption | verified 2026-09-14 | run-to-run std reaches 2 pp of accuracy for some recipes on most tasks |
| Miller 2024 (not model-specific) | — | eval-gate | size of a new evaluation | at least 1,000 questions; n ≈ 969 for δ = 3 pp, α = 0.05, power 0.80, ω² = 1/9, no generation variance | arXiv:2411.00640v1 §5, Eq. 9 | verified 2026-09-15 ([[adding-error-bars-evals]]) | the §5 worked example is the stated basis for the recommendation |
| Miller 2024 (Anthropic models) | — | eval-gate | standard error on clustered evaluations | clustered SE; DROP 1.34 vs naive 0.44 (3.05×); MGSM 1.62% vs 0.86% (1.88×) | Table 4 | verified 2026-09-15 | Table 4, described as non-fictional numbers |
| Miller 2024 | — | eval-gate | generations per question K | choose K with E[σ²_i]/K ≪ Var(x); at n = 198, K from 1 to 10 moves MDE 13.2% → 7.5% | §3.1, §5 | verified 2026-09-15 | §5 worked example |
| FormatSpread study | LLaMA-2 7B/13B/70B, Falcon-7B(-Instruct), GPT-3.5-Turbo | eval-gate | number of prompt formats evaluated | 10 sampled formats as a lower bound; 320 formats under Thompson sampling in the GPT-3.5 study | arXiv:2310.11324 §4.1-4.2 | verified 2026-09-15 ([[prompt-format-sensitivity-formatspread]]) | median spread 7.5 points at 10 formats; spread up to 76 points |
| Madaan et al. 2024 | 7B, 10 seeds | eval-gate | MCQ formulation for small-scale ablations | MMLU-Cloze instead of standard MMLU | arXiv:2406.10229v1 §3.3, Table 1 | verified 2026-09-15 ([[benchmark-variance-quantified]]) | monotonicity 0.09 → 0.95; seed std 0.57 → 0.22; Pearson 0.92 with the standard format on 70 fully trained models |
| Madaan et al. 2024 | 7B, 10 seeds | eval-gate | 100-item IRT subsets (tinyBenchmarks) | not recommended for pretraining ablations | §5, Tables 3-4 | verified 2026-09-15 | seed std ARC-C 0.80 → 1.80/1.86, HellaSwag 0.21 → 2.06/2.42; monotonicity falls |

**Starting point for a small general-purpose run.** Every number here comes from a verified row above, under the
conditions its source used. For a checkpoint comparison at 1B-8B: evaluate both checkpoints on identical item ids
and report paired standard errors with the score correlation (Miller §4.2); use the clustered estimator on any
suite with grouped questions and print the cluster count (Tables 3-4); target at least 1,000 questions for a newly
built evaluation that must resolve a 3-point difference (§5, Eq. 9); set K from the generation-variance rule
rather than lowering the temperature (§3.1, §3.3); for pre-training or mid-training ablations read bits-per-byte
and average the final checkpoints instead of reading the last one (Heineman §5.2-5.3) and use cloze formulations
for MCQ tasks at small scale (Madaan §3.3). The seed-replication row (3 reruns at 1B) comes from a pre-training
data study; treat it as a cost reference, not a value transferable to post-training runs, which no source here
measures.

---

## Generalization lens

**(a) What increases breadth.** Measuring more than one thing, and pre-registering what will be measured. The
two-part gate of §6 makes non-regression a condition for shipping rather than a post-hoc check, and the
multiplicity control of §5 stops the memo from being written around whichever slice happened to move. Reporting
across meaning-equivalent formats rather than one ([[prompt-format-sensitivity-formatspread]] §4.2) measures a
capability instead of a capability-under-one-template.

**(b) What causes narrowing.** Three mechanisms, each visible in the sources. Selection: with 20 slices at
α = 0.05 the probability of at least one false positive is 0.642, so a pipeline that ships on "the best slice
improved" drifts toward whichever narrow capability was lucky. Format overfitting: a gate fixed to one template
can be passed by a model that improved on that template, and the rank of two models reverses under another format
with probability about 0.14, with both directions significant in 76% of the 13B-vs-70B reversals (§4.2).
Under-powered non-inferiority: an interval wide enough to contain any regression the general suite could
plausibly show passes every checkpoint, so forgetting accumulates across rounds without failing a gate — the
statistical counterpart of the forgetting slices in [[ch-50]].

**(c) How to measure it for this stage.** Report, for every gated number: n and the cluster count; the estimator;
the score correlation for paired comparisons; K and the sampling temperature; the number of training runs and the
measured or assumed seed σ with its source; the number of prompt formats and the observed spread; and the minimum
detectable effect at the chosen design. When decisions are taken at a small scale and transferred to a large one,
report decision accuracy — the share of model pairs ranked the same at both scales ([[datadecide]] §2.3, Eq. 3;
equal to (Kendall's τ + 1)/2 with ties excluded) — and check the benchmark's SNR at your scale
([[signal-and-noise-eval]] §4.1), because a benchmark whose SNR is low there cannot support the decision however
the interval is computed.

---

## Common mistakes and how to detect them

| Mistake | Observable symptom | Check |
|---|---|---|
| Subtracting two independent intervals to compare checkpoints | The difference interval is far wider than the observed per-item disagreement; no score correlation is logged | Recompute with Eq. 7 and report Corr(s_A, s_B); if it was never logged, the comparison was not paired |
| Ignoring clustering | Suites built from passages or translations (DROP, RACE, QuAC, SQuAD, MGSM) use the same formula as MMLU | Compute Eq. 4; the DROP ratio was 3.05× ([[adding-error-bars-evals]] Table 4). Print cluster counts beside item counts |
| Using SE_Bernoulli on fractional scores | F1, partial credit or per-item pass rates carry a Bernoulli interval | Use Eq. 1; the Bernoulli form is too wide here (§2.1) |
| One training seed treated as the recipe's score | A 1-point difference between separately trained checkpoints is called a result | Compare against a measured seed σ for that benchmark (0.21–2.15 pp, [[benchmark-variance-quantified]] Table 1) or run a second seed |
| Lowering temperature to reduce noise | Evaluation temperature differs from the deployed one and is justified by variance | Both conditions are legitimate to measure; §3.3 shows T=0 can raise Var(x). State which condition the claim is about |
| Pooling K generations as K·n independent items | The n in the interval exceeds the number of questions | The unit is the per-question mean over its K generations (§3.1) |
| One prompt format behind a gate | The harness has a single template per task and reports no spread | Evaluate a fixed set of meaning-equivalent formats; median spread 7.5 points over 10 formats and 14.1% of pairs reverse rank ([[prompt-format-sensitivity-formatspread]] §4.2) |
| Slice shopping | The headline slice was chosen after seeing results | Pre-register target slices and apply Holm across them; 1 − 0.95²⁰ = 0.642 without correction |
| Reading "the interval contains 0" as "no regression" | The no-regression line cites a wide interval and no margin | State δ_NI in advance and test the lower bound against −δ_NI (§6) |
| Judge win rate without swap-and-average | A pairwise win rate comes from one presentation order | Swap and re-judge; GPT-4 changed its verdict on 35.0% of similar pairs ([[judge-llm-bias]] Table 2) |
| Small-scale decision on a benchmark that cannot support it | An ablation is decided on a benchmark near chance at that scale | Check SNR or decision accuracy at your scale ([[signal-and-noise-eval]] App. B.3; [[datadecide]] Fig. 2); MMLU sat at 25.86 against chance 25 after 210B tokens at 7B |
| Pruned item subset used for ablations | A 100-item subset replaces a full benchmark to save compute | Measure seed variance on the subset; it rose by factors of 2 to 10 in [[benchmark-variance-quantified]] Tables 3-4 |

---

## Check your understanding

1. Pairing reduced the interval by a factor of 1.61 in §3.1. What would have to be true of the two checkpoints for
   pairing to give no reduction, and why does that point at something other than a statistical problem?
2. An evaluation of 9,622 questions reports a naive standard error of 0.44 and a clustered one of 1.34. Why is the
   clustered figure the correct one, and how many independent questions is the set worth?
3. A team lowers the sampling temperature to 0 "to remove noise" and the interval narrows. Using
   Var(s) = Var(x) + E[σ²_i], explain how the estimate can still be worse.
4. §1.4 reports that 76% of the 13B-vs-70B rank reversals were significant in both directions. Why does this not
   contradict §2-§3, and what does a significance statement therefore not license?
5. A gate runs 20 non-inferiority tests with no multiple-comparison correction. Why is omitting it defensible
   there but not on the gain side, and what is the real failure mode of that gate?
6. [[ai2-benchmirt]] reports that keeping 10% of items preserves a benchmark's picture; [[benchmark-variance-quantified]]
   reports that 100-item IRT subsets raise seed variance by factors of 2 to 10. Give one account of both that
   requires neither study to be wrong, and state which decision each conclusion applies to.
7. A checkpoint curve shows a step between two evaluation points. Name three mechanisms from this chapter that
   could produce it with no change in capability, and the measurement that separates each.
8. A non-inferiority test fails only because the interval is too wide. Which lever from §4 would you spend compute
   on first, and what would you need to know about the suite to choose it?

---

## Connections

- **ch-50 — Slice Analysis, Forgetting Slices, and Failure Bucketing** (dependency): supplies the slices this
  chapter attaches intervals and multiplicity control to; its forgetting slices are the content of the
  non-inferiority half of the gate.
- **ch-49 — Judge Models: Bias, Calibration, and Judge-Specific Overfitting**: the full treatment of the judge
  noise term introduced in §1.5.
- **ch-51a — Evaluating Agent Generality and Reliability** (next): run-to-run variance for agent tasks, pass@k
  versus pass^k, scaffold sensitivity, and cost-aware gates, where the §1 noise sources grow by another order.
- **ch-52 — Safety Evaluation, Over-Refusal, and Red-Teaming**: the safety and over-refusal suites that enter the
  §6 gate as non-inferiority claims, and where construct validity ([[ai2-benchmirt]]) matters most.
- **ch-53 — Lab: Evaluation Harness with a Held-Out Suite, Forgetting Report, and Perturbation Robustness**:
  implements the estimators, the format perturbations and the memo of this chapter.

---

## Sources

- [[benchmark-variance-quantified]] — 10 same-recipe 7B runs; measured seed variance per benchmark; analytic
  versus bootstrapped intervals; MMLU vs MMLU-Cloze; the negative result on 100-item IRT subsets.
- [[signal-and-noise-eval]] — signal, noise and SNR definitions; SNR versus decision accuracy; subtask filtering,
  checkpoint averaging and bits-per-byte; SNR by benchmark and scale.
- [[adding-error-bars-evals]] — standard errors (CLT, Bernoulli, clustered); paired comparison and its variance
  reduction; repeated generations; the temperature argument; the power formulas used in §4 and §6.
- [[datadecide]] — decision accuracy as a metric; which decisions single-scale small experiments support; 1B seed
  reruns and their reported run-to-run standard deviation.
- [[prompt-format-sensitivity-formatspread]] — measured format spread; rank reversals under meaning-equivalent
  formats; the share of reversals significant in both directions.
- [[emergent-abilities-mirage]] — metric shape and the 1/(test-set size) resolution limit as sources of apparent
  steps in a checkpoint curve.
- [[ai2-benchmirt]] — IRT-based item pruning and held-out item prediction; benchmark construct validity (BBQ,
  WMDP, XSTest).
- [[judge-llm-bias]] — swap-consistency rates used as the judge noise term in §1.5.
- [[reward-model-overoptimization]] — proxy versus gold score under optimization pressure, cited in §6 for the
  limit of monitoring a proxy curve; the transfer to benchmark curves is labelled an Interpretation.
- [[llama-3]] — the report named in [[adding-error-bars-evals]] §2.1 as computing all its standard errors with the
  Bernoulli form.
