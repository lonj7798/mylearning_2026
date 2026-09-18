<!-- scope: statistical treatment of language-model evaluations — standard errors (CLT, Bernoulli, clustered), variance reduction, paired model comparison, and power analysis for sizing an eval
     see-also: [[signal-and-noise-eval]], [[ai2-benchmirt]], [[llama-3]]
-->

# Adding Error Bars to Evals: A Statistical Approach to Language Model Evaluations
- **Core Insight:** Treating eval questions as an independent draw from an unseen super-population yields closed-form standard errors for eval scores; on two evals with clustered questions measured on Anthropic models, the clustered standard error is 3.05× the naive one on DROP and 1.88× on MGSM (§2.2, Table 4), so unclustered intervals on reading-comprehension evals are too narrow.
- **Guideline:** When reporting an eval score, report the number of questions and the CLT standard error beside the mean; when comparing two models on the same questions, compute the standard error on the question-level paired differences rather than on the two summary statistics, because paired differences remove the shared question-difficulty term and, in the paper's worked case at score correlation 0.5, reduce estimator variance by 1/3 (§4.2). When questions are drawn in groups, use the clustered form (Eq. 4, Eq. 8). Do not lower the sampling temperature to reduce variance (§3.3).
- **Authors:** Evan Miller (Anthropic)
- **Year:** 2024 (arXiv v1 2024-11; dated November 4, 2024 on the title page)
- **URL:** https://arxiv.org/abs/2411.00640
- **Source type:** paper
- **Relevant topics:** evaluation statistics, standard errors, clustered sampling, variance reduction, resampling, next-token probabilities, paired comparison, power analysis, minimum detectable effect, eval reporting practice

## Abstract
Evaluations are experiments, but the eval literature has largely ignored the experiment-analysis and experiment-planning literature of other sciences. The article shows researchers with some statistical training how to analyze eval data. Conceptualizing eval questions as drawn from an unseen super-population, it presents formulas for analyzing eval data, for measuring differences between two models, and for planning an eval experiment, and makes specific recommendations for running and reporting evals so as to minimize statistical noise and maximize informativeness (Abstract).

## Key Contributions
- Five stated recommendations: CLT standard errors; clustered standard errors when questions come in groups; variance reduction by resampling answers and by using next-token probabilities; inference on question-level paired differences rather than population summaries; power analysis to decide whether an eval can test the hypothesis of interest (§1).
- Standard-error formulas for the independent case (Eq. 1), the Bernoulli special case (Eq. 2), and the clustered case (Eq. 4, derived in App. A).
- Variance decomposition `Var(µ̂) = Var(s)/n = (Var(x) + E[σ²_i])/n` separating question-selection variance from per-question sampling variance, and the two levers that act only on the second term (§3).
- Paired and paired-clustered standard errors for model comparison (Eq. 7, Eq. 8) and the equivalent form from single-model standard errors and the score correlation (§4.2).
- A sample-size formula (Eq. 9) and its inverse, the minimum detectable effect (Eq. 10), with cluster-adjusted versions in App. C.
- A concrete criticism of published practice: the Llama 3 report computes all its standard errors with the Bernoulli form even where scores are fractional (for example F1), which is conservative there, and does not cluster reading-comprehension evals, which makes those intervals too narrow (§2.1, §2.2).

## Key Figures/Tables to Study
- **Table 1-2:** the motivating fictional three-eval comparison, then the same table with question counts and standard errors added.
- **Table 3:** suggested reporting format with cluster counts beside question counts.
- **Table 4:** the only non-fictional numbers in the paper — clustered versus naive standard errors on DROP, RACE-H, and MGSM for Anthropic models.
- **Table 5:** suggested presentation of pairwise differences, standard errors, 95% intervals, and per-item score correlations.

## Technical Details
- **Model of a score:** `s_i = x_i + ε_i`, with `x_i` the conditional mean of question `i` and `σ²_i = Var(ε_i)` the conditional variance; `µ = E[s]` is the super-population mean, estimated by `µ̂ = s̄` (§2.1).
- **CLT standard error:** `SE_CLT = sqrt(Var(s)/n) = sqrt((1/(n−1)) Σ_i (s_i − s̄)² / n)` (Eq. 1); Bernoulli case `SE = sqrt(s̄(1 − s̄)/n)` (Eq. 2); `CI_95% = s̄ ± 1.96 × SE_CLT` (Eq. 3).
- **On bootstrapping:** "we regard bootstrapping as unnecessary unless a complicated sampling scheme or estimator is being used" (§2.1). The Inspect framework is named as computing `SE_CLT` correctly in its `stderr()` metric; OpenAI evals is named as bootstrapping (§2.1).
- **Clustered standard error:** `SE_clustered = [SE²_CLT + (1/n²) Σ_c Σ_i Σ_{j≠i} (s_{i,c} − s̄)(s_{j,c} − s̄)]^{1/2}` (Eq. 4). It interpolates between "each cluster is one observation" (perfect within-cluster correlation) and the unclustered case (no correlation) (§2.2). Named clustered evals: DROP, QuAC, RACE, SQuAD, and multilingual evals such as MGSM, where the same question is translated into many languages (§2.2).
- **Measured cluster effect (Anthropic models, real numbers):** DROP `SE_clustered` 1.34 vs `SE_CLT` 0.44, ratio 3.05; RACE-H 0.51% vs 0.46%, ratio 1.10; MGSM 1.62% vs 0.86%, ratio 1.88 (§2.2, Table 4).
- **Resampling:** with `K` answers per question, `Var(s_i) = σ²_i / K`; once `E[σ²_i]/K ≪ Var(x)`, further `K` has little effect (§3.1). Worked example with binary scores and uniformly distributed difficulty `x ~ U[0,1]`: `Var(x) = 1/12`, `E[σ²_i] = 1/6`, giving `Var(µ̂|K) = Var(µ̂|K=1) × (1 + 2/K)/3` — variance reduced by 1/3 at K=2, 1/2 at K=4, 5/9 at K=6, with an upper limit of 2/3 (§3.1). Pooling all `K·n` answers as independent draws is inconsistent (§3.1).
- **Next-token probabilities:** for evals without chain of thought where the answer is the first token, scoring the correct-token probability gives `s_i = x_i = p_i`, `ε_i = 0`, so `Var(µ̂) = Var(p)/n`; in the same uniform-difficulty example this reaches the 2/3 reduction that resampling only approaches (§3.2).
- **Temperature:** in a single-token true/false example, moving from T=1 with `x ~ U[0,1]` to T=0 rounds the distribution to Bernoulli(1/2), raising `Var(x)` from 1/12 to 1/4; in a second example with `x_{T=1} ~ U[1/3, 1]`, T=0 also shifts the expected score from 2/3 to 3/4 and raises `Var(x)` from 1/27 to 3/16 (§3.3). The recommendation is to use next-token probabilities when available, otherwise choose `K` with `E[σ²_i]/K ≪ Var(x)`, and in neither case to change temperature for variance reduction (§3.3).
- **Unpaired comparison:** `µ̂_{A−B} = µ̂_A − µ̂_B`, `SE_{A−B} = sqrt(SE²_A + SE²_B)`, with the 95% interval (Eq. 5) and z-score (Eq. 6). This form works even when the two reports used non-identical random subsets of questions (§4.1).
- **Paired comparison:** `SE_{A−B,paired} = sqrt((1/(n−1)) Σ_i (s_{A−B,i} − s̄_{A−B})² / n)` (Eq. 7), equivalently `sqrt(SE²_A + SE²_B − 2 SE_A SE_B Corr(s_A, s_B))`; the reduction over the unpaired variance is `2 Cov(x_A, x_B)/n` (§4.2). Worked case: next-token-probability scores uniform on [0,1] for both models with correlation 0.5 give `Var(s_A) = Var(s_B) = 1/12`, `Cov(x_A, x_B) = 1/24`, and a relative variance reduction of 1/3 (from 1/6 to 1/9) (§4.2). The paper recommends the paired form "wherever practicable" (§4.2). Clustered paired form: Eq. 8.
- **Power analysis:** with `ω² = Var(x_A) + Var(x_B) − 2Cov(x_A, x_B)`, `σ²_A = E[σ²_{A,i}]`, `σ²_B = E[σ²_{B,i}]`, and `K_A`, `K_B` answers per question, `n = (z_{α/2} + z_β)² (ω² + σ²_A/K_A + σ²_B/K_B) / δ²` (Eq. 9) and `δ = (z_{α/2} + z_β) sqrt((ω² + σ²_A/K_A + σ²_B/K_B)/n)` (Eq. 10). Cluster-adjusted versions are in App. C.
- **Worked power examples:** with `σ²_A = σ²_B = 0`, `ω² = 1/9`, `δ = 0.03`, `α = 0.05`, `β = 0.20`, `n ≈ 969`, from which the paper concludes that "new evals should contain at least 1,000 questions in order to have good signaling ability" (§5). With `σ²_A = σ²_B = 1/6`, `ω² = 1/9`, `n = 198`, `α = 0.05`, `β = 0.20`, raising `K_A = K_B` from 1 to 10 lowers the minimum detectable effect from 13.2% to 7.5% (§5).

## Findings relevant to generality
- The super-population framing makes an eval score an estimate of an underlying skill rather than a property of the specific question set, which is the assumption under which a score is read as evidence about capability beyond the items tested (§2).
- The cluster result bounds how much a passage-based or translation-based suite can say: DROP's effective precision is about one third of what the naive interval reports (§2.2, Table 4), so held-out-suite comparisons on such evals need the clustered form before a difference is called real.
- Paired analysis is the form that applies to before-and-after comparisons of two checkpoints on a fixed suite, and it is the form under which the paper's own motivating example reverses conclusion: Galleon beats Dreadnought on MATH significantly, while the HumanEval and MGSM differences are indistinguishable from noise (§4.2, Table 5).
- The paper does not address pass@k, pass^k, agentic or multi-turn evaluation, contamination, or judge-model scoring. Any use of its estimators for those settings is an extension not made by the source.

## Connections
- [[signal-and-noise-eval]] — measures benchmark noise empirically across checkpoints; this card supplies the sampling-theory estimators for a single comparison.
- Madaan et al., "Quantifying Variance in Evaluation Benchmarks" (arXiv:2406.10229) — cited in the paper as [15] for the "highest number is best" reporting practice; no library card on 2026-09-18.
- [[llama-3]] — the report the paper criticizes for computing all standard errors with the Bernoulli form and for unclustered reading-comprehension intervals (§2.1, §2.2).
- [[ai2-benchmirt]] — item-level modeling of benchmark items, a different route to the same question of what an item set measures.

## Verification
- Checked on 2026-09-18 against: https://arxiv.org/abs/2411.00640 (arXiv:2411.00640v1 [stat.AP], 1 Nov 2024; title page dated November 4, 2024)
- Created on 2026-09-18 from https://arxiv.org/abs/2411.00640 (arXiv v1).
- Corrections to the previous card version: none (no prior card; chapters ch-32f, ch-46a, ch-47a, ch-51 carried excerpt files noting "no library card exists on 2026-09-15").
- Removed as unsupported by the source: none.
- Chapter claims not found in the source: `ch-51/read.md` L118 attributes the decomposition `Var(s) = Var(x) + E[σ²_i]` to §3.1; the equation appears in the §3 preamble, and §3.1 is the resampling subsection. The decomposition itself is correct as stated. No other chapter claim checked (ch-32f L254, ch-46a L239/L243/L352, ch-47a L28/L449/L451/L457/L550, ch-51 L20/L25/L69/L172/L185/L202/L224/L238/L267/L318/L509/L562) contradicts the source; the ch-46a L243 use of correlation 0.7 is labeled there as the chapter's own choice rather than the paper's 0.5, which matches the source.
- Not reported by the source: any empirical standard errors for models other than the unnamed Anthropic models in Table 4; the identity of those models; estimators for pass@k or pass^k; any treatment of multi-turn or agentic evals.
