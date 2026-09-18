<!-- scope: GSM1k (arXiv:2405.00332, May 2024) — a 1,205-problem human-written parallel set for GSM8k, used to measure benchmark-specific inflation and to correlate it with test-set log-likelihood
     see-also: [[gsm-symbolic]], [[swe-bench-illusion]], [[livecodebench]], [[quantifying-memorization]]
-->

# A Careful Examination of Large Language Model Performance on Grade School Arithmetic
- **Core Insight:** On GSM1k, 1,205 newly written grade-school problems matched to GSM8k in difficulty, the worst models score up to 8 percentage points lower than on GSM8k, and the Phi and Mistral families show this gap for almost every release and scale, while frontier models show minimal difference (Abstract; §5.1, §5.2).
- **Guideline:** When a reported gain on a public benchmark is used as evidence of capability, re-measure on a fresh set built to the same difficulty distribution under one fixed evaluation setting, because holding prompt, shot source, decoding, and answer extraction constant across both sets is what makes the difference readable as benchmark-specific inflation (§4). Otherwise report the score as benchmark-specific and not as general skill.
- **Authors:** Hugh Zhang, Jeff Da, Dean Lee, Vaughn Robinson, Catherine Wu, Will Song, et al. (Scale AI)
- **Year:** 2024 (arXiv v1 2024-05-01; v4 2024-11-22; NeurIPS 2024 Datasets and Benchmarks)
- **URL:** https://arxiv.org/abs/2405.00332
- **Source type:** paper
- **Relevant topics:** benchmark contamination, overfitting audits, fresh parallel test sets, GSM8k, evaluation protocol, memorization proxies

## Abstract
Performance on mathematical-reasoning benchmarks may partly reflect dataset contamination rather than reasoning ability. The authors commission GSM1k, a set of 1,205 grade-school math problems written by human annotators to mirror the style and complexity of GSM8k, and evaluate leading open and closed models on both sets under one evaluation configuration. Accuracy drops of up to 8% appear on GSM1k, several model families show systematic gaps across releases and scales, and a model's probability of generating GSM8k test examples is positively related to the size of its gap (Spearman's r² = 0.36 as printed in the abstract). The authors also find that the most overfit models still solve many novel problems, and conclude that contamination is not the full explanation.

## Key Contributions
- A human-written parallel benchmark for GSM8k with an explicit difficulty-matching procedure and three quality-check layers (§3, §3.1, §3.2).
- A held-back release policy: publication only when three open models from different pre-trained lineages reach 95% on GSM1k, or June 2025, whichever is earlier; a reserve of unreleased items is kept in case of leakage through API evaluation (§3).
- A single fixed evaluation setting applied to 63 models in the standard-prompt table, with prompt, shot-count, and answer-extraction ablations (§4; App. E, F, J, K, L).
- A memorization proxy: per-character log-likelihood of the GSM8k test set, correlated with the GSM8k-minus-GSM1k gap (§5.4, Eq. 1).

## Key Figures/Tables to Study
- Fig. 3 (final winnowing to 1,205 problems to match the GSM8k answer-magnitude distribution); Fig. 4 (models above 70% GSM8k against the line of no overfit); Fig. 5 (gap against per-character log-likelihood); Fig. 12 (pre-GSM8k models GPT-2 and GPT-NeoX-20B); App. F (full results table, standard and alternative prompts).

## Technical Details
**Construction (§3, §3.1, §3.2)**
- 1,205 problems; annotators were shown 3 GSM8k examples and asked for novel problems of similar difficulty; solutions are positive integers requiring only the four basic operations; no language model was used at any point (§3).
- Three review layers: trusted-annotator review, an independent second solve with mismatches discarded, and a Scale quality audit. Of 2,108 initial problems, 1,419 passed the second solve and 1,375 passed the audit (§3.1).
- Difficulty target N was the count of calculator tags in GSM8k solutions; the requested distribution matched GSM8k's. The final step discards problems so that the answer-magnitude distributions match; 1,205 survive (§3.2, Fig. 3).
- The authors state the two sets are "only highly similar, but not identically distributed" (§3).

**Difficulty checks (§3.2.1–§3.2.3)**
- Human distinguishability: 19 annotators picked the single GSM1k item out of five (four from GSM8k) in 21.83% of 1,205 attempts; chance is 20% (§3.2.1).
- Human solve rate under 15 minutes of time pressure, 14 annotators: 4.07 ± 0.93 problems on GSM8k and 4.36 ± 1.11 on GSM1k, standard deviations. The authors read this as GSM1k being comparable or slightly easier (§3.2.2).
- Models released before GSM8k (GPT-2, GPT-NeoX-20B) show minimal difference between the two sets (§3.2.3, Fig. 12).

**Evaluation setting (§4)**
- A fork of EleutherAI's LM Evaluation Harness with default settings except the maximum generated tokens raised from 256 to 1000, because the default truncated some chains of thought (§4).
- Both sets use the same prompt: 5 examples drawn at random from the GSM8k **train** split (§4, App. D).
- Open models at temperature 0, vLLM where compatible, otherwise HuggingFace; proprietary models through LiteLLM at provider defaults; API queries between 2024-04-16 and 2024-07-10 (§4).
- Scoring extracts the last numeric answer in the response; a manual-extraction ablation on a subset of models did not change the findings (§4, App. J).
- Ablations: an alternative prompt with non-GSM8k shots and different answer phrasing (App. E), and varying number and source of n-shot examples (App. K, L). Absolute accuracies vary; the overfitting trends hold (§4).

**Results (Abstract; §5; App. F)**
- The standard-prompt table in App. F lists 63 models sorted by GSM8k-minus-GSM1k difference, with a Z-score and p-value from a two-tailed two-proportion Z-test (App. F).
- Selected standard-prompt rows (Diff, GSM8k, GSM1k, Z, p): Yi-6B-Chat 0.080, 0.437, 0.357, 4.135, 0.000; math-shepherd-mistral-7b-rl 0.072, 0.826, 0.754, 4.488, 0.000; phi-2 0.063, 0.566, 0.504, 3.167, 0.001; Meta-Llama-3-8B-Instruct 0.062, 0.752, 0.690, 3.532, 0.000; Phi-3-medium-128k-instruct 0.044, 0.869, 0.825, 3.103, 0.001; Phi-3-mini-4k-instruct 0.040, 0.788, 0.748, 2.385, 0.009; gpt-4-turbo 0.003, 0.898, 0.895, 0.270, 0.394; gpt-4o 0.002, 0.931, 0.929, 0.219, 0.413; claude-3-opus-20240229 −0.022, 0.802, 0.824, −1.421, 0.922; gemini-1.5-flash-preview-0514 −0.038, 0.797, 0.835, −2.507, 0.994 (App. F).
- Under the alternative prompt (App. E table), the same model can change verdict: Phi-3-mini-4k-instruct shows Diff 0.007, GSM8k 0.807, GSM1k 0.800, Z 0.474, p 0.318, and Phi-3-medium-128k-instruct shows Diff −0.005 (App. F, alternative-prompt table).
- Lesson 1 (§5.1): "Several families of models, including the Phi and Mistral families of models, show systematic tendencies to perform stronger on GSM8k compared to GSM1k for almost every release and scale of models"; Yi, Xwin, Gemma, and CodeLlama show it to a lesser extent.
- Lesson 2 (§5.2): frontier and near-frontier models, including the proprietary Mistral Large, perform similarly on both sets.
- Lesson 3 (§5.3): Phi-2 drops 6% between the two sets and still solves over half of GSM1k, comparable to Llama2-70B, which has over 25× as many parameters.
- Lesson 4 (§5.4, Eq. 1, Fig. 5): the per-character log-likelihood of generating the GSM8k test set, `(1/c) Σ_i log p(x_i | x_<i)` with c the number of characters, has Spearman rank correlation 0.36 with the gap (p = 0.03); every percentage point of gap is associated with an increase of 1.2 × 10⁻² in that log-likelihood; Pearson r² = 0.26 and Kendall τ = 0.29 are also reported, with the note that Pearson r² is not ideal because the fit is not linear. Outliers (Mixtral-8x22b and Mixtral-8x22b-Instruct at the extremes of log-likelihood with similar overfit; Math-Shepherd-Mistral-7B-RL) lead to the conclusion that contamination is not the full story.
- Models near the top of the Open LLM Leaderboard perform substantially worse on GSM1k, which the authors read as evidence of Goodhart's law (§4).

## Recipe ledger
| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| GSM1k audit, all 63 table rows | 0.1B–70B+ and API models | eval-gate | prompt; shot source | 5-shot, examples drawn at random from the GSM8k **train** split, identical for both sets | arXiv:2405.00332v4 §4 | verified 2026-09-18 | App. E, K, L: alternative prompt and varied shot number/source preserve the overfitting trends |
| GSM1k audit | same | eval-gate | harness; max generated tokens | LM Evaluation Harness fork; 1000, raised from the default 256 | §4 | verified 2026-09-18 | raised because the default truncated some chains of thought (§4) |
| GSM1k audit | same | eval-gate | decoding; serving | temperature 0 for open models, vLLM where compatible; API models at provider defaults via LiteLLM, queried 2024-04-16 to 2024-07-10 | §4 | verified 2026-09-18 | no ablation reported |
| GSM1k audit | same | eval-gate | answer extraction | last numeric answer in the response | §4 | verified 2026-09-18 | App. J: manual extraction on a subset does not change the findings |
| GSM1k set | n/a | eval-gate | set size; construction | 1,205 items from 2,108 written (1,419 after second solve, 1,375 after audit); no language model used | §3, §3.1, §3.2 | verified 2026-09-18 | §3.2.1–3.2.3: distinguishability 21.83% vs 20% chance, solve rates 4.07 ± 0.93 vs 4.36 ± 1.11, pre-2021 models show minimal difference |
| GSM1k release | n/a | eval-gate | release condition | earlier of three open models from different pre-trained lineages above 95% on GSM1k, or June 2025; a reserve of passing items is withheld | §3 | verified 2026-09-18 | no ablation reported |

## Findings relevant to generality
- A drop on a matched fresh set bounds how much of a benchmark score is benchmark-specific, but does not identify the cause: §5.4 reports only a rank correlation of 0.36 between the gap and test-set likelihood, and the authors name benchmark-like training data and benchmark-based model selection as alternatives to verbatim leakage.
- Overfitting and capability are separable: the most overfit models still solve novel problems of the same difficulty (§5.3).
- The audit is domain-bounded: GSM1k is grade-school arithmetic with positive-integer answers and the four basic operations (§3), so a gap there does not transfer to other capability claims.

## Connections
- [[gsm-symbolic]] — the perturbation counterpart: same items re-instantiated from templates, instead of new items.
- [[swe-bench-illusion]] — the same audit logic applied to agentic coding through memorization probes.

## Verification
- Created on 2026-09-18 from https://arxiv.org/abs/2405.00332 (arXiv v4, 2024-11-22).
- Corrections to the previous card version: none (no card existed; ch-21, ch-34, ch-47a, and ch-53 carried chapter excerpts instead).
- Removed as unsupported by the source: none.
- Chapter claims not found in the source: none. Two source-internal inconsistencies that chapters flag are reproduced here rather than resolved. (1) The abstract writes "Spearman's r² = 0.36" while §5.4 writes "Spearman's rank correlation of 0.36" and gives Pearson r² separately as 0.26; §5.4 then refers back to "the r² = 0.36 value". (2) App. F states the p-values come from a two-tailed test, but the printed pairs are upper-tail values: the Phi-3-mini-4k-instruct row prints Z 2.385 with p 0.009, and P(Z > 2.385) = 0.0086 while the two-tailed value would be 0.017. Chapter arithmetic in ch-34 that treats these p-values as upper-tail is consistent with the printed table, not with its caption.
- Not reported by the source: the GSM8k test-set size used by the harness (checked §4 and App. F); per-model GSM1k item-level results; any evaluation outside grade-school arithmetic.
- Errata in v4: App. G notes that a previous version of the paper included some questions from a nonfinal version of GSM1k and prints a corrected table of 50 example items.
