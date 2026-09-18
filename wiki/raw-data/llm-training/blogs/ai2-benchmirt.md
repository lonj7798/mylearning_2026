<!-- scope: Ai2 blog (2026-09) introducing BenchMIRT, a two-dimensional item response theory model fit on 100 open-weight LLMs x 16 reasoning and safety benchmarks, used to audit what each benchmark and subset measures and to prune benchmarks to their most informative items
     deps: [[olmo-3]]
     see-also: [[signal-and-noise-eval]], [[olmes]], [[xstest]], [[harmbench-data]], [[multi-prompt-evaluation]]
-->

# BenchMIRT: What are LLM benchmarks actually measuring?
- **Core Insight:** A two-dimensional item response theory model fit without benchmark labels on 100 open-weight LLMs, 16 benchmarks, and more than 34K questions recovered a safety ability and a general-reasoning ability, and showed that BBQ (a safety benchmark) correlates 0.85 with general reasoning and -0.06 with safety (blog, "What BenchMIRT finds each benchmark measures" table).
- **Guideline:** When an evaluation suite averages benchmarks or subsets that are labelled with one construct (for example "safety"), check item-level or subset-level alignment before reading the average as that construct, because in this analysis BBQ, WMDP, the WildJailbreak benign set, and the HarmBench copyright subset aligned with general reasoning rather than safety, and XSTest split between the two abilities (blog table and "What BenchMIRT reveals" section).
- **Authors:** Ai2 (blog byline). Companion technical report: Malia Morgan, Vasudha Varadarajan, Faeze Brahman, Maarten Sap (Allen Institute for AI; Carnegie Mellon University).
- **Year:** 2026 (blog published 2026-09-01; no arXiv version linked from the blog)
- **URL:** https://allenai.org/blog/benchmirt (report: https://allenai.org/papers/benchmirt; code: https://github.com/allenai/BenchMIRT)
- **Source type:** official blog (reliability: official, Ai2 built and ran the method)
- **Relevant topics:** evaluation, construct validity, item response theory, safety vs capability measurement, benchmark pruning, held-out item prediction

## Summary
The post introduces BenchMIRT, a method for auditing LLM benchmarks at the level of individual prompts. It builds on Item Response Theory (IRT), a psychometrics model in which each item has a difficulty and a discrimination (how well it separates stronger from weaker test takers) and each test taker has an ability. Earlier Ai2 work (Fluid Benchmarking) used single-dimensional IRT; BenchMIRT uses multidimensional IRT (MIRT), so one item can depend on more than one ability. The model was fit on results from 100 LLMs across 16 benchmarks: 6 general-reasoning benchmarks (including MMLU-Pro, GPQA, MATH, BBH) and 10 benchmarks from the Olmo 3 safety suite (including HarmBench, StrongReject, WildJailbreak, BBQ, WMDP, XSTest). The model was not told which benchmark measures which ability; it recovered safety and general reasoning as the two dominant dimensions, and repeated fits from scratch recovered the same two dimensions. The post then reports which benchmarks align with their stated goal, shows that pruning to the most discriminative items keeps the ranking picture, and reports held-out item prediction accuracy. It closes with limitations and a dual-use risk.

## Key Contributions
- An unsupervised two-ability decomposition (safety, general reasoning) of 16 benchmarks that is stable across repeated fits ("Finding the signals inside a benchmark").
- A per-benchmark audit table: Pearson correlation of each benchmark score with each ability score across 100 open-weight LLMs, with a verdict per benchmark.
- Subset-level findings: WildJailbreak benign vs harmful prompts, XSTest over-refusal vs safety items, HarmBench copyright vs standard and contextual items.
- Item pruning: 10% of items generally preserved the model-strength picture; 50% often matched the full benchmark more closely.
- Held-out item prediction: 79% correct vs 70% for a per-benchmark-average baseline.

## Key Figures/Tables to Study
- **"What BenchMIRT finds each benchmark measures" table:** the two correlations and the verdict for all 16 benchmarks.
- **HarmBench item difficulty/discrimination plot:** the copyright subset separates from standard and contextual items. Caption note: independent runs may flip axis signs; in the plotted run a lower Dimension 0 score means higher safety ability.

## Technical Details
Sign convention. In the table, safety correlations for jailbreak and harmful-content benchmarks are negative (for example WildJailbreak -0.90*). The blog marks these benchmarks "Aligned with stated benchmark goal", and its HarmBench caption states that in this run lower Dimension 0 corresponds to higher safety ability. A negative safety value therefore means alignment with safety. Asterisks mark p < 0.01 (table caption).

| Benchmark | Stated use | r (general reasoning) | r (safety axis) | Blog verdict |
|---|---|---|---|---|
| MMLU-Pro | reasoning | 0.97* | -0.21 | aligned |
| BBH | reasoning | 0.94* | -0.20 | aligned |
| GPQA | reasoning | 0.81* | -0.12 | aligned |
| IFEval | reasoning | 0.72* | -0.34* | aligned |
| MATH | reasoning | 0.70* | -0.41* | aligned |
| MuSR | reasoning | 0.67* | 0.00 | aligned |
| WildJailbreak | safety | 0.14 | -0.90* | aligned; benign set (250 of 2,250 items) leans reasoning |
| JailbreakTrigger | safety | 0.27* | -0.91* | aligned |
| Do-Anything-Now | safety | 0.20 | -0.88* | aligned |
| HarmBench | safety | 0.32* | -0.90* | aligned except copyright subset, which leans reasoning |
| StrongReject | safety | -0.16 | -0.84* | aligned |
| WildGuardTest | safety | 0.40* | -0.87* | aligned |
| XSTest | safety | 0.46* | -0.53* | split, consistent with its even over-refusal/safety split |
| ToxiGen | safety | 0.40* | -0.32* | weak on both; saturated, 92% average score |
| BBQ | safety | 0.85* | -0.06 | tracks reasoning |
| WMDP | safety | -0.89* | 0.21 | runs opposite to reasoning; no significant safety correlation |

- BBQ: a low BBQ score "may partly reflect difficulty understanding or reasoning through certain questions, rather than safety behavior alone" (blog, "What BenchMIRT reveals").
- WMDP: stronger general reasoning is associated with lower WMDP scores because the benchmark counts refusing or failing to provide dangerous dual-use knowledge as the desired response (same section).
- Pruning procedure: items ranked by how well they distinguish stronger from weaker models while keeping a mix of easy and hard items; 10% kept "generally preserved nearly the same picture"; 50% "often matched the full benchmark's measure of those capabilities even more closely" ("Doing more with fewer questions").
- Held-out prediction: 79% correct on whether a model answers a held-out question correctly, vs 70% for assuming per-question performance equals the model's benchmark average (same section).
- Ranking trade-off: for ranking models by predicted performance on randomly held-out items, the benchmark average "performs slightly better than BenchMIRT" ("What this could mean").

Method details from the companion report (checked, not in the blog):
- Model: two-parameter, two-dimensional MIRT, P(U_ij = 1 | θ_i, a_j, b_j) = 1 / (1 + exp(-a_j · (θ_i - b_j))); θ_i = ability vector of LLM i, a_j = discrimination vector of item j, b_j = difficulty vector of item j, U_ij = 1 if LLM i answers item j correctly (report §2.2, Eq. 3).
- Items: 34,301 in total; items with mean accuracy above 95% or below 5% across models removed, leaving 29,574 (report §2.3). Fit with py-irt, 5,000 epochs, initial learning rate 0.2, decay 0.9999 (report §2.3).
- Reasoning results come from Open LLM Leaderboard v2 item-level logs; safety outputs were generated by the authors for the same 100 models (report §2.3).
- Stability: ten independent runs, Tucker congruence ϕ from .882 to .998, dimension means .907 and .947 (report §3.1.4).
- Prediction (10-fold CV over model-item pairs): micro-average accuracy .79 for BenchMIRT vs .70 for benchmark average; macro-average rank correlation .81 vs .89 (report Table 1).
- 3-D and 4-D MIRT gain 1 point macro-average accuracy over 2-D, while 2-D gains 24 points over 1-D (report §3.1.7).

## Findings relevant to generality
- The measured structure depends on the benchmark set: safety and reasoning emerged for these 16 benchmarks, and "a different mix of evaluations could surface different underlying capabilities" (blog, limitations).
- All models used were released by March 2025, so behaviour on newer LLMs was not tested (blog, limitations).
- A benchmark grouped under one capability can mostly measure another (BBQ, WMDP), and averaging subsets with different alignments (XSTest, WildJailbreak) hides that difference (blog table).
- Dual-use risk stated by the authors: the same item estimates could be used to remove the most informative safety questions and produce a weaker evaluation that an unsafe model could pass (blog, "What this could mean").

## Connections
- [[olmo-3]] — source of the 10-benchmark safety suite used here.
- [[xstest]], [[harmbench-data]], [[wildguard-data]] — benchmarks whose subsets the audit separates.
- [[mmlu-pro]], [[ifeval]] — reasoning-side benchmarks in the fit.
- [[signal-and-noise-eval]], [[olmes]], [[multi-prompt-evaluation]], [[benchmark-variance-quantified]] — other work on benchmark reliability and variance.
- [[leaderboard-illusion]] — another source on how leaderboard aggregates can mislead.
- [[shallow-safety-alignment]], [[finetuning-compromises-safety]] — safety-capability interaction during training.

## Verification
- Created on 2026-09-14 from https://allenai.org/blog/benchmirt (blog dated 2026-09-01); method details cross-checked in the linked report https://allenai.org/papers/benchmirt (PDF, undated version fetched 2026-09-14).
- Audit claims not found in the source: none. Note: the audit's "BBQ loads 0.85 / -0.06" values are Pearson correlations between ability scores and benchmark scores, not IRT loadings.
- Not reported in the blog: authors (in the report only), the list of 100 models (report App. G), per-benchmark pruning numbers (report Figs. 7-9).
