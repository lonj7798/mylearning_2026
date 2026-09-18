<!-- scope: DataDecide suite (25 pretraining corpora x 14 sizes x 3 seeds, 4M-1B) and how accurately small-scale experiments, scaling-law fits, and proxy metrics predict which pretraining data wins at 1B
     deps: [[olmes]], [[task-scaling-model-ladders]]
     see-also: [[signal-and-noise-eval]], [[dclm]], [[overtraining-downstream-scaling]], [[emergent-abilities-mirage]]
-->

# DataDecide: How to Predict Best Pretraining Data with Small Experiments
- **Core Insight:** Across 25 pretraining corpora, ranking models of a single small size (for example 150M parameters) picks the corpus that wins at the 1B target scale in about 80% of pairwise comparisons, and none of 8 scaling-law baselines exceeds the compute-to-decision-accuracy frontier of such single-scale rankings (Abstract; §3.2, Fig. 3).
- **Guideline:** When choosing between pretraining corpora with small models, rank them with a continuous likelihood metric (character-normalized probability of the correct answer) on benchmarks that separate recipes at that scale, such as MMLU and ARC, because at small scales these metrics are better or equivalent predictors of 1B accuracy rankings and make MBPP and HumanEval decisions go from trivial to 80% decision accuracy (§3.3-3.4, Fig. 4, Fig. 6). Otherwise, for tasks such as SocialIQA or BoolQ, small experiments give unreliable decisions at the scales tested (§3.1, Fig. 2).
- **Authors:** Ian Magnusson, Nguyen Tai, Ben Bogin, David Heineman, Jena Hwang, Luca Soldaini, et al. (Allen Institute for AI; University of Washington; University of Pennsylvania)
- **Year:** 2025 (arXiv v1 2025-04; ICML 2025, PMLR 267)
- **URL:** https://arxiv.org/abs/2504.11393 (official blog: https://allenai.org/blog/datadecide, 2025-04-15)
- **Source type:** paper
- **Relevant topics:** pretraining data selection, small-scale proxy experiments, decision accuracy, scaling-law extrapolation, proxy metrics, evaluation noise, benchmark choice

## Abstract
Pretraining a large model on each candidate dataset is too expensive, so data decisions are made with smaller experiments. The paper asks which benchmarks and which decision methods applied to small-scale results best predict the dataset that produces the best large model. It releases DataDecide, a suite of models pretrained on 25 corpora that differ in source, deduplication, and filtering, with up to 100B tokens, sizes up to 1B parameters, and 3 random seeds. Ranking models at one small size (for example 150M) predicts the best 1B models in about 80% of comparisons. No scaling-law method among 8 baselines exceeds the compute-decision frontier of single-scale ranking. Using continuous likelihood metrics as proxies in small experiments makes MMLU, ARC, HellaSwag, MBPP, and HumanEval more than 80% predictable at 1B with 0.01% of the target compute.

## Key Contributions
- An open suite of 1,050 models (25 data recipes × 14 model sizes × 3 seeds) with more than 30K checkpoints, the corpora, and evaluations (§1, §2.1).
- Decision accuracy: a metric that scores a prediction method by the share of corpus pairs whose winner it predicts correctly, measured against 1B results averaged over 3 seeds (§2.3, Eq. 3).
- A compute-versus-decision-accuracy comparison of single-scale ranking and 8 scaling-law variants (§3.1-3.2, Fig. 1-3).
- A comparison of five proxy metrics (accuracy and four likelihood-based metrics) for predicting accuracy rankings at the target scale (§2.5, §3.3, Table 3, Fig. 4).
- An explanation of per-task differences in decision accuracy through run-to-run noise and spread across recipes (§3.4, Fig. 5).

## Key Figures/Tables to Study
- **Fig. 1 (right):** decision accuracy against percent of target compute for all model sizes and intermediate checkpoints; stars mark scaling-law predictions.
- **Fig. 2:** the same curve per OLMES task; shows which benchmarks are predictable at small compute.
- **Fig. 3 and Table 4:** 8 scaling-law variants; decision accuracy and 1B prediction error.
- **Fig. 4 and Fig. 6:** proxy metrics per task; code tasks become predictable with CORRECT PROB, math tasks do not.
- **Fig. 5:** noise (std over seeds) versus spread (std across recipes) at 150M.
- **Table 1 and Table 2:** the 25 recipes and the 14 model configurations.

## Technical Details
- **Suite:** 25 recipes × 14 scales × 3 seeds = 1,050 models (§2.1); sizes 4M to 1B parameters, up to 100B tokens (§1, Table 2). Previous suites compared 2 or 6 recipes (§1).
- **Recipes (Table 1):** Dolma1.7 (original, no code, no math/code, no Reddit, no Flan); Dolma1.6++; C4; FineWeb-Pro; FineWeb-Edu; Falcon RefinedWeb; Falcon+Dolma1.7 Common Crawl with quality filters (top 10% or 20%, reproduced or original DCLM classifier, or a classifier retrained on pre-release Tulu-v3); DCLM-Baseline with 7 filter variants (original, top 7% plus FineWeb-edu score 2+ or 3+, FineWeb-edu top 3% or 10%, reproduced DCLM top 10% or 20%); DCLM-Baseline/Dolma1.7 mixes with λ ∈ {25%, 50%, 75%}.
- **Token-to-parameter ratio:** 100, described as 5× the Chinchilla-optimal ratio (§2.1; §5).
- **Configurations:** OLMo model ladder; global batch size and learning rate from heuristics in Porian et al. (2024); layers, width, heads, and MLP size chosen by OLMo developers per scale (§2.1; Table 2). Per-size values: [[datadecide-recipe]].
- **Seeds:** all 1B models have 3 full reruns; for other sizes the second and third seeds stop at 25% of the target compute budget (§2.1; Table 2 caption). Standard deviation between 1B 5×C runs reaches 2 percentage points of accuracy for some recipes on most tasks (§2.1).
- **Decision accuracy (Eq. 3):** (1/|P|) Σ_{(A,B)∈P} 1[sign(ŷ_A − ŷ_B) = sign(y_A − y_B)]. P is the set of recipe pairs, y the observed 1B performance averaged over 3 seeds, ŷ the prediction. The paper states it is nearly equivalent to Kendall's τ but ranges from 0 to 1 (§2.3).
- **Compute budget:** FLOPs = 6ND (N parameters, D tokens); efficiency is %C = c/C × 100%, the experiment budget divided by the target budget (§2.3).
- **Scaling-law baseline:** L(C) = A/C^α + E, then Acc(L) = a/(1 + e^{−k(L−L0)}) + b (Eq. 1-2), following Bhagia et al. (2024). Eq. 1 is fit on final checkpoints only, with the final loss averaged over the last 10% of checkpoints; Eq. 2 is fit on all checkpoints (§2.2). Size subsets used: {s1..sk | 3 ≤ k ≤ 14} and {sk..s14 | 2 ≤ k ≤ 11} (§3.2).
- **Evaluation:** 10 OLMES multiple-choice tasks (MMLU, HellaSwag, ARC-C, ARC-E, PIQA, CommonsenseQA, SocialIQA, OpenBookQA, BoolQ, WinoGrande), macro-averaged; targets use cloze-formulation accuracy with per-task normalization; all items of each split are used instead of OLMES subsampling (§2.4).
- **Proxy metrics (Table 3):** CORRECT PROB (mean probability of the correct continuation), MARGIN (correct minus most likely incorrect), NORM CORRECT PROB (correct probability renormalized over the answer set), TOTAL PROB (sum over all answer options), ACCURACY; each normalized per token or per character, per character by default (§2.5).
- **Compute to decision accuracy:** roughly log-linear across the aggregate of 10 tasks; intermediate checkpoints decide as well as compute-equivalent final checkpoints (§3.1, Fig. 1).
- **Per task:** ARC Easy is predictable with 5 orders of magnitude less compute; BoolQ exceeds trivial decision accuracy only with intermediate checkpoints of the target runs; HellaSwag, SocialIQA, and WinoGrande are insensitive until a compute threshold, then rise roughly log-linearly (§3.1, Fig. 2). The blog states "MMLU and ARC Easy are highly predictable with as little as 4 orders of magnitude less compute" (blog, "What did we learn?").
- **Scaling laws:** the 2- and 3-parameter variants are among the top decision accuracies of the 8, and none exceeds single-scale ranking (§3.2, Fig. 3). 1B prediction error, relative/absolute: 5.6/2.6 (3-parameter with helper points and >50% checkpoints), 6.5/3.1 (3-parameter), 42.9/42.3 (3-parameter single step), 230.8/65.4 (5-parameter) (Table 4).
- **Crossovers:** single-scale ranking cannot predict recipes whose trends cross between the small and target scale; the authors report that observed scaling trends cross frequently but that noise and true crossovers are hard to separate (§3.2).
- **Proxy metrics result:** CORRECT PROB or TOTAL PROB give decision accuracy at least as good as any other metric for most small scales; for tasks where CORRECT PROB and TOTAL PROB are flat with scale, ACCURACY and the metrics that penalize probability on incorrect answers (MARGIN, NORM CORRECT PROB) tend to overtake them in the last order of magnitude below target compute (§3.3, Fig. 4).
- **Noise and spread:** at 150M, MMLU combines low run-to-run noise with high decision accuracy; ARC Easy spreads recipes widely; CORRECT PROB gains often align with lower noise or wider spread (§3.4, Fig. 5).
- **Code and math:** with CORRECT PROB, MBPP and HumanEval decision accuracy goes from trivial to 80%; Minerva and GSM8K stay near trivial, but exceed 80% if the target metric is also CORRECT PROB (§3.4, Fig. 6).
- **Cost:** about 820K H100 GPU hours for the pretraining experiments (Impact Statement).

## Recipe ledger
Per-size rows (batch, LR, width, heads, layers, steps, tokens for all 14 sizes): [[datadecide-recipe]].

| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| DataDecide 150M (all 25 recipes) | 151.9M non-embedding | pretrain-stable | global batch (sequences); peak LR; tokens | 192; 4.2e-03; 15.0B | arXiv:2504.11393v2 App. A Table 2 | verified 2026-09-14 | set by ladder heuristics (Porian et al. 2024); no ablation reported |
| DataDecide 1B (all 25 recipes) | 1176.8M non-embedding | pretrain-stable | global batch (sequences); peak LR; tokens; steps | 704; 2.1e-03; 100.0B; 69,369 | App. A Table 2 | verified 2026-09-14 | same |
| DataDecide, all 14 sizes | 4M-1B | pretrain-stable | sequence length; MLP ratio; tokens per parameter | 2024 (as printed); 8; 100 | Table 2 caption; §2.1 | verified 2026-09-14 | 100 chosen as typical overtraining (§2.1, §5); no ablation |
| DataDecide, all 14 sizes | 4M-1B | pretrain-stable | optimizer, warmup, LR schedule shape, weight decay | not reported | checked §2, App. A-C, blog | not reported | none |
| DataDecide decision gate | 4M-750M → 1B | eval-gate | target ranking rule | mean OLMES cloze accuracy over 3 full 1B seeds | §2.3-2.4 | verified 2026-09-14 | not applicable |

## Findings relevant to generality
- **Benchmark choice affects whether a data comparison transfers across scale.** The compute needed for a given decision accuracy "depends heavily on task": MMLU and ARC are cheaper to predict than HellaSwag, and SocialIQA is difficult to predict at all scales tested (§1 recommendations; Fig. 2). Result (single study).
- **Scope limits:** one token-to-parameter ratio (100); 14 configurations up to 1B; 25 recipes may not represent future recipes; only multiple-choice cloze tasks; tasks for larger target scales would have to be selected differently (§2.4, §5).

## Connections
- [[signal-and-noise-eval]]: same group; uses DataDecide models to explain decision accuracy with a signal-to-noise ratio.
- [[task-scaling-model-ladders]]: source of the two-step scaling-law baseline and the model ladder (Bhagia et al. 2024).
- [[resolving-scaling-discrepancies]]: Porian et al. (2024), the batch-size and LR heuristics behind Table 2.
- [[olmes]]: evaluation standard for the 10 target tasks.
- [[dclm]], [[dolma]], [[fineweb]], [[c4]]: corpora among the 25 recipes; DCLM also ranks single-scale experiments (§4).
- [[emergent-abilities-mirage]]: cited motivation for continuous proxy metrics (§2.5).
- [[overtraining-downstream-scaling]]: two-step prediction of downstream performance (Gadre et al. 2024).
- [[pythia]], [[paloma]]: earlier suites with 2 and 6 data recipes (§1, §4).
- [[data-mixing-laws]]: related work that optimizes mixing proportions with scaling laws (§4).
- [[olmo-2]]: OLMo model ladder lineage for the configurations.

## Verification
- Created on 2026-09-14 from https://arxiv.org/abs/2504.11393 (v2, 2025-07-13; v1 2025-04-15) and https://allenai.org/blog/datadecide (2025-04-15).
- Audit claims not found in the source: "the approach behind OLMo 3's OlmoBaseEval Easy suite" (not in paper or blog); "continuous likelihood proxies beat discrete accuracy" (source says "better or equivalent", §3.3); "small-model accuracy sits at chance" for code (source reports trivial decision accuracy, not task accuracy, Fig. 6).
- Source-internal difference: ARC Easy "5 orders of magnitude less compute" (paper §3.1) versus MMLU and ARC Easy "as little as 4 orders of magnitude" (blog).
