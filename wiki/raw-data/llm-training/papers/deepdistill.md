<!-- scope: a-m-team difficulty-graded long-CoT distillation data (3.34M queries, about 40M responses from three samplers), pass-rate and coefficient-of-variation selection, and two-stage SFT of Qwen2.5-32B/72B base models
     deps: [[am-deepseek-r1-distilled-1-4m]]
     see-also: [[distillation-source-matters]], [[am-thinking-v1]], [[light-r1]], [[omnithought]]
-->

# DeepDistill: Enhancing LLM Reasoning Capabilities via Large-Scale Difficulty-Graded Data Training
- **Core Insight:** SFT of the Qwen2.5-72B base model on about 5M DeepSeek-R1 responses, selected by a per-category verify-score threshold and the coefficient of variation (CV) of verify scores, reaches 79.2 pass@1 on AIME2024 (Table 1); in a 72B ablation, lowering the SFT learning rate from 8e-5 to 8e-6 lowers AIME2024 to 72.5 and LiveCodeBench from 63.8 to 60.2 (§4.4.2).
- **Guideline:** When running long-CoT SFT from a base model at 32B-72B scale, include a learning rate near 8e-5 in the sweep rather than only lower rates such as 8e-6, because the authors' 72B run at 8e-6 lost 6.7 AIME2024 points (§4.4.2, Fig. 4). When adding an annealing stage on the highest-CV data, evaluate code separately, because in this paper it raised 32B AIME2024 from 75.8 to 77.9 but lowered LiveCodeBench from 64.2 to 61.7 (Table 2).
- **Authors:** Xiaoyu Tian, Sitong Zhao, Haotian Wang, Shuaiting Chen, Yiping Peng, Yunjie Ji, et al. (a-m-team)
- **Year:** 2025 (arXiv v1 2025-04-24; v3 2025-05-13; preprint, "under review")
- **URL:** https://arxiv.org/abs/2504.17565 (dataset card: https://huggingface.co/datasets/a-m-team/AM-DeepSeek-Distilled-40M)
- **Source type:** paper (with dataset card)
- **Relevant topics:** reasoning distillation, difficulty grading, pass rate, coefficient of variation, data selection, long-CoT SFT learning rate, annealing, verification pipelines, decontamination

## Abstract
The authors build a reasoning dataset of about 3.34 million unique queries and about 40 million distilled responses produced by several models over several passes. They use pass rate and the coefficient of variation to select training data. They report a "training pattern shift": reasoning-focused SFT on base models needs higher learning rates. With the selected data, a base model reaches 79.2% pass@1 on AIME2024. Data and methods are released.

## Key Contributions
- A difficulty-graded dataset in which each query has 4 responses from each of DeepSeek-R1-Distill-Qwen-1.5B, DeepSeek-R1-Distill-Qwen-7B, and DeepSeek-R1, with per-response verify scores and per-model pass rates (§2.3-2.4; dataset card).
- Difficulty measured from pass rates of models of different strength, motivated by the bias of difficulty ratings from a single judge model (§1; dataset card).
- A selection rule (Alg. 1) that keeps queries with high CV of verify scores and discards stable queries, except a random 50% of "other" and multi-turn queries (§3.2).
- Two-stage SFT: Stage I on about 5M samples, Stage II annealing on the highest-CV data (§4.1-4.2).
- A 72B learning-rate ablation supporting the higher-LR claim (§4.4.2).

## Key Figures/Tables to Study
- **Alg. 1 (§3.2):** the full selection procedure.
- **Figs. 2-3:** Stage I and Stage II category shares at instance level and answer-token level.
- **Table 1:** 32B/72B results against DeepSeek-R1-Distill, QwQ-32B, and DeepSeek-R1. **Table 2:** Stage I vs Stage II.
- **Fig. 4:** 72B training loss at high and low LR. **Fig. 5:** AIME2024 score, generation stop ratio, and average generated length over training steps.
- **App. A, Figs. 7-8:** query category shares and per-model pass-rate distributions.

## Technical Details
- **Query pool:** math (OpenR1-Math-220k, Big-Math-RL-Verified, data_ablation_full59K, NuminaMath, MetaMathQA, DeepMath-103K, AIME_1983_2024 without AIME2024, and others), code (PRIME, DeepCoder, KodCode, codeforces_cots, opencoder, AceCode-87K, and others), science, instruction following (including AutoIF generated with Qwen2.5-72B-Instruct), multi-turn, and other (§2.1). The dataset card states 30 sources in total; the first listed is OpenHermes-2.5 with 762,544 queries (card "Instruction sources").
- **Query processing:** exact dedup; removal of queries with a high Unicode ratio (threshold not reported), empty or incomplete queries, and queries with URLs or tables; exact-match and bge-m3 semantic decontamination against AIME2024 at similarity > 0.9; result about 3.34M queries (§2.2). §2.3 instead says "approximately 4 million preprocessed queries".
- **Category counts (card "Data Type Distribution"):** math 828,639; code 489,363; science 91,637; instruction follow 76,520; other 1,850,985. Paper shares: math 24.8%, code 14.7%, science 2.7%, IF 2.3%, other 55.5% (App. A.2, Fig. 7).
- **Sampling:** 4 responses per query per model, about 40M responses; only the DeepSeek-R1 responses are used for training and analysis (§2.3, footnote 2). Temperature and top-p are not reported.
- **verify_score:** math uses Math-Verify, then Qwen2.5-7B-Instruct as a second binary check when Math-Verify marks a response wrong (Eq. 1); code uses sandbox-fusion tests on the first 10 open-source test cases (Python stdin/stdout and assert, C++ stdin/stdout), score = passed/total (Eq. 2); science uses Qwen2.5-7B-Instruct similarity to ground truth on [0, 5]; IF uses the ifeval validator with extra constraints generated by Qwen2.5-72B-Instruct, score = mean pass over constraints (Eq. 3); multi-turn and other use Decision-Tree-Reward-Llama-3.1-8B, score = (coherence + correctness + helpfulness)/12, each on [0, 4] (Eq. 4) (§2.4.1).
- **Pass thresholds:** verify_score > 0.99 for math, code, IF; > 4.99 for science; > 0.7 for multi-turn and other. pass_rate = (1/n) Σ 1(verify_score_i > threshold), n = 4 (§2.4.2, Eq. 5). The dataset card writes pass_rate as the mean of verify_score over n = 4 without the indicator.
- **Quality filters:** perplexity from "our previously trained 32B model" [60], removal above 20; removal of 20-token strings occurring more than 20 times; even turn count for multi-turn data; required think content and answer content (§2.5). The card describes its ppl field as "calculated by the 7b model".
- **CV:** CV = σ/μ over the n verify scores of one query, where μ is the mean and σ the standard deviation (§3.1, Eq. 6, which divides by n). Paper example: [0.9, 0.1, 0.7, 0.3, 0.5] has μ = 0.5, σ ≈ 0.32, CV ≈ 0.63, while [0.5 ×5] has CV = 0. The example σ values match the n−1 sample standard deviation (0.316), not the 1/n form (0.283) (derived from §3.1).
- **Alg. 1 (Stage I):** discard the query if its maximum verify_score is below the category threshold; if CV > CV_t, keep only responses above the threshold; otherwise keep 50% of "other" or multi-turn queries at random and discard the rest. CV_t differs by category and its values are not reported (§3.2, footnote 4). Output: "5 million" samples (§3.2).
- **Stage II selection:** same categories, verify_score threshold 0.99, only queries above CV_t with the highest CV, one random response per query, "nearly 200k" samples (§3.2).
- **Stage I composition (Fig. 2):** code 1,397,676 (25.7%), math 1,582,693 (29.1%), other 1,930,647 (35.5%), science 402,407 (7.4%), IF 122,667 (2.3%); answer tokens code 7,824.5M (40.7%), math 7,686.3M (40.0%), other 2,817.3M (14.6%), science 770.7M (4.0%), IF 134.7M (0.7%).
- **Stage II composition (Fig. 3):** code 69,725, math 84,883, other 67,932, science 20,000, IF 10,000; answer tokens 535.6M, 581.0M, 63.0M, 56.7M, 8.3M (math 46.7%, code 43.0% of tokens).
- **Evaluation:** pass@1 on AIME2024 (30 questions), LiveCodeBench, and GPQA-Diamond (198 questions) (§4.3). Sampling temperature, samples per question, and the LiveCodeBench version are not reported.
- **Results (Table 1):** Ours-Distill-32B 75.8 / 66.3 / 64.2 and Ours-Distill-72B 79.2 / 65.7 / 63.8 (AIME2024 / GPQA-Diamond / LiveCodeBench); DS-Distill-32B 72.6 / 62.1 / 57.2; DS-Distill-70B 70.0 / 65.2 / 57.5; QwQ-32B 79.5 / 65.9 / 63.4; DeepSeek-R1 79.8 / 71.5 / 65.9.
- **Stage II (Table 2):** 32B AIME2024 75.8 → 77.9, GPQA-Diamond 66.3 → 66.9, LiveCodeBench 64.2 → 61.7.

## Recipe ledger
| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| Ours-Distill-32B and Ours-Distill-72B, Stage I (Qwen2.5-32B / Qwen2.5-72B base) | 32B, 72B | distill-SFT | samples; answer tokens | "5 million"; Fig. 2 counts sum to 5,436,090 samples and 19,233.5M answer tokens | arXiv:2504.17565v3 §3.2; Fig. 2 | verified 2026-09-14 (sums derived) | no comparison against unselected data reported |
| same | 32B, 72B | distill-SFT | peak LR; schedule | 8e-5; cosine, linear warmup over first 5% of steps, decay to 0 | §4.1 | verified 2026-09-14 | 72B at 8e-6: AIME2024 72.5 vs 79.2, LCB 60.2 vs 63.8 (§4.4.2, Fig. 4); ablation at 72B only |
| same | 32B, 72B | distill-SFT | epochs; packing; max length; global batch | 1; packing on; 32k tokens; 64 | §4.1 | verified 2026-09-14 | no ablation reported |
| Ours-32B (Stage II) | 32B | distill-SFT | samples; answer tokens | "nearly 200k" (prose) vs Fig. 3 counts summing to 252,540; 1,244.6M answer tokens | §3.2 vs Fig. 3 | conflict (prose vs figure) | Table 2 compares Stage I and Stage II |
| same | 32B | distill-SFT | init; LR; schedule | best Stage I 32B checkpoint; 8e-6; cosine, 5% warmup, decay to 0 | §4.2 | verified 2026-09-14 | Table 2: AIME2024 +2.1, LCB −2.5; not a single-variable ablation |
| same | 32B | distill-SFT | epochs; packing; max length; global batch | 2; no packing; 32k tokens; batch not reported | §4.2 | verified 2026-09-14; batch not reported | no ablation reported |
| both stages | 32B, 72B | distill-SFT | optimizer, betas, weight decay, loss masking, seeds, compute, sampling temperature for distillation and evaluation | not reported | checked §2-§5, App. A, dataset card | not reported | none |

## Findings relevant to generality, negative feedback, distillation
- **Distillation setup:** stage is SFT from base models. Prompts come from about 30 open datasets, 55.5% "other" by count (Fig. 7). The teacher is DeepSeek-R1; the 1.5B and 7B distilled models only supply difficulty labels (§2.3). Quality control is per-category verification, a PPL < 20 filter, n-gram repetition removal, and structural checks (§2.4-2.5).
- **Narrowing risk reported by the authors:** Stage II's higher-difficulty data lowered LiveCodeBench by 2.5 points; the authors state the code mixing ratio and hyperparameters need further tuning (§4.4.2). Result (single study, one run).
- **Measurement limits:** decontamination is described only against AIME2024 (§2.2.2). The evaluation has three reasoning benchmarks and no instruction-following or chat benchmark, although 55.5% of queries are in the "other" category (§4.3; Fig. 7).
- **Format adaptation:** the generation stop ratio on AIME2024 starts low and rises with training, which the authors attribute to base models not being trained on QA format (§4.4.2, Fig. 5).
- **Negative samples:** responses below the verify threshold and low-CV queries are discarded (negative marginal value). No negative is used as gradient. The release keeps all 12 responses per query with scores and both paper and card name DPO and GRPO as possible uses (§1; dataset card summary and "Limitation and Usage Limits").

## Connections
- [[am-deepseek-r1-distilled-1-4m]]: earlier a-m-team dataset; reference [60], whose 32B model computes the PPL filter.
- [[distillation-source-matters]]: same team; keeps this paper's 8e-5 LR and verifiers and changes only the teacher.
- [[am-thinking-v1]]: later a-m-team 32B reasoning model.
- [[light-r1]]: cited as prior difficulty-based data selection for long-CoT training (§1, ref [50]).
- [[omnithought]]: another CoT dataset with difficulty annotations.
- [[s1]], [[openr1]], [[tulu-3-sft-mix]], [[openhermes-2-5]]: query sources used here (§2.1).
- [[deepseek-r1]]: teacher model.

## Verification
- Created on 2026-09-14 from https://arxiv.org/abs/2504.17565 (v3, 2025-05-13; v1 2025-04-24) and the AM-DeepSeek-Distilled-40M dataset card (fetched 2026-09-14).
- Audit claims not found in the source: global batch 64 for Stage II (§4.2 gives no batch size); "30+ open datasets" (card says "A total of 30 data sources"); "(Beike)" affiliation (not stated in this paper; stated in arXiv:2505.14464 footnote 1).
- Internal inconsistencies recorded: 3.34M vs "approximately 4 million" queries (§2.2.2 vs §2.3); Stage II "nearly 200k" vs 252,540 in Fig. 3; CV example uses n−1 while Eq. 6 uses n; PPL model 32B (paper) vs 7B (card); pass_rate as mean indicator (paper) vs mean verify_score (card). The card states that only a subset of the complete dataset is open-sourced and that ground_truth labels may contain errors.
