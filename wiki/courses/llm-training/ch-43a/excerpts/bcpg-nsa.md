---
chapter: ch-43a
course: llm-training
phase: read
excerpt_of: primary source (no library card at wiki/raw-data/llm-training/papers/bcpg-nsa.md on 2026-09-15)
source_url: https://arxiv.org/abs/2505.14403
source_version: arXiv v4 (2025-09-15); v1 2025-05-20
created_at: "2026-09-15"
---

# Excerpt: Unearthing Gems from Stones — Policy Optimization with Negative Sample Augmentation for LLM Reasoning (Yang, Ye, Jiang, Hu, Deng, Li, Jiang; CAS, BIT, StepFun)

Facts used by [[read]], read in the arXiv v4 PDF on 2026-09-15.

## Preliminary experiment (§3, Table 1)
- Two 19,000-sample SFT sets with identical prompts from an open-source R1 dataset: SFT-pos holds only responses with correct final answers, SFT-neg only responses with incorrect final answers. Table 1 reports AIME24 / AIME25: Qwen2.5-14B-Instruct 10.00 / 13.33; trained on SFT-pos 52.75 / 39.42; trained on SFT-neg 41.67 / 34.00. (The table's first row names Qwen2.5-14B-Instruct while §3 names Qwen2.5-14B-Base as the model being trained.)
- Conclusions stated: training on wrong answers alone gives a large gain over the base model, because long-CoT responses contain self-reflection and alternative approaches; it still trails SFT-pos "by 5-10% across benchmarks" because of the flawed steps they also contain.

## Method (§4)
- Step segmentation of negative responses, then per-step correctness labels from an LLM judge (Claude-3.7 with a taxonomy of reasoning errors plus two context rules: Error Propagation and Error Termination) and a discriminative PRM with threshold λ; a step counts as correct only when both judges agree (Eqs. 3-5, consensus filtering).
- Objective (Eq. 6): `J = (1/G) Σ_i (1/|y_i|) [ Σ_j log π_θ(y_{i,j}|·) β_{i,j} (r_i − r̄) − (τ/2) (log π_θ(y_i|x)/π_ref(y_i|x))² ]`, where r̄ is the group mean reward and τ the behaviour-constraint factor.
- Token weight (Eq. 7): `β_{i,j} = β` when token y_{i,j} lies in a step labelled correct inside an incorrect response, and 1 otherwise. β ∈ [−1, 1]: β < 1 reduces the penalty on those tokens and β < 0 reverses it into an increase. β = 1 recovers vanilla BCPG, which penalises every token of a negative response equally.

## Data and setup (§5.1, App. C)
- DeepSeek-R1-Distill-Qwen-14B generates 32 responses per question at temperature 0.7 with a 22,000-token limit; math_verify labels final answers; questions whose responses are all correct or all incorrect are removed. Final offline set: 2,069 questions, 66,208 samples, 14,896 negative samples, 470M tokens (Table 2).
- Offline RL hyperparameters (Table 8): max learning rate 5e-7, min 2.5e-7, 8 epochs, batch 512, sequence length 32k, 64 × H800/H100, about 14 hours for BCPG-NSA; τ = 1e-3; DPO baseline β_DPO = 0.5; RFT uses max learning rate 2.5e-6 and 6 epochs (Table 7).

## Results (Table 3, β = 0.5)
| Method | AIME24 | AIME25 | MATH500 | LiveCodeBench | Average |
|---|---|---|---|---|---|
| DS-R1-14B | 70.58 | 49.58 | 91.80 | 52.40 | 66.09 |
| RFT | 62.92 | 50.75 | 92.20 | 52.69 | 64.64 |
| DPO | 69.83 | 49.00 | 92.88 | 51.61 | 65.83 |
| TOPR | 69.75 | 52.67 | 93.12 | 52.90 | 67.11 |
| GRPO-offline | 70.50 | 50.90 | 92.08 | 52.97 | 66.61 |
| BCPG | 70.50 | 52.00 | 93.98 | 53.26 | 67.44 |
| BCPG-NSA | 72.17 | 54.42 | 93.36 | 53.84 | 68.45 |

- Every offline RL method that uses negatives beats RFT, and RFT falls 7.5 points below the starting model on AIME24; the authors attribute this to the model memorising its own weaker answers for questions already used in distillation (§5.2).
- Annotation ablation (Tables 4-5, β = 0.5): PRM-only labels 38M correct and 80M incorrect tokens, LLM-only 65M and 53M, consensus 26M and 92M; consensus gives the best results (AIME24 72.17, AIME25 54.42) although it mines the fewest correct tokens.
- β ablation (§6.2, Figure 2): as β falls the average of AIME24 and AIME25 rises and then declines; BCPG-NSA beats vanilla BCPG "across a wide range of β values, including at the relatively aggressive setting of β = −0.5".
