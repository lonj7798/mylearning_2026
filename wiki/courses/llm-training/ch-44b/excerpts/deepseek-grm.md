---
chapter: ch-44b
course: llm-training
phase: read
excerpt_of: arXiv:2504.02495v3 (no library card at the time of writing; chapter-local verified extract)
source_url: https://arxiv.org/abs/2504.02495
created_at: "2026-09-15"
---

# Excerpt: Inference-Time Scaling for Generalist Reward Modeling (DeepSeek-GRM, SPCT)

- **Authors:** Zijun Liu, Peiyi Wang, Runxin Xu, Shirong Ma, Chong Ruan, Peng Li, Yang Liu, Yu Wu (DeepSeek-AI; Tsinghua University)
- **Year:** 2025 (arXiv v1 2025-04; v3 2025-09-25)
- **Source type:** paper
- **Used in:** [[read]] §2.4, Recipe

## Method
- Pointwise generative reward model: given a query and n ≥ 1 responses, the model generates principles, then critiques, then one pointwise score per response; the same format covers single, paired, and multi-response inputs (§2.1, §3.1).
- Self-Principled Critique Tuning (SPCT) = rejective fine-tuning + rule-based online RL. In rejective fine-tuning, N_RFT = 3 trajectories are sampled per example (generator DeepSeek-v2.5-0905); a trajectory is kept only if the ground-truth best response receives a strictly higher score than every other response; examples that are always correct are dropped (§3.2, Eq. 10).
- RL stage: GRPO with reward +1 when the extracted scores satisfy the ground-truth ordering and −1 otherwise, no format reward, and a larger KL coefficient to keep the output format (§3.2, Eq. 11). KL grid {0.00, 0.01, 0.02, 0.08}; β = 0.08 is most stable for the 27B model and smaller values collapse on some subsets (App. D).
- Inference-time scaling: sample k trajectories with shuffled response order and sum the pointwise scores; a meta reward model keeps the top k_meta trajectories for voting (§4).
- Settings (App. C.1): Gemma-2-27B base; RFT LR 5e-6, batch 1024; RL LR 4e-7, batch 512; 900 steps each; 128 A100 GPUs; 1256K RFT examples and 237K RL examples.

## Results (Table 2; accuracy of selecting the best response)
| Model | RewardBench | PPE Pref. | PPE Correct. | RMB | Overall |
|---|---|---|---|---|---|
| Skywork-Reward-Gemma-2-27B | 94.1 | 56.6 | 56.6 | 60.2 | 66.9 |
| Nemotron-4-340B-Reward | 92.0 | 59.3 | 60.8 | 69.9 | 70.5 |
| DeepSeek-GRM-27B | 86.0 | 64.7 | 59.8 | 69.0 | 69.9 |
| DeepSeek-GRM-27B, Voting@32 | 88.5 | 65.3 | 60.4 | 69.7 | 71.0 |
| DeepSeek-GRM-27B, MetaRM Voting@32 | 90.4 | 67.2 | 63.2 | 70.3 | 72.8 |
Scalar and semi-scalar reward models score highest on PPE Correctness and fall behind on the other benchmarks, which the authors describe as domain bias (§5.2). Ablations (Table 4, greedy overall): full 69.9; without principle generation 67.5; without general instruction data 63.3.

## Limits stated by the authors
The GRM is evaluated as a judge; "integrating GRMs into online RL pipelines" is listed as future work (§7). The authors also report that they did not find an effective way to incentivize long-horizon reward generation for generalist reward modeling (§6).

## Verification
- Checked on 2026-09-15 against https://arxiv.org/abs/2504.02495 (v3, 2025-09-25): Abstract, §2–§7, Tables 2, 4, 5, App. C.1, D.
- Not reported: policy-model results from using DeepSeek-GRM as an RL reward; judging latency inside an RL loop.
