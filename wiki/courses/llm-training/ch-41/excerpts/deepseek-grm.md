---
chapter: ch-41
course: llm-training
phase: read
excerpt_of: arXiv:2504.02495v3 (Inference-Time Scaling for Generalist Reward Modeling)
source_url: https://arxiv.org/abs/2504.02495
created_at: "2026-09-15"
revised: "2026-09-15 (generality revision; written from the primary source because no library card exists yet)"
---

# Excerpt: DeepSeek-GRM and Self-Principled Critique Tuning (SPCT)

Used by [[read]] §4.3 (RewardBench value for Skywork-Reward-Gemma-2-27B), §4.4, the Generalization lens, and the Recipe. Authors: Zijun Liu, Peiyi Wang, Runxin Xu, Shirong Ma, Chong Ruan, Peng Li, et al. (DeepSeek-AI; Tsinghua University). arXiv v1 2025-04; checked against v3 (2025-09-25) on 2026-09-15.

## Method
- **Pointwise generative RM.** Given a query and n ≥ 1 responses, the model generates principles, then critiques, then a pointwise score per response; one format covers single, paired, and multiple responses (§2.1, §3.1).
- **Rejective fine-tuning (cold start).** Trajectories are sampled N_RFT times per example; a trajectory is correct if the ground-truth best response gets a strictly higher score than every other response (or, for n = 1, the score equals the label) (Eq. 10). Incorrect trajectories are rejected, and examples with all N_RFT trajectories correct are removed as too easy. Optional hinted sampling appends "The best response is: Response j"; hinted trajectories "sometimes take shortcuts in the generated critique" (§3.2).
- **Rule-based RL.** GRPO with reward +1 when the extracted scores satisfy Eq. 10 and −1 otherwise (Eq. 11); no format reward; a larger KL coefficient is used to keep format and avoid biases (§3.2).
- **Inference-time scaling.** Sample k trajectories with shuffled response order and sum pointwise scores (Eq. 14); a meta RM (pointwise scalar, binary cross-entropy on Eq. 10 correctness) keeps the top k_meta trajectories for voting (§4).

## Training settings (App. C.1)
- Data: 1256K RFT examples (1070K general instruction + 186K rejective-sampled) and 237K RL examples, from internal data and MATH, UltraFeedback (partly re-labelled), OffsetBias, Skywork-Reward-Preference-80K-v0.2, HelpSteer2-Preference.
- Rejective sampling with DeepSeek-v2.5-0905, N_RFT = 3.
- Gemma-2-27B models: RFT LR 5e-6, batch 1024; RL LR 4e-7, batch 512; 900 steps each; 128 A100 GPUs.
- KL coefficient grid {0.00, 0.01, 0.02, 0.08}; β = 0.08 most stable for 27B; smaller values collapsed on some subsets (RewardBench Chat, RMB Harmlessness). β = 0.002 for 16B. Group size G = 4.
- Inference-time scaling temperature 0.5; greedy otherwise (App. D.1).

## Results (Table 2; accuracy of picking the best response)
| Model | RewardBench | PPE Pref. | PPE Correct. | RMB | Overall |
|---|---|---|---|---|---|
| Skywork-Reward-Gemma-2-27B (reported) | 94.1 | 56.6 | 56.6 | 60.2 | 66.9 |
| Nemotron-4-340B-Reward (reported) | 92.0 | 59.3 | 60.8 | 69.9 | 70.5 |
| GPT-4o (reported) | 86.7 | 67.1 | 57.6 | 73.8 | 71.3 |
| DeepSeek-BTRM-27B | 81.7 | 68.3 | 66.7 | 57.9 | 68.6 |
| DeepSeek-PairRM-27B | 87.1 | 65.8 | 64.8 | 58.2 | 69.0 |
| DeepSeek-GRM-27B-RFT | 84.5 | 64.1 | 59.6 | 67.0 | 68.8 |
| DeepSeek-GRM-27B | 86.0 | 64.7 | 59.8 | 69.0 | 69.9 |
| DeepSeek-GRM-27B, Voting@32 | 88.5 | 65.3 | 60.4 | 69.7 | 71.0 |
| DeepSeek-GRM-27B, MetaRM Voting@32 | 90.4 | 67.2 | 63.2 | 70.3 | 72.8 |

- Scalar and semi-scalar RMs score higher on PPE Correctness than all generative RMs but fail on other benchmarks; the authors describe this as domain bias (§5.2).
- Ablations (Table 4, greedy overall): full 69.9; without principle generation 67.5; without rejective sampling 68.7 (after RL; 66.1 before RL); RFT model 68.8; without hinted sampling 68.0; without non-hinted sampling 67.4; without general instruction data 63.3.
- Voting@32 with the 27B GRM is comparable to the 671B MoE model on RewardBench; DeepSeek-R1 on a 300-sample subset scores below the 236B RFT model (§5.2, Fig. 4).

## Limits
The GRM is evaluated as a judge; integration into online policy RL is future work (§7). Automated principles and critiques may be unfaithful or amplify biases (Ethics Statement).
