<!-- scope: DeepSeek-GRM / SPCT (arXiv:2504.02495) — pointwise generative reward model that writes principles and critiques then scores 1–10; rejective fine-tuning + rule-based online RL; inference-time scaling by sampling k and voting with a meta RM
     deps: [[generative-reward-models]], [[grpo]]
     see-also: [[rubrics-as-rewards]], [[rlcf-checklists]], [[master-rm-one-token]], [[judge-llm-bias]], [[rl-design-guide]]
-->

# Inference-Time Scaling for Generalist Reward Modeling (DeepSeek-GRM, SPCT)
- **Core Insight:** A pointwise generative RM that first generates its own principles and critiques and then a 1–10 score per response can be improved at inference by sampling it k times and summing scores: DeepSeek-GRM-27B goes from 67.9 (Voting@1) to 70.6 (Voting@8), and to 72.8 at Voting@32 when a scalar meta RM keeps the better half of samples, above GPT-4o's 71.3 overall on the paper's RM benchmarks (Tables 2–3).
- **Guideline:** When a reward model must cover open-ended domains and the grading budget allows several judge calls per response, use a generative pointwise RM with sampled principles, shuffle response order per sample, and aggregate k samples with a meta-RM filter, because this gained +4.9 overall at k = 32 for the 27B model (Table 3) and the authors report the 27B model with 32-sample voting comparable to a 671B MoE (Fig. 4); scalar RMs scored unevenly across benchmarks (BTRM-27B 66.7 on PPE Correctness but 57.9 on RMB), which the authors call biased results.
- **Authors:** Zijun Liu, Peiyi Wang, Runxin Xu, Shirong Ma, Chong Ruan, Peng Li, Yang Liu, Yu Wu (DeepSeek-AI; Tsinghua University)
- **Year:** 2025 (arXiv:2504.02495; v3 read, dated 25 Sep 2025)
- **URL:** https://arxiv.org/abs/2504.02495
- **Source type:** paper
- **Relevant topics:** generative reward models, principles/critiques, inference-time scaling of graders, voting, meta RM, judge training

## Summary
SPCT (Self-Principled Critique Tuning) trains a GRM in two stages. Rejective fine-tuning: sample trajectories (principles, critiques, scores) and reject those whose predicted reward is wrong, and also prompts where all N_RFT samples are correct. Then rule-based online RL with GRPO: outcome reward +1 if the GRM's scores pick the correct response, −1 otherwise; no format reward, larger KL coefficient (§3.2). At inference, k samples are drawn and scores summed; a meta RM (pointwise scalar RM on Gemma-2-27B trained with binary cross-entropy) selects k_meta = k/2 samples (§4).

## Key Contributions
- Argument for pointwise generative RMs: one format scores single, paired or multiple responses (§2.1).
- SPCT training: rejective fine-tuning + rule-based RL on the RM (§3.2).
- Inference-time scaling with voting and meta-RM guidance (§4, Table 3).

## Key Figures/Tables to Study
- Table 2 (overall across Reward Bench, PPE, RMB), Table 3 (Voting@k), Fig. 4 (inference vs model scaling).

## Technical Details
**Overall scores (Table 2).** DeepSeek-BTRM-27B (scalar) 68.6; CLoud-Gemma-2-27B (semi-scalar) 68.7; DeepSeek-GRM-27B-RFT 68.8; DeepSeek-GRM-27B 69.9; Voting@32 71.0; MetaRM Voting@32 72.8; GPT-4o 71.3; Nemotron-4-340B-Reward 70.5.

**Meta-RM ablation at k = 32.** k_meta = 1: 71.5; k_meta = 8: 72.7; k_meta = 16: 72.8.

**Hinted sampling.** Appending the best response as a hint during RFT produced shortcut trajectories (§3.2).

**Limitations stated.** Efficiency and some specific tasks remain hard; no effective way found to incentivize long-horizon reward generation; risk of unfaithful principles and critiques (Ethics Statement).

## Verification
- Read arXiv HTML v3, first 100K of 176K characters, on 2026-11-24. N_RFT value and Appendix B limitations were not in the portion read.
- The 671B comparison value is figure-only (Fig. 4) and not transcribed.
- These are RM-benchmark accuracies; the paper read does not report a policy-RL result using DeepSeek-GRM as the reward.
