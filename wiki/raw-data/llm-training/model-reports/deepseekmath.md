<!-- scope: DeepSeekMath — the paper that introduced GRPO
     deps: [[README]]
     see-also: [[deepseek-r1]], [[grpo]], [[ppo]]
-->

# DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models
- **Core Insight:** PPO's critic is expensive and unnecessary for LLM RL; replace it with the group mean/std of K samples per prompt — this is GRPO.
- **Guideline:** For verifiable-reward RL on LLMs, use GRPO: sample G completions, compute group-relative advantage, add KL penalty to reference, optimize with clipped PPO-style ratio.
- **Authors:** Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, Y.K. Li, Y. Wu, Daya Guo
- **Year:** 2024
- **URL:** https://arxiv.org/abs/2402.03300
- **Relevant topics:** GRPO algorithm, math continued pre-training, verifiable reward RL, critic-free policy gradient

## Abstract
DeepSeekMath 7B continues pre-training DeepSeek-Coder-Base-v1.5 7B with 120B math-related tokens sourced from Common Crawl, plus natural language and code data. The model reaches 51.7% on the MATH benchmark without external tools or voting, approaching Gemini-Ultra and GPT-4 levels at 7B scale. The paper introduces Group Relative Policy Optimization (GRPO), a variant of PPO that removes the value network by computing advantages as within-group z-scores across G sampled completions per prompt.

## Key Contributions
- 120B-token math corpus extracted from Common Crawl via a math-topic classifier — shows large-scale math text mining works.
- GRPO: critic-free PPO variant; advantage = (r_i - mean(r_{1..G})) / std(r_{1..G}).
- Unified comparison of SFT, RFT, DPO, online RFT, PPO, GRPO on the same base.
- DeepSeekMath-RL achieves 51.7% MATH (pass@1), 88.2% GSM8K.

## Key Figures/Tables to Study
- **Figure of GRPO vs PPO architecture:** shows the missing critic in GRPO.
- **Table comparing RL algorithms:** SFT 46.8 -> RFT 49.0 -> online RFT 49.4 -> DPO 49.0 -> PPO 51.0 -> GRPO 51.7 on MATH.
- **Equation 20 (or similar)**: the full GRPO objective.
- **KL penalty equation:** the unbiased k3 estimator `KL[pi_theta || pi_ref] ~ pi_ref/pi_theta - log(pi_ref/pi_theta) - 1`.

## Technical Details — GRPO and Training

### GRPO Objective
For each prompt q, sample G outputs {o_1, ..., o_G} from pi_theta_old. Each gets scalar reward r_i. Advantage for output i is:

```
A_i = (r_i - mean(r_1..r_G)) / std(r_1..r_G)
```

The GRPO loss (per-token form):

```
J_GRPO(theta) = E_q ~ P, {o_i} ~ pi_theta_old [
  (1/G) Σ_i (1/|o_i|) Σ_t min(
    ratio_{i,t} * A_i,
    clip(ratio_{i,t}, 1-eps, 1+eps) * A_i
  ) - beta * D_KL(pi_theta || pi_ref)
]
```
where `ratio_{i,t} = pi_theta(o_{i,t} | q, o_{i,<t}) / pi_theta_old(o_{i,t} | q, o_{i,<t})`. The KL divergence uses the unbiased k3 estimator computed at the token level.

### Hyperparameters (DeepSeekMath-RL)
- **Learning rate:** 1e-6 (policy).
- **KL coefficient beta:** 0.04.
- **Group size G:** 64 samples per question.
- **Max generation length:** 1024 tokens.
- **Training batch size:** 1024 (16 prompts x 64 completions).
- **Clip ratio eps:** 0.2.
- **Reward model:** a 7B RM trained on math preference data; also supports rule-based outcome reward (exact-match on final answer).

### Training data
- **SFT stage:** math instruction-tuning corpus (776K problems w/ CoT).
- **RL stage:** prompts drawn from the GSM8K + MATH training set's problems.

### Benchmark results
- **MATH pass@1:** 51.7% (base model); 60.9% with self-consistency (64 samples).
- **GSM8K:** 88.2%.
- **CMATH:** 88.8%.

### Why GRPO > PPO here
PPO's value network is (a) another 7B model to train, (b) a persistent source of bias (value estimates are imperfect), and (c) a memory hog. Replacing it with group z-score advantages trades a learned baseline for a sample-based one — empirically matches or beats PPO while halving the memory footprint.

## Connections
- [[ppo]] — parent algorithm; GRPO inherits clipped ratio + KL penalty, drops the critic.
- [[deepseek-r1]] — scales GRPO to full-model reasoning emergence.
- [[deepseek-v3]] — R1's base; uses GRPO in its own SFT+RL stage.
- [[dr-grpo]] — 2025 follow-up correcting bias in advantage normalization.
- [[rloo]] — parallel critic-free PPO variant (REINFORCE leave-one-out).
- [[rlvr-tulu3]] — Tulu's RLVR uses PPO + verifier; DeepSeek uses GRPO + verifier or RM.
