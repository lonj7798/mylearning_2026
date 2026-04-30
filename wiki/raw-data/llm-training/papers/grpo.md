<!-- scope: Group Relative Policy Optimization — PPO without a value network, using group-mean baseline
     deps: [[ppo]], [[rloo]]
     see-also: [[dr-grpo]], [[deepseek-r1]], [[deepseekmath]]
-->

# DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models (GRPO)
- **Core Insight:** Replace PPO's learned value function with the mean/std of rewards across G sampled responses to the same prompt — removing the critic halves memory and avoids value-function-fit bias, while group statistics give a low-variance baseline.
- **Guideline:** Use GRPO when (a) you have verifiable rewards (math/code), (b) can afford G rollouts per prompt (G=8–64), and (c) want to skip value-network training; apply the k3 unbiased KL estimator to π_ref in the loss (not in the reward) to keep KL positivity.
- **Authors:** Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, Y.K. Li, Y. Wu, Daya Guo
- **Year:** 2024
- **URL:** https://arxiv.org/abs/2402.03300
- **Relevant topics:** RL for reasoning, critic-free RL, group baseline, KL approximation, DeepSeek R1 ancestry

## Abstract
Introduces DeepSeekMath 7B, a math-specialized LM that reaches 51.7% on MATH (approaching GPT-4 / Gemini Ultra). The training recipe contributes two pieces: a large-scale math pretraining corpus (120B tokens filtered from Common Crawl) and **Group Relative Policy Optimization (GRPO)**, a variant of PPO that discards the value network and uses per-group statistics for advantages. GRPO improves both in-domain (MATH, GSM8K) and out-of-domain (CMATH) reasoning and memory footprint.

## Key Contributions
- **GRPO algorithm** — on-policy RL without a critic; became the core of DeepSeek-R1.
- **Group baseline**: for each question, sample G outputs; advantage = (reward − group mean) / group std.
- **KL term inside the loss (not the reward)** — uses the k3 unbiased positive KL estimator.
- **Process reward variant** — per-step advantages when PRMs are available.
- DeepSeekMath-RL beats DeepSeekMath-SFT by 4–6 points on MATH with minimal extra compute.

## Key Figures/Tables to Study
- **Figure 4 / Algorithm 1:** GRPO pseudocode — canonical reference.
- **Section 4.1.2 / Equation 3:** The GRPO objective — most-cited formula in post-R1 RL.
- **Figure 5:** Per-question reward variance with and without group baseline.
- **Table 5:** Outcome RM vs Process RM with GRPO — PRM helps slightly but is not strictly required.

## Technical Details

### Rollout
For each question q in batch, sample G outputs {o_1, …, o_G} from π_θ_old. Score each with reward model R → (r_1, …, r_G).

### Advantage (outcome supervision)
`Â_{i,t} = r̃_i = (r_i − mean({r_1,...,r_G})) / std({r_1,...,r_G})`
Same value for every token t in o_i (outcome-level).

### GRPO loss (Equation 3)
`J_GRPO(θ) = E[q ~ P(Q), {o_i} ~ π_θ_old(·|q)]`
`  (1/G) Σ_i (1/|o_i|) Σ_t { min[ρ_{i,t} Â_{i,t},  clip(ρ_{i,t}, 1-ε, 1+ε) Â_{i,t}]  −  β D_KL(π_θ || π_ref) }`
where `ρ_{i,t} = π_θ(o_{i,t} | q, o_{i,<t}) / π_θ_old(o_{i,t} | q, o_{i,<t})`.

### KL approximation (Equation 4, "k3" / Schulman estimator)
`D_KL[π_θ || π_ref] ≈ π_ref(o_{i,t}|·)/π_θ(o_{i,t}|·) − log[π_ref(o_{i,t}|·)/π_θ(o_{i,t}|·)] − 1`
- **k3 vs k1 vs k2:** k1 = log(π_θ/π_ref) (low variance, biased sign), k2 = 0.5·(log ratio)^2 (unbiased but sign-insensitive), k3 above (unbiased, always ≥0). GRPO uses k3.
- Applied **token-wise inside the loss**, not as a per-token reward penalty.

### Process-supervision variant
If a Process RM gives per-step reward r_i^{index(t)}:
`Â_{i,t} = Σ_{k: index(k) ≥ t} r̃_i^{(k)}` — discounted sum from step onward, group-normalized.

### Hyperparameters (paper recipe)
| Knob | Value |
|------|-------|
| Group size G | 64 (main runs) |
| Clip ε | 0.2 |
| KL coefficient β | 0.04 |
| Learning rate | 1e-6 |
| Batch size (prompts) | 1024 |
| Max response length | 1024 tokens |
| π_ref | SFT model, frozen |
| Epochs per rollout μ | 1 (single-step) |
| Sampling T | 1.0 (rollouts) |

## Connections
- Parent algorithm: [[ppo]] — GRPO replaces V-function and removes GAE.
- Leave-one-out cousin: [[rloo]] — same "use other samples as baseline" idea with a different estimator.
- Bias-corrected successor: [[dr-grpo]] — drops 1/|o_i| normalization shown to favor short wrong answers.
- Verifiable rewards this enables: [[deepseek-r1]], [[rlvr-tulu3]].
- Framework implementations: [[verl-grpo]], [[trl-grpo]], [[openrlhf-ppo]].
