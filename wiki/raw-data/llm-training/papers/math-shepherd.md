<!-- scope: automatic PRM labels via Monte-Carlo rollouts; step-level PPO
     deps: [[prm800k]]
     see-also: [[deepseek-r1]], [[rlvr-tulu3]]
-->

# Math-Shepherd: Verify and Reinforce LLMs Step-by-Step
- **Core Insight:** You do not need humans to label step correctness — the empirical probability that the rest of a rollout reaches the correct final answer is a scalable automatic process label.
- **Guideline:** For each prefix ending at a step boundary, sample K completions; label that step with `(# correct completions) / K` or with its binary "hard" version; train a PRM on these soft labels; then do either Best-of-N verification or step-level PPO where the per-step reward is the PRM score.
- **Authors:** Peiyi Wang, Lei Li, Zhihong Shao, R.X. Xu, Damai Dai, Yifei Li, Deli Chen, Y. Wu, Zhifang Sui
- **Year:** 2023 (v3 2024)
- **URL:** https://arxiv.org/abs/2312.08935
- **Relevant topics:** automatic PRM, MC rollout estimation, step-level PPO, process verification, GSM8K, MATH

## Abstract
Math-Shepherd removes the human-labeling bottleneck of PRM800K by replacing step-level human judgments with Monte-Carlo estimates of eventual success. For every intermediate state `s_t` in a math rollout, the method samples K continuations from a strong LM, scores them against the ground-truth answer, and assigns `y_t = (# correct continuations) / K` (hard: 1 if any are correct; soft: the ratio). The resulting dataset trains a PRM that can be used as (1) a Best-of-N verifier or (2) a dense reward for step-level PPO. Applied to Mistral-7B and DeepSeekMath-7B, Math-Shepherd lifts GSM8K from 77.9% to 84.1% with PPO and to 89.1% with verification, and MATH from 28.6% to 33.0% with PPO and 43.5% with verification.

## Key Contributions
- **Auto-labeling algorithm:** for every step `s_t` in a rollout, sample K completions and compute:
  - `y_hard(s_t) = 1` iff at least one of K completions reaches the correct final answer, else 0.
  - `y_soft(s_t) = (# correct completions) / K`.
- **PRM training:** BCE loss at every step token; soft labels work better than hard for noisy domains.
- **Best-of-N verification:** rerank N sampled solutions by aggregated PRM score; aggregation is `min_t PRM(step_t)` — shown to outperform `prod` and `mean`.
- **Step-level PPO:** at every step boundary, add the PRM score as an intermediate dense reward; the trajectory's total reward is `sum_t PRM(step_t)` plus the standard final-correctness reward.
- **Results (published):**
  - Mistral-7B GSM8K: 77.9 → 84.1 (+PPO), 89.1 (+PRM verify).
  - Mistral-7B MATH: 28.6 → 33.0 (+PPO), 43.5 (+PRM verify).
  - DeepSeekMath-7B gains of a similar magnitude.
- **Data efficiency:** only ~1K problems × modest K → hundreds of thousands of step-level labels, at zero human cost.

## Key Figures/Tables to Study
- **Fig. 2** (the auto-labeling diagram) — prefix → K MC rollouts → fraction correct → label.
- **Fig. 4 / Table 2** (GSM8K and MATH accuracy across verification / RL configurations).
- **Table 4** (`min` vs `mean` vs `prod` aggregation) — `min` wins.
- **Table 5** (soft vs hard labels) — soft wins on MATH, comparable on GSM8K.

## Technical Details
- **Rollout LM for labels:** a stronger generator than the one being trained (DeepSeekMath-7B in the paper); K = 8 or 16 per state.
- **Step boundary detection:** text-based split on "\n" or "Step i:"; PRM emits score at the first token after the boundary.
- **PPO reward composition:**
  `R_total = r_final + λ · sum_{t ∈ steps} PRM(step_t)`
  with λ ≈ 0.1–1.0; the final answer correctness reward is still included.
- **Verification at inference:** sample N = 256 or 1024 chains, aggregate step scores with `min`, pick max.
- **Noise mitigations:** steps with too-short or template-only content are filtered out before labeling.

## Connections
- Automatic, scalable version of **[[prm800k]]**; most open reasoning stacks (Qwen-Math, DeepSeekMath, OpenRLHF PRM recipes) use Math-Shepherd-style labeling.
- Process-level RL it enables is a middle ground between outcome-only RLVR (**[[rlvr-tulu3]]**, **[[deepseek-r1]]**) and preference-based RM training.
- Still susceptible to proxy misalignment (**[[reward-model-overoptimization]]**) but less so than preference RMs because labels are grounded in end-to-end correctness.
