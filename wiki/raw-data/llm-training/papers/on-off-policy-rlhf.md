<!-- scope: on-policy vs off-policy preference optimization analysis
     deps: [[dpo]], [[ppo]]
     see-also: [[iterative-sft-rl]], [[trl-online-dpo]]
-->

# Understanding the Performance Gap Between On-Policy and Off-Policy RLHF
- **Core Insight:** On-policy RLHF (PPO, online DPO) outperforms off-policy DPO primarily because on-policy methods *cover the policy's own output distribution*, while offline DPO trains on a static preference dataset collected from a different (usually stronger) model — the distribution mismatch, not the algorithm family, is what causes DPO to underperform.
- **Guideline:** If offline DPO is underperforming PPO on your task, the fix isn't switching to PPO — it's making DPO on-policy by sampling the chosen/rejected pairs from the current policy each step.
- **Authors:** Yunhao Tang, Daniel Zhaohan Guo, Zeyu Zheng, Daniele Calandriello, Yuan Cao, Eugene Tarassov, Rémi Munos, Bernardo Ávila Pires, Michal Valko, Yong Cheng, Will Dabney
- **Year:** 2024 (DeepMind)
- **URL:** https://arxiv.org/abs/2404.14367
- **Relevant topics:** on-policy vs off-policy RLHF, DPO vs PPO, distribution shift, iterative DPO

## Abstract
Reinforcement learning (RL) has been used to fine-tune large language models (LLMs) using human feedback in what is termed reinforcement learning from human feedback (RLHF). The dominant algorithms used in industrial RLHF have been dubbed off-policy algorithms (e.g. direct preference optimization (DPO)) and on-policy algorithms (e.g. proximal policy optimization (PPO)). We identify fundamental differences between off-policy algorithms (e.g., DPO) and their more on-policy counterparts (e.g., iterative DPO), with the off-policy algorithms mostly lagging behind in performance. We rigorously characterize the performance differences across a range of RLHF tasks and find consistent evidence that (1) the primary cause of the gap is distribution shift — DPO trains on samples from a distribution different from the policy's own; (2) iterative (on-policy) DPO largely closes this gap; (3) PPO's advantage over DPO vanishes when DPO is made on-policy.

## Key Contributions
- Isolates the "on-policy bonus" as the cause of PPO > offline DPO — not the algorithm class (policy-gradient vs closed-form).
- Demonstrates **iterative DPO ≈ PPO** on summarization, helpfulness, and math reasoning across Gemma-2B and Gemma-7B.
- Formalizes a coverage argument: DPO on off-policy pairs is biased because the implicit reward's normalization constant depends on the sampling distribution.
- Provides a decomposition of the performance gap into (i) distribution-shift contribution (≈80%) and (ii) variance-reduction contribution (≈20%).

## Key Figures/Tables to Study
- **Figure 1 (TL;DR win-rate vs training compute):** PPO and iterative DPO overlap; offline DPO sits 5–10 pts below.
- **Figure 3 (KL-vs-reward Pareto frontier):** iterative DPO dominates offline DPO at every KL budget.
- **Figure 6 (synthetic distribution-shift control):** as you increase the mismatch between DPO training data and policy, the gap grows predictably.
- **Table 2 (helpfulness + math):** iterative DPO matches PPO on both, with lower variance across seeds.

## Technical Details
- **Task suite:** TL;DR summarization (Reddit), HH-RLHF helpfulness, GSM8K math.
- **Iterative DPO recipe:**
  - Each step: sample 2 responses per prompt from current π_t.
  - Label with a frozen RM → chosen/rejected.
  - DPO update with β=0.1, 1 grad step per pair, reference = π_0 (fixed).
- **PPO recipe:** standard Ouyang-style, clip 0.2, KL coef adaptive.
- **Offline DPO baseline:** trained on fixed Anthropic HH pairs.
- **Scale:** Gemma-2B, Gemma-7B; 100K–1M prompts.

## Connections
- Contextualizes the [[trl-online-dpo]] implementation — this paper is the theoretical justification for making DPO on-policy.
- Complements the iteration argument of [[iterative-sft-rl]]: same lesson from a different angle.
- The distribution-shift framing matches [[ppo]]'s importance-sampling motivation.
- Motivates [[self-play-preference]] (Nash-MD) which takes on-policy preference optimization to its logical extreme with self-play.
- Practical corollary: offline DPO is only "safe" when the preference dataset was collected from a policy very close to the current one — otherwise switch to iterative / online DPO.
