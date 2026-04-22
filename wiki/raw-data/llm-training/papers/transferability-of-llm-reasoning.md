<!-- scope: whether math-reasoning gains transfer to other capabilities
     deps: [[deepseek-r1]], [[qwen-3]]
     see-also: [[front-loading-reasoning]], [[prorl]], [[lima]]
-->

# Does Math Reasoning Improve General LLM Capabilities? Understanding Transferability of LLM Reasoning
- **Core Insight:** Better math reasoning does not automatically mean better general capability; RL-based math tuning transfers better than SFT-based math tuning because RL preserves broader latent structure while SFT can induce harmful drift.
- **Guideline:** Separate "math leaderboard gains" from "general capability gains"; if you tune on math-only data, prefer RL when cross-domain retention matters, and audit for representation drift after SFT.
- **Authors:** Maggie Huan, Yuetai Li, Tuney Zheng, Xiaoyu Xu, Seungone Kim, Minxin Du, Radha Poovendran, Graham Neubig, Xiang Yue
- **Year:** 2025
- **URL:** https://arxiv.org/abs/2507.00432
- **Relevant topics:** transferability, math reasoning, RL vs SFT, representation drift, capability retention

## Abstract
The paper asks whether the rapid gains in math reasoning reflect broad problem-solving improvement or narrow specialization. Evaluating more than 20 open-weight reasoning-tuned models across math, science QA, agent planning, coding, and instruction following, the authors find that most math-successful models do not transfer their gains broadly. In controlled Qwen3-14B experiments, RL-tuned models generalize better while SFT-tuned models often forget general capabilities.

## Key Contributions
- Broad cross-domain audit of reasoning-tuned models, not just math benchmarks.
- Shows that **math gains often fail to transfer** to science, agents, coding, and instruction following.
- Finds a clean method difference: **RL transfers better than SFT** in math-only tuning setups.
- Uses latent-space and token-space analyses to show **SFT causes stronger representation and output drift**.

## Key Figures/Tables to Study
- **Cross-domain comparison table of reasoning-tuned models:** the main empirical message.
- **Controlled Qwen3-14B experiment:** strongest evidence because it isolates tuning method.
- **Representation / token distribution analyses:** useful for thinking mechanistically about forgetting and drift.

## Technical Details

### Main empirical finding
- Strong performance on MATH/AIME-style benchmarks is often a **narrow capability improvement**, not a general reasoning lift.

### RL vs SFT
- **RL-tuned** math models preserve broader structure and generalize better.
- **SFT-tuned** math models can forget general capabilities, likely because token distributions and internal representations drift too far toward the narrow domain.

### Practical lesson
- If a team claims "reasoning improved," ask:
  - On which domains?
  - Under which tuning method?
  - With what retention of chat / coding / agent / knowledge behavior?

## Connections
- [[front-loading-reasoning]] complements this by asking when reasoning data should be injected, not only how.
- [[prorl]] argues RL can widen the reasoning frontier; this paper shows RL can also preserve cross-domain structure better than SFT.
- [[lima]] is a useful foil because it values small high-quality SFT, whereas this paper highlights a major risk of narrow SFT specialization.
