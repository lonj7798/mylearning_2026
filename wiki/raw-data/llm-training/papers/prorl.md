<!-- scope: counter-argument that longer and better-controlled RL can expand reasoning boundaries
     deps: [[grpo]], [[kl-control-rlhf]]
     see-also: [[rlvr-beyond-base-model]], [[spurious-rewards-rlvr]], [[echo-chamber-rl-post-training]]
-->

# ProRL: Prolonged Reinforcement Learning Expands Reasoning Boundaries in Large Language Models
- **Core Insight:** Short or poorly controlled RL may only sharpen pretrained behaviors, but prolonged RL with KL control, reference-policy resetting, and task diversity can push the model into genuinely new reasoning regions unreachable by the base model under extensive sampling.
- **Guideline:** If RL seems to saturate early, do not conclude the boundary is fixed; extend training duration, control KL drift carefully, reset the reference policy when needed, and diversify the task suite before writing off RL as mere mode concentration.
- **Authors:** Mingjie Liu, Shizhe Diao, Ximing Lu, Jian Hu, Xin Dong, Yejin Choi, Jan Kautz, Yi Dong
- **Year:** 2025
- **URL:** https://arxiv.org/abs/2505.24864
- **Relevant topics:** prolonged RL, KL control, reference-policy reset, reasoning boundary, pass@k

## Abstract
ProRL is an explicit rebuttal to the claim that RL only amplifies what the base model already knows. The paper argues that sufficiently long RL training, paired with KL divergence control, reference-policy resetting, and diverse tasks, can reveal reasoning strategies inaccessible to the base model even under extensive sampling. The reported improvements persist across pass@k evaluations and include cases where the base model fails entirely.

## Key Contributions
- Argues for **reasoning-boundary expansion**, not just pass@1 concentration.
- Introduces the **ProRL** recipe: prolonged RL + KL control + reference policy resetting + task diversity.
- Shows cases where RL models solve problems that the base model cannot solve even with aggressive sampling.
- Connects improved boundary expansion to both **base-model competence** and **training duration**.

## Key Figures/Tables to Study
- **Pass@k comparisons against the base model:** this is the core evidence against the strict "RL only sharpens priors" thesis.
- **Ablations on training duration / resets / KL control:** essential for understanding what made the difference.
- **Task-diversity analysis:** useful for thinking about RL data mixture rather than single-benchmark optimization.

## Technical Details

### Proposed recipe
- **KL divergence control** to prevent destructive drift while keeping exploration alive.
- **Reference policy resetting** to avoid locking the run to a stale anchor policy.
- **Diverse task suite** instead of single-benchmark overfitting.
- **Longer training horizon** than typical short RL runs.

### Main claim
- Novel reasoning strategies can emerge after enough RL compute, even when extensive sampling from the base model never finds them.

### Important nuance
- The paper does not say all RL does this by default.
- It says that **training duration and control strategy matter**, which is a narrower and more useful claim.

### Practical implication
- Many negative conclusions about RL may be conclusions about **short-horizon RL**.
- If you want RL to truly expand the boundary, optimize for sustained exploration and dynamic stabilization, not just immediate reward gain.

## Connections
- [[rlvr-beyond-base-model]] is the direct foil; read them together.
- [[spurious-rewards-rlvr]] and [[echo-chamber-rl-post-training]] explain why naive RL can look like prior amplification.
- [[kl-control-rlhf]] is a prerequisite because ProRL leans on KL management as a core ingredient.
- [[deepseek-r1]] is the obvious empirical reference point for long-horizon RL optimism.
