<!-- scope: pass@k critique of whether RLVR expands reasoning beyond the base model
     deps: [[grpo]], [[rloo]], [[deepseek-r1]]
     see-also: [[spurious-rewards-rlvr]], [[echo-chamber-rl-post-training]], [[prorl]]
-->

# Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model?
- **Core Insight:** Under large-pass@k evaluation, many RLVR gains look like improved sampling efficiency rather than genuine expansion of the base model's reasoning boundary.
- **Guideline:** Measure reasoning with large-`k` coverage, not just pass@1; if the base model surpasses the RL model at high `k`, your RL run likely improved concentration over existing good paths rather than discovering new ones.
- **Authors:** Yang Yue, Zhiqi Chen, Rui Lu, Andrew Zhao, Zhaokai Wang, Shiji Song, Gao Huang
- **Year:** 2025
- **URL:** https://arxiv.org/abs/2504.13837
- **Relevant topics:** RLVR evaluation, pass@k, reasoning boundary, exploration collapse, distillation vs RL

## Abstract
The paper argues that RLVR does not, in its current form, reliably create fundamentally new reasoning patterns. Across math, coding, and visual reasoning tasks, RL-trained models beat base models at small `k`, but at large `k` the base models often match or exceed them. The authors interpret this as evidence that RLVR mostly redistributes probability mass toward already-existing successful paths, while narrowing exploration and reducing the broader coverage of solvable problems.

## Key Contributions
- Introduces **large-`k` pass@k** as a probe of reasoning-capacity boundary rather than average-case performance.
- Shows a recurring pattern: **RL improves pass@1, base model wins at high `k`**.
- Frames RLVR as **sampling-efficiency improvement** rather than capability-boundary expansion.
- Contrasts RL with **distillation**, arguing that distillation can truly introduce knowledge beyond the base model.

## Key Figures/Tables to Study
- **Figure 1:** conceptual search-tree picture of RL narrowing the distribution toward rewarded paths.
- **Figure 2:** pass@k curves where base models overtake RL-trained models at large `k`.
- **Algorithm comparison section:** important because the critique is not aimed at just one RLVR variant.

## Technical Details

### Evaluation move
- The paper's core move is to redefine the question from "is pass@1 better?" to "what is the model's coverage ceiling under many tries?"
- This reframing matters because average-case improvement can coexist with **reduced exploration breadth**.

### Main mechanism claim
- RLVR biases the model toward high-reward paths already present in the base distribution.
- This improves **sampling efficiency** but can reduce the **scope of reasoning coverage**.

### Additional claims
- Different RL algorithms perform similarly under this lens and remain far from optimal.
- For visual reasoning, the same pattern appears.
- **Distillation** is presented as the cleaner route for importing new capabilities.

### Practical implication
- High pass@1 after RL is not sufficient evidence that the model learned a new reasoning strategy.
- Track both:
  - `pass@1` or average-case quality
  - `pass@large-k` or coverage / boundary

## Connections
- [[spurious-rewards-rlvr]] and [[echo-chamber-rl-post-training]] reinforce this "RL as prior sharpening" interpretation.
- [[prorl]] directly disputes the strongest version of this conclusion by claiming prolonged RL can reach new regions of solution space.
- [[deepseek-r1]] is an important empirical backdrop because it motivated much of the field's optimism about zero-RL and RLVR.
