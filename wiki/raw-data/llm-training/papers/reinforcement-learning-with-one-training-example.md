<!-- scope: data efficiency and exploration in RLVR for math reasoning
     deps: [[rlvr-tulu3]], [[math-shepherd]], [[deepseek-r1]]
     see-also: [[entropy-mechanism-llm-rl]], [[rlvr-beyond-base-model]], [[spurious-rewards-rlvr]], [[rloo-vs-grpo]]
-->

# Reinforcement Learning for Reasoning in Large Language Models with One Training Example
- **Core Insight:** RLVR can be surprisingly data-frugal: on a capable math base model, one carefully chosen training example can unlock large gains, and the main driver is not memorizing that example but reshaping exploration and policy-gradient updates around a high-variance training signal.
- **Guideline:** When RLVR is unstable or sample budgets are tiny, prioritize examples with high historical training variance, keep an entropy bonus, and watch for post-saturation generalization rather than stopping once the single training example reaches near-100% accuracy.
- **Authors:** Yiping Wang, Qing Yang, Zhiyuan Zeng, Liliang Ren, Liyuan Liu, Baolin Peng, Hao Cheng, Xuehai He, Kuan Wang, Jianfeng Gao, Weizhu Chen, Shuohang Wang, Simon Shaolei Du, Yelong Shen
- **Year:** 2025
- **URL:** https://arxiv.org/abs/2504.20571
- **Relevant topics:** RLVR, one-shot data efficiency, exploration, entropy bonus, GRPO, PPO, math reasoning

## Abstract
The paper shows that reinforcement learning with verifiable reward can substantially improve math reasoning using only one training example. On Qwen2.5-Math-1.5B, a single selected example lifts MATH500 from 36.0% to 73.6% and raises average performance across six math benchmarks from 17.6% to 35.7%, matching the reported 1.2k-example DeepScaleR subset. The authors also show that two examples can do slightly better, that the effect transfers across several base models and both GRPO and PPO, and that the main mechanism is policy-gradient-driven exploration rather than grokking. A notable empirical pattern is post-saturation generalization: training accuracy saturates quickly, yet test performance keeps improving for a long time.

## Key Contributions
- Shows that RLVR does not need a large curated dataset to produce large reasoning gains.
- Introduces a simple selection heuristic based on **historical variance score** to pick the single example that works best.
- Demonstrates robustness across models, including Qwen2.5-Math-7B, Llama3.2-3B-Instruct, and DeepSeek-R1-Distill-Qwen-1.5B.
- Finds that **entropy loss** materially helps exploration and that entropy alone can improve MATH500 even without outcome reward.
- Separates the main effect from grokking: the observed gains come primarily from the policy-gradient term.

## Key Figures/Tables to Study
- **Figure 1:** 1-shot RLVR versus 1.2k-example RLVR on MATH500 and average math performance.
- **Figure 2:** detailed 1/2-shot curves showing that the best checkpoint keeps improving after the training example is solved.
- **Tables on example selection:** show that many different single examples can work, and that historical-variance ranking beats random selection.
- **Non-math transfer table:** shows that math-focused 1-shot RLVR can also help unrelated reasoning tasks.

## Technical Details

### Training setup
- Default base model: `Qwen2.5-Math-1.5B`.
- RL algorithm: `GRPO` by default, with `PPO` also tested.
- Training data for the one-shot setting is duplicated until it reaches the batch size, so the optimizer still sees full batches even though the semantic dataset is tiny.

### Reward and optimization
- Outcome reward is binary: correct final answer gets `1`, otherwise `0`.
- The RL loss combines policy-gradient updates with KL regularization and an entropy term.
- The paper’s main ablation suggests the policy-gradient term is what actually drives the jump in capability.

### Example selection
- The best one-shot example is selected by a **historical variance score** computed from training accuracy across epochs on the full dataset.
- High-variance examples are more informative for RLVR because they expose the model to reward-sensitive decision boundaries.
- The paper reports that even moderately chosen examples can produce large gains, so the heuristic is useful but not unique.

### Reported phenomena
- **Post-saturation generalization:** the model keeps getting better on held-out math problems after it has already memorized the training example.
- **Cross-category generalization:** training on one math domain can improve others.
- **Self-reflection increase:** downstream outputs contain more reflective language as training progresses.
- **Entropy matters:** exploration collapse hurts; a properly tuned entropy bonus is an important stabilizer.

### Concrete numbers
- Qwen2.5-Math-1.5B: `36.0% -> 73.6%` on MATH500 with one example.
- Average across six math benchmarks: `17.6% -> 35.7%`.
- Two examples: `74.8%` on MATH500 and `36.6%` average, slightly above the 1.2k-example DeepScaleR subset.

## Connections
- Directly complements [[rlvr-tulu3]]: both argue that verifier-grounded RL can outperform much heavier supervision when the reward is clean.
- Mechanistically adjacent to [[entropy-mechanism-llm-rl]]: both find that exploration/entropy is a first-class control knob in LLM RL.
- Closely related to [[deepseek-r1]], but this paper shows the extreme low-data edge of the same RLVR family.
- Pairs well with [[spurious-rewards-rlvr]] because both study how RLVR behaves when the reward signal is sparse, brittle, or oddly structured.
- Useful foil for [[rlvr-beyond-base-model]]: this paper suggests RL can move performance a lot even when the training signal is tiny, but it does not by itself prove a broader reasoning boundary was created.
