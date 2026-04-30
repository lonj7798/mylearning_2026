<!-- scope: why RLVR can improve with weak or spurious rewards
     deps: [[grpo]], [[reward-model-overoptimization]]
     see-also: [[echo-chamber-rl-post-training]], [[rlvr-beyond-base-model]], [[prorl]]
-->

# Spurious Rewards: Rethinking Training Signals in RLVR
- **Core Insight:** RLVR can improve reasoning even with random or negatively correlated rewards because GRPO's clipped objective can preferentially amplify strong pretrained behaviors without truly informative training signals.
- **Guideline:** Treat apparent RLVR gains with caution; verify whether your reward is actually teaching the model or merely triggering distributional sharpening through clipping bias.
- **Authors:** Rulin Shao, Shuyue Stella Li, Rui Xin, Scott Geng, Yiping Wang, Sewoong Oh, Simon Shaolei Du, Nathan Lambert, Sewon Min, Ranjay Krishna, Yulia Tsvetkov, Hannaneh Hajishirzi, Pang Wei Koh, Luke Zettlemoyer
- **Year:** 2025
- **URL:** https://arxiv.org/abs/2506.10947
- **Relevant topics:** RLVR, spurious rewards, GRPO clipping bias, code reasoning, training-signal quality

## Abstract
This paper shows that RLVR can substantially improve mathematical reasoning even when the reward has little or no positive correlation with correctness. On Qwen2.5-Math-7B, GRPO with randomly assigned rewards improves MATH-500 by 21.4 points, close to the 29.1-point gain from ground-truth rewards. The proposed explanation is a clipping bias in GRPO that amplifies high-prior pretrained behaviors, including a specific "code reasoning" mode.

## Key Contributions
- Demonstrates a striking empirical result: **random rewards can still produce strong RLVR gains**.
- Attributes this to **GRPO clipping bias**, not to meaningful signal extraction from the reward.
- Identifies **code reasoning** as a concrete pretrained behavior that gets amplified.
- Forces a reinterpretation of many RLVR results: some gains may come from **prior exploitation**, not reward informativeness.

## Key Figures/Tables to Study
- **Main MATH-500 improvement table:** random vs ground-truth reward gains.
- **Code-reasoning frequency analysis:** useful because it gives a visible mechanistic behavior rather than a purely abstract argument.
- **GRPO clipping discussion:** central for understanding how the objective can induce this effect.

## Technical Details

### Main empirical result
- On **Qwen2.5-Math-7B**, random-reward GRPO yields a large benchmark improvement close to the gain from true rewards.

### Proposed mechanism
- The **clip term** in GRPO introduces a bias that can favor and reinforce already high-probability behaviors.
- If the base model already contains a productive reasoning mode, RLVR may surface it even when the reward is nearly useless.

### Code reasoning case study
- The authors identify a behavior they call **code reasoning**: the model reasons in code-like form without actually executing code.
- RL with spurious rewards pushes this behavior from roughly **65% to above 90%** frequency.

### Practical implication
- RLVR results should be audited against:
  - base-model latent behaviors
  - reward informativeness
  - algorithm-specific objective biases

## Connections
- [[echo-chamber-rl-post-training]] is the closest companion paper: both emphasize RL as amplification of prior structure.
- [[grpo]] provides the algorithmic background for the clipping mechanism discussed here.
- [[rlvr-beyond-base-model]] similarly argues that observed gains can overstate true capability expansion.
- [[prorl]] is a useful counterweight because it argues that longer and more diverse RL can genuinely widen the reasoning frontier.
