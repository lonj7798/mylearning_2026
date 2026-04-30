<!-- scope: reinforcement-style pretraining objective that turns chain-of-thought into dense exploration reward
     deps: [[quiet-star]], [[front-loading-reasoning]]
     see-also: [[deepseek-r1]], [[rlvr-tulu3]], [[spurious-rewards-rlvr]], [[rlvr-beyond-base-model]], [[echo-chamber-rl-post-training]]
-->

# RLP: Reinforcement as a Pretraining Objective
- **Core Insight:** RLP moves RL-style exploration into pretraining by treating chain-of-thought as an exploratory action and rewarding it by the information gain it gives for predicting future tokens. The result is a dense, verifier-free objective that pushes models to think before they predict.
- **Guideline:** If you want reasoning to be learned earlier than SFT or RLVR, make the reward a direct function of predictive improvement with and without sampled reasoning, and apply it over ordinary text rather than only on curated reasoning datasets.
- **Authors:** Ali Hatamizadeh, Syeda Nahida Akter, Shrimai Prabhumoye, Jan Kautz, Mostofa Patwary, Mohammad Shoeybi, Bryan Catanzaro, Yejin Choi
- **Year:** 2026
- **URL:** https://openreview.net/forum?id=9Gp45bnDrJ
- **Relevant topics:** reinforcement pretraining, dense reward, verifier-free RL, reasoning priors, chain-of-thought exploration

## Abstract
The paper argues that the usual pipeline, next-token pretraining followed by SFT and then RL, delays reasoning too late. RLP instead defines a pretraining-time reinforcement objective whose reward is the increase in next-token log-likelihood obtained by conditioning on a sampled reasoning chain. That makes chain-of-thought a training-time exploratory action rather than a post-training artifact. The objective is dense, position-wise, and verifier-free, so it can be applied across the full document stream during pretraining.

## Key Contributions
- Recasts reasoning as a **pretraining objective** rather than a post-training add-on.
- Uses a **dense information-gain reward** instead of sparse outcome verification.
- Makes the reward **verifier-free**, so it can run on ordinary pretraining text.
- Shows that the gains persist through identical post-training, not just in the base checkpoint.
- Reports large improvements on reasoning-heavy benchmarks, including especially strong gains on AIME25 and MMLU-Pro.

## Key Figures/Tables to Study
- **Overview figure comparing NTP vs RLP:** this is the cleanest statement of how the objective differs from next-token prediction.
- **Benchmark table for Qwen3-1.7B-Base:** shows the reported average uplift of about 19% on an eight-benchmark math-and-science suite.
- **Scaling result on NVIDIA-Nemotron-Nano-12B-v2-Base:** shows the 42.81% to 61.32% jump and the stronger scientific-reasoning scores.

## Technical Details

### Reward definition
- For a context token sequence, sample a reasoning chain and compare next-token log-likelihood with and without that chain.
- The reward is the **incremental predictive gain** from conditioning on the chain.
- Because the reward is computed from the model's own predictive improvement, no external verifier is required.

### Training setup
- RLP is applied during **pretraining**, not after SFT.
- The paper frames chain-of-thought as an **exploratory action** that can be credited wherever it improves future prediction.
- This produces a **dense, position-wise signal** rather than one sparse score per completed answer.

### Empirical takeaway
- RLP lifts reasoning-heavy metrics on smaller and larger base models.
- The gains are not confined to the pretraining checkpoint; they still appear after the same post-training recipe is applied.
- The result is best read as evidence that some reasoning behavior can be installed earlier, before classic post-training begins.

## Connections
- [[quiet-star]] is the closest conceptual ancestor on this page: both try to push latent reasoning into a pretraining-like phase, but RLP does it with an explicit RL-style reward.
- [[front-loading-reasoning]] provides the broader stage-ordering argument that early reasoning data has a durable effect.
- [[deepseek-r1]] and [[rlvr-tulu3]] are the downstream verifier-grounded counterpart: they show what happens when reasoning is learned by reward after pretraining.
- [[spurious-rewards-rlvr]] and [[echo-chamber-rl-post-training]] are the main cautionary counterpoints, because they ask whether RL is mostly amplifying existing priors rather than creating new reasoning capacity.
- [[rlvr-beyond-base-model]] is the useful evaluation warning: pass@1 gains do not by themselves prove that a model's reasoning boundary expanded.
