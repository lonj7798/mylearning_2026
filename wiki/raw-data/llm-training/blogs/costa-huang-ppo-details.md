<!-- scope: Costa Huang's "37 Implementation Details of PPO" — the canonical PPO reproducibility reference
     deps: [[README]]
     see-also: [[ppo]], [[hf-rlhf-illustrated]]
-->

# The 37 Implementation Details of Proximal Policy Optimization
- **Core Insight:** The performance gap between a "canonical" PPO paper and a tuned open-source PPO is dominated by implementation choices (observation normalization, value clipping, orthogonal init, learning-rate annealing, advantage normalization) — not by the loss formula.
- **Guideline:** When PPO fails to reproduce, go through the 37-item checklist before changing hyperparameters.
- **Author:** Shengyi "Costa" Huang, Rousslan F. J. Dossa, Antonin Raffin, Anssi Kanervisto, Weixun Wang
- **Year:** 2022 (ICLR Blog Track)
- **URL:** https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/
- **Relevant topics:** PPO reproducibility, reference implementation, orthogonal init, value clipping, LR annealing

## Summary
The post catalogs 37 concrete implementation choices that separate the PPO paper's described algorithm from the working reference implementation in OpenAI Baselines / Stable-Baselines. Each item is named, located in source code, and ablated where possible. The companion repo (`vwxyzjn/ppo-implementation-details`) provides reproducible CleanRL variants. The post is the first-stop reference whenever a PPO run diverges from published numbers.

A 2024 follow-up, "The N+ Implementation Details of RLHF with PPO" (Huang, Liu, von Werra — HF), extends the same analysis to RLHF-specific details (TL;DR-summarization reproduction of OpenAI's 2019 RLHF paper).

## Key Contributions
- 37 named implementation tricks, each with code pointer and one-line rationale.
- Open CleanRL reference implementation matching the tricks.
- Quantitative ablation on the Atari + MuJoCo stacks where omission of each trick is measurable.
- Spiritual successor "N+ Implementation Details of RLHF with PPO" specialized for LLM fine-tuning.

## Key Figures/Tables to Study
- **Annotated pseudocode block:** PPO loop with each of the 37 details labeled inline.
- **Ablation bar chart:** per-trick performance delta on standard benchmarks.
- **Code-location table:** mapping each trick to OpenAI Baselines line number.

## Technical Details
Selected items from the list most relevant to RLHF fine-tuning:

1. **Orthogonal initialization + layer scaling** — policy and value heads initialized with orthogonal matrix and specific gain (sqrt(2) for hidden, 0.01 for policy output, 1 for value).
2. **Learning-rate annealing** — linear decay from LR_0 to 0 over training; missing this degrades terminal performance.
3. **Observation normalization** — running mean/std, clipped to [-10, 10]. In LLM RLHF the analog is reward whitening.
4. **Value function loss clipping** — clip value predictions around the old value (mirrors policy ratio clipping). Optional but common.
5. **Advantage normalization** — per-minibatch whitening of advantages.
6. **Generalized Advantage Estimation (GAE)** with lambda ~0.95.
7. **Global gradient clipping** at max-norm 0.5 (RL) or 1.0 (RLHF typical).
8. **Minibatch shuffle** inside each PPO update epoch.
9. **Separate policy/value clip ranges** — not always enabled by default.
10. **PPO clip epsilon** 0.2 is the most common default.
Plus 27 more, several of which (reward scaling, policy-value sharing, entropy coefficient schedule) are the direct ancestors of modern RLHF defaults.

### RLHF-specific follow-up highlights
- Whitening the scalar reward before injecting into the PPO reward stream.
- Handling per-token vs per-sequence log-prob correctly when computing the ratio.
- KL-to-reference is added to per-token reward, not to loss, in the canonical impl (equivalent but commonly mis-coded).
- Value head initialization: start from the RM's value head to avoid warmup regression.

## Connections
- [[ppo]] — the paper the blog post operationalizes.
- [[hf-rlhf-illustrated]] — HF's illustrated RLHF post covers the same ground at a higher level.
- [[trl-ppo]] — HF TRL's PPO trainer implements many of these details.
- [[openrlhf-ppo]] — another reference implementation incorporating the 37 tricks.
- [[john-schulman-kl-tricks]] — complementary blog on the KL-estimator side of PPO for LLMs.
