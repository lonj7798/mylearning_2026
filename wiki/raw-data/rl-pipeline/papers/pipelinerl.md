<!-- scope: PipelineRL (arXiv:2509.19128, ServiceNow) — concurrent generation and training with in-flight weight updates; stale KV cache kept; mixed-policy sequences; ~2x speedup vs G=32 conventional RL on Qwen2.5-7B math
     deps: [[async-rollout]], [[grpo]]
     see-also: [[scalerl]], [[slime]], [[areal-async-rl]], [[verl-rollout]], [[rollout-training-mismatch-tis]], [[rl-design-guide]]
-->

# PipelineRL: Faster On-policy Reinforcement Learning for Long Sequence Generation
- **Core Insight:** If the inference engine swaps in new weights between decoding steps without discarding in-progress sequences (in-flight weight update), generation never waits for training; on Qwen2.5-7B math RL this reached the same reward about 2× faster than a conventional generate-then-train loop with G = 32 optimizer steps per RL step, while its effective sample size stayed similar to the G = 8 conventional setting (§5, Figs. 5–6).
- **Guideline:** When rollouts are long and vary in length, update weights in flight and keep the stale KV cache instead of recomputing it, because mixed-policy sequences with stale cache showed only slightly higher KL divergence than recomputed ones (§5.1, Fig. 7) and conventional RL became unstable beyond G = 32 (G = 64 diverged, Fig. 10); apply clamped importance weights (the paper clamps at 5) because early tokens in long sequences carry high lag (§4, §7).
- **Authors:** Alexandre Piché, Ehsan Kamalloo, Rafael Pardinas, Xiaoyin Chen, Dzmitry Bahdanau (ServiceNow AI Research, Mila, McGill University)
- **Year:** 2025 (arXiv:2509.19128; v2 read, 26 Sep 2025)
- **URL:** https://arxiv.org/abs/2509.19128 (code: https://github.com/ServiceNow/PipelineRL)
- **Source type:** paper (system + experiments)
- **Relevant topics:** asynchronous rollout, in-flight weight updates, staleness/lag, effective sample size, KV-cache staleness, importance weighting

## Summary
Conventional RL alternates generation and training and takes G optimizer steps per RL step, so lag grows with G (Algorithm 1). PipelineRL runs actor and trainer concurrently: when the trainer publishes new weights, the generation engine pauses briefly to load them and continues the sequences already in progress (Algorithm 2). A sequence therefore contains tokens from several policy versions; earlier tokens have higher lag (Fig. 3a). The paper argues and measures that this keeps data closer to on-policy than conventional RL at the same throughput.

## Key Contributions
- In-flight weight update with KV cache retained by default (§1, Algorithm 2).
- Measurements of lag, ESS and KL for mixed-policy sequences (§5, §5.1, Figs. 6–7).
- About 2× faster time-to-reward than the most aggressive stable conventional setting (§5, Fig. 5a).

## Key Figures/Tables to Study
- Fig. 3 (lag structure), Fig. 5 (reward vs time, throughput), Fig. 6 (max lag, ESS), Fig. 7 (stale vs recomputed KV), Fig. 10 (G = 64 divergence), Table 1.

## Technical Details
**Setup (§5).** Qwen2.5-7B base; 17K math problems from Open-Reasoner-Zero; 1,000 optimizer steps; batch B = 1,024; Adam, LR 1e-6; 16 DGX-H100 nodes (128 GPUs): 48 generation, 80 training. Reward 1 for a correct answer, 0 otherwise, plus a soft penalty near max length. Loss: importance-weighted REINFORCE with weights clamped to 5.

**Speed.** About 2× faster to the same reward, driven by roughly 2× sample throughput (Fig. 5a, 5c). The conventional baseline's throughput was estimated from 2 generation + 2 training nodes plus a correction, not a direct 128-GPU run. App. A.4's analytical simulation (not an experiment) gives up to 1.57× at g_max ≈ 133.

**On-policyness (Fig. 6).** PipelineRL has a higher max lag than every conventional run (some tokens lag by more than 50K samples), but its ESS is similar to G = 8, while ESS for G = 16 and G = 32 drops during training.

**Results (Table 1).** PipelineRL batch 1,024: MATH500 81, AIME24 17.5 (2.0M samples); batch 4,096: 84.6 / 19.8 (6.2M samples); SimpleRL-Zero 78.2 / 20.0; Open-Reasoner-Zero ~82.0 / ~20.0.

**Limitations (§7).** Gains shrink when generation lengths are uniform or compute is very scarce or very abundant; high lag of early tokens in long sequences is an open question.

## Verification
- Read arXiv HTML v2 (full page) on 2026-11-24.
- [[scalerl]] independently adopts PipelineRL with k = 8 and finds it improves efficiency B but not asymptote A relative to PPO-off-policy (its Fig. 4a).
