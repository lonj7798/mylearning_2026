<!-- scope: BroRL (arXiv:2510.01180, NVIDIA) — scaling RLVR by rollouts per prompt (N = 16 → 512) instead of training steps; mass-balance argument for unsampled tokens; throughput gain from larger N
     deps: [[prorl]], [[grpo]]
     see-also: [[scalerl]], [[dapo]], [[reasoning-gym]], [[rl-design-guide]]
-->

# BroRL: Scaling Reinforcement Learning via Broadened Exploration
- **Core Insight:** In a one-step analysis of the policy update, the change in total probability on correct tokens has two non-negative terms from sampled tokens and one possibly negative term from unsampled tokens whose expected size scales as p²(1 − p)^N, so it shrinks as rollouts per prompt N grow (§2, Theorem 1, Lemma 2); continuing a 3,000-step ProRLv2 1.5B checkpoint with N = 512 instead of N = 16 at the same compute raised the math average from 61.69 to 63.03 while N = 16 stayed at 62.02 (Table 3).
- **Guideline:** When a long RL run at small group size has plateaued, increase rollouts per prompt (N = 512 with fewer prompts per step) before adding more steps, and scale the learning rate by the square root of the batch increase, because at equal compute BroRL beat further ProRL steps on math, code and Reasoning Gym (Tables 3–5) and doubled generation throughput from 36.5 to 72.4 samples/s (Table 2); treat the gain as modest (paired Δ = 0.0033 per instance, §4.2.2) and only shown for N ∈ {16, 512}.
- **Authors:** Jian Hu, Mingjie Liu, Ximing Lu, Fang Wu, Zaid Harchaoui, Shizhe Diao, Yejin Choi, Pavlo Molchanov, Jun Yang, Jan Kautz, Yi Dong (NVIDIA; Stanford; University of Washington)
- **Year:** 2025 (arXiv v1, 1 Oct 2025)
- **URL:** https://arxiv.org/abs/2510.01180
- **Source type:** paper
- **Relevant topics:** group size / rollouts per prompt, exploration, dynamic sampling pass rate, generation throughput, RL plateau

## Summary
ProRL scaled RL by training steps and reported gains that flatten after thousands of steps ([[prorl]]). BroRL instead scales N per prompt. A token-level simulator (vocabulary 128,000, 10,000 correct tokens, N ∈ {4, 8, 16, 512, 51,200}) shows faster growth of correct-token mass and fewer drops of individual correct tokens at large N (§4.1, Fig. 3). The LLM experiments start from the public ProRLv2 1.5B checkpoint (3,000 RL steps, context extended to 16,384) on 64 H100 GPUs with veRL.

## Key Contributions
- A decomposition of correct-mass change into sampled and unsampled terms (Theorem 1) and the p²(1 − p)^N bound on the unsampled term (Lemma 2).
- Equal-compute comparison: ProRL N = 16 × 512 prompts per step vs BroRL N = 512 × 128 prompts per step (§4.2).
- Systems argument: larger N moves generation from memory-bound toward compute-bound and raises prefix-cache hits (Table 2).

## Key Figures/Tables to Study
- Fig. 3 (simulation), Table 1 (wall-clock), Table 2 (pass rate, throughput), Tables 3–5 (math, code, Reasoning Gym).

## Technical Details
**Results.**

| Method | N | Extra steps | Math avg | Code avg | RG avg |
|---|---|---|---|---|---|
| Baseline | 16 | 0 (3,000) | 61.69 | 52.00 | 61.29 |
| ProRL | 16 | +535 | 62.02 | 52.74 | 61.45 |
| BroRL | 512 | +191 | 63.03 | 54.20 | 63.09 |

(Math rows from Table 3; code and RG final values from Tables 4–5.)

**Time (Table 1).** ProRL +535 steps: 133.8 h. BroRL +107 steps: 98.1 h, already above ProRL's final scores. BroRL +191 steps: 173.8 h.

**Rollout statistics (Table 2).** Dynamic-sampling pass rate (share of groups with non-zero variance kept) 41% at N = 16 vs 62% at N = 512; throughput 36.5 vs 72.4 samples/s.

**Significance (§4.2.2).** Over more than 10,000 per-instance points, paired t-test Δ = 0.0033, t = 4.84, one-tailed p = 6.5×10⁻⁷.

## Verification
- Read arXiv HTML v1 (full page) on 2026-11-24.
- The theory assumes one-step updates, a linear surrogate and i.i.d. sampling; intermediate N values are left to future work (App. A).
- Contrast with [[scalerl]] App. A.13, which found 8–32 generations per prompt at fixed total batch gave nearly identical curves; the two studies cover different N ranges and starting points.
