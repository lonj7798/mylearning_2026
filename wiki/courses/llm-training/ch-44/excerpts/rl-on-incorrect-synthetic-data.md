---
chapter: ch-44
course: llm-training
phase: read
excerpt_of: arXiv:2406.14532v1 (RL on Incorrect Synthetic Data Scales the Efficiency of LLM Math Reasoning by Eight-Fold)
source_url: https://arxiv.org/abs/2406.14532
created_at: "2026-09-15"
revised: "2026-09-15 (generality revision; written from the primary source because no library card exists yet)"
---

# Excerpt: Per-step credit from incorrect traces

Used by [[read]] §4 and the negatives section. Authors: Amrith Setlur, Saurabh Garg, Xinyang (Young) Geng, Naman Garg, Virginia Smith, Aviral Kumar (CMU, Google DeepMind, MultiOn). arXiv v1 2024-06-20; checked on 2026-09-15.

## What is measured (§1, §4)
- Positive synthetic data are model-generated solutions whose final answer is correct; negative synthetic data are generated solutions whose final answer is wrong.
- Error rate scales as about `D^-0.05` to `D^-0.15` in the number `D` of synthetic examples under the scaling law of Zhang et al. (§1).
- Self-generated positives (RFT) are worth about 2x the synthetic problems of teacher positives; per-step use of negatives is worth about 8x (§1, Fig. 7a-b). Models: DeepSeek-Math-7B and Llama2-7B; datasets GSM8K and MATH.

## Construction of the negatives (§4, §6.1, Algorithm 1)
1. Sample incorrect responses from the SFT policy for a problem with a known answer.
2. For each step of an incorrect response, estimate `Q^π̃(x, ŷ_1:i−1; ŷ_i)` by Monte-Carlo rollouts from the prefix, where `π̃` is a sampling policy. The practical version uses 8 negative responses per question and `π̃ = BoK(π_sft)` with `K = 5`.
3. The "first pit" `ŷ_c` is the first step with the lowest `Q`; the pair `(x, y, ŷ_1:c)` — the correct solution against the prefix ending at the first bad step — is added to the preference set.
4. Optimize the DPO loss on these per-step pairs with `π_sft` as reference.

Step advantage (Eq. 3), with deterministic dynamics so that the value of a state equals the Q-value of the previous step:

```
A^π̃(x, ŷ_1:i−1; ŷ_i) = Q^π̃(x, ŷ_1:i−1; ŷ_i) − Q^π̃(x, ŷ_1:i−2; ŷ_i−1)
```

## Result and the negative control (§6.2, Fig. 7)
- Per-step DPO improves over the SFT policy and keeps improving as `|D_syn|` grows on GSM8K and MATH, for both models; the paper summarizes the gain as equivalent to 8x more synthetic data.
- Standard DPO on arbitrary correct/incorrect response pairs does not improve over the SFT policy on MATH, and the authors report they could not remove the degradation by tuning `β` (Fig. 7c).
- Pairing by highest edit distance (Pal et al.) improves over standard DPO and stays below per-step DPO (Fig. 7c).
- Theorem 6.1: DPO on pairs that share a prefix sampled from `π_sft`, with the positive continuation drawn according to the advantage under `π̃`, has the same optimum as advantage-weighted RL (Eq. 4).
- Pass@5 of the per-step DPO policy also improves as data grows, although advantages were estimated under a Best-of-5 policy (§6.2).
- Advantage estimates on correct responses identify spurious steps: steps that are correct but do not make progress receive low advantage (§6.3.1, Figs. 8-9).
