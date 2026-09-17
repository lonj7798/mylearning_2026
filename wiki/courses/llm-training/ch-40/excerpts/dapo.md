---
chapter: ch-40
course: llm-training
phase: read
excerpt_of: arXiv:2503.14476v2 (DAPO: An Open-Source LLM Reinforcement Learning System at Scale)
source_url: https://arxiv.org/abs/2503.14476
created_at: "2026-09-15"
revised: "2026-09-15 (generality revision; written from the primary source — the library has no DAPO card)"
---

# Excerpt: DAPO — four changes to GRPO, measured cumulatively

Used by [[read]] §7, the Recipe, and the Common-mistakes table. ByteDance Seed, Institute for AI Industry Research (AIR) Tsinghua, HKU, SIA-Lab. Project lead Qiying Yu; algorithm authors Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, Yu Yue. Paper date 17 March 2025; text read from arXiv v2 (20 May 2025) on 2026-09-15.

## Setting
Qwen2.5-32B base, mathematics only, verl framework, DAPO-Math-17K (17K prompts, integer answers obtained by rewriting problems so the answer is easy to parse, §3.5). Reward: `R(ŷ, y) = 1` if `is_equivalent(ŷ, y)` else `−1` (Eq. 7). No KL term: "during training the long-CoT reasoning model, the model distribution can diverge significantly from the initial model, thus this restriction is not necessary" (§2.3).

## Objective (Eqs. 8–12)
```
J_DAPO(θ) = E[(q,a)~D, {o_i}~π_θold]
    1/(Σ_i |o_i|) Σ_i Σ_t min( r_{i,t}(θ) Â_{i,t}, clip(r_{i,t}(θ), 1−ε_low, 1+ε_high) Â_{i,t} )
    s.t. 0 < |{o_i : is_equivalent(a, o_i)}| < G
Â_{i,t} = (R_i − mean({R_i})) / std({R_i})     r_{i,t}(θ) = π_θ(o_{i,t}|q,o_{i,<t}) / π_θold(o_{i,t}|q,o_{i,<t})
```
The advantage keeps GRPO's std normalization; the changes are the clip bounds, the group constraint, and the `1/Σ_i |o_i|` normalizer.

## The four techniques
1. **Clip-Higher (§3.1).** ε_low = 0.2, ε_high = 0.28. Argument: with ε = 0.2 and `Â > 0`, a token at π_θold = 0.9 may rise to 1.08 (unbounded in effect) while one at 0.01 is capped at 0.012. ε_low is not raised because that would suppress token probabilities to 0 and collapse the sampling space. Effect shown on entropy and AIME accuracy (Fig. 2); mean probability of up-clipped tokens is below 0.2 (Fig. 3a).
2. **Dynamic Sampling (§3.2).** Prompts whose G samples are all correct or all wrong give a zero advantage and no gradient; the share of prompts with accuracy 1 increases during training (Fig. 3b). Over-sample and refill the batch with groups that have non-zero advantage. Convergence time was not increased despite extra sampling (Fig. 6).
3. **Token-level policy-gradient loss (§3.3).** Sample-level averaging gives tokens in long responses a lower weight, so gibberish and repetition in long samples are under-penalized; entropy and length rise (Fig. 4). Token-level normalization weights every token in the batch equally.
4. **Overlong Reward Shaping (§3.4).** *Overlong Filtering* masks the loss of truncated samples. *Soft Overlong Punishment* (Eq. 13): `R_length = 0` for `|y| ≤ L_max − L_cache`; `((L_max − L_cache) − |y|)/L_cache` inside the buffer; `−1` beyond `L_max`. Reason given: "a sound reasoning process can be penalized solely due to its excessive length".

## Results (Table 1, AIME 2024 avg@32, cumulative)
Naive GRPO 30 → +Overlong Filtering 36 → +Clip-Higher 38 → +Soft Overlong Punishment 41 → +Token-level Loss 42 → +Dynamic Sampling (DAPO) 50. Reference: DeepSeek-R1-Zero-Qwen-32B 47, reached by DAPO in 50% of the training steps (Fig. 1). One run per row.

## Training details (§4.1)
AdamW, constant LR 1e-6 with linear warm-up over 20 rollout steps; rollout prompt batch 512 with 16 responses per prompt; training mini-batch 512, i.e. 16 gradient updates per rollout step; expected max length 16,384 with a 4,096-token soft-punish cache, so generation cap 20,480; evaluation avg@32 at temperature 1.0, top-p 0.7.

## Monitoring statements (§4.3)
Response length can stagnate or decline for long stretches and is read together with validation accuracy. "The final reward on the training set often exhibits little correlation with the accuracy on the validation set, which indicates overfitting to the training set." Entropy is kept in a range and a slow upward trend was associated with better performance (Figs. 7c–7d).

## Notes for the chapter
- Training only on high-entropy tokens is *not* a DAPO technique; it comes from arXiv:2506.01939 ("Beyond the 80/20 Rule"), which TRL cites separately ([[trl-grpo]] grpo_config.py L276–281).
- Conflict on overlong filtering: Table 1 lists it as a cumulative step, while the verl documentation states that most experiments in the paper, including the best-performing one, ran without it because it overlaps with Overlong Reward Shaping ([[verl-grpo]], docs/algo/dapo.md FAQ).
- verl reproduction on Qwen2.5-32B: 52% with dynamic sampling, 50% without, 44% without token-level loss and dynamic sampling (docs/algo/dapo.md L36–38; one run each, the 44% run on different hardware).
