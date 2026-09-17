---
chapter: ch-43
course: llm-training
phase: read
excerpt_of: "no library card exists for this source as of 2026-09-15; extracted directly from the primary text"
source_url: https://arxiv.org/abs/2503.14476
primary_version: arXiv:2503.14476v2 (20 May 2025)
created_at: "2026-09-15"
---

# Excerpt: DAPO — An Open-Source LLM Reinforcement Learning System at Scale

Yu et al. (ByteDance Seed; Tsinghua AIR), 2025. ch-43 uses two of DAPO's four changes: the removal of the KL
term and Clip-Higher. The full algorithm is covered in ch-40.

## Removing the KL term (§2.3)
> "The KL penalty term is used to regulate the divergence between the online policy and the frozen reference
> policy. In the RLHF scenario, the goal of RL is to align the model behavior without diverging too far from
> the initial model. However, during training the long-CoT reasoning model, the model distribution can
> diverge significantly from the initial model, thus this restriction is not necessary. Therefore, we will
> exclude the KL term from our proposed algorithm."

This is an argument, not a measurement: no ablation with and without the KL term is reported, and no retention
benchmark outside math is evaluated (checked §2.3, §4, Table 1).

## Clip-Higher (§3.1)
- Observation: with naive PPO or GRPO "the entropy of the policy decreases quickly as training progresses
  (Figure 2b). The sampled responses of certain groups tend to be nearly identical."
- Mechanism: at `ε = 0.2` and a positive advantage, a token at `π_old = 0.01` can rise at most to 0.012, while
  a token at `π_old = 0.9` can rise to 1.08 (the bound `π_old·(1 + ε)` is not binding). The measured mean
  probability of up-clipped tokens is below 0.2 (Fig. 3a).
- Change: decouple the clip range into `1 − ε_low` and `1 + ε_high` (Eq. 10) and raise `ε_high`. `ε_low` is
  left at its value "because increasing it will suppress the probability of these tokens to 0, resulting in
  the collapse of the sampling space".
- Values used (§4.1): `ε_low = 0.2`, `ε_high = 0.28`; Qwen2.5-32B base; verl; AdamW at constant 1e-6 with a
  20-step linear warmup; prompt batch 512 with 16 responses per prompt; mini-batch 512 (16 gradient updates
  per rollout step); maximum generation 20,480 tokens (16,384 expected plus a 4,096 soft-punish cache);
  evaluation avg@32 on AIME at temperature 1.0 and top-p 0.7.
- Table 1 (AIME24 avg@32, progressive stack on Qwen2.5-32B base): naive GRPO 30; + overlong filtering 36;
  + Clip-Higher 38; + soft overlong punishment 41; + token-level loss 42; + dynamic sampling (full DAPO) 50.
  DeepSeek-R1-Zero-Qwen-32B is listed at 47. One run per row.

## Verification
- Read on 2026-09-15 from the cached primary text of arXiv:2503.14476v2 (scratchpad `sources/rc03-dapo.txt`).
- Not reported: entropy values in nats; any evaluation outside mathematics; an ablation of `ε_high`.
