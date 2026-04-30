---
chapter: ch-38
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/rlhf-instructgpt.md
source_url: https://arxiv.org/abs/2203.02155
created_at: "2026-04-23"
---

# Excerpt: InstructGPT Equation 2 — the PPO-ptx objective

**Source library:** `wiki/raw-data/llm-training/papers/rlhf-instructgpt.md`
**Artifact:** Equation 2, β=0.02, γ=27.8, hyperparameter table.

---

## Why this source anchors ch-38

Ch-38 §4 is built around Equation 2. Every other RLHF variant (Llama-2, Anthropic RLHF, DeepSeek) is best read as a modification of this equation: dual RM (two `r_φ` terms), γ=0 (plain InstructGPT), different β schedule.

---

## Equation 2 verbatim

From the source:

> `objective(φ) = E_{(x,y)~D_RL}[ r_φ(x,y) − β log(π_φ^RL(y|x) / π^SFT(y|x)) ] + γ · E_{x~D_pretrain}[ log π_φ^RL(x) ]`
> — First term = RM score. `β log(π/π_ref)` = per-token KL penalty folded into the reward (not into the loss) — "KL-control" / "KL-reward" style. `γ · L_ptx` = pretraining loss mixed back in; prevents alignment tax. Optimized with standard PPO-clip (ε=0.2) over this shaped reward.

Ch-38 §4 walks through the three terms one by one and emphasizes the "folded into the reward not the loss" point because it is the single most commonly mis-coded detail in PPO-LLM implementations (see [[trl-ppo]] `non_score_reward` vs `pg_loss`).

---

## Canonical hyperparameters ch-38 quotes

From the source table:

> SFT LR 9.65e-6 (cosine), SFT epochs 16, RM size 6B, RM LR 9e-6, **PPO LR 1.41e-5 (fixed)**, PPO batch size 512 prompts, PPO rollout length ≤ 2048 tokens, **KL coef β 0.02** (adaptive controller optional), **Pretraining coef γ 27.8** (InstructGPT-ptx) or 0 (InstructGPT), Clip ε 0.2, Epochs per rollout 4.

The bolded entries are what ch-38 §4 puts in the hyperparameter table and then contrasts against Llama-2's (§5).

---

## Entropy handling

The source is explicit about why no entropy bonus:

> No explicit entropy bonus in the objective; the β KL term to π_SFT serves the same regularizing role. Entropy collapse is instead tracked as a failure signal and controlled via β annealing.

Ch-38 §4 uses this verbatim when explaining why `c_2 = 0` in the RLHF version of PPO's combined objective.

---

## Adaptive KL — implementation pointer

The source mentions "adaptive controller optional." Ch-38 §7 picks this up through [[openrlhf-ppo]]'s `AdaptiveKLController` and [[kl-control-rlhf]]'s description of "multiplicatively raise β when KL exceeds target, lower when below" — the original InstructGPT recipe, now standard.

---

## What ch-38 keeps, changes, drops

Keeps: Equation 2, β=0.02, γ=27.8 option, ε=0.2, 4 epochs. Changes: γ=1.0 (Schulman discount, not the ptx coef), uses k3 KL estimator not k1. Drops: labeler-protocol details (covered in a different chapter on preference data).
