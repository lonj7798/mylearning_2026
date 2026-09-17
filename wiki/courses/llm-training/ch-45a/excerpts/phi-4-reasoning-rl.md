---
chapter: ch-45a
course: llm-training
phase: read
excerpt_of: "the Phi-4-reasoning technical report's §4; the library card [[phi-4]] summarizes this stage from search-surfaced text and states a reward shape the report does not use"
source_url: https://arxiv.org/abs/2504.21318
primary_version: arXiv:2504.21318v1 (30 Apr 2025)
created_at: "2026-09-15"
---

# Excerpt: Phi-4-reasoning-plus — the 90-step GRPO stage

Microsoft Research, 2025. ch-45a uses §4 for the shortest RL run in the ledger.

## Data and scale (§4)
> "The RL training focused exclusively on mathematical reasoning. The seed dataset for GRPO consisted of 72,401
> mathematical problems (prompts without solutions), from which we subsample 64 problem seeds per RL iteration...
> even performing RL over a small set of 6,400 problems significantly improved accuracy... the seed data
> contained no coding exercises."

## Hyper-parameters (§4.2, verbatim)
> "We leverage the verl framework [49] for GRPO training with the reward signal defined above. Hyper-parameters
> for the RL training are: a global batch size of 64 across 32 Nvidia H100 GPUs, Adam optimizer learning rate
> 5 × 10⁻⁸ with cosine warm-up in the first 10 steps, GRPO group size of G = 8, KL regularization of β = 0.001
> and entropy coefficient of γ = 0.001. The Phi-4-reasoning-plus was trained with 32k maximum length."

The printed objective places the KL against π_θold, the rollout policy, not against a frozen reference:
`… − β D_KL(π_θ ‖ π_θold) + γ Entropy(π_θ)`, with the group-relative advantage standardized by the group's
reward mean and standard deviation.

> "We select as our RL checkpoint the model with the best observed AIME 2024 score, which is the model trained
> for 90 steps, over only ~6k examples (and 8 trajectories of responses per example)."

> "additional GRPO training for only 90 steps boosts AIME performance by more than 10% (Figure 7a). Further
> training for more steps does not translate to additional gains."

## Reward (§4.1)
The reward is not "+1 correct, −0.5 incorrect". It is a length-aware cosine-scaled accuracy term plus a
repetition penalty:
- Correct answers: R_acc_scaled ranges from +0.5 to +1.0, decreasing with length once the response exceeds
  L_pos_control = 25,600 tokens, with L_max = 31,744.
- Incorrect answers: R_acc_scaled ranges from −1.0 to −0.5, *increasing* with length up to
  L_neg_control = 3,702 tokens, so a longer wrong answer is penalized less than a short wrong answer.
- Format overrides: a missing `<|im_end|>` sets R_acc_scaled = −0.5; an incorrect or missing `<think>` tag sets
  it to −1.0.
- Repetition penalty R_rep from 5-gram frequencies.
- Final reward `R_final = w_acc · R_acc_scaled + w_rep · R_rep` with `w_acc = 8/13` and `w_rep = 1/13`.

Outputs longer than 31k tokens are clipped to their first 31k tokens during GRPO, and the report names this cap
as a limit on how far the stage can go: "the fact that we clip responses beyond 31k output tokens during GRPO,
which limits the extent to which GRPO can help".

## SFT stage for comparison (§3)
About 16K steps, global batch 32, context 32K, AdamW, learning rate 1e-5 with linear warm-up over 450 steps,
weight decay 1e-4; the learning rate was chosen by a grid search over [1e-6, 2e-5].

## Verification
- Read on 2026-09-15 from the cached primary text of the Phi-4-reasoning technical report (scratchpad
  `sources/phi-4-reasoning.txt`), §3, §4, §4.1, §4.2, Figure 7.
- Not reported: the clip parameter ε (it appears only as a symbol in the objective); rollout temperature;
  optimizer betas.
