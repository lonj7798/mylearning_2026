---
chapter: ch-45a
course: llm-training
phase: read
excerpt_of: "no library card exists for this source as of 2026-09-15; extracted directly from the primary text"
source_url: https://arxiv.org/abs/2505.07608
primary_version: arXiv:2505.07608v2 (5 Jun 2025)
created_at: "2026-09-15"
---

# Excerpt: MiMo-7B — RL hyper-parameters, data re-sampling, difficulty-driven code reward

LLM-Core Xiaomi, 2025. ch-45a uses §3.3 for an RL-ledger row whose batch unit is stated unambiguously.

## Hyper-parameters (§3.3.3, verbatim)
> "In our experiment, we employed a training batch size of 512, with an actor mini-batch size of 32. We executed
> 16 gradient updates per training iteration at a learning rate of 1e-6. The maximum sequence length was set to
> 32,768 tokens to facilitate complex reasoning tasks. During the training phase, both temperature and top-p
> parameters were configured at 1.0 to promote output diversity."

512 / 32 = 16 matches the stated 16 gradient updates, so batch and mini-batch are in the same unit and the run
is 16-step off-policy within each rollout.

## Algorithm (§3.3.2)
Modified GRPO with three changes taken from other work: removal of the KL loss; dynamic sampling (over-sample
and filter out prompts whose group is all-correct or all-wrong); and Clip-Higher (raise ε_high, hold ε_low).
Eq. (1) is written with separate ε_low and ε_high, but no numeric value is printed for either.

## Easy-data re-sampling (§3.3.2)
> "During the training process, we maintain an easy data pool, where problems with perfect pass rates are stored.
> When performing rollouts, there is a probability α (10% in our experiments) to sample data from this easy data
> pool."

The stated purpose is to keep sampling efficiency from collapsing late in training without the instability the
authors saw when re-using easy data directly.

## Data and reward
130K verifiable mathematics and programming problems (§1, §3.2). Code problems use a test-difficulty-driven
reward: test cases are clustered into difficulty levels by pass rate, and a soft scheme awards partial credit
per level instead of a single all-tests-pass bit (§3.3.1). Difficulty filtering for code: problems solved in all
16 rollouts by an SFT version of MiMo-7B are removed (§3.2).

SFT for comparison (§3.1): constant learning rate 3 × 10⁻⁵, batch size 128, samples packed to 32,768 tokens.

## Verification
- Read on 2026-09-15 from the cached primary text of arXiv:2505.07608v2 (scratchpad `sources/c1-mimo7b.txt`),
  §3.1, §3.2, §3.3.1–§3.3.3, §3.4.
- Not reported: group size G for the released run; ε_low and ε_high values; number of RL steps; KL placement is
  moot because the KL loss is removed.
