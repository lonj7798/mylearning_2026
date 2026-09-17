---
chapter: ch-45a
course: llm-training
phase: read
excerpt_of: "no library card exists for this source as of 2026-09-15; extracted directly from the primary text, the released launch scripts, and the dataset card"
source_url: https://arxiv.org/abs/2505.22312
primary_version: "arXiv:2505.22312v2 (29 May 2025); github.com/SkyworkAI/Skywork-OR1 launch scripts; huggingface.co/datasets/Skywork/Skywork-OR1-RL-Data card"
created_at: "2026-09-15"
---

# Excerpt: Skywork-OR1 — multi-stage RL settings and prompt filter

He, Liu, Liu et al. (Skywork AI, Kunlun Inc), 2025. ch-45a uses this source for a multi-stage RL-ledger row and
for the prompt-filter column.

## Shared settings (§8.1)
> "All three models are fine-tuned by optimizing the policy loss (3.1) with a constant learning rate of 1e-6,
> clip ratio of 0.2, target entropy of 0.2, sampling temperature of 1.0, and rejection sampling. Notably, we do
> not apply any KL loss in our training process."

Eq. (3.1) is a GRPO-style clipped objective with an added entropy term α_k H, where α_k is adjusted online to
hold entropy at the target (§3.2.5, "adaptive entropy control"). §3.2.6 states the reason for dropping the KL
loss: "The KL penalty hinders further improvements in test performance during multi-stage training."

"Rejection sampling" here has a specific meaning (§3.1): "Responses in the zero-advantage group ... do not
contribute to the policy loss but may influence the KL loss or entropy loss, potentially leading to a more
unstable training process due to the implicitly increased relative weight of these losses. To mitigate this
issue, our training batches include only groups with non-zero advantages". A separate online filter drops
prompts the actor solved with correctness 1 in the previous stage.

Truncated responses are *not* masked: "penalizing truncated responses does not hinder later-stage improvements
and enhances token efficiency. Based on these results, we do not employ any advantage mask strategy in our
training pipeline" (§3.1 item 2, evidence in §3.2.3).

## Per-stage configurations (Tables 10–12)
Skywork-OR1-Math-7B (released checkpoint: step 2160)

| Stage | Steps | Context length | Batch | Mini-batch | Group size |
|---|---|---|---|---|---|
| 1 | 0–740 | 8K | 256 | 128 | 16 |
| 2 | 740–1740 | 16K | 256 | 128 | 16 |
| 3 | 1740–2080 | 32K | 256 | 128 | 16 |
| 3.5 | 2080–2160 | 32K | 128 | 64 | 64 |

Skywork-OR1-7B (released checkpoint: step 1320)

| Stage | Steps | Context length | Batch | Mini-batch | Group size |
|---|---|---|---|---|---|
| 1 | 0–660 | 16K | 256 | 256 | 16 |
| 2 | 660–1320 | 32K | 160 | 160 | 32 |

Skywork-OR1-32B (released checkpoint: step 1000)

| Stage | Steps | Context length | Batch | Mini-batch | Group size |
|---|---|---|---|---|---|
| 1 | 0–760 | 16K | 256 | 256 | 16 |
| 2 | 760–1130 | 24K | 160 | 160 | 32 |

In the two general models, batch equals mini-batch, so each rollout produces exactly one gradient update
(strictly on-policy). In Math-7B, batch is twice the mini-batch, so each rollout produces two updates.

## Launch script (`32b_16k`, released repo)
```
ENTROPY_COEFF=0.0
USE_ADAPTIVE_ENT=True
TGT_ENTROPY=0.2
MAX_ENT_COEF=0.005
MIN_ENT_COEF=0
DELTA_ENT_COEF=0.0001
ROLLOUT_BATCH_SIZE=256
PPO_MINI_BATCH=256
MAX_PROMPT_LENGTH=2048
RES_LENGTH=16384
GROUP_SIZE=16
TRAIN_TEMPERATURE=1.0
```
with `actor_rollout_ref.actor.optim.lr=1e-6`, `actor_rollout_ref.rollout.val_temperature=0.6`, run through
`verl.trainer.main_ppo`.

## Prompt filter
Dataset card (huggingface.co/datasets/Skywork/Skywork-OR1-RL-Data): math split 105,055 examples, code split
14,057 examples.
> "For our final training phase, we filtered problems based on their difficulty levels (0-16, higher values
> indicate harder problems) relative to specific model variants (DeepSeek-R1-Distill-Qwen-{1.5,7,32}B. For each
> model variant, we excluded problems with difficulty values of 0 and 16 specific to that model from its
> training data."

The released 32B preprocessing script implements exactly that: `if difficulty < 1 or difficulty > 15: return
False`, keyed on `model_difficulty['DeepSeek-R1-Distill-Qwen-32B']`. The difficulty field is per problem *and*
per model, so the same prompt pool yields a different filtered set for each policy size.

The same script builds the training file list as
`["…train_32b_code.pkl", "…train_32b_code.pkl", "…train_32b_math.pkl"]`, i.e. the code split is listed twice and
the math split once, while the comment above the line reads "Since math queries are much more than code queries,
we duplicate the math data when mixing the datasets". The comment and the code describe opposite duplications.

## Evaluation setup (§8.1)
Maximum generation length 32,768 for all models; AIME24/25 avg@32; LiveCodeBench (2024-08 to 2025-02) avg@4;
temperature 1, top-p 1. Results (Table 13): Skywork-OR1-32B 82.2 AIME24, 73.3 AIME25, 63.0 LiveCodeBench;
Skywork-OR1-7B 70.2 / 54.6 / 47.6.

## Verification
- Read on 2026-09-15 from the cached primary text of arXiv:2505.22312v2 (scratchpad `sources/skywork-or1.txt`,
  §3.2, §8.1, Tables 10–13), the cached launch scripts (`sources/skywork-or1-script-32b_16k.txt`), the cached
  preprocessing script (`sources/skywork-or1-filter-32b.txt`), and the cached dataset card
  (`sources/skywork-or1-rl-data-readme.txt`).
- Not reported: total GPU-hours; optimizer betas; how "difficulty" is computed from the rollout counts.
