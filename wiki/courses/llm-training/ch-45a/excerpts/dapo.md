---
chapter: ch-45a
course: llm-training
phase: read
excerpt_of: "no library card exists for this source as of 2026-09-15; extracted directly from the primary text"
source_url: https://arxiv.org/abs/2503.14476
primary_version: arXiv:2503.14476v2 (20 May 2025)
created_at: "2026-09-15"
---

# Excerpt: DAPO — training details and the progressive ablation

Yu et al. (ByteDance Seed; Tsinghua AIR), 2025. ch-45a uses §4.1 as an RL-ledger row and Table 1 as the
"evidence for this value" column of that row. The algorithm itself is covered in ch-40 and ch-43a.

## §4.1 Training details (verbatim)
> "We adopt the verl framework [20] for training. We use naive GRPO [38] as our baseline algorithm and estimate
> advantages using group reward normalization. For hyper-parameters, we utilize the AdamW [39] optimizer with a
> constant learning rate of 1 × 10⁻⁶, incorporating a linear warm-up over 20 rollout steps. For rollout, the
> prompt batch size is 512 and we sample 16 responses for each prompt. For training, the mini-batch size is set
> to 512, i.e., 16 gradient updates for each rollout step. For Overlong Reward Shaping, we set the expected
> maximum length as 16,384 tokens and allocate additional 4,096 tokens as the soft punish cache. Therefore, the
> maximum number of tokens for generation is set to 20,480 tokens. As for the Clip-Higher mechanism, we set the
> clipping parameter ε_low to 0.2 and ε_high to 0.28... For evaluation on AIME, we repeat the evaluation set for
> 32 times and report avg@32 for results stability. The inference hyperparameters of evaluation are set to
> temperature 1.0 and top_p 0.7."

Unit note: "prompt batch size is 512" counts prompts, "mini-batch size is set to 512" counts samples. The two
512s are different quantities; 512 prompts × 16 responses = 8,192 samples, and 8,192 / 512 = the stated 16
gradient updates per rollout step.

Base model: Qwen2.5-32B base (§4.2). Data: DAPO-Math-17K, 17K prompts each with an integer answer (§3.5).

## Table 1 — progressive techniques, AIME 2024 avg@32, one run per row
| Configuration | AIME24 avg@32 |
|---|---|
| DeepSeek-R1-Zero-Qwen-32B (reference) | 47 |
| Naive GRPO | 30 |
| + Overlong Filtering | 36 |
| + Clip-Higher | 38 |
| + Soft Overlong Punishment | 41 |
| + Token-level Loss | 42 |
| + Dynamic Sampling (full DAPO) | 50 |

§4.2: DAPO reaches this "with only 50% of the training steps required by DeepSeek-R1-Zero-Qwen-32B".

## Verification
- Read on 2026-09-15 from the cached primary text of arXiv:2503.14476v2 (scratchpad `sources/rc03-dapo.txt`),
  §3.5, §4.1, §4.2, Table 1.
- Not reported: total training steps of the released run, KL coefficient (the paper removes the KL term, §2.3),
  the over-sampling factor used to refill a batch after dynamic sampling, any evaluation outside mathematics.
