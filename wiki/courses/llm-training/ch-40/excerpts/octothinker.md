---
chapter: ch-40
course: llm-training
phase: read
excerpt_of: arXiv:2506.20512v1 (OctoThinker: Mid-training Incentivizes Reinforcement Learning Scaling)
source_url: https://arxiv.org/abs/2506.20512
created_at: "2026-09-15"
revised: "2026-09-15 (generality revision; written from the primary source — the llm-training library has no OctoThinker card; ch-32 keeps a separate excerpt)"
---

# Excerpt: the base model decides how far RL goes

Used by [[read]] §9 and the Generalization lens. Authors: Zengzhi Wang, Fan Zhou, Xuefeng Li, Pengfei Liu (SJTU, SII, GAIR Lab). arXiv v1, 25 June 2025; read on 2026-09-15.

## Question
Why does R1-Zero-style RL reproduce on Qwen base models and not on Llama? The paper holds the RL recipe fixed and changes what happens before RL.

## RL setup (§2)
GRPO; MATH8K prompt set; batch size 128; 16 rollout responses per query; PPO mini-batch 64. Base models: Llama-3.2-3B-Base and Qwen2.5-3B-Base.

## Headline numbers (Fig. 1, MATH500, values as labelled in the figure)
Llama-3.2-3B-Base 7.4 → 10.0 after zero RL. Qwen2.5-3B-Base 38.2 → 66.4 after zero RL. After mid-training: OctoThinker-Stable-3B-Base 22.4, OctoThinker-Long-3B-Base 25.8, and OctoThinker-Long-3B-Zero 65.2 after the same RL.

## Mid-training findings (Abstract, §3–§5)
1. Corpus quality decides the RL outcome: MegaMath-Web-Pro improves both the base model and its RL result, while FineMath-4plus does not in the tested setting.
2. Adding QA-style data, particularly long chain-of-thought examples, improves RL outcomes further; a small amount of instruction data adds to this.
3. Long-CoT data increases reasoning depth but also verbosity and RL instability, so data formatting matters.
4. Scaling mid-training helps: checkpoints at 20B, 70B, and 100B tokens give increasing post-RL performance, with the 70B and 100B checkpoints close to each other and both above 20B.
5. Recipe: Stable-then-Decay — 200B tokens at a constant learning rate, then 20B tokens with learning-rate decay across three CoT-focused branches. A 70B-token math corpus, MegaMath-Web-Pro-Max, is released.

## Statement the chapter uses
RL performance is not a property of the RL algorithm alone: two base models of the same size under the same GRPO recipe end 56 MATH500 points apart, and most of that gap is closed by changing what the Llama model saw before RL. Algorithm rankings measured on one family therefore carry limited information about another.

## Limits
3B models, mathematics, one RL algorithm, figure-level reporting for the headline comparison.
