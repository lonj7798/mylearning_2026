---
chapter: ch-45a
course: llm-training
phase: read
excerpt_of: "no library card exists for this source as of 2026-09-15; extracted directly from the primary text"
source_url: https://arxiv.org/abs/2505.16400
primary_version: arXiv:2505.16400 (AceReason-Nemotron, 23 May 2025)
created_at: "2026-09-15"
---

# Excerpt: AceReason-Nemotron — stage-wise length extension and cross-domain transfer

Liu, Yang, Chen, Lee, Shoeybi, Catanzaro, Ping (NVIDIA), 2025. ch-45a uses this source for a staged RL-ledger row and for the
generalization lens.

## Objective and settings (§3.2.2)
The objective is GRPO with the importance weight fixed at r_i,t(θ) = 1 (one gradient update per generation) and
β = 0: "we eliminate the KL divergence term by setting β = 0".

> "We used a batch size of 128, sampling G = 8 responses per prompt for 8K length training and 16 responses
> otherwise. We adopted a learning rate of 1 × 10⁻⁶ with AdamW (Kingma, 2014), and set both the entropy loss
> coefficient and KL loss coefficient β to 0."

Code-only RL (§3.3.2): two stages; stage 1 at maximum response length 24,000, temperature 0.6, 8 rollouts;
stage 2 at 32,768 with temperature raised from 0.6 to 1.0 and rollouts raised from 8 to 16; batch size 128,
learning rate 5 × 10⁻⁶ with AdamW.

## The three design decisions (§3.2.2, verbatim summaries)
1. Strict on-policy: "applying multiple (2 or 4) gradient updates after model generation with a group of G
   rollouts per prompt led to rapid entropy collapse around 100 steps (see Figure 3c). In contrast, using
   exactly one gradient update after model generation... consistently prevented collapse."
2. Stage-wise length extension 8K → 16K → 24K → 32K: "directly starting from 16K or 24K resulted in suboptimal
   results (see Figure 3b)". At 8K the model first loses accuracy while compressing its reasoning and recovers
   "after approximately 1K–2K RL steps"; extending to 16K then produces an immediate gain (§4.3.3).
3. Curriculum by pass rate: at the 24K and 32K stages, "We filtered prompts by model pass rate, filtering out
   those with pass rate > 6/16, which significantly improves model performance (Table 3)."

## Cross-domain transfer (Table 1)
Math-only RL, evaluated with the DeepSeek-R1 protocol (temperature 0.6, top-p 0.95, max 32,768 tokens):

| Model | AIME24 avg@64 | AIME25 avg@64 | LiveCodeBench v5 avg@8 |
|---|---|---|---|
| DeepSeek-R1-Distill-Qwen-7B | 55.5 | 39.0 | 37.6 |
| AceReason-Nemotron-7B (math-only RL) | 69.0 | 53.6 | 44.4 (+6.8) |
| DeepSeek-R1-Distill-Qwen-14B | 69.7 | 50.2 | 53.1 |
| AceReason-Nemotron-14B (math-only RL) | 78.6 | 67.4 | 58.9 (+5.8) |
| OpenMath-Nemotron-14B (math-only SFT) | 76.3 | 63.0 | 19.3 |

The paper's reading: "math-only RL improves coding performance across all problem topics—not just math-related
coding tasks (see Figure 4)", while "domain-specific supervised fine-tuning (SFT) often results in poor
performance on other domains".

## Verification
- Read on 2026-09-15 from the cached primary text of arXiv:2505.16400 (scratchpad
  `sources/acereason-nemotron.txt`), §3.2.2, §3.3.2, §4.3.3, Tables 1 and 3.
- Not reported: clip ε; total number of RL steps per stage; GPU count and hours.
