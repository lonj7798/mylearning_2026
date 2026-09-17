---
chapter: ch-35a
course: llm-training
phase: read
excerpt_of: primary source (no library card at wiki/raw-data/llm-training/papers/acereason-nemotron-1-1.md on 2026-09-15)
source_url: https://arxiv.org/abs/2506.13284
source_version: arXiv v1 (2025-06-16)
created_at: "2026-09-15"
---

# Excerpt: AceReason-Nemotron 1.1: Advancing Math and Code Reasoning through SFT and RL Synergy (Liu, Yang, Chen, Lee, Shoeybi, Catanzaro, Ping; NVIDIA)

Facts used by [[read]], read in the arXiv v1 PDF text on 2026-09-15.

## SFT prompts and responses (§3.1)
- Math prompts: AceMath, NuminaMath, OpenMathReasoning. Code prompts: TACO, APPs, OpenCoder-Stage2, OpenCodeReasoning. Deduplicated so each prompt is unique.
- Decontamination: samples with "a 9-gram overlap with any test sample in math and coding benchmarks" are filtered.
- Teacher: DeepSeek-R1.
- Difficulty balancing: "longer model responses often correspond to more difficult questions"; many responses were "around or below 2,000 tokens", so a subset of these simpler prompts was randomly filtered out and other difficulty levels were resampled. Final pool: 247K math and 136K code prompts (383K).
- Seven SFT sets v1-v7 from 36K to 2.2M samples, with similar response-length distributions; v1 has 18K math and 18K code samples, v7 has 1.2M math and 1.0M code samples (§3.1.2, §4.4.1).
- Student: Qwen2.5-Math-7B; rope_theta changed from 10,000 to 1,000,000 for a 128K context.

## Scaling results (§4.4)
- v1-v4 increase unique prompts with one response each; from v5, prompts and responses per prompt both increase; v7 keeps a similar number of prompts as v6 but "nearly twice as many responses per prompt", and AIME25 rises from 41.3 to 49.3.
- Regression over the 7 datasets: z = a·log2 x + b·log2 y + c, where x is unique prompts, y responses per prompt, z the average accuracy over AIME24, AIME25, LiveCodeBench v5 and v6; log2 x and log2 y are standardized before fitting. Estimates a = 4.831, b = 2.635, R² = 0.989.
- Authors' reading: increasing unique prompts "may have a greater impact"; when prompts are hard to collect, more responses per prompt is "a practical alternative".
- Epochs (Fig. 6, v6 and v7): accuracy improves from epoch 1 to 5 and plateaus around epoch 5-6; the authors attribute the benefit of "a certain degree of 'overfitting'" to exposure bias (Interpretation).
- SFT learning rate, batch size, and maximum training length are not printed.

## SFT then RL (§3.2, §4.3, §4.5)
- RL: GRPO, strictly on-policy, G = 8 or 16 rollouts, 128 prompts per batch, token-level loss, no KL term; stages math 8K → 16K → 24K, code 24K → 32K, math 32K.
- Table 1 (avg@64 AIME, avg@8 LCB): Our SFT-7B 62.0 / 48.4 AIME24/25, 48.8 / 43.8 LCB v5/v6; AceReason-Nemotron-1.1-7B 72.6 / 64.8, 57.2 / 52.1; DeepSeek-R1-Distill-Qwen-7B 55.5 / 39.0, 37.6 / 34.1.
- Fig. 7: RL from SFT v5 and v7 narrows the AIME24 gap from 6.6% to 1.6%.
- Evaluation noise (§4.1): on AIME2024, avg@16, avg@32, avg@64 give standard deviations of 1.8, 1.2, 0.7.
- RL data (§3.2.2): "incorrect test cases can lead to false negative rewards, while overly simple test cases may result in false positives".

## Not reported
SFT optimizer settings; general-domain evaluations; safety evaluations.
