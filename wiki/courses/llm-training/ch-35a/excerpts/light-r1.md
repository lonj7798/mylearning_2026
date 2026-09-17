---
chapter: ch-35a
course: llm-training
phase: read
excerpt_of: primary source (no library card at wiki/raw-data/llm-training/papers/light-r1.md on 2026-09-15)
source_url: https://arxiv.org/abs/2503.10460
source_version: arXiv v4 (2025-05-28); v1 2025-03
created_at: "2026-09-15"
---

# Excerpt: Light-R1: Curriculum SFT, DPO and RL for Long COT from Scratch and Beyond (Wen, Cai, Xiao, He, An, Duan, et al.; Qiyuan Tech, Renmin University)

Facts used by [[read]], read in the arXiv v4 PDF text on 2026-09-15.

## Data (§3.1, App. B-C)
- About 1000k math questions collected; only questions with ground-truth answers kept; categories with excessive data downsampled using an in-house tagging system (§3.1.1).
- Decontamination: exact matching "excluding digits to filter questions with only numerical changes" and 32-gram matching against AIME24 and AIME25, MATH-500, GPQA (§3.1.2).
- App. C Table 7, matched prompts against MATH-500: OpenThoughts-114k 100; OpenR1-Math-220k 10; DeepScaleR-Preview-Dataset 196; LIMO 0; Bespoke-Stratos-17k 125; Open-Reasoner-Zero 325; data_ablation_full59K 244; s1K-1.1 3; Light-R1 data 0. AIME24+25 matches are 0 for all listed datasets.
- Stage-1 difficulty filter: DeepScaleR-1.5B-Preview generates responses; "only questions with a pass rate < α were selected for DeepSeek-R1 queries, resulting in approximately 76k data points". The value of α is not printed. For questions with several correct R1 responses, one is chosen at random (§3.1.3).
- Stage-2 filter: DeepSeek-R1 replaces the 1.5B model; questions with pass rate < α and with R1 samples "neither uniformly correct nor uniformly incorrect" are kept; about 3k examples (§3.1.3).
- Motivation for a second stage: in preliminary 32B experiments "approximately 20% of training data still exhibited pass rates below 50% across 10 runs" (§1).

## Training (§3.2-3.4, Table 8)
- Curriculum: SFT stage 1 (76k), SFT stage 2 (3k), DPO with NCA loss; rejected responses sampled from the SFT-stage-2 model and verified incorrect; chosen responses are verified correct DeepSeek-R1 answers; "using chosen responses from significantly stronger models yielded better results" than fully on-policy DPO for hard math (§3.2).
- Table 8 (LR / batch / sequence length): Light-R1-32B SFT stage 1 5.0e-5 / 96 / 20k; SFT stage 2 1.0e-5 / 32 / 20k; DPO 5.0e-7 / 16 / 32k; Light-R1-7B-DS 5.0e-6 / 32 / 20k; Light-R1-14B-DS-SFT 5.0e-6 / 32 / 20k; Light-R1-14B-DS GRPO 1.0e-6 / 128 / 24k; Light-R1-32B-DS 5.0e-6 / 32 / 20k. Epochs are not printed.
- Table 2 (Light-R1-32B from Qwen2.5-32B-Instruct; AIME24 / AIME25 / GPQA / LCB): base 16.6 / 13.6 / 48.8 / 24.6; +SFT stage 1 69.0 / 57.4 / 64.3 / 42.9; +SFT stage 2 73.0 / 64.3 / 60.6 / 42.0; +DPO 75.8 / 63.4 / 61.8 / N/A; +merging 76.6 / 64.6 / 61.8 / 44.7. Caption: "a decrease in GPQA (Science QA) scores beginning from SFT-stage2, indicating a partial degradation of the model's generalization capabilities during extensive math-focused training".
- Table 3 (3k stage-2 data on DeepSeek-R1-Distill models): 7B 55.5 / 39.2 / 49.1 / 34.6 → 59.1 / 44.3 / 49.4 / 38.4; 32B 72.6 / 54.9 / 62.1 / 58.8 → 78.1 / 65.9 / 68.0 / 66.1. Light-R1-7B-DS "exhibits improvements confined solely to in-domain tasks" (§3.4).

## RL data selection (§4)
- Light-R1-7B-DS samples RL prompts; prompts with pass rate between 0.25 and 0.625 are kept.
- "When manually checking data with a pass rate of 0, we found that over half of the prompt answers are either unverifiable ... or incorrect"; a model verifier re-checks pass-rate-0 data.
- Evaluation: 64 samples per query; "large deviation of over 3 points using 16 responses or fewer across different runs of the same model" (§2).

## Not reported
α; teacher sampling temperature and maximum length; SFT epochs; loss masking.
