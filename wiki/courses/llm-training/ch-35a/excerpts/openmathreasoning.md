---
chapter: ch-35a
course: llm-training
phase: read
excerpt_of: primary source (no library card at wiki/raw-data/llm-training/papers/openmathreasoning.md on 2026-09-15)
source_url: https://arxiv.org/abs/2504.16891
source_version: arXiv v1 (2025-04-23)
created_at: "2026-09-15"
---

# Excerpt: AIMO-2 Winning Solution: Building State-of-the-Art Mathematical Reasoning Models with OpenMathReasoning dataset (Moshkov, Hanley, Sorokin, Toshniwal, Henkel, Schifferer, et al.; NVIDIA)

Facts used by [[read]], read in the arXiv v1 PDF text on 2026-09-15.

## Problem pool (§2.1, Tables 1-3)
- Source: Art of Problem Solving (AoPS) forums, all discussions except "Middle School Math", which the authors "found to be too elementary and unhelpful for training in our preliminary experiments". Qwen2.5-32B-Instruct performs all processing steps unless stated otherwise.
- Steps: (1) LLM problem extraction from first posts; (2) LLM classification as proof or not, multiple choice or not, binary (yes/no) or not, valid or not; multiple-choice, binary, and invalid problems are removed; (3) proof questions are converted into answer-based questions; (4) final-answer extraction for non-proof questions; (5) LLM-based benchmark decontamination following Yang et al. (2023).
- Sizes after each stage (Table 1): 620K forum discussions → 580K extracted problems → 550K after removing "bad" problems → 540K after decontamination.
- Final composition (Table 2): 260K converted proofs, 190K with extracted answer, 90K without extracted answer; 540K total.
- Validation set Comp-Math-24-25 (Table 4): 256 problems from AIME 2024 and 2025 and HMMT 2024-2025, restricted to 2024-2025 competitions "to minimize potential data contamination".

## Solution generation and filtering (§2.3, Table 5)
- Teachers: DeepSeek-R1 and QwQ-32B; up to 32 candidates per problem; temperature 0.7, top-p 0.95, generations limited to 16,384 tokens.
- More solutions are generated for harder problems with known answers; hardness is "an average pass-rate across 32 generations from the Qwen2.5-72B-Math-Instruct model" (model name as printed).
- Solutions that do not reach the expected answer are removed; Qwen2.5-32B-Instruct judges answer equivalence. For problems without an extracted answer and for all converted proofs, the most common answer across candidates is treated as ground truth.
- Table 5, CoT solutions after filtering / all: QwQ-32B 0.5M / 1.0M; DeepSeek-R1 2.7M / 4.2M; total 3.2M / 5.2M.

## Tool-integrated reasoning data (§3)
- Stage-0: LIMO-Qwen-32B prompted to use Python produced at least one code block for roughly half of the problems; 1.2M solutions at temperature 0.7, top-p 0.95, 16,384 tokens, stopped after more than 8 code executions.
- Filters: Qwen2.5-32B-Instruct classifies each code block as novel calculation or verification, and as significant, moderate, or trivial; solutions are kept with at least one novel and significant block or more than half novel and moderate blocks; incorrect answers, no-code solutions, and more than two code blocks are removed; result 15k stage-0 samples.
- Next round: QwQ-32B fine-tuned on stage-0 for 7 epochs at constant LR 5e-6; 700K generated, 260K kept; "novelty and significance filters degrade the performance at this stage, so we do not use them". Final TIR set 1.7M.

## Student training (§5.1, Table 6)
- Students: Qwen2.5-Base 1.5B, 7B, 14B, 32B; the 1.5B and 7B start from the math versions with RoPE base changed to 500K.
- Data: 5.5M samples (3.2M CoT, 1.7M TIR, 566K GenSelect); each task has its own prompt; mixing all tasks gave "similar accuracy" to sequential training.
- Optimization: six epochs; AdamW, weight decay 0.01; cosine decay with 10% linear warmup; peak LR 3e-4 (1.5B), 2e-4 (7B), 1e-4 (14B, 32B); final LR 1000 times smaller; batch 1024 samples; sequence packing and context parallelism; 4 equally spaced checkpoints averaged.
- Second SFT round: Olympiad-forum problems only; problems with pass rate above 0.3 out of 32 generations of the Qwen2.5-Math-72B-Instruct TIR model are discarded; solutions under 5000 tokens are filtered; 2.2M samples; 4 epochs; not applied to 32B "as we found some degradation in results".
- Table 6 (Comp-Math-24-25 maj@64, first SFT → second SFT): 1.5B CoT 55.1 → 58.2; 1.5B TIR 64.1 → 64.5; 7B CoT 61.3 → 62.5; 7B TIR 71.1 → 70.7; 14B CoT 62.9 → 65.2; 14B TIR 74.6 → 73.4.
- Fig. 4: "smaller models need to be trained for longer to achieve meaningful improvements".

## Not reported
Maximum SFT sequence length; teacher sampling seeds; general-domain (non-math) evaluations of the students.
