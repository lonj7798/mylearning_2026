---
chapter: ch-35a
course: llm-training
phase: read
excerpt_of: primary source (no library card at wiki/raw-data/llm-training/papers/opencodereasoning.md on 2026-09-15)
source_url: https://arxiv.org/abs/2504.01943
source_version: arXiv v2 (2025-08-07), COLM 2025; v1 2025-04
created_at: "2026-09-15"
---

# Excerpt: OpenCodeReasoning: Advancing Data Distillation for Competitive Coding (Ahmad, Narenthiran, Majumdar, Ficek, Jain, Huang, et al.; NVIDIA)

Facts used by [[read]], read in the arXiv v2 PDF text on 2026-09-15.

## Questions (§2.1, Table 1)
- Sources: TACO, APPS, CodeContests, and CodeForces from the OpenR1 project; exact-match deduplication gives 28,904 distinct questions.
- Table 1 lists 10 platforms (questions / Python samples): AIZU 2,151 / 62,476; AtCoder 2,080 / 47,222; CodeChef 3,869 / 72,925; CodeForces 10,403 / 388,405; Codewars 2,515 / 34,326; GeeksForGeeks 2,674 / 37,602; HackerEarth 2,285 / 59,181; HackerRank 914 / 10,955; Kattis 1,236 / 13,095; LeetCode 777 / 10,525; total 28,904 / 736,712.
- Contamination check against LiveCodeBench, CodeContests, HumanEval, MBPP: nearest benchmark neighbor by cosine similarity (threshold 0.7), then Llama-3.3-70B-Instruct and Qwen2.5-32B-Instruct as semantic-similarity judges; manual inspection of the 90 flagged samples (≤ 0.3% of the dataset) found no paraphrases, so all 28,904 questions were kept.

## Generation and post-processing (§2.2-2.3)
- Teacher: DeepSeek-R1; multiple solutions per question; temperature 0.6, top-p 0.95; `<think>` tag injected to force reasoning; SGLang with maximum output length 16k tokens.
- Post-processing: reasoning enclosed in `<think>`/`</think>`; a python or cpp code block present in the solution segment; responses with code blocks inside the reasoning removed; syntax checked with Tree Sitter. Result: 736,712 Python and 355,792 C++ samples.

## Scaling (§2.4, Fig. 3)
- Data expanded in stages from 25k to 736k samples: 13k CodeContests questions first; then about 4.5k hard CodeContests questions with multiple solutions; then the full 28k question set.
- "The scaling curve in Figure 3 does not plateau despite the use of 736k samples"; the largest gains came from "increasing the number of unique, varied, and difficult questions".

## Training (§3)
- Qwen2.5 base and instruct, 7B, 14B, 32B; 3 epochs; AdamW; batch size 256; maximum sequence length 32,768; LR grid [1e-5, 3e-5, 5e-5, 8e-5, 1e-4], 5e-5 "optimal across model sizes"; cosine annealing, warmup ratio 0.1; final checkpoint; sequence packing; BF16.
- Inference temperature sweep [0.0, 0.2, 0.6, 0.7, 1.0]: 0.6 and 0.7 best; 0.6 used; maximum generation 30,720 tokens. LiveCodeBench 2408-2502 (279 problems), average of 64 runs; CodeContests 16 runs.
- Table 2 (LiveCodeBench avg): OCR-Qwen-7B-Instruct 51.3; OCR-Qwen-14B-Instruct 59.4; OCR-Qwen-32B 61.8; R1-Distill-Qwen-7B 38.0; R1-Distill-Qwen-32B 58.1; DeepSeek-R1 65.6; QwQ-32B 61.3. Baselines were run once.

## Execution-filtering ablation (§4.1, Table 3)
- Qwen2.5-14B-Instruct on a 445k CodeContests-derived subset with unit tests (up to 50 tests per problem).
- No filtering: 445,618 samples, LiveCodeBench 54.1, CodeContests 16.59. Correct solutions only: 151,251, 47.0, 15.34. Incorrect solutions (fail all tests): 151,251, 52.3, 15.53.
- Authors' explanation: "incorrect solutions span questions that are more challenging than the ones associated with the correct solutions" (Fig. 4).

## Other ablations
- Adding 356k C++ samples: "no positive impact on Python benchmark performance" but improves the IOI C++ benchmark (Table 4).
- Extending the evaluation budget from 16k to 32k tokens "did not translate to improved accuracy" on hard problems (§4.3).

## Not reported
General-domain or instruction-following evaluations of the students; seeds.
