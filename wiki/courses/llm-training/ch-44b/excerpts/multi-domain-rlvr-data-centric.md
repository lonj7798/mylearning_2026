---
chapter: ch-44b
course: llm-training
phase: read
excerpt_of: arXiv:2507.17512v1 (no library card at the time of writing; chapter-local verified extract)
source_url: https://arxiv.org/abs/2507.17512
created_at: "2026-09-15"
---

# Excerpt: Can One Domain Help Others? A Data-Centric Study on Multi-Domain Reasoning via Reinforcement Learning

- **Authors:** Yu Li, Zhuoshi Pan, Honglin Lin, Mengyuan Sun, Conghui He, Lijun Wu (OpenDataLab; Shanghai AI Laboratory)
- **Year:** 2025 (arXiv v1 2025-07-23)
- **Source type:** paper (code: https://github.com/Leey21/A-Data-Centric-Study)
- **Used in:** [[read]] §3, Common mistakes

## Setup
GRPO on the Qwen2.5-7B family. Domains and data: Math (DeepScaleR 10K, CountDown 10K), Code (CodeR1-12K), Puzzle (Knights-and-Knaves 5.4K binary reward, Logic Puzzle Baron 2.4K proportional reward). Math-domain hyperparameters (Table 3), used for every combination containing math: max 8,192 tokens, rollout batch 256, mini-batch 128, LR 1e-6, 8 rollouts per prompt, 12 epochs.

## Domain combinations (Table 9 and Table 12; Qwen2.5-7B-Base; "All Avg" is the mean of the seven benchmark scores)
| Training data | Math Avg | Code Avg | Puzzle Avg | All Avg |
|---|---|---|---|---|
| Base | 22.48 | 67.46 | 9.07 | 31.50 |
| Math | 47.48 | 64.23 | 22.42 | 45.11 |
| Puzzle | 29.47 | 71.35 | 61.98 | 50.72 |
| Code | 19.17 | 73.95 | 22.55 | 35.78 |
| Math + Puzzle | 49.72 | 44.90 | 49.78 | 48.36 |
| Puzzle + Code | 32.06 | 74.88 | 55.15 | 50.89 |
| Math + Code | 47.22 | 75.06 | 25.34 | 48.92 |
| Math + Puzzle + Code | 49.75 | 73.63 | 49.73 | 56.57 |
Benchmarks behind the averages (Table 12): MATH500, CountDown, AIME24; HumanEval, MBPP; KK, Zebra. Stated observations: pairs can help in domain (Math + Puzzle raises Math above math-only), every combination is worse on Puzzle than puzzle-only training, Math + Puzzle drops Code by 22.56 points below base, and the triple combination has the highest overall average while giving up peak Puzzle accuracy. Single-domain math training lowers Code from 67.46 to 64.23; the CountDown-only run lowers Code to 29.59 (§3.1).

## Template sensitivity (§5, Table 11, Figure 5)
The same model changes by large margins with the prompt template: Qwen2.5-7B-Base scores 0.00 on CountDown and 3.00 on MBPP under the mismatched "base" template, and the Instruct model scores 1.80 on MATH500 and 0.29 on CountDown under the mismatched Qwen template. Average test performance with the matched R1 template is 47.84 (base) and 54.56 (instruct).

## Verification
- Checked on 2026-09-15 against https://arxiv.org/abs/2507.17512 (v1, 2025-07-23): Abstract, Overall Takeaways, §2, §3.1, §4.1–4.2, §5, Tables 1, 3, 9, 11, 12.
- Not reported: seeds; whether the per-combination runs use equal total prompts or equal steps per domain.
