---
chapter: ch-44a
course: llm-training
phase: read
excerpt_of: arXiv:2509.23863v4 (SPELL, ICLR 2026), §3.2-§3.3, §4.1-§4.3, Tables 1-2, 7, Figures 3-4 (chapter-local verified extract; no library card exists for this slug on 2026-09-15)
source_url: https://arxiv.org/abs/2509.23863
created_at: "2026-09-15"
---

# Excerpt: SPELL — Self-Play Reinforcement Learning for Evolving Long-Context Language Models

- **Authors:** Ziyi Yang, Weizhou Shen, Chenliang Li, Ruijun Chen, Fanqi Wan, Ming Yan, Xiaojun Quan, Fei Huang (Sun Yat-sen University; Tongyi Lab, Alibaba Group)
- **Year:** 2025 (arXiv v1 2025-09; this extract read from v4, 2026-03-13, ICLR 2026 version)
- **Source type:** paper
- **Used in:** ch-44a §5, §6, Generalization lens.

## Method (§3.2)
One policy plays three roles. The **questioner** writes a question and reference answer from sampled documents, conditioned
on a memory of previously validated pairs; the **responder** answers with all n documents present, so unseen documents act
as distractors; the **verifier** judges semantic equivalence between the responder's answer and the reference.
Questioner reward (Eq. 7): a Gaussian in the responder's average success rate `r̄_res`, `exp(−(r̄_res − µ)²/2σ²)` with
µ = 0.5 and σ = 0.5/3, zero when `r̄_res` is 0 or 1, −0.5 for a question not grounded in the documents, and −1 for a
formatting error. Responder reward: maximum of a rule-based check and the verifier consensus. Role-specific dynamic
sampling keeps only responder groups with non-zero reward variance and only verifier instances whose majority vote agrees
with the rule-based check.

## Settings (§4.1)
VeRL; temperature 0.7, top-p 0.95; **maximum input 16K tokens; maximum output 4K for non-reasoning models and 20K for
reasoning models**; group size G = 8; history memory L = 3; m = 5 candidate documents per proposal; purely on-policy,
batch size 128, constant learning rate 2 × 10⁻⁶. The RLVR baseline uses a dataset synthesized by DeepSeek-R1-0528 over the
same corpus with identical hyperparameters. Evaluation at maximum input lengths of 16K and 100K, averaged over eight runs.

## Results (Table 1, average over DocMath, Frames, LongBench multi-hop QA, LongBench-v2)
| Model | 16K base → +RLVR → +SPELL | 100K base → +RLVR → +SPELL |
|---|---|---|
| Qwen2.5-14B | 37.3 → 49.9 → 51.7 | 36.1 → 50.0 → 51.1 |
| Qwen2.5-32B-Instruct | 51.8 → 56.8 → 59.5 | 52.5 → 57.1 → 60.1 |
| Qwen3-30B-A3B-Thinking | 60.7 → 60.7 → 62.7 | 63.6 → 64.5 → 65.9 |

"All models are trained with a 16K input limit and evaluated at 100K without additional tuning ... For Qwen2.5-14B, the
average improvement is 14.4 at 16K and increases to 15.0 at 100K."
Test-time scaling (§4.3, Figure 3, 100K input): Qwen3-30B-A3B-Thinking pass@8 is 74.5 with SPELL, 68.1 with RLVR, 66.9 for
the base model; SPELL's pass@4 exceeds gemini-2.5-pro.
Ablation (Table 2, Qwen2.5-7B-Instruct, 16K): removing the verifier lowers the average from 47.2 to 44.0; freezing the
questioner lowers it by 4.6.

## Short-context transfer (App. F.2, Table 7)
Trained only on long-context self-play data, average over AIME24, AIME25, AMC23, MATH500, GSM8K, MMLU-Pro:
Qwen2.5-7B 35.43 → 42.36 (+6.93); Qwen2.5-14B 40.50 → 49.24 (+8.74); Qwen2.5-32B 43.80 → 51.04 (+7.24). These rows are
base models, not instruct models; maximum output 4K, temperature 0.7, 8 runs.

## Not reported
Short-context results for the instruct and reasoning models, truncation rate at the 4K/20K output limits, and the number
of RL steps.
