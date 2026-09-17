---
chapter: ch-32a
course: llm-training
phase: read
excerpt_of: primary source arXiv:2409.12186v3 (no library card as of 2026-09-15)
source_url: https://arxiv.org/abs/2409.12186
created_at: "2026-09-15"
---

# Excerpt: Qwen2.5-Coder Technical Report

**Source:** Qwen Team, Alibaba Group. arXiv v1 2024-09; v3 2024-11-12 read. Source type: official technical report.

## Initialization and stages (§2, §3.2, Fig. 2, Table 1)
- "Qwen2.5-Coder models are derived from the Qwen2.5 LLMs"; six sizes 0.5B–32B, each listed with 5.5T trained tokens (Table 1).
- Stage 1, file-level pretraining: maximum sequence 8,192 tokens, 5.2T tokens, next-token prediction plus fill-in-the-middle (FIM).
- Stage 2, repo-level pretraining: context 8,192 → 32,768 tokens; "RoPE's base frequency is adjusted from 10,000 to
  1,000,000"; YaRN for up to 131,072 tokens; about 300B tokens of long-context code data; repo-level FIM format.
- Stage 3: instruction tuning (SFT and DPO).

## Data (§3.1)
- Five types: source code (GitHub before February 2024, 92 languages, plus pull requests, commits, notebooks, Kaggle),
  text-code grounding data from Common Crawl, synthetic data from CodeQwen1.5 filtered by an executor, math data (the
  Qwen2.5-Math pre-training corpus), and text data.
- Text data: "high-quality general natural language data from the pre-training corpus of the Qwen2.5 model to preserve
  Qwen2.5-Coder's general capabilities"; code segments removed from it.
- Four-stage text-code grounding filtering raised the HumanEval+MBPP average of Qwen2.5-Coder-1.5B from 41.6% to 46.8% (Fig. 1).

## Mixture ablation (§3.1.2, Table 3; Qwen2.5-Coder-7B)
| Code:Text:Math | Common | BCB | MATH | GSM8K | MMLU | CEval | HellaSwag | Average (as printed) |
|---|---|---|---|---|---|---|---|---|
| 100:0:0 | 49.8 | 40.3 | 10.3 | 23.8 | 42.8 | 35.9 | 58.3 | 31.3 |
| 85:15:5 (text says 85:10:5) | 43.3 | 36.2 | 26.1 | 52.5 | 56.8 | 57.1 | 70.0 | 48.9 |
| 70:20:10 | 48.3 | 38.3 | 33.2 | 64.5 | 62.9 | 64.0 | 73.5 | 55.0 |
- Final mixture: 70% code, 20% text, 10% math; "the 7:2:1 ratio outperformed the others, even surpassing the performance
  of groups with a higher proportion of code". Training tokens of the ablation runs are not printed.
- Consistency note (this course): the mean of the seven printed scores for 100:0:0 is 37.3, not 31.3; the other two rows
  match their means.

## General-capability evaluation (§6.5, Tables 13–14)
- Base models are compared with other code models, not with the Qwen2.5 checkpoints they were initialized from.
- Qwen2.5-Coder-7B: MMLU 68.0, MMLU-Pro 40.1, MMLU-Redux 66.6; ARC-C 60.9, TruthfulQA 50.6, WinoGrande 72.9, HellaSwag 76.8.
- Qwen2.5-Coder-32B: MMLU 79.1, MMLU-Pro 50.4; ARC-C 70.5, TruthfulQA 54.2, WinoGrande 80.8, HellaSwag 83.0.

## Verification
- Read on 2026-09-15 against the arXiv:2409.12186v3 PDF text (§2–§3, §6.4–§6.6, Tables 1–3, 12–14).
- Not reported: learning rate, batch size, and schedule for either pretraining stage; general-benchmark scores of the
  Qwen2.5 initialization checkpoints in this report (so the retention delta is not reported here). The 5.5T in Table 1
  equals the 5.2T file-level plus about 300B repo-level tokens (derived from §3.2).
