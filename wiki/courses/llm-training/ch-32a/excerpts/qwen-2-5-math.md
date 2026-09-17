---
chapter: ch-32a
course: llm-training
phase: read
excerpt_of: primary source arXiv:2409.12122v1 (no library card as of 2026-09-15)
source_url: https://arxiv.org/abs/2409.12122
created_at: "2026-09-15"
---

# Excerpt: Qwen2.5-Math Technical Report: Toward Mathematical Expert Model via Self-Improvement

**Source:** Qwen Team, Alibaba Group. arXiv v1 2024-09-18 read. Source type: official technical report.

## Math continued pre-training (§2)
- Qwen Math Corpus v1: math web text recalled with a fastText classifier (iteratively retrained), URL-based pool
  expansion, MinHash deduplication, Qwen2-0.5B-Instruct quality scoring, and question-answer data synthesized by
  Qwen2-72B-Instruct; mixture chosen by ablations on Qwen2-Math-1.5B; 700B tokens in total.
- Qwen2-Math-1.5B/7B/72B are initialized "with intermediate checkpoints from the corresponding Qwen2-1.5B/7B/72B base
  models" and continuously pre-trained on v1 at 4K context.
- Qwen Math Corpus v2: adds synthetic data from Qwen2-Math-72B-Instruct and more (especially Chinese) math data;
  "from 700B to over 1T" tokens; 4K context.
- Qwen2.5-Math-1.5B/7B/72B are initialized from Qwen2.5 base models and continuously pre-trained on v2 "under a math
  pre-training setup similar to Qwen2-Math".

## Base-model evaluation (§5.1, Table 2; few-shot chain-of-thought)
| Model | GSM8K | MATH | MMLU-STEM | CMATH | GaoKao Math Cloze | GaoKao Math QA |
|---|---|---|---|---|---|---|
| Qwen2-7B | 79.9 | 44.2 | 67.6 | 76.7 | 37.3 | 51.6 |
| Qwen2-Math-7B | 80.4 | 50.4 | 65.7 | 83.2 | 48.3 | 57.3 |
| Qwen2.5-Math-7B | 91.6 | 55.4 | 67.8 | 85.0 | 57.6 | 69.5 |
| Qwen2-72B | 89.5 | 51.1 | 79.9 | 85.4 | 55.9 | 72.6 |
| Qwen2-Math-72B | 89.1 | 60.5 | 79.1 | 86.4 | 72.9 | 69.5 |
| Qwen2.5-Math-72B | 90.8 | 66.8 | 82.8 | 89.7 | 72.9 | 86.3 |
- Qwen2.5 base models (the initialization of Qwen2.5-Math) are not rows of Table 2.
- The Qwen2 rows are final released base models, while Qwen2-Math was initialized from intermediate Qwen2 checkpoints,
  so Qwen2 → Qwen2-Math differences are not clean before/after deltas (this course's reading of §2 and Table 2).

## Verification
- Read on 2026-09-15 against the arXiv:2409.12122v1 PDF text (§1–§2, §5.1, Table 2).
- Not reported: data-mixture percentages, share of general (non-math) data, learning rate and batch size for math
  continued pre-training; non-math general benchmarks (for example full MMLU) for the base models; token count seen per model.
