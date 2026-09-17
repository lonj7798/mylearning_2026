---
chapter: ch-29a
course: llm-training
phase: read
excerpt_of: arXiv:2409.02897v3 (LongCite), §3-4, Tables 2-5 (chapter-local verified extract; no library card exists yet)
source_url: https://arxiv.org/abs/2409.02897
created_at: "2026-09-15"
---

# Excerpt: LongCite — sentence-level citations as SFT targets for long-context QA

- **Authors:** Jiajie Zhang, Yushi Bai, Xin Lv, Wanjun Gu, Danqing Liu, Minhao Zou, et al. (Tsinghua University; Zhipu AI)
- **Year:** 2024 (arXiv v1 2024-09; v3 used here)
- **Source type:** paper
- **Used in:** ch-29a §6, Negative samples

## CoF pipeline (§3.1)
1. QA generation by Self-Instruct over the long document; the answer is produced by vanilla long-context QA before citations are added.
2. "we split the context into 128-token chunks and use each sentence in the answer to retrieve lmax chunks"; retained chunks total about k;
   the LLM segments the answer into statements and adds chunk-level citations. Validation used lmax = 10, k = 40 (§3.2).
3. Each cited chunk is expanded with its neighbours; the LLM outputs sentence spans such as "[6-8]" or "No relevant information".
4. "we discard the instance if less than 20% of the statements in the answer have citations. If an answer has too few citations, we assume
   it is not factual-grounded enough in the context and may leverage the internal knowledge of LLMs, which often results in hallucinations."
- LongCite-45k: 50K documents (256 to 128K tokens, 9 domains) from the GLM-4 pretraining corpus → 44,600 instances after filtering (§3.3).

## Training (§4.1)
GLM-4-9B and Llama-3.1-8B (both 128K base models); LongCite-45k plus 76K ShareGPT instances; packing with loss weighting; batch 8, LR 1e-5,
4,000 steps (about 2 epochs). Control models LongSFT-9B/8B are trained on the same QA pairs without citations.

## Results
- Citation F1 on LongBench-Cite (Table 2): LongCite-8B 72.0; LongCite-9B 69.2; GPT-4o 65.6 (derived from "+6.4" in §4.2.1); CoF pipeline 65.8.
- Correctness ratio C/C_LQA, average (Table 3): LongCite-8B 107%; LongCite-9B 109%; on GovReport 116% and 128%. The §4.2.1 sentence
  "increased by 16%/28%" matches the GovReport column, not the average.
- Table 5 (LongCite-9B on LongBench-Chat): citation F1 63.6 with filtering vs 61.2 without data filtering; correctness 67.6 vs 67.4;
  standard long SFT (no citation data) F1 6.3.
