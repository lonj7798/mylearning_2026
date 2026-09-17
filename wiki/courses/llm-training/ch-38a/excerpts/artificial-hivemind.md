---
chapter: ch-38a
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/artificial-hivemind.md (planned card; not present on 2026-09-15)
source_url: https://arxiv.org/abs/2510.22954
created_at: "2026-09-15"
---

# Excerpt: Artificial Hivemind: The Open-Ended Homogeneity of Language Models (and Beyond)

**Authors:** Liwei Jiang, Yuanjun Chai, Margaret Li, Mickel Liu, Raymond Fok, Nouha Dziri, Yulia Tsvetkov, Maarten Sap, Alon Albalak, Yejin Choi.
**Version read:** arXiv:2510.22954v1 (27 Oct 2025), NeurIPS 2025.
**Status:** no library card existed for this slug on 2026-09-15; numbers read at the stated loci in the v1 PDF.

## Data (§2)
INFINITY-CHAT: 26K real open-ended user queries mined from WildChat, with a taxonomy of 6 top-level and 17 subcategories, plus 31,250 human annotations (absolute ratings and pairwise preferences, 25 annotators per query–response pair). INFINITY-CHAT 100 is a human-verified 100-query subset used for the homogeneity study.

## Measurement (§3, App. C.1)
For each model, 50 responses per query, maximum 2048 tokens. Two decoding configurations: top-p = 0.9 with temperature 1.0, and min-p = 0.1 with top-p = 1.0 and temperature 2.0. Similarity is cosine similarity between OpenAI `text-embedding-3-small` embeddings; the reported quantity is the average pairwise similarity within a query's response pool. More than 70 models were measured; 25 appear in the main paper.

## Intra-model repetition (§3, Fig. 4)
Under top-p = 0.9, temperature 1.0: "in 79% of cases, the average similarity exceeds 0.8". Baseline: randomly paired responses drawn from the global pool fall 100% in the 0.1–0.2 range. Under min-p decoding, "81% of response pairs still exceed 0.7 similarity and 61.2% exceed 0.8" (Fig. 5), with fewer pairs above 0.9 than under top-p.

## Inter-model homogeneity (§3, Fig. 6)
"the average pairwise similarity between responses from different models ranges from 71% to 82%". DeepSeek-V3 and qwen-max-2025-01-25 reach 0.82; DeepSeek-V3 and gpt-4o-2024-11-20 reach 0.81. The authors state that the causes "remain unclear due to proprietary training details" and name shared data pipelines and synthetic-data contamination as possible explanations.

## Calibration finding (§4)
Language models, reward models and LM judges "are less well calibrated to human ratings on model generations that elicit differing idiosyncratic annotator preferences, despite maintaining comparable overall quality".

## How ch-38a uses it
§6 (repetition within and across models under high-temperature decoding; decoding is not a sufficient control), §9 (measurement caution about embedding-similarity thresholds), Generalization lens.
