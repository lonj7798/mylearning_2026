---
chapter: ch-23
course: llm-training
phase: read
excerpt_of: primary source arXiv:2510.22954v1 (no library card as of 2026-09-15; card planned under slug artificial-hivemind)
source_url: https://arxiv.org/abs/2510.22954
created_at: "2026-09-15"
---

# Excerpt: Artificial Hivemind: The Open-Ended Homogeneity of Language Models (and Beyond)

**Paper:** Liwei Jiang, Yuanjun Chai, Margaret Li, Mickel Liu, Raymond Fok, Nouha Dziri, et al. (UW, CMU, AI2, Lila Sciences, Stanford). arXiv v1 2025-10-27 read (NeurIPS 2025). Source type: paper (observational measurement study).

## Data (§2)
- INFINITY-CHAT: 26,070 open-ended and 8,817 closed-ended queries mined from WildChat (37,426 single-turn GPT-4 queries, English, non-toxic, 15–200 characters).
- Taxonomy of 6 top-level and 17 subcategories; Creative Content Generation 58.0% of queries, Brainstorm & Ideation 15.2% (Fig. 2).
- 31,250 human annotations with 25 annotators per example (Abstract).

## Intra-model repetition (§3, Figs. 4–5)
- INFINITY-CHAT 100 subset; 50 responses per query per model; similarity by OpenAI text-embedding-3-small.
- With top-p 0.9 and temperature 1.0, average pairwise similarity within a query's response pool exceeds 0.8 in 79% of cases; randomly paired responses from the global pool fall in 0.1–0.2.
- With min-p 0.1, top-p 1.0, temperature 2.0: 81% of response pairs exceed 0.7 similarity and 61.2% exceed 0.8.

## Inter-model homogeneity (§3, Figs. 1, 6, 8)
- 25 models in the main analysis (70+ overall): average pairwise similarity between different models' responses ranges from 0.71 to 0.82 (DeepSeek-V3 vs qwen-max-2025-01-25: 0.82; DeepSeek-V3 vs gpt-4o-2024-11-20: 0.81).
- "Write a metaphor about time": 50 responses from each of 25 models form two main clusters ("time is a river", "time is a weaver").
- Among the top-50 most similar responses per query, an average of about 8 distinct models contribute.
- Causes are not tested; the authors list shared data pipelines and contamination from synthetic data as possible explanations (§3; Interpretation).

## Judges and reward models (Abstract, §4)
- LMs, reward models, and LM judges are less calibrated to human ratings on responses where annotators disagree, at comparable overall quality.

## Verification
- Read on 2026-09-15 against arXiv:2510.22954v1 PDF text (Abstract, §1–3).
