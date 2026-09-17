---
chapter: ch-10a
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/nemotron-cc.md (planned card; not present on 2026-09-15)
source_url: https://arxiv.org/abs/2412.02595
created_at: "2026-09-15"
---

# Excerpt: Nemotron-CC: Transforming Common Crawl into a Refined Long-Horizon Pretraining Dataset

**Authors:** Dan Su, Kezhi Kong, Ying Lin, Joseph Jennings, Brandon Norick, Markus Kliegl, et al. (NVIDIA)
**Version read:** arXiv:2412.02595v2 (30 May 2025); v1 December 2024.
**Status:** no library card existed for this slug on 2026-09-15; values read in the v2 PDF text.

## Problem statement (Abstract, §1, §2.2)
- FineWeb-Edu and DCLM remove "around 90%" of data; both "contain around 80% near-duplicates (1T and 0.2T unique tokens, respectively)" (§1; order as printed: DCLM 1T, FineWeb-Edu 0.2T per Table 4).
- The two classifiers "have a limited recall (around 10%) of high-quality tokens (see Table 9)" (§2.2).

## Method
- Extraction: Justext instead of Trafilatura; over 13 snapshots, high-quality tokens (FineWeb-Edu classifier score 3-5) 80B (Trafilatura-filtered), 104B (Justext-filtered), 127B (Justext, no filter) (Table 1). The heuristic filter pipeline "removes a non-trivial portion of high-quality tokens (-18.1%)" (§2.1). Heuristic filters are applied only to low- and medium-quality data (§2.1, Fig. 3).
- English ID: pycld2 and fastText lid176, threshold 0.3; global fuzzy dedup and exact substring dedup (§2.1).
- Three classifiers (§2.2): two linear-regression heads on Snowflake-arctic-embed-m trained on Mistral 8x22B-instruct and Nemotron-340B-instruct educational-value annotations of the same 460K FineWeb-Edu-Annotation documents (20 epochs, LR 3e-4, frozen encoder, best validation F1), plus the released DCLM fastText classifier. The FineWeb-Edu classifier itself is not in the ensemble "for license issue" (footnote 18).
- Scoring: each classifier's scores are rounded to integer buckets 0-19 so each bucket has around 5% of documents; ensemble score = maximum of the three integers (§2.2).
- Quality labels from annealing tests: each bucket tested by continued pretraining for 50B tokens on a "70% trained 8B" model with 34% bucket data / 66% default mix, average over 9 tasks (§2.2); App. C describes a 900B-token checkpoint and 13 tasks.
- Table 2: High = bucket 19 (553B tokens, 12.63%); Medium-High = 18 (504B, 11.52%); Medium = 12-17 (2,023B, 46.24%); Medium-Low = 7-11 (894B, 20.43%); Low = 0-6 (402B, 9.18%).
- Rephrasing (§2.3): Mistral NeMo 12B instruct, FP8, top-p 0.9, temperature 0.5. Low-quality: Wikipedia-style prompt. High-quality: Wikipedia, Diverse QA pairs, Distill, Extract knowledge, Knowledge list. Table 3 synthetic tokens (B): low → Wikipedia 336.3; high → Wikipedia 372.9, Diverse QA 499.5, Distill 157.6, Extract 303.6, Knowledge list 203.2. Medium-quality data is not rephrased.

## Results
- Table 4 (trillions): FineWebEdu-2 5.4 total / 1.1 unique; FineWebEdu 1.3 / 0.2; DCLM 3.8 / 1.0; Nemotron-CC 6.3 / 4.4 / 1.9 synthetic; Nemotron-CC-HQ 1.1 / 0.6 / 0.5.
- Table 5 (8B, 1T tokens, 73% tested CC + 27% fixed non-CC): MMLU / 10-task avg: FineWebEdu-2 42.4 / 53.2; FineWebEdu 42.9 / 53.2; DCLM 53.4 / 57.0; Nemotron-CC 53.0 / 57.8; Nemotron-CC-HQ 59.0 / 60.1. HQ gains over DCLM "on all tasks except RACE" (36.4 vs 36.5).
- Table 6 (8B, 15T tokens) vs Llama 3.1 8B in the authors' harness: MMLU 70.3 vs 65.3; ARC-C 58.1 vs 55.0; avg 64.7 vs 64.2; lower on Winogrande (73.8 vs 74.7), RACE (37.8 vs 39.1), SIQA (47.4 vs 48.3), CSQA (69.9 vs 70.6), OBQA (45.4 vs 46.0), PIQA (81.1 vs 81.2).
- Table 7 (8B-1T): MMLU / avg non-MMLU: Trafilatura filtered 55.4 / 60.6; Justext filtered 54.1 / 60.9; Justext unfiltered 55.5 / 60.3; Justext HQ-unfiltered 57.5 / 60.6.
- Table 8 (CC-MAIN-2021-21): of 11,359,655 documents labeled high quality by either classifier, 10.1% by both, 35.4% by FineWeb-Edu only, 54.4% by DCLM only. App. B: 368 of the top 1k domains shared.
- Table 9 (8B-1T, 13 snapshots): HQ% / SIQA / MMLU / Avg: FineWeb-Edu 8% / 45.8 / 55.4 / 59.0; DCLM 11% / 33.9 / 56.0 / 58.4; Ours-mistral 9% / 46.2 / 53.2 / 58.1; Ours-nemotron-340B 14% / 34.3 / 54.9 / 58.0; Ours-ensembled 25% / 45.7 / 56.4 / 59.4.
- Table 10 (8B-1T) avg: LQ-Base 52.5; LQ-Synthetic 54.0 (MMLU 48.2 → 47.1); HQ-Base 55.8; HQ-Synthetic 56.7. HQ-Synthetic swaps 4 of 8 repetitions of high-quality documents for synthetic data.

## Training details
- App. D: 8B, 32 layers, hidden 4096, 32 heads, GQA 8 groups, SwiGLU; Adam (0.9, 0.95, 1e-8), WD 0.1, cosine peak 3e-4 to 3e-6; 1T tokens; ~40 hours on 1024 H100. Sequence length and global batch: not reported in the paper body or App. D.
- Table 12 blend: English CC 73%, books and patents 9%, papers 9%, code 5%, conversational 3%, Wikipedia 1%.
- App. E (15T run): phase 1 9T tokens with 59% English CC (5.31T), medium, medium-high, high; phase 2 6T tokens with 31% (1.86T), high only; total 7.17T (47.8%).

## Limitations stated (§6)
One ensembling and bucketing strategy tried; factual accuracy of rephrased data not verified; English only; dataset not decontaminated.

## How ch-10a uses it
§4 (ensembles, rephrasing, unique tokens), Negative samples (low-quality data rephrased instead of discarded), Recipe rows.
