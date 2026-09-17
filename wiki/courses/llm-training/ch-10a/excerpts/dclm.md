---
chapter: ch-10a
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/dclm.md (planned card; not present on 2026-09-15)
source_url: https://arxiv.org/abs/2406.11794
created_at: "2026-09-15"
---

# Excerpt: DataComp-LM: In search of the next generation of training sets for language models

**Authors:** Jeffrey Li, Alex Fang, Georgios Smyrnis, Maor Ivgi, Matt Jordan, Samir Gadre, et al. (University of Washington, Apple, Toyota Research Institute, and others)
**Version read:** arXiv:2406.11794v4 (21 Apr 2025); v1 June 2024.
**Status:** no library card existed for this slug on 2026-09-15; every value below was read in the v4 PDF text at the stated locus.

## Setup
- DCLM-Pool: all Common Crawl before 2023, re-extracted with resiliparse; 200B documents, 240T GPT-NeoX tokens (§3.1).
- Competition scales (Table 1): 400M-1x (412M params, 8.2B tokens), 1B-1x (1.4B, 28.8B), 3B-1x (2.8B, 55.9B), 7B-1x (6.9B, 138B), 7B-2x (6.9B, 276B). Tokens = 20 × parameters × multiplier (§3.2).
- Metrics (§3.5, App. G): MMLU 5-shot; CORE = centered accuracy over 22 low-variance tasks, "linearly rescaling the accuracy per task so that 0 corresponds to random guessing and 1 corresponds to perfect accuracy"; EXTENDED = centered accuracy over all 53 tasks. EXTENDED adds, among others, GSM8K, GPQA, MMLU, SVAMP, AGI Eval LSAT/SAT, BBQ, Winogender (App. G.1).
- Rank transfer across scales (§3.2, Fig. 3): Pearson r = 0.838, 0.956, 0.982 between 400M-1x, 1B-1x, 3B-1x and 7B-1x CORE for 10 methods.

## Pipeline percentages (Fig. 4, share of original documents)
Heuristic cleaning (RefinedWeb reproduction) keeps 19.9%; Bloom-filter dedup removes 6.2%; the fastText filter removes 12.3%; DCLM-Baseline keeps 1.4% of DCLM-Pool documents.

## Model-based filter comparison (Table 4, 1B-1x)
| Filter | CORE | EXTENDED |
|---|---|---|
| RefinedWeb reproduction | 27.5 | 14.6 |
| Top 20% by PageRank | 26.1 | 12.9 |
| SemDedup | 27.1 | 13.8 |
| Classifier on BGE features | 27.2 | 14.0 |
| AskLLM | 28.6 | 14.3 |
| Perplexity filtering | 29.0 | 15.0 |
| Top-k average logits | 29.2 | 14.7 |
| fastText OH-2.5 + ELI5 | 30.2 | 15.4 |

## fastText ablations (Table 5 and Table 14, 7B-1x; Table 5 runs used LR 3e-4 and weight decay 0.33, App. F)
| Positives | Threshold | Features | CORE | MMLU | EXTENDED |
|---|---|---|---|---|---|
| OH-2.5 + ELI5 | 10% | uni+bigrams | 41.0 | 29.2 | 21.4 |
| Wikipedia | 10% | uni+bigrams | 35.7 | 27.0 | 19.1 |
| OpenWebText2 | 10% | uni+bigrams | 34.7 | 25.0 | 18.7 |
| GPT-3 Approx (Wiki + OWT2 + RPJ Books) | 10% | uni+bigrams | 37.5 | 24.4 | 20.0 |
| OH-2.5 + ELI5 | 15% | uni+bigrams | 39.8 | 27.2 | 21.5 |
| OH-2.5 + ELI5 | 20% | uni+bigrams | 38.7 | 24.2 | 20.3 |
| OH-2.5 + ELI5 | 10% | unigrams | 40.0 | 28.3 | 22.1 |

## Classifier training details (App. J.1)
- 400K training examples: 200K positive, 200K negative. Negatives: random documents from an earlier RefinedWeb reproduction extracted with trafilatura. Score = predicted probability of the positive label; percentile threshold.
- Positives OH-2.5 + ELI5: 100K OpenHermes 2.5 examples without preprocessing; ELI5 post plus top-scoring answer, kept if post score ≥ 0, best comment score ≥ 5, and at least 3 comments.
- fastText defaults except `wordNgrams` = 2.

## Statements used in ch-10a
- §4.4: the OH-2.5 + ELI5 positives give "a 3.5 percentage point lift on CORE compared to the other more conventional choices"; App. Q shows filtering with OH-2.5 does not preclude instruction-tuning gains.
- App. N: 16 annotators, ~500 documents, 71% inter-annotator agreement; fastText filters ~73% ROC-AUC on majority labels, AskLLM ~82% ROC-AUC but ~28.5% CORE vs > 31% for several fastText filters; "human intuition may not reliably identify the most useful documents".
- Table 6 (1B-1x): adding RedPajama extras (33%) changes CORE by +2.2 (C4), +1.7 (RPJ CC), +1.4 (RefinedWeb), −1.2 (DCLM-Baseline, 31.1 → 29.9).
- Table 7 (7B-2x): removing MMLU/HellaSwag overlaps: MMLU 51.8 → 52.7, HellaSwag 77.9 → 78.4. Table 25: 0.007% (DCLM-Baseline), 0.001% (Dolma-V1.7), 0.009% (FineWeb-Edu) of samples flagged for MMLU.
- Table 8 (7B, 0.28T tokens): FineWeb-Edu CORE 41.9 / MMLU 37.3 / EXTENDED 24.5; DCLM-Baseline 48.9 / 50.8 / 31.8.
- §6: models trained on DCLM-Baseline "do not perform as well on code and math". Table 29: DCLM-Baseline 7B (2.5T) GSM8K 2.1; after instruction tuning CORE 56.0 → 55.0, EXTENDED 43.7 → 46.5, GSM8K 2.1 → 52.5.
- App. S: the DCLM-Baseline model "does not identify toxic comments as accurately as other base models, according to CivilComments".

## Final 7B run (§5, App. Q)
Stage 1 hyperparameters Table 27 (schedule length 4.4T, stopped at 2T; LR 2e-3 → 3e-5; warmup 10,000; WD 0.05; batch 2048). Two cooldowns (200B, 270B) on 70% DCLM-Baseline at top-7% threshold + 30% ProofPile; soup weights 0.2 / 0.8 (App. Q). Context extension to 8192: "100B tokens" (§5) vs "∼120B tokens" (App. Q.2, Table 30).

## How ch-10a uses it
§3 (instruction-shaped positives, CORE vs EXTENDED), §6 (contamination checks), §7 (rank transfer), Recipe rows, Negative samples (classifier negatives).
