---
chapter: ch-17
course: llm-training
phase: read
excerpt_of: primary source (no library card exists for this artifact)
source_url: https://arxiv.org/abs/2406.11794
source_version: arXiv v4, 2025-04-21
created_at: "2026-09-17"
---

# Excerpt: DataComp-LM (DCLM) — competition scales, CORE, and the filter/dedup ablation tables ch-17 reuses

**Primary source:** Jeffrey Li, Alex Fang, Georgios Smyrnis, Maor Ivgi, Matt Jordan, Samir Gadre, et al.,
*DataComp-LM: In search of the next generation of training sets for language models*, arXiv:2406.11794 (v1 2024-06; v4 2025-04-21).
**Source type:** paper. Read on 2026-09-17 from the cached full text of v4.

This excerpt exists because `wiki/raw-data/llm-training/` has no `dclm.md` card. Every line below was
read at the stated locus. Nothing here is estimated.

## Competition scales (§3.2, Table 1)

Five fixed scales. Train tokens = 20 × parameters × Chinchilla multiplier. Train FLOPs = 6ND.
H100 hours are the paper's own estimate for the OpenLM framework.

| Scale | Model parameters | Train tokens | Train FLOPs | Train H100 hours | Pool size |
|---|---|---|---|---|---|
| 400M-1x | 412M | 8.2B | 2.0e19 | 26 | 469B |
| 1B-1x | 1.4B | 28.8B | 2.4e20 | 240 | 1.64T |
| 3B-1x | 2.8B | 55.9B | 9.4e20 | 740 | 3.18T |
| 7B-1x | 6.9B | 138B | 5.7e21 | 3,700 | 7.85T |
| 7B-2x | 6.9B | 276B | 1.1e22 | 7,300 | 15.7T |

## Rank transfer across scales (§3.2, Fig. 3)

Performance of 10 curation methods at 7B-1x plotted against their performance at smaller scales.
Pearson r = 0.838 (400M-1x), 0.956 (1B-1x), 0.982 (3B-1x). The authors state this suggests that
better curation strategies at smaller scales transfer to larger scales.

## Metrics (§3.5)

- Full suite: 53 downstream tasks for base models, via LLM-Foundry.
- **CORE**: centered accuracy over a subset of **22 tasks** (examples given: HellaSwag, ARC-E) "that
  provide a low-variance signal even at small scales", each task linearly rescaled so 0 is random
  guessing and 1 is perfect.
- **EXTENDED**: centered accuracy averaged over all 53 tasks.
- MMLU is reported separately at 5-shot.

## Pipeline and per-stage results

- DCLM-POOL: all Common Crawl before 2023, re-extracted from HTML with resiliparse; 200B documents,
  240T GPT-NeoX tokens (§3.1).
- Text extraction (§4.2, Table 3, 1B-1x): resiliparse CORE 24.1 / EXTENDED 13.4; trafilatura 24.5 / 12.5;
  Common Crawl WET files 20.7 / 12.2. resiliparse is reported as 8× faster than trafilatura.
- Existing datasets (§4.1, Table 2, 7B-1x): C4 34.2 / 18.0; Dolma-V1 35.0 / 18.4; RedPajama 35.3 / 18.2;
  RefinedWeb 36.9 / 19.8.
- Model-based filters (§4.4, Table 4, 1B-1x): RefinedWeb reproduction 27.5 / 14.6; top 20% by PageRank
  26.1 / 12.9; SemDedup 27.1 / 13.8; classifier on BGE features 27.2 / 14.0; AskLLM 28.6 / 14.3;
  perplexity filtering 29.0 / 15.0; top-k average logits 29.2 / 14.7; fastText OH-2.5 + ELI5 30.2 / 15.4.
- fastText reference-data and threshold ablations (§4.4, Table 5, 7B-1x), CORE / MMLU / EXTENDED:
  OH-2.5 + ELI5 at top 10% → 41.0 / 29.2 / 21.4; Wikipedia 35.7 / 27.0 / 19.1; OpenWebText2 34.7 / 25.0 / 18.7;
  "GPT-3 Approx" (Wikipedia + OpenWebText2 + RPJ Books) 37.5 / 24.4 / 20.0; OH-2.5 + ELI5 at top 15%
  39.8 / 27.2 / 21.5, at top 20% 38.7 / 24.2 / 20.3.
- Mixing (§4.5, Table 6, 1B-1x): adding the Llama/RedPajama "extras" (Wikipedia, books, StackExchange,
  arXiv, GitHub) raises C4 from CORE 23.7 to 25.9, and the paper reports that mixing lowers the average
  for DCLM-BASELINE.

## Deduplication ablation table (App. L.2.1, Table 17, 1B-1x)

Pool: 76B tokens from Common Crawl with the RefinedWeb heuristic filters applied, then subsampled to
the 28B tokens the 1B-1x scale needs.

| Exact | MinHash | Suffix array | Bloom filter | Tokens | Removal rate | CORE | Δ from baseline |
|---|---|---|---|---|---|---|---|
| — | — | — | — | 76B | 0% | 24.7 | +0.0 |
| ✓ | — | — | — | 66B | 13% | 26.0 | +1.3 |
| — | ✓ | — | — | 62B | 18% | 25.6 | +0.9 |
| — | — | ✓ | — | 51B | 33% | 26.6 | +1.9 |
| — | — | — | ✓ | 56B | 26% | 26.8 | +2.1 |
| ✓ | ✓ | — | — | 58B | 24% | 25.0 | +0.3 |
| ✓ | — | ✓ | — | 49B | 36% | 26.2 | +1.5 |
| — | ✓ | ✓ | — | 48B | 37% | 26.3 | +1.6 |
| ✓ | ✓ | ✓ | — | 45B | 41% | 26.8 | +2.1 |

The authors state that the apparent advantage of the suffix-array pipeline over MinHash alone "falls
within the range of variance for the CORE score due to the nondeterminism in subsampling the dataset
and training a model" (App. L.2.1).

## Decontamination (§4.6, Table 7)

DCLM-POOL is released without decontamination; tooling is released instead. For MMLU and HellaSwag the
authors flag pages containing the question text plus at least one answer option, and for MMLU detect only
the last sentence of each question to raise recall; they state this "still incurs many false positives".
At 7B-2x, removing the detected overlaps did not lower scores: MMLU 51.8 → 52.7, HellaSwag 77.9 → 78.4.

## Not reported / not usable from this source

- Per-seed standard deviations for CORE at any scale (variance is described qualitatively in App. L.2.1).
- Any per-domain held-out perplexity breakdown (use [[paloma]] for that).
- Optimizer, learning rate, and batch size per scale were not transcribed into this excerpt.
