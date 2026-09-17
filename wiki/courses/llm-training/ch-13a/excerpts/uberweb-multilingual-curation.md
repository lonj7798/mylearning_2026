---
chapter: ch-13a
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/uberweb-multilingual-curation.md (planned card; not present on 2026-09-15)
source_url: https://arxiv.org/abs/2602.15210
created_at: "2026-09-15"
---

# Excerpt: ÜberWeb: Insights from Multilingual Curation for a 20-Trillion-Token Dataset

**Authors:** DatologyAI Team (core contributors Aldo Gael Carranza, Kaleigh Mentzer, Ricardo Pio Monti; leadership Bogdan Gaza, Ari Morcos, Matthew Leavitt) (§6)
**Version read:** arXiv:2602.15210v3 (25 Feb 2026); full PDF text including App. A.1-A.4.1.
**Source reliability:** paper from a data-curation company that describes its own commercial curation pipeline; the pipeline itself is not released or specified in detail (§3 "Data curation").
**Status:** no library card existed for this slug on 2026-09-15; every value below was read at the stated locus.

## Setup (§3)
- English sources: DCLM, FineWeb, non-synthetic Nemotron-CC v1. Non-English: FineWeb2. "Uncurated" means random samples from DCLM or FineWeb2, which the authors note were already "heavily curated" (§3, footnote 2).
- 13 non-English languages (Table 1): Russian, Chinese, German, Spanish, Japanese, French, Portuguese, Indonesian, Arabic, Vietnamese, Korean, Hindi, Bengali. Table 1 FineWeb2 documents range from 699.1M (Russian) to 15.2M (Bengali). App. A.1 Table 2 additionally lists Italian, for which no results table is given.
- Curation per language: selecting, validating, or training language-appropriate models for filtering, embedding-based selection, and synthetic rephrasing; filtering and mixing adapted to script- and language-specific artifacts (§3). No thresholds, models, or removal rates are reported.
- Models: Llama-based 3B and 8B, Llama-3.2 tokenizer, context 4,096, identical training configurations except data (§3). Hyperparameters not reported.
- Evaluation: zero-shot lighteval; multilingual MMLU (Global MMLU; Indic MMLU for Hindi, Bengali), multilingual ARC (Okapi, Ko-ARC, LumiOpen, Indic ARC; none for Japanese), Belebele; English MMLU and ARC-Challenge only. Cloze formulation for 3B/60B runs, multiple-choice formulation for large runs; all models are base models (§3, App. A.1).

## Controlled bilingual experiments (§4.1-4.2)
- 3B models, 60B tokens, 50/50 English:target mixture, 13 pairs, three regimes: (i) uncurated English + uncurated target; (ii) curated English + uncurated target; (iii) curated English + curated target (§4.1.1).
- English curation raised non-English scores in 12 of 13 languages (not Bengali), average relative gain 3.91% (§4.1.1, Fig. 2).
- Gains were larger for Spanish, French, German than for Hindi and Arabic: 8.56% vs 3.94% relative (§4.1.2). Correlation of gain with distance to English on FLORES parallel text: embedding distance r = −0.62 (p = 0.024), log perplexity under an English-only model r = −0.70 (p = 0.018) (§4.1.2, Fig. 3).
- Bespoke per-language curation: 16.87% relative gain over the uncurated baseline (§1, §5); per-language values are shown only as bars (Fig. 2).
- Non-English curation raised English MMLU+ARC in 12 of 13 pairs, average relative gain 1.21% (§4.1.4, Fig. 4; the abstract rounds to 1.2%).
- Translation (Hindi, Bengali, Arabic; 3B, 60B): adding translations of random English documents gave "marginal" gains; translating fastText-score-filtered English documents gave 5.09% average relative gain over the uncurated baseline; bespoke curation remained higher (§1, §4.2, Fig. 5).

## Large-scale mixture (§4.3)
- 20T-token corpus including over 8T synthetic tokens. 3B and 8B models trained on a random 1T subset.
- Phases: 650B tokens at 5% multilingual, 250B at 10%, 100B at 20%; overall 7.75% across 13 languages, "an average of 6B tokens per language" (§4.3). The multilingual total is given as "∼80B tokens" (§1) and "75B" (App. A.4); 7.75% of 1T is 77.5B.
- FLOPs (App. A.2 Table 3, 6ND): DatologyAI 3B 1.8 × 10^22; 8B 4.8 × 10^22; SmolLM3-3B (11T) 1.9 × 10^23; Qwen3-4B (36T) 8.6 × 10^23; LFM-2.5-1.2B (28T) 1.9 × 10^23.
- Stated comparison shares: LFM uses 20% multilingual tokens; SmolLM3 12% (§4.3).
- Per-language base-model accuracy, multiple-choice (App. A.4.1). Spanish (Table 4) MMLU / ARC / Belebele: DatologyAI 3B 0.46 / 0.63 / 0.62; 8B 0.55 / 0.73 / 0.76; SmolLM3 3B 0.53 / 0.66 / 0.62; Qwen3 4B 0.67 / 0.86 / 0.85. Bengali (Table 11): DatologyAI 3B 0.32 / 0.42 / 0.43; 8B 0.36 / 0.48 / 0.51; SmolLM3 3B 0.30 / 0.28 / 0.30; Qwen3 4B 0.45 / 0.67 / 0.71; Sarvam1 2B 0.41 / 0.56 / 0.55.
- Per-language token estimates for other models are the authors' estimates from public information (App. A.4), for example Llama 3.2 "approximately 100B tokens per language".

## Claims and their status
- The claim that "many reported regressions … stem from correctable deficiencies in data quality and composition rather than fundamental capacity limits" (Abstract) is the authors' interpretation. The controlled evidence is 50/50 bilingual runs at 3B; the number of languages at fixed capacity is not varied.
- The Pareto-frontier claim (Fig. 1) is accuracy per training FLOP; absolute accuracies of the 3B and 8B models are below Qwen3-4B on every per-language table in App. A.4.1.

## How ch-13a uses it
§3.2 (bilingual transfer under curation, translation, phased shares), Generalization lens, Recipe row, Common mistakes.
