---
chapter: ch-13a
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/model-reports/apertus.md (planned card; not present on 2026-09-15)
source_url: https://arxiv.org/abs/2509.14233
created_at: "2026-09-15"
---

# Excerpt: Apertus v1 Technical Report: Democratizing Open and Compliant LLMs for Global Language Environments

**Authors:** Project Apertus (Swiss AI Initiative: EPFL, ETH Zurich, CSCS, and others); leads Antoine Bosselut, Martin Jaggi, Imanol Schlag.
**Version read:** arXiv:2509.14233v2 (1 Dec 2025); v1 September 2025. Sections 1-5 and App. C, G, H.3, I read for this excerpt.
**Source type:** official technical report (the organization that trained the models).
**Status:** no library card existed for this slug on 2026-09-15; every value below was read at the stated locus. This excerpt covers multilingual coverage, tokenizer, and data-stage facts only; post-training, Goldfish loss, and safety are outside ch-13a.

## Coverage claims (Abstract, §1)
- Pretrained on 15T tokens "from over 1800 languages, with ∼40% of pretraining data allocated to non-English content" (Abstract). §1 gives 1,811 languages from FineWeb-2. Post-training data covers 149 languages; evaluation covers 94 languages (§1, Table 22 caption).
- The report gives no ablation of the 40% share; it lists a separate publication on multilingual data mixtures (Foroutan et al., 2025c) (§1).

## Models (§2.1, Table 1-2)
- Apertus 8B: 32 layers, d 4096, 32/8 query/KV heads; Apertus 70B: 80 layers, d 8192, 64/8. Untied input and output embeddings (§2.1). Pretraining context 4,096, extended to 65,536.
- Tokens: 15T for both; 8B max LR 1.1e-4, batch 4.2M → 8.4M tokens; 70B max LR 1.0e-5, batch 8.4M → 16.8M tokens (Table 2).

## Tokenizer (§2.2, App. I)
- Byte-level BPE adapted from the Mistral-Nemo-Base-2407 "v3 tekken" tokenizer; vocabulary 2^17 = 131,072 (§2.2).
- Chosen by comparing Llama-3.1, Mistral-Nemo, Qwen-2.5, and Gemma-2 tokenizers on the FLORES+ development set in 55 languages with four metrics: fertility, compression ratio, vocabulary utilization, Gini coefficient (§2.2, Fig. 1). Mistral-Nemo had the lowest Gini coefficient; it and Gemma-2 were similar on fertility and compression; Mistral-Nemo was preferred "because it is fairer across languages and uses a smaller vocabulary (128k vs. 256k)" (§2.2). Numeric metric values appear only in Fig. 1.
- Definitions (App. I): fertility = Σ_b |τ(b)| / Σ_b |b|_words (words by the HuggingFace Whitespace pre-tokenizer); vocabulary utilization = distinct tokens observed / |V|; Gini = (1/n)·(n + 1 − 2·Σ_i (n + 1 − i)·c_i / Σ_i c_i) for costs c_1 ≤ … ≤ c_n per language, 0 = equal costs.

## Multilingual data (§3.2.2, §3.3, App. G, Table 6)
- FineWeb-2 (v2.0.1) with all 1,811 languages "in their natural frequency"; FineWeb-2-HQ (XLM-RoBERTa classifiers) for 20 high-resource languages; no quality or toxicity filtering beyond those 20 languages "since the available multilingual web-crawl data quickly drops off in volume" (§3.2.2). The 20 languages: Russian, Chinese, German, Spanish, Japanese, French, Italian, Portuguese, Polish, Dutch, Indonesian, Turkish, Czech, Arabic, Persian, Hungarian, Swedish, Greek, Danish, Vietnamese (App. G).
- Parallel data (EuroParl, ParaDocs) and multilingual Clean Wikipedia are added in Stage 5 (§3.3).
- Table 6 pool tokens (billions), 70B stage boundaries; caption: "not necessarily all tokens of each stage data were consumed":

| Stage (70B tokens) | Multilingual pool | English web pool | Code (StarCoder etc.) | Math |
|---|---|---|---|---|
| 1 (0-5T) | 3,557 (FineWeb-2-HQ top 33% + FineWeb-2 random 33%) | 4,815 (FineWeb-Edu score-2) | 235 | 32 |
| 2 (5-9T) | 3,557 | 4,064 (FineWeb-HQ 33%) + 1,179 (FineWeb-Edu score-3) | 235 | 32 |
| 3 (9-12T) | 3,556 | 4,064 + 1,179 | 235 | 32 + 19 + 260 |
| 4 (12-13.5T) | 986 (top 10% + random 10%) | 1,619 (DCLM-Edu) | 234 | 32 + 19 + 15 |
| 5 (13.5-15T) | 986 + 33 (Clean Wikipedia) + 21 (parallel) | 1,619 | 182 + 68 | 32 + 19 + 15 |

- Stage 1 also lists 2B tokens of Gutenberg V1 and poison data; Stage 3 lists 1B of Gutenberg V2; Stage 5 lists "3 replica of Task data 3×1". The 8B model skipped Stage 2 and switched from Stage 1 to Stage 3 at 7,038B consumed tokens (App. H.3 Table H.8).
- Mixture selection: cooldown ablations on 1.5B checkpoints over 100B tokens (70% Stage 1 data + 30% candidate), reporting English and multilingual macro accuracy (§3.3, Table 7). Regular: English 0.45175, multilingual 0.44301; 30% DCLM-edu: 0.46158, 0.44608. Candidates were English datasets; no multilingual-share ablation is reported.
- Long-context mixture: FineWeb-Long from FineWeb-HQ and FineWeb-2-HQ top 10% documents longer than 4k tokens (§3.4).

## Evaluation (§5.1-5.3, Tables 14-15, 20-21)
- Pretrained Apertus-8B: XNLI 45.2, XCOPA 66.5, ARC 72.7, HellaSwag 59.8, WinoGrande 70.6; OLMo2-7B: 40.4, 55.2, 72.9, 60.4, 74.5; Llama3.1-8B: 45.3, 61.8, 71.6, 60.0, 73.4 (Table 14). Apertus-70B XCOPA 69.8, highest in the table.
- Table 15 groups MMLU and Global-MMLU as "Factual Agnostic" and INCLUDE V1, INCLUDE V2, CulturalBench, BLEND, SwitzerlandQA as "Factual Regional" (region-specific factual knowledge, §5.1). Pretrained Global-MMLU: Apertus-8B 55.3, OLMo2-7B 41.1, Qwen2.5-7B 60.3; INCLUDE V1 (44 languages): 54.8, 33.8, 53.9 (Table 15).
- Table 21 benchmarks "were held-out during model development and were not used for making decisions" (§5.2). ARC Challenge Multilingual: Apertus-8B-Instruct 36.8, Llama-3.1-8B-Instruct 32.0, Qwen3-8B 30.2 (Table 21).
- Low-resource translation: Romansh WMT24++ preliminary benchmark, German ↔ six Romansh varieties, 3-shot, greedy, BLEU (§5.3, Table 24).

## How ch-13a uses it
§3.3 (curation coverage by language), §4.3-§4.4 (vocabulary parameters, tokenizer selection metrics and Gini), §5 (held-out multilingual suites), Recipe rows, Generalization lens.
