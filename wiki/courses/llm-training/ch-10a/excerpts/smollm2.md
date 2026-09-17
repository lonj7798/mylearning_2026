---
chapter: ch-10a
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/model-reports/smollm2.md (planned card; not present on 2026-09-15)
source_url: https://arxiv.org/abs/2502.02737
created_at: "2026-09-15"
---

# Excerpt: SmolLM2: When Smol Goes Big — Data-Centric Training of a Small Language Model

**Authors:** Loubna Ben Allal, Anton Lozhkov, Elie Bakouch, Gabriel Martín Blázquez, Guilherme Penedo, Lewis Tunstall, et al. (Hugging Face)
**Version read:** arXiv:2502.02737v1 (4 Feb 2025).
**Status:** no library card existed for this slug on 2026-09-15; values read in the v1 PDF text. Source type: official technical report.

## English web ablations (§3.1-3.2)
Ablation model: 1.7B Llama-architecture, sequence 2048, global batch ≈ 2M tokens, GPT-2 tokenizer, cosine LR 3.0e-4, 350B tokens sampled from each dataset; lighteval on MMLU, HellaSwag, OpenBookQA, PIQA, WinoGrande, ARC, CommonSenseQA.

Table 1 (FW-Edu/DCLM ratio for mixes):
| Task | FW-Edu | DCLM | 40/60 | 60/40 |
|---|---|---|---|---|
| MMLU | 37.5 | 35.5 | 36.5 | 37.0 |
| ARC | 57.5 | 53.5 | 53.2 | 56.0 |
| OpenBookQA | 41.9 | 40.8 | 39.0 | 41.9 |
| HellaSwag | 60.1 | 62.3 | 61.4 | 62.2 |
| CommonsenseQA | 36.2 | 40.1 | 39.9 | 38.5 |
| PIQA | 76.2 | 76.9 | 75.7 | 76.4 |

§3.2: "FineWeb-Edu achieves higher scores on the educational benchmarks MMLU, ARC, and OpenBookQA, while DCLM performs better on HellaSwag and CommonsenseQA ... FineWeb-Edu prioritizes educational material, while DCLM captures more diverse, conversational styles."

## Mixture decisions in the 1.7B run (§4)
- Principle (1) "Performance-driven interventions, where we monitor evaluation metrics on key benchmarks and adapt dataset mixtures" (§4). Pretraining cost "around $250,000 USD of GPU compute".
- Stage 1 (0-6T tokens): web data 60% FineWeb-Edu / 40% DCLM; StarCoderData 10% of total; no math (§4.2). Stage 2 (6-8T): 75% web (60/40), 20% code, 5% OWM (§4.3). Stage 3 (8-10T): FineWeb-Edu/DCLM ratio changed to 40/60 after annealing ablations found that more DCLM "slightly improves MMLU MCF" (§4.3-4.4). Stage 4 decay (10-11T): includes 0.02% AugGSM8K, "an augmented version of the GSM8K benchmark's training set" (§4.5).
- FineMath and Infi-WebMath were decontaminated against GSM8K, MATH, and MMLU with 13-gram matching and LCS overlap ratio 0.6 (§3.3.2).
- §4.7: MMLU-Pro, TriviaQA, and Natural Questions are "held-out benchmarks not monitored during training"; Table 4 base 1.7B: MMLU-Pro 19.4, NQ 8.7, TriviaQA 36.7 (Llama3.2-1B 11.7 / 6.2 / 28.1; Qwen2.5-1.5B 13.7 / 10.5 / 20.9).
- §6: for SmolLM2-135M and 360M, "filtering DCLM with the FineWeb-Edu classifier, removing samples with score 0, and downsampling those with scores 1 and 2 worked best".

## How ch-10a uses it
§2 (FineWeb-Edu vs DCLM per-task split), §6 (benchmark-monitored mixture changes and held-out benchmarks), Recipe rows.
