---
chapter: ch-13a
course: llm-training
phase: read
excerpt_of: "huggingface/smollm text/pretraining/smollm3 nanotron configs (multilingual weights) and the SmolLM3 blog"
source_url: https://github.com/huggingface/smollm/tree/main/text/pretraining/smollm3
created_at: "2026-09-15"
---

# Excerpt: SmolLM3-3B multilingual data weights in the released pretraining configs

This excerpt stands in for multilingual values that the library card [[smollm-3]] does not contain (that card has not been verified and describes post-training).
Source types: released config (GitHub `huggingface/smollm`, main branch, raw YAML cached 2026-09-14, read 2026-09-15) and official blog (huggingface.co/blog/smollm3, July 2025).

## Blog statements ("Model summary", "Data mixture and training stages")
- "Multilingual support for 6 languages: English, French, Spanish, German, Italian, and Portuguese."
- Stage 1 (0T → 8T): "Web: 85% (12% multilingual) - FineWeb-Edu, DCLM, FineWeb2 and FineWeb2-HQ"; Stage 2 (8T → 10T): "Web: 75% (12% Multilingual)"; Stage 3 decay (10T → 11.1T): "Web: 63% (12% Multilingual)".
- Mixture ratios were set by "extensive ablations on 3B models trained on 50B to 100B tokens"; no ablation numbers are given.
- Evaluation of multilingual ability: Global MMLU, MLMM HellaSwag, Flores-200, Belebele across "five major European languages" (numbers only in figures).

## Released configs
All stage files use `tokenizer_name_or_path: meta-llama/Llama-3.2-1B` with `vocab_size: 128256` and `tie_word_embeddings: true`, `hidden_size: 2048` (`stage1_8T.yaml` L144-146, L180, L196-198, L244). Tokens per step: dp 192 × micro batch 3 × sequence 4,096 = 2,359,296 (L225, L253-254); `train_steps: 4720000` (L255).

The comment at `stage1_8T.yaml` L20 reads: "we use FineWeb2-HQ for all the languages below except Hindi, Thai, Korean for which we use FineWeb2". Weights of the 14 FineWeb2 entries:

| Language entry | `stage1_8T.yaml` L105-118 | `stage3_9T_11T.yaml` stage 2, L251-264 | `stage3_9T_11T.yaml` decay, L426-439 |
|---|---|---|---|
| fra | 0.016 | 0.016 | 0.018 |
| spa | 0.02 | 0.02 | 0.022 |
| deu | 0.022 | 0.0232 | 0.023 |
| ita | 0.0105 | 0.0105 | 0.0125 |
| por | 0.01 | 0.01 | 0.0045 |
| cmn | 0.01 | 0.01 | 0.01 |
| rus | 0.01 | 0.01 | 0.01 |
| fas | 0.003 | 0.002 | 0.009 |
| jpn | 0.00325 | 0.00325 | 0.0032 |
| kor | 0.00325 | 0.00325 | 0.0032 |
| hin | 0.00325 | 0.00325 | 0.0032 |
| tha | 0.00325 | 0.00325 | 0.0032 |
| vie | 0.00325 | 0.00005 "# downsample viet, too many epochs" | 0.00005 (same comment) |
| ell | 0.00225 | 0.00225 | 0.0022 |
| **Sum of FineWeb2 weights** | **0.12** | **0.117** | **0.12405** |
| Sum of all weights in the stage | 1.0 (41 entries) | 1.00005 (46 entries) | 1.00565 (56 entries) |

- Stage start steps: stage 2 `start_training_step: 3450001` (L301), decay `start_training_step: 4198001` (L484). Derived token positions: 3,450,000 × 2,359,296 = 8.14T; 4,198,000 × 2,359,296 = 9.90T; 4,720,000 × 2,359,296 = 11.14T.
- Derived normalized multilingual share in the decay stage: 0.12405 / 1.00565 = 12.3%.
- The long-context configs keep 14 FineWeb2 entries; the eighth entry is `fw2-hq-arb_Arab` (Arabic) instead of `fas`. Sums: `long_context_4k_to_32k.yaml` L135-148 = 0.11108 of 1.0; `long_context_32k_to_64.yaml` L137-150 = 0.10808 of 1.0, with comments such as "# remove 0.001 from Jpn".
- The weights are sampling weights over tokenized dataset folders, not token shares of unique data.

## Observations used in ch-13a (course reading, not stated by the authors)
- The configs sample 14 non-English languages, while the blog lists 5 non-English supported languages; 9 of the 14 (cmn, rus, fas, jpn, kor, hin, tha, vie, ell) are sampled but not listed as supported.
- The "12% multilingual" in the blog corresponds to the sum of the FineWeb2 weights over the whole mixture (0.12), not 12% of the web portion.

## How ch-13a uses it
Recipe rows for SmolLM3-3B; §3.4 (per-language weights and the repetition comment for Vietnamese); Common mistakes (reading a share statement without the config).
