---
chapter: ch-13a
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/multilinguality-curse-250-languages.md (planned card; not present on 2026-09-15)
source_url: https://arxiv.org/abs/2311.09205
created_at: "2026-09-15"
---

# Excerpt: When Is Multilinguality a Curse? Language Modeling for 250 High- and Low-Resource Languages

**Authors:** Tyler A. Chang, Catherine Arnett, Zhuowen Tu, Benjamin K. Bergen (UC San Diego)
**Version read:** arXiv:2311.09205v1 (15 Nov 2023), marked "Preprint"; full PDF text including App. A.1-A.7.
**Status:** no library card existed for this slug on 2026-09-15; every value below was read at the stated locus.

## Data (§3)
- 24 multilingual sources (OSCAR, Wikipedia, No Language Left Behind, and others); sequences of 100 or more repeated UTF-8 bytes deduplicated.
- 41.4B tokens in 1,572 languages after capping each language at 1B tokens; 1,329 languages with at least 100K tokens "largely due to Bible translations"; 252 languages with the 1.5M tokens needed (1M training, 500K evaluation), covering 29 families and 30 scripts.

## Monolingual baselines and metric (§4)
- GPT-2 models: tiny 4.6M, mini 11.6M, small 29.5M parameters (§4.1). Dataset sizes 1M, 10M, 100M, 1B tokens ("low", "med-low", "med-high", "high"); available for 252, 167, 48, 28 languages (§4.1). §4 counts 1,989 monolingual baselines; §1 says "over 1900".
- One SentencePiece tokenizer per language, vocabulary 32K, trained on 10K lines; a 10K-line tokenizer covers on average 93.7% of the 4K most frequent tokens of a 10M-line tokenizer (§4.1).
- Relative log-likelihood (Eq. 1): `mean_w log2 P_M(w) − mean_w log2 P_Baseline_L(w)`, where the baseline is the tiny model trained on 1M tokens of language L. A value ℓ means the model assigns the evaluation set 2^ℓ times the baseline likelihood, equivalently 2^ℓ times lower perplexity (§4.2).
- Performance is re-expressed as "estimated monolingual tokens": the monolingual data size that gives the same log-likelihood at the same model size, from a fitted curve `−a·x^(−b) + c`, x = log10 of monolingual tokens; for lower-resource languages a is fixed to the median of languages where the curve can be fit (§4.3, App. A.4).

## Multilingual models (§5)
- 8,454 multilingual models, 8M-45M parameters (§5); §1 says "over 8400". With added vocabulary embeddings: about 8.7M (tiny), 19.8M (mini), 45.8M (small).
- Each model keeps the target language's monolingual tokenizer fixed; the added languages share a separate 32K multilingual tokenizer and token ids are merged. Reason given: 252 × 32K = 8.1M vocabulary would need 1.0B embedding parameters at embedding size 128 (§5).
- Added data: always 10 languages, equal tokens each, total 10M, 100M, or 1B tokens, one epoch, interspersed randomly; chosen from the 48 med-high languages as the 10 most or 10 least similar (§5).
- Similarity = mean of z-scored syntactic (lang2vec), geographic, and lexical (log shared tokens) similarity. Most similar to English: Dutch, Swedish, Norwegian, German (§5).

## Results (§6)
- Low-resource: gains when 100M or 1B tokens are added (p < 0.001 in 11 of 12 comparisons). Similar vs dissimilar added languages: equivalent to +33% vs +22% target data for small models in the best scenario; small vs tiny: +33% vs +12% (§6.1). Adding 10M tokens (1M per added language) leaves performance "essentially unaffected" (§6.1).
- Fig. 3 caption: small model, 1M monolingual + 1B similar-language tokens ≈ 1.2M monolingual tokens. Fig. 1 caption: adding 1B multilingual tokens ≈ adding 22% (low-resource) or removing 63% (high-resource) of the monolingual data.
- Gains plateau between 100M and 1B added tokens; in med-low resource settings, adding 1B tokens hurts (p < 0.001) except in the largest models (§6.1).
- Similarity analysis (low-resource, small, 100M added): Pearson r = 0.494 (syntactic), 0.341 (geographic), 0.346 (lexical); syntactic similarity explains 24.2% of variance, all three together 26.4% (§6.1, Fig. 4).
- High-resource: multilingual data hurts at all sizes (p < 0.001 when adding 1B tokens). Small model with 1B added ≈ removing 63% of target data; tiny ≈ 88% (p < 0.001). More similar added languages give slightly larger degradations (p < 0.05 in 7 of 12 scenarios) (§6.2). §7: degradations "similar to reducing high-resource dataset sizes by over 85%".
- Statistics: paired t-tests by language with Bonferroni correction; 95% confidence intervals in §6 plots (App. A.5).

## Training details (App. A.3, Table 1)
| Setting | Tiny | Mini | Small |
|---|---|---|---|
| Layers / embedding / heads | 2 / 128 / 2 | 4 / 256 / 4 | 4 / 512 / 8 |
| Learning rate | 1e-3 | 7e-4 (4e-4 low-resource) | 5e-4 (2e-4 low-resource) |

Shared: max sequence length 128; batch size 128; linear decay; warmup 10% of steps; Adam β1 0.9, β2 0.999, ε 1e-6; dropout 0.1. Epochs of target data: 20 (low, med-low), 10 (med-high), 2 (high); more than 20 epochs "often leads to overfitting" in low-resource runs. Example step counts: 1B monolingual (2 epochs) + 1B multilingual = 187,500 steps. All runs: 1.87 × 10^20 FLOPs, about 17,700 A6000 GPU hours.

## Limitations stated (§7.1)
Models up to 45M parameters; no languages indigenous to Australia and few from the Americas; results cover language modeling in individual languages, not downstream tasks or cross-lingual transfer.

## How ch-13a uses it
§1 (study design, relative log-likelihood, equivalent-token metric, low- and high-resource results), Generalization lens, Recipe row.
