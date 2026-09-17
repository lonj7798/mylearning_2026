---
chapter: ch-09
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/the-pile.md (the library card has no numbers and no Verification section on 2026-09-15; values below are taken from the primary source)
source_url: https://arxiv.org/abs/2101.00027
primary_version: arXiv:2101.00027v1 (2020-12-31)
created_at: "2026-04-23"
revised_at: "2026-09-15"
---

# Excerpt: The Pile: An 800GB Dataset of Diverse Text for Language Modeling

Values and quotations used by ch-09 `read.md`, checked against the v1 PDF on 2026-09-15. Authors: Leo Gao, Stella Biderman, Sid Black, Laurence Golding, Travis Hoppe, Charles Foster, et al. (EleutherAI).

## Weights and epochs (§2)
> "Following Brown et al. (2020), we increase the weights of higher quality components, with certain high-quality datasets such as Wikipedia being seen up to 3 times ("epochs") for each full epoch over the Pile."

## Table 1 (all 22 components)
| Component | Raw Size | Weight | Epochs | Effective Size | Mean Document Size |
|---|---|---|---|---|---|
| Pile-CC | 227.12 GiB | 18.11% | 1.0 | 227.12 GiB | 4.33 KiB |
| PubMed Central | 90.27 GiB | 14.40% | 2.0 | 180.55 GiB | 30.55 KiB |
| Books3 | 100.96 GiB | 12.07% | 1.5 | 151.44 GiB | 538.36 KiB |
| OpenWebText2 | 62.77 GiB | 10.01% | 2.0 | 125.54 GiB | 3.85 KiB |
| ArXiv | 56.21 GiB | 8.96% | 2.0 | 112.42 GiB | 46.61 KiB |
| Github | 95.16 GiB | 7.59% | 1.0 | 95.16 GiB | 5.25 KiB |
| FreeLaw | 51.15 GiB | 6.12% | 1.5 | 76.73 GiB | 15.06 KiB |
| Stack Exchange | 32.20 GiB | 5.13% | 2.0 | 64.39 GiB | 2.16 KiB |
| USPTO Backgrounds | 22.90 GiB | 3.65% | 2.0 | 45.81 GiB | 4.08 KiB |
| PubMed Abstracts | 19.26 GiB | 3.07% | 2.0 | 38.53 GiB | 1.30 KiB |
| Gutenberg (PG-19) | 10.88 GiB | 2.17% | 2.5 | 27.19 GiB | 398.73 KiB |
| OpenSubtitles | 12.98 GiB | 1.55% | 1.5 | 19.47 GiB | 30.48 KiB |
| Wikipedia (en) | 6.38 GiB | 1.53% | 3.0 | 19.13 GiB | 1.11 KiB |
| DM Mathematics | 7.75 GiB | 1.24% | 2.0 | 15.49 GiB | 8.00 KiB |
| Ubuntu IRC | 5.52 GiB | 0.88% | 2.0 | 11.03 GiB | 545.48 KiB |
| BookCorpus2 | 6.30 GiB | 0.75% | 1.5 | 9.45 GiB | 369.87 KiB |
| EuroParl | 4.59 GiB | 0.73% | 2.0 | 9.17 GiB | 68.87 KiB |
| HackerNews | 3.90 GiB | 0.62% | 2.0 | 7.80 GiB | 4.92 KiB |
| YoutubeSubtitles | 3.73 GiB | 0.60% | 2.0 | 7.47 GiB | 22.55 KiB |
| PhilPapers | 2.38 GiB | 0.38% | 2.0 | 4.76 GiB | 73.37 KiB |
| NIH ExPorter | 1.89 GiB | 0.30% | 2.0 | 3.79 GiB | 2.11 KiB |
| Enron Emails | 0.88 GiB | 0.14% | 2.0 | 1.76 GiB | 1.78 KiB |
| The Pile | 825.18 GiB | | | 1254.20 GiB | 5.91 KiB |

Caption: "Raw Size is the size before any up- or down-sampling. Weight is the percentage of bytes in the final dataset occupied by each dataset. Epochs is the number of passes over each constituent dataset during a full epoch over the Pile. Effective Size is the approximate number of bytes in the Pile occupied by each dataset."

## Bits per byte (§3.1)
> "Our preferred metric is bits per UTF-8 encoded byte (BPB). [...] BPB = (L_T/L_B) log2(e^ℓ) = (L_T/L_B) ℓ/ln(2), where L_T is the length of the dataset in tokens and L_B is the length of the dataset in UTF-8 encoded bytes. We find that L_T/L_B is 0.29335 GPT-2-tokens/byte across the Pile"

## Evaluation setup (§4, §4.1)
> "we train architecturally-identical 1.3 billion parameter models based on those in Brown et al. (2020) on different datasets"

> "we decontaminate any instances of the evaluation sets using the same 13-gram overlap filtering as in Brown et al. (2020) and downsample to 40GB to control for dataset size. As we control for dataset size, we emphasize that our evaluation is generous to CC-100 (en), which is about 1/3 the size of the Pile in reality."

Token counts and training steps for these models are not stated in §4.

## Table 3 (size-controlled results)
| Trained on | Pile val (BPB) | Pile test (BPB) | WikiText (PPL) | LAMBADA (PPL) | LAMBADA (ACC) |
|---|---|---|---|---|---|
| The Pile | 0.9281 | 0.9433 | 5.59 | 12.78 | 50.1 |
| CC-100 (en) | 1.3143 | 1.3293 | 8.27 | 11.78 | 49.7 |
| Raw CC | 1.1180 | 1.1275 | 11.75 | 19.84 | 43.8 |

## Table 4 (BPB on each Pile test component; columns are training corpora)
| Evaluated on | The Pile | CC-100 (en) | Raw CC (en) |
|---|---|---|---|
| Pile-CC | 0.9989 | 1.0873 | 1.0287 |
| PubMed Central | 0.6332 | 1.1311 | 0.9120 |
| Books3 | 1.0734 | 1.2264 | 1.1366 |
| OpenWebText2 | 0.9938 | 1.2222 | 1.0732 |
| ArXiv | 0.7945 | 1.8159 | 1.2642 |
| Github | 0.5597 | 1.6509 | 0.9301 |
| FreeLaw | 0.6978 | 1.0221 | 0.9468 |
| Stack Exchange | 0.8152 | 1.5414 | 1.1292 |
| USPTO Backgrounds | 0.6731 | 0.8772 | 0.8455 |
| PubMed Abstracts | 0.7313 | 1.0193 | 0.9718 |
| Gutenberg (PG-19) | 1.1426 | 1.2780 | 1.2235 |
| OpenSubtitles | 1.0909 | 1.1827 | 1.2139 |
| Wikipedia (en) | 0.8961 | 1.1807 | 1.0252 |
| DM Mathematics | 1.5206 | 3.1774 | 2.6229 |
| Ubuntu IRC | 1.4085 | 2.1243 | 1.5691 |
| BookCorpus2 | 1.0613 | 1.1346 | 1.0914 |
| EuroParl | 1.1202 | 2.7141 | 1.4917 |
| HackerNews | 1.0968 | 1.4352 | 1.2305 |
| YoutubeSubtitles | 1.4269 | 2.3287 | 1.5607 |
| PhilPapers | 1.1256 | 1.4269 | 1.2090 |
| NIH ExPorter | 0.7347 | 0.9713 | 0.9225 |
| Enron Emails | 0.8301 | 1.3300 | 1.0483 |

## Results text (§4.2)
> "models trained on Pile improve significantly over both Raw CC and CC-100 on all components of the Pile, as shown in Table 4. This indicates that models trained on the Pile have greater cross-domain generalization capabilities without compromising performance on traditional benchmarks."

> "Surprisingly, raw Common Crawl performs better on the Pile BPB than CC-100, despite losing by a significant margin on LAMBADA and WikiText. We hypothesize that this is due to the perplexity based filtering used in CC-100, where a language model is trained on Wikipedia and all data with a perplexity too high or too low is discarded."

## Tokens per byte and language (§5.1, §5.2)
> "many of the sets with the lowest bytes per token are those which consist in large part of non-text content (Github, ArXiv, Stack Exchange, and DM Mathematics) or languages other than English (EuroParl)."

> "Using fasttext (Suárez et al., 2019a), we determine that the Pile is 97.4% English."

## Correction to earlier ch-09 material
The April 2026 version of this excerpt and of ch-09 listed percentages computed by dividing raw sizes by the 1254.20 GiB effective size (for example PubMed Central 7.6%) and called them token-weighted. The printed weights are byte shares of the effective dataset (PubMed Central 14.40%, Books3 12.07%, ArXiv 8.96%).
