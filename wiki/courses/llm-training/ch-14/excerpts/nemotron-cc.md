---
chapter: ch-14
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/nemotron-cc.md (library card not present on 2026-09-15; quotations below are taken from the primary source)
source_url: https://arxiv.org/abs/2412.02595
primary_version: arXiv:2412.02595v2 (2025-05-30; v1 2024-12)
created_at: "2026-09-15"
---

# Excerpt: Nemotron-CC: Transforming Common Crawl into a Refined Long-Horizon Pretraining Dataset

Verbatim quotations and table values used by ch-14 `read.md`, with loci. Authors: Dan Su, Kezhi Kong, Ying Lin, Joseph Jennings, Brandon Norick, Markus Kliegl, et al. (NVIDIA). Checked against the v2 PDF on 2026-09-15. Source type: paper (official dataset release).

## Motivation (Abstract, §1)
> "Recent English Common Crawl datasets like FineWeb-Edu and DCLM achieved significant benchmark gains via aggressive model-based filtering, but at the cost of removing 90% of data. This limits their suitability for long token horizon training, such as 15T tokens for Llama 3.1."
> "Both DCLM and FineWeb-Edu contain around 80% near-duplicates (1T and 0.2T unique tokens, respectively) ... and to train on these datasets for many trillions of tokens implies seeing essentially the same samples many times during training." (§1)

## Table 4 (dataset sizes, trillions of tokens; "Unique" is after global fuzzy deduplication of real tokens)
| Dataset | Total | Unique | Synthetic |
|---|---|---|---|
| FineWebEdu-2 | 5.4 | 1.1 | - |
| FineWebEdu | 1.3 | 0.2 | - |
| DCLM | 3.8 | 1.0 | - |
| Nemotron-CC | 6.3 | 4.4 | 1.9 |
| Nemotron-CC-HQ | 1.1 | 0.6 | 0.5 |

## Rephrasing (§2.3)
> "For low-quality data, our goal is to improve the quality by reducing noise and errors while preserving useful information"
> "For high-quality data, we aim to obtain more unique tokens and condense essential knowledge. According to (Muennighoff et al., 2024), adding repeated tokens yields a diminishing return, especially after 4 epochs."
> "We adopt the Wikipedia style prompt from (Maini et al., 2024) to rewrite low-quality documents"
> "For high-quality documents, we generate synthetic data using four additional prompts: (1) Diverse Question-Answer (QA) pairs ... (2) Distill ... (3) Extract knowledge ... (4) Knowledge list"
> "As we increase the length of provided text, the model shows a tendency to produce over-simplified outputs with reduced detail. Therefore, we chunk each document into segments"
- Table 3 (synthetic tokens, billions): low-quality source 403.0 → Wikipedia 336.3; high-quality source 451.3 → Wikipedia 372.9, Diverse QA Pairs 499.5, Distill 157.6, Extract Knowledge 303.6, Knowledge List 203.2. "We do not use medium-quality documents for synthetic data generation".
> "Using the instruct version of Mistral NeMo 12B with FP8 inference, a top-p value of 0.9, and a sampling temperature of 0.5, we synthesize over 1.8T tokens as Table 3 shows, including 336.3B tokens from low-quality documents and 1.5T tokens from high-quality documents."

## Table 5 (8B models, 1T tokens each; 73% English Common Crawl from the tested dataset, 27% fixed non-Crawl data)
| Dataset | MMLU | Avg (10 tasks) |
|---|---|---|
| FineWebEdu-2 | 42.4 | 53.2 |
| FineWebEdu | 42.9 | 53.2 |
| DCLM | 53.4 | 57.0 |
| Nemotron-CC | 53.0 | 57.8 |
| Nemotron-CC-HQ | 59.0 | 60.1 |

> "Our complete 6.3T token dataset (Nemotron-CC) gives MMLU and average accuracies roughly on par with DCLM. But since this dataset contains 4× more unique real tokens, we expect it to be superior in data-constrained settings like 15T token training runs." (§3.2)

## Table 10 (8B models, 1T tokens each; blend is 73% English Common Crawl variant and 27% fixed non-Crawl data, §3.1)
> "(1) LQ-Base: original Common Crawl data including low-quality documents; (2) LQ-Synthetic: an augmented version of LQ-Base where the low-quality documents are rephrased; (3) HQ-Base: a blend containing eightfold high-quality documents and less low- and medium-quality documents; (4) HQ-Synthetic: a variant of HQ-Base where 4 repetitions of the high-quality documents are swapped out for synthetic datasets."

| Blend | ARC-E | ARC-C | MMLU | Avg (10 tasks) |
|---|---|---|---|---|
| LQ-Base | 67.7 | 41.8 | 48.2 | 52.5 |
| LQ-Synthetic | 71.3 | 45.2 | 47.1 | 54.0 |
| HQ-Base | 74.2 | 47.7 | 53.4 | 55.8 |
| HQ-Synthetic | 76.7 | 49.2 | 53.6 | 56.7 |

> "rephrasing low-quality data leads to 1.50 absolute gains on average score ... however, we also encounter slight accuracy drops on some tasks, which may indicate potential misinformation introduced by data synthesis."
> "swapping 4 out of 8 epochs of high-quality data with a mix of synthetic datasets improves accuracy on most benchmarks. This improvement could potentially result from two factors: the incorporation of fresh unique tokens and styles that enable the model to learn specific abilities (e.g., question answering) or absorb knowledge more efficiently."

## Long horizon (§3.2, Table 6, App. E)
> "Our dataset contributed 7.2T of the tokens used to train an 8B model for 15T tokens. As shown in Table 6, our model achieves a higher average accuracy than Llama 3.1 8B, which was also trained for 15T tokens, including an MMLU score of 70.3 vs. Llama's 65.3."
- Table 6 caption: the Llama 3.1 numbers "are from our own lm-evaluation-harness setup ... and may not match Meta's publicly reported numbers".
> "The first phase of 9T tokens used 59% English Common Crawl data (5.31T) and the second phase of 6T tokens used 31% (1.86T), for a combined total of 47.8% (7.17T)." (App. E)

## Limitations (§6)
> "For the rephrased data, we did not verify the factual accuracy or fidelity to the original contents. More work is required to understand the risks of hallucinations or loss of content diversity in this setting"
> "Finally, we did not decontaminate the dataset, as there is not yet a strong consensus on how to best do this and the impact is uncertain and debated, especially for large models trained over large token horizons. We note that the datasets we compare against (FineWeb-Edu, DCLM) were released without decontamination, and the model we compare against (Meta Llama 3.1) was also trained on contaminated data."
