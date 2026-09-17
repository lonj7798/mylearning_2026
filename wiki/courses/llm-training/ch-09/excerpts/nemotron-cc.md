---
chapter: ch-09
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/nemotron-cc.md (library card not present on 2026-09-15; quotations below are taken from the primary source)
source_url: https://arxiv.org/abs/2412.02595
primary_version: arXiv:2412.02595v2 (2025-05-30; v1 2024-12)
created_at: "2026-09-15"
---

# Excerpt: Nemotron-CC: Transforming Common Crawl into a Refined Long-Horizon Pretraining Dataset

Verbatim quotations and table values used by ch-09 `read.md`, with loci. Authors: Dan Su, Kezhi Kong, Ying Lin, Joseph Jennings, Brandon Norick, Markus Kliegl, et al. (NVIDIA). Checked against the v2 PDF on 2026-09-15.

## Motivation (§1)
> "Both DCLM and FineWeb-Edu contain around 80% near-duplicates (1T and 0.2T unique tokens, respectively) [...] and to train on these datasets for many trillions of tokens implies seeing essentially the same samples many times during training."

## Dataset (§1 contributions, §2.4)
> "We propose a method for transforming English Common Crawl into a 6.3T token long-horizon pretraining dataset, consisting of 4.4T globally deduplicated original tokens and 1.9T synthetically generated tokens. We release the dataset under the Common Crawl Terms of Use and a reference implementation as part of the Apache 2.0 open-source NeMo Curator library. The quality classifier models have been released as well."

> "Combining the techniques above to the 99 snapshots CC-MAIN-2013-20 through CC-MAIN-2024-30 of Common Crawl, we create a 6.3T token dataset (Nemotron-CC)"

Table 4 (trillions of tokens; Total / Unique / Synthetic): FineWebEdu-2 5.4 / 1.1 / -; FineWebEdu 1.3 / 0.2 / -; DCLM 3.8 / 1.0 / -; Nemotron-CC 6.3 / 4.4 / 1.9; Nemotron-CC-HQ 1.1 / 0.6 / 0.5. Caption: ""Unique" shows the estimated number of tokens after global fuzzy deduplication of the real tokens."

## Training blend (§3.1, App. D Table 12)
> "Unless otherwise noted, we train for 1T tokens on a blend of 73% English Common Crawl data and 27% a fixed mix of specialized code, papers, books, patents, and Wikipedia datasets (Adler et al., 2024). When comparing datasets, we vary only the 73% English Common Crawl portion."

Table 12 (Blend %): English Common Crawl 73; Books and patents 9; Papers 9; Code 5; Conversational 3; Wikipedia 1.

## Results at 1T tokens (Table 5, 8B)
| Dataset | MMLU | Avg (10 tasks) |
|---|---|---|
| FineWebEdu-2 | 42.4 | 53.2 |
| FineWebEdu | 42.9 | 53.2 |
| DCLM | 53.4 | 57.0 |
| Nemotron-CC | 53.0 | 57.8 |
| Nemotron-CC-HQ | 59.0 | 60.1 |

## Long horizon (§3.2, Table 6, App. E)
> "Our dataset contributed 7.2T of the tokens used to train an 8B model for 15T tokens. As shown in Table 6, our model achieves a higher average accuracy than Llama 3.1 8B, which was also trained for 15T tokens, including an MMLU score of 70.3 vs. Llama's 65.3."

Table 6 caption: "The numbers for Llama 3.1 are from our own lm-evaluation-harness setup described in Section 3.1 and may not match Meta's publicly reported numbers, as Meta made various customizations to the benchmarks."

> "For the 15T token training run, a two-phase curriculum was employed that is described in more detail in Feng et al. (2024). The first phase of 9T tokens used 59% English Common Crawl data (5.31T) and the second phase of 6T tokens used 31% (1.86T), for a combined total of 47.8% (7.17T)." (App. E)
