---
chapter: ch-09
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/epoch-data-stock.md (library card not present on 2026-09-15; quotations below are taken from the primary source)
source_url: https://arxiv.org/abs/2211.04325
primary_version: arXiv:2211.04325v2 (2024-06-04; v1 2022-11)
created_at: "2026-09-15"
---

# Excerpt: Will we run out of data? Limits of LLM scaling based on human-generated data

Verbatim quotations and table values used by ch-09 `read.md`, with loci. Authors: Pablo Villalobos, Anson Ho, Jaime Sevilla, Tamay Besiroglu, Lennart Heim, Marius Hobbhahn (Epoch). Checked against the v2 PDF on 2026-09-15. Source type: paper (forecast based on a model of data stock and demand).

## Abstract
> "Our findings indicate that if current LLM development trends continue, models will be trained on datasets roughly equal in size to the available stock of public human text data between 2026 and 2032, or slightly earlier if models are overtrained."

## Figure 1 caption
> "The intersection of the stock and dataset size projection lines indicates the median year (2028) in which the stock is expected to be fully utilized if current LLM development trends continue. At this point, models will be trained on dataset sizes approaching the total effective stock of text in the indexed web: around 4e14 tokens, corresponding to training compute of ∼5e28 FLOP for non-overtrained models."

## Table 1 (estimates of the stock of data on the web, in tokens)
| Estimate | Median | 95% CI |
|---|---|---|
| Common Crawl | 130T | [100T, 260T] |
| Indexed web | 510T | [130T, 2100T] |
| Whole web | 3100T | [1900T, 5200T] |
| Images | 300T | N/A |
| Video | 1350T | N/A |

Tokenizer (footnote 6): "We make our estimates based on cl100k base, a byte-pair encoding (BPE) tokenizer from OpenAI".

## Quality adjustment (§2.3.1)
> "we believe with 95% certainty that between 10% and 40% of deduplicated web data can be used for training without significantly compromising performance."

## Exhaustion date (§2.5)
> "The median exhaustion year is 2028, and by 2032 exhaustion becomes very likely."
