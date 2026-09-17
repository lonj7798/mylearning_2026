---
chapter: ch-14
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/epoch-data-stock.md (library card not present on 2026-09-15; quotations below are taken from the primary source)
source_url: https://arxiv.org/abs/2211.04325
primary_version: arXiv:2211.04325v2 (2024-06-04; v1 2022-11)
created_at: "2026-09-15"
---

# Excerpt: Will we run out of data? Limits of LLM scaling based on human-generated data

Verbatim quotations used by ch-14 `read.md`, with loci. Authors: Pablo Villalobos, Anson Ho, Jaime Sevilla, Tamay Besiroglu, Lennart Heim, Marius Hobbhahn (Epoch). Checked against the v2 PDF on 2026-09-15. Source type: paper (forecast based on a model of data stock and data demand). Token counts use the cl100k_base tokenizer (footnote 6).

## Abstract
> "Our findings indicate that if current LLM development trends continue, models will be trained on datasets roughly equal in size to the available stock of public human text data between 2026 and 2032, or slightly earlier if models are overtrained."

## Quality adjustment (§2.3.1)
> "we believe with 95% certainty that between 10% and 40% of deduplicated web data can be used for training without significantly compromising performance."

## Multi-epoch adjustment (§2.3.2)
> "The authors estimate the maximum increase in the effective dataset size that can be gained from multiple epochs at between 3x and 15x, and we anchor to this estimate in adjusting our model. Because additional epochs yield diminishing returns, the upper extreme of 15x would require a very inefficient training procedure with a large number of epochs that does not correspond to common practices. For this reason we reduce it to 5x."
- Footnote 16: "Typical numbers are between 1 and 4 epochs".

## Figure 3 (adjusted stock sizes, median [95% CI], tokens)
- Initial deduplicated stock (indexed web): 510T [130T, 2100T]
- Quality-adjusted stock: 100T [22T, 490T]
- Repetition-adjusted stock: 320T [65T, 1700T]

## Exhaustion date (§2.5)
> "The median exhaustion year is 2028, and by 2032 exhaustion becomes very likely."

## Overtraining (§2.5, footnotes 23-24)
> "Overtraining by 5x means that the tokens/parameter ratio is 5x higher than that of a compute-optimal model"
> "normally only smaller models are heavily overtrained. For example, Llama 3 8B is overtrained by close to 100x, while Llama 3 70B is only overtrained by 10x."
