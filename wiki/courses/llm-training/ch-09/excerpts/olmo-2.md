---
chapter: ch-09
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/model-reports/olmo-2.md (the library card has no Verification section on 2026-09-15; values below are taken from the primary source)
source_url: https://arxiv.org/abs/2501.00656
primary_version: arXiv:2501.00656v3 (2025-10-08)
created_at: "2026-09-15"
---

# Excerpt: 2 OLMo 2 Furious — pretraining data composition

Values and quotations from the OLMo 2 report used by ch-09 `read.md`, with loci. Authors: Team OLMo (Pete Walsh, Luca Soldaini, Dirk Groeneveld, Kyle Lo, Shane Arora, Akshita Bhagia, et al.; Allen Institute for AI). Checked against the v3 PDF on 2026-09-15.

## Stage token counts (§2.3 "Overall")
> "In total, OLMo 2 7B is trained on 4.05 trillion tokens (3.90 trillion for pretraining stage), OLMo 2 13B is trained on 5.6 trillion tokens (5 trillion for pretraining stage), and OLMo 2 32B is trained on 6.6 trillion tokens (6.06 trillion for pretraining stage)."

The Table 9 caption states: "Pretrain checkpoints have been trained on 4 trillion (1B, 7B), 5 trillion (13B) and 7 trillion (32B) tokens respectively." The two statements differ for 7B and 32B.

## Table 4: OLMo 2 1124 Mix (§2.4.1)
| Source | Type | Tokens | Words | Bytes | Docs |
|---|---|---|---|---|---|
| DCLM-Baseline | Web pages | 3.71T | 3.32T | 21.32T | 2.95B |
| StarCoder (filtered version from OLMoE Mix) | Code | 83.0B | 70.0B | 459B | 78.7M |
| peS2o (from Dolma 1.7) | Academic papers | 58.6B | 51.1B | 413B | 38.8M |
| arXiv | STEM papers | 20.8B | 19.3B | 77.2B | 3.95M |
| OpenWebMath | Math web pages | 12.2B | 11.1B | 47.2B | 2.89M |
| Algebraic Stack | Math proofs code | 11.8B | 10.8B | 44.0B | 2.83M |
| Wikipedia & Wikibooks (from Dolma 1.7) | Encyclopedic | 3.7B | 3.16B | 16.2B | 6.17M |
| Total | | 3.90T | 3.48T | 22.38T | 3.08B |

> "It consists of approximately 3.9 trillion tokens, with over 95% derived from web data. We refer to this set as OLMo 2 Mix 1124."

> "In an attempt to include higher quality code, we remove any document from a repository with fewer than 2 stars on GitHub."

## Table 10: "PT Mix" column (§4.3)
Share of a 50B-token sample of the pretraining mix, used as the reference column for mid-training mixes: DCLM 95.2; StarCoder 2.1; peS2o 1.5; ArXiv 0.5; Algebraic Stack 0.3; OpenWebMath 0.3; Wikipedia 0.1. Caption: "PT Mix is sampled (with repetition) from the pretraining stage."

## Mid-training purpose (§2.4.2)
> "After the initial pretraining stage on mostly web data, we further train with a mixture of web data that has been more restrictively filtered for quality and a collection of domain-specific high quality data, much of which is synthetic."

Mid-training mixtures and their effects are taught in ch-32.
