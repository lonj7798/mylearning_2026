---
chapter: ch-11
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/model-reports/llama-3.md
source_url: https://arxiv.org/abs/2407.21783
created_at: "2026-04-23"
revised: 2026-09 (generality revision) — rewritten from arXiv:2407.21783 v3 and the card verified on 2026-09-14; the earlier version described the tokenizer as SentencePiece-style, invented shard sizes, and attributed execution-based filtering to pre-training code data
---

# Excerpt: The Llama 3 Herd of Models — tokenizer and PII filtering

Llama Team, AI @ Meta, arXiv 2024-07 (v3 2024-11-23). Library card: [[llama-3]].

## Tokenizer (§3.2, Table 3)
- Table 3 lists "Vocabulary Size 128,000" for the 8B, 70B, and 405B models.
- §3.2: "Our token vocabulary combines 100K tokens from the tiktoken tokenizer with 28K additional tokens to better support non-English languages. Compared to the Llama 2 tokenizer, our new tokenizer improves compression rates on a sample of English data from 3.17 to 3.94 characters per token. This enables the model to 'read' more text for the same amount of training compute. We also found that adding 28K tokens from select non-English languages improved both compression ratios and downstream performance, with no impact on English tokenization."
- The report does not give the size of the English sample, the languages of the 28K tokens, or the downstream numbers for the addition.

## PII (§3.1, §3.1.1)
- §3.1: "We remove domains that contain large amounts of personally identifiable information (PII), and domains with known adult content."
- §3.1.1: "we implement filters designed to remove data from websites [that] are likely to contain unsafe content or high volumes of PII, domains that have been ranked as harmful according to a variety of Meta safety standards, and domains that are known to contain adult content."
- No detector, threshold, or removal rate is reported.

## Pre-training mix (card, §3.1.2)
Roughly 50% general knowledge, 25% math and reasoning, 17% code, 8% multilingual tokens.

## How ch-11 uses it
§1-§2 (vocabulary construction by extension of an existing BPE vocabulary), §6 (domain-level PII filtering), Recipe rows.
