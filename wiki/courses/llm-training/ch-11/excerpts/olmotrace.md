---
chapter: ch-11
course: llm-training
phase: read
excerpt_of: none (no library card planned on 2026-09-15)
source_url: https://arxiv.org/abs/2504.07096
created_at: "2026-09-15"
---

# Excerpt: OLMoTrace: Tracing Language Model Outputs Back to Trillions of Training Tokens

**Authors:** Jiacheng Liu, Taylor Blanton, Yanai Elazar, Sewon Min, YenSung Chen, Arnavi Chheda-Kothary, et al. (AI2, University of Washington, UC Berkeley, Stanford).
**Version read:** arXiv:2504.07096v2 (8 Jul 2025); v1 Apr 2025.
**Status:** no library card existed; values read in the v2 PDF text at the stated locus.

## Indexed data (Table 1, §2)
Full training data of OLMo-2-32B-Instruct: pre-training olmo-mix-1124 (3,081M documents, 4,575B tokens), mid-training dolmino-mix-1124 (81M, 34B; sources already in pre-training excluded), post-training SFT, DPO, RLVR (1.7M, 1.6B); total 3,164M documents and 4,611B tokens counted with the Llama-2 tokenizer.

## Pipeline (§3)
1. Find maximal spans of the model output that appear verbatim in the training data. Criteria: existence (at least once), self-contained (no period or newline token except at the end; no partial words), maximality (not a subspan of another qualifying span).
2. Keep the K = ⌈0.05 × L⌉ spans with the smallest span unigram probability (product of token unigram probabilities over the training data), where L is the output length in tokens.
3. Retrieve up to 10 enclosing document snippets per span.
4. Merge overlapping spans and snippets from the same document.
5. Rank documents with BM25 (query = prompt + response); buckets by normalized score: ≥ 0.7 high, 0.5-0.7 medium, < 0.5 low.
Longest matching prefix of each suffix is found with one FIND query on the infini-gram suffix array (§3.1, Algorithm 1); index shards are limited to 500B tokens.

## Measurements (§3.2, §4)
- 98 internal conversations; mean response 458 tokens; mean latency 4.46 s (steps 1-3) on a 64-vCPU, 256 GB RAM node with 40 TB SSD.
- Spans after filtering: mean 10.4 tokens, median 10.
- Retrieved documents by stage: 96.7% pre-training, 0.9% mid-training, 2.4% post-training (0.9% SFT, 1.5% DPO, 0% RLVR); "this distribution heavily depends on the topic".
- Relevance: human expert score (0-3) of first displayed document 1.90, top-5 1.43; gpt-4o judge Spearman 0.73 with human; final setting 1.82 and 1.50 (judge).
- Case study (§5): an OLMo answer to AIME 2024 I Problem 4 contains "\binom{10}{4} = \frac{10!}{4!(10-4)!} = 210", which appears verbatim in the post-training data.

## Limitations (paper section "Limitations")
"The retrieved documents should not be interpreted as having a causal effect on the LM output, or as supporting evidence or citations for the LM output." Red-teaming covered copyright, PII, and toxicity; the team "were unable to find any PII data in OLMoTrace results" and added a regex-based filter that blocks documents with PII.

## How ch-11 uses it
§6 (tracing outputs to training data), Generalization lens (memorization versus generalization evidence), Common mistakes.
