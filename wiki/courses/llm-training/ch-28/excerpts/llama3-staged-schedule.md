---
chapter: ch-28
course: llm-training
phase: read
excerpt_of: Llama Team, AI @ Meta — "The Llama 3 Herd of Models" (§3.2, §3.4.2, §4.3.4, Table 7, Table 21)
source_url: https://arxiv.org/abs/2407.21783
created_at: "2026-04-23"
revised: "2026-09-15 (rewritten against arXiv v3; the previous version quoted an unverified card)"
---

# Excerpt: Llama 3 — long-context pre-training stage and long-context SFT data

**Report:** arXiv 2407.21783 v3 (2024-11-23). All results in the report are for Llama 3.1 models (§1).
**Verified library cards:** [[llama-3]], [[llama-3-recipe]].

## RoPE base is an architecture setting for all of pre-training (§3.2)

> "We increase the RoPE base frequency hyperparameter to 500,000. This enables us to better support longer contexts; Xiong et al. (2023) showed this value to be effective for context lengths up to 32,768." (§3.2)

The same section lists an attention mask "that prevents self-attention between different documents within the same sequence" (§3.2). The report does not describe a RoPE base change during the long-context stage.

## Long-context pre-training (§3.4.2)

> "We increase the supported context length in increments, pre-training until the model has successfully adapted to the increased context length. We assess successful adaptation by measuring whether (1) model performance on short-context evaluations has recovered completely and (2) the model perfectly solves 'needle in a haystack' tasks up to that length. In Llama 3 405B pre-training, we increased context length gradually in six stages, starting from the original 8K context window and ending in the final 128K context window. This long-context pre-training stage was performed using approximately 800B training tokens." (§3.4.2)

Not printed: per-stage lengths, per-stage token counts, per-stage data mixture. Context parallelism uses an all-gather of K and V tensors, chosen partly because it supports "different types of attention masks ... such as the document mask" (§3.3.2); Table 4 lists CP 16 at sequence length 131,072.

## Long-context SFT data (§4.3.4)

> "Naively applying our existing SFT recipe with only short-context data resulted in significant regressions in long-context capabilities from pre-training ... We use earlier versions of Llama 3 to generate synthetic data based on the key long-context use-cases: (possibly multi-turn) question-answering, summarization for long documents, and reasoning over code repositories" (§4.3.4)

- **Question answering:** "We split these documents into chunks of 8K tokens, and prompted an earlier version of the Llama 3 model to generate QA pairs conditional on randomly selected chunks. During training, the whole document is used as context."
- **Summarization:** "hierarchical summarization of long-context documents by first summarizing the chunks of 8K input length using our strongest Llama 3 8K context model and then summarizing the summaries ... We also generate QA pairs based on the summaries of the documents and prompt the model with questions that require global understanding of the whole long document."
- **Long context code reasoning:** "We parse Python files to identify import statements and determine their dependencies. From here, we select the most commonly depended-upon files, specifically those referenced by at least five other files. We remove one of these key files from a repository and prompt the model to identify which files depended on the missing file and to generate the necessary missing code."
- **Length buckets:** "16K, 32K, 64K and 128K".
- **Mixing:** "Through careful ablations, we observe that mixing 0.1% of synthetically generated long-context data with the original short-context data optimizes the performance across both short-context and long-context benchmarks." No ablation table is printed.
- **DPO:** "using only short context training data in DPO did not negatively impact long-context performance as long as the SFT model is high quality in long context tasks. We suspect this is due to the fact that our DPO recipe has fewer optimizer steps than SFT."

## SFT data statistics (Table 7)

| Dataset | % of examples | Avg. turns | Avg. tokens | Avg. tokens in context | Avg. tokens in final response |
|---|---|---|---|---|---|
| Long context | 0.11% | 6.7 | 38,135.6 | 37,395.2 | 740.5 |
| Total | 100% | 4.7 | 846.1 | 535.7 | 310.4 |

Derived: a long-context example has 38,135.6 / 846.1 ≈ 45 times the average token count, so its token share is larger than its 0.11% example share. The token share is not printed.

## Long-context evaluation reported (§5.2.6, Table 21)

Needle-in-a-Haystack: "successfully retrieving 100% of needles at all document depths and context lengths". Multi-needle ("insert 4 needles in the context and test if a model can retrieve 2 needles", average recall over 10 lengths up to 128K): 8B 98.8, 70B 97.5, 405B 98.1. InfiniteBench En.MC: 8B 65.1, 70B 78.2, 405B 83.4. The report does not include RULER.

## Connections

- [[ruler]] Table 3 is the source of the claimed 128K / effective 64K result for Llama 3.1 70B.
- ch-28 §4; the long-context stage as a mid-training step is covered in ch-32b.
