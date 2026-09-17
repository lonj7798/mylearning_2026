---
chapter: ch-29a
course: llm-training
phase: read
excerpt_of: arXiv:2407.21783v3 (The Llama 3 Herd of Models), §3.4.2, §4.3.4, Table 7 (chapter-local verified extract; the older card long-context-llama3 has not been re-verified)
source_url: https://arxiv.org/abs/2407.21783
created_at: "2026-09-15"
---

# Excerpt: Llama 3 — long-context pre-training gate and long-context SFT data

- **Authors:** Llama Team, AI @ Meta
- **Year:** 2024 (arXiv v1 2024-07; v3 2024-11 used here)
- **Source type:** official technical report
- **Used in:** ch-29a §3.1, §3.2, §7, Recipe

## Long-context pre-training (§3.4.2)
- "We increase the supported context length in increments, pre-training until the model has successfully adapted to the
  increased context length. We assess successful adaptation by measuring whether (1) model performance on short-context
  evaluations has recovered completely and (2) the model perfectly solves 'needle in a haystack' tasks up to that length."
- "In Llama 3 405B pre-training, we increased context length gradually in six stages, starting from the original 8K context
  window and ending in the final 128K context window. This long-context pre-training stage was performed using approximately
  800B training tokens."
- Per-stage token counts and per-stage data mixtures are not printed in §3.4.2.

## Long-context SFT data (§4.3.4)
- "Naively applying our existing SFT recipe with only short-context data resulted in significant regressions in long-context
  capabilities from pre-training."
- "We use earlier versions of Llama 3 to generate synthetic data based on the key long-context use-cases: (possibly multi-turn)
  question-answering, summarization for long documents, and reasoning over code repositories."
- Question answering: "We split these documents into chunks of 8K tokens, and prompted an earlier version of the Llama 3 model
  to generate QA pairs conditional on randomly selected chunks. During training, the whole document is used as context."
- Summarization: "first summarizing the chunks of 8K input length using our strongest Llama 3 8K context model and then
  summarizing the summaries. During training we provide the full document and prompt the model to summarize the document while
  preserving all the important details. We also generate QA pairs based on the summaries of the documents and prompt the model
  with questions that require global understanding of the whole long document."
- Code: "We parse Python files to identify import statements and determine their dependencies. From here, we select the most
  commonly depended-upon files, specifically those referenced by at least five other files. We remove one of these key files
  from a repository and prompt the model to identify which files depended on the missing file and to generate the necessary
  missing code."
- "We further categorize these synthetically generated samples based on the sequence length (16K, 32K, 64K and 128K)."
- "Through careful ablations, we observe that mixing 0.1% of synthetically generated long-context data with the original
  short-context data optimizes the performance across both short-context and long-context benchmarks." No ablation table is printed.
- DPO: "using only short context training data in DPO did not negatively impact long-context performance as long as the SFT model
  is high quality in long context tasks. We suspect this is due to the fact that our DPO recipe has fewer optimizer steps than SFT."

## SFT data statistics (Table 7)
| Dataset | % of examples | Avg. # turns | Avg. # tokens | Avg. # tokens in context | Avg. # tokens in final response |
|---|---|---|---|---|---|
| General English | 52.66% | 6.3 | 974.0 | 656.7 | 317.1 |
| Code | 14.89% | 2.7 | 753.3 | 378.8 | 374.5 |
| Multilingual | 3.01% | 2.7 | 520.5 | 230.8 | 289.7 |
| Exam-like | 8.14% | 2.3 | 297.8 | 124.4 | 173.4 |
| Reasoning and tools | 21.19% | 3.1 | 661.6 | 359.8 | 301.9 |
| Long context | 0.11% | 6.7 | 38,135.6 | 37,395.2 | 740.5 |
| Total | 100% | 4.7 | 846.1 | 535.7 | 310.4 |

Derived (this course, not printed in the report): long-context token share of SFT data =
0.0011 × 38,135.6 / 846.1 ≈ 0.0496, about 5% of SFT tokens, while it is 0.11% of examples.

## Not reported
Number of long-context examples, generator sampling settings, any filtering or verification of the generated QA,
the ablation table behind the 0.1% choice.
