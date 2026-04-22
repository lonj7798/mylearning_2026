<!-- scope: tool-calling synthesis — retriever-augmented API-calling LLM with self-instruct-style data
     deps: [[self-instruct]]
     see-also: [[toolllm]], [[api-bank]], [[bfcl]]
-->

# Gorilla: Large Language Model Connected with Massive APIs
- **Core Insight:** Pairing a fine-tuned API-calling LLM with an **API retriever** (BM25 or GPT-embedding) substantially reduces hallucinated API calls and enables zero-shot generalization to new APIs via retrieval-augmented fine-tuning; the retriever at train time teaches the model to lean on retrieved documentation rather than memorize API names.
- **Guideline:** Always train function-calling models with retrieval-in-the-loop — include the retrieved API doc in the prompt context during SFT, so the model learns `condition on doc → emit call` rather than `recall name from memory`.
- **Authors:** Shishir G. Patil, Tianjun Zhang, Xin Wang, Joseph E. Gonzalez (UC Berkeley)
- **Year:** 2023
- **URL:** https://arxiv.org/abs/2305.15334
- **Relevant topics:** function calling, API retrieval, retrieval-augmented fine-tuning, Gorilla

## Abstract
Gorilla is a Llama-7B fine-tuned for API-call generation across three ML hubs (TorchHub, TensorHub, HuggingFace — ~1,600 APIs total). The key insight is retriever-aware training: at training time, each example's prompt includes the retrieved top-k API documentation; the model is trained to produce the correct call given the docs. At inference, the retriever is still used, so the model generalizes to new/updated APIs without retraining. Gorilla outperforms GPT-4 on the three-hub eval in direct API-call accuracy.

## Key Contributions
- **Retriever-aware fine-tuning** — first to systematically train with retrieved context.
- **APIBench dataset** (~16K instruction-API pairs) from three ML-library docs.
- Demonstration that AST-level accuracy and hallucination rate both improve with retriever context.
- Released Gorilla-7B, Gorilla-MPT-7B, and a Gradio demo.

## Synthesis pipeline (REQUIRED — concrete, modality-specific)
- **Step 1 — API scraping:** crawl TorchHub, TensorHub, HuggingFace model documentation. Each API has a function signature, description, arg types, and example invocation.
- **Step 2 — Instruction generation (self-instruct style):** for each API, prompt GPT-4 with the doc and ask it to generate 10 diverse user instructions that would plausibly call this API. Produces ~16K (instruction, API) pairs.
- **Step 3 — Retriever training:** off-the-shelf BM25 + a contrastively fine-tuned dense retriever (GPT-embeddings → Gorilla-retriever).
- **Step 4 — Retriever-augmented SFT:** training input includes the user instruction + top-1 retrieved API doc; target is the correct API invocation code. For 20% of examples, the retriever is "oracled" (perfect); 80% are real retrieval results (noisy).
- **Step 5 — Filtering:** accept pairs where the generated instruction's gold API matches the one documented.
- **Output shape:** 16,450 (instruction, retrieved-doc, API-call) triplets; single-turn. Average call length ~50 tokens.
- **Teacher model:** GPT-4 (for instruction generation).
- **Cost / compute:** ~$5K GPT-4 API + standard Llama-7B fine-tuning.

## Modality-specific technical details (REQUIRED — tool-calling)
- **API registry size:** ~1,645 APIs (TorchHub 94 + TensorHub 626 + HuggingFace 925).
- **Exact verification rules:** AST match between predicted and gold API invocation — checks function name, args, arg types.
- **Hallucination-rate measurement:** 11% for Gorilla-7B with retriever, vs 40% for GPT-4 without retriever. Hallucination defined as predicting an API that doesn't exist in the hub.
- **Call format:** natural Python code (e.g., `model = torch.hub.load('pytorch/vision', 'resnet50', pretrained=True)`), not JSON tool-calls.
- **Why retrieval-in-training matters:** models trained without retriever context memorize APIs from the training distribution and fail on updated/new APIs; retriever-trained models learn `doc → call`, which transfers.

## Quality / diversity evaluation
- Gorilla-7B with retriever: **AST accuracy ~72%** on APIBench (TorchHub 67, TensorHub 87, HuggingFace 71).
- Beats GPT-4 with retriever (~63%) on HuggingFace and TensorHub.
- Hallucination rate cut to 11% from 40% baseline.
- Zero-shot generalization demo: model still produces correct calls for APIs added after training cutoff, given retrieved docs.

## Risks + gotchas
- **Retriever failure = model failure:** if retriever returns wrong API, model cheerfully hallucinates a call for the retrieved-but-wrong API.
- **Narrow domain:** ML-library APIs only; does not cover general REST APIs or long-tail enterprise tools.
- **Python-code format** not aligned with modern OpenAI tool-call JSON convention; newer Gorilla releases migrate.
- **Benchmark overlap with training** — APIBench generated from the same hubs used for training, risk of memorization.

## Connections
- Parallel 2023 tool-use work: [[toolformer]], [[toolllm]] (ToolBench 16K real APIs, DFS-DT).
- Retriever lineage: every modern FC pipeline uses retrieval at inference when API pool is large.
- Evaluation superseded by [[bfcl]] (same Berkeley team).
- Related: [[api-bank]] (evaluation-first contemporaneous benchmark).
