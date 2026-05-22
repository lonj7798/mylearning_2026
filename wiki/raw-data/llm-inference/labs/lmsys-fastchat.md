<!-- scope: LMSYS FastChat project summary for chat serving and evaluation
     see-also: lmsys-serving, llmperf
-->

# LMSYS FastChat
- **Core Insight:** FastChat provided an early open serving and evaluation stack for chat-tuned LLMs, including API servers and multi-model demos.
- **Guideline:** Study FastChat for historical serving patterns and evaluation workflows; use newer engines for peak inference performance.
- **Authors:** LMSYS
- **Year:** 2023-2026
- **URL:** https://github.com/lm-sys/FastChat
- **Relevant topics:** chat serving, controller-worker architecture, OpenAI-compatible API, evaluation, arena

## Abstract
FastChat is an open platform for training, serving, and evaluating chatbots. It includes controller/worker serving components, a web UI, OpenAI-compatible API support, and evaluation utilities. It was central to the Vicuna release and early LMSYS Chatbot Arena infrastructure.

## Key Contributions
- Provides a simple multi-model chat serving architecture.
- Helped standardize open chat model demos and evaluation.
- Includes OpenAI-compatible API serving for local models.
- Connects serving infrastructure to human preference evaluation workflows.

## Key Figures/Tables to Study
- FastChat serving docs: controller, model worker, API server, and web UI.
- Vicuna/LMSYS blog posts: deployment and evaluation context.

## Technical Details
FastChat's architecture separates request routing from model workers. It supports multiple backends, but its core contribution is orchestration and evaluation rather than cutting-edge KV-cache or scheduler optimization. For course purposes, it is a useful contrast with vLLM and SGLang: simple serving control plane versus specialized inference engine.

## Connections
- [[lmsys-serving]] provides the related practitioner post.
- [[vllm-project]] represents the later high-performance engine path.
