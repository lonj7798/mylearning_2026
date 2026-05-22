<!-- scope: LMSYS serving posts and FastChat/Vicuna deployment context
     see-also: lmsys-fastchat, vllm-project
-->

# LMSYS Serving
- **Core Insight:** LMSYS early chat-serving work tied open model release, serving infrastructure, and human preference evaluation into one feedback loop.
- **Guideline:** Use LMSYS serving material for historical architecture and evaluation context; use newer runtimes for current high-throughput engine design.
- **Authors:** LMSYS
- **Year:** 2023-2026
- **URL:** https://lmsys.org/blog/2023-03-30-vicuna/
- **Relevant topics:** FastChat, Vicuna, chatbot arena, controller-worker serving, evaluation

## Abstract
LMSYS blog posts around Vicuna, FastChat, and Chatbot Arena describe how open chat models were served, compared, and evaluated at scale. The serving stack emphasized accessibility, model workers, web demos, and human evaluation rather than the later specialized KV-cache schedulers.

## Key Contributions
- Shows an early open chat-serving architecture.
- Connects serving infrastructure to evaluation and data collection.
- Popularized public multi-model chat comparison workflows.
- Provides historical context for FastChat and Chatbot Arena.

## Key Figures/Tables to Study
- Vicuna blog: FastChat release and demo/evaluation framing.
- FastChat docs: controller, worker, API server, and web UI.
- Chatbot Arena posts: serving as part of evaluation infrastructure.

## Technical Details
The LMSYS stack is useful for understanding routing and evaluation loops. It is less focused on low-level inference optimizations than vLLM, SGLang, or TensorRT-LLM. In course material, it should appear as a control-plane and evaluation case study rather than a kernel/runtime benchmark source.

## Connections
- [[lmsys-fastchat]] summarizes the project.
- [[vllm-project]] and [[sglang-project]] represent later optimized serving engines.
