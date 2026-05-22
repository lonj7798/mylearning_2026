<!-- scope: inference-relevant facts from OpenAI gpt-oss model card
     see-also: qwen-3-inference, vllm-project
-->

# GPT-OSS Inference
- **Core Insight:** gpt-oss exposes open-weight MoE reasoning models designed to fit practical GPU memory targets through sparse activation and MXFP4 MoE weights.
- **Guideline:** Serve gpt-oss with the Harmony format and explicit reasoning effort; memory planning must account for MXFP4 weights, 128k context, and MoE routing.
- **Authors:** OpenAI
- **Year:** 2025
- **URL:** https://openai.com/index/introducing-gpt-oss/
- **Relevant topics:** MoE, MXFP4, grouped multi-query attention, 128k context, reasoning effort

## Abstract
OpenAI's gpt-oss-120b and gpt-oss-20b are open-weight text-only reasoning models. They are transformer MoE models with sparse active parameters per token, alternating dense and locally banded sparse attention, grouped multi-query attention, RoPE, and native 128k context support. The models are distributed with MXFP4 quantization for MoE weights and require the Harmony response format.

## Key Contributions
- Releases 120B and 20B open-weight reasoning models under Apache 2.0.
- Uses MoE so active parameters per token are far smaller than total parameters.
- Uses grouped multi-query attention to reduce KV-cache cost.
- Supports configurable reasoning effort that trades latency for quality.
- Targets practical deployment: 120B on a single 80GB-class GPU and 20B within smaller memory budgets.

## Key Figures/Tables to Study
- OpenAI announcement architecture table: layers, total parameters, active parameters, experts, and context length.
- gpt-oss model card: Harmony format, quantization, and safety/deployment notes.
- Hugging Face model cards: runtime examples for Transformers, vLLM, Ollama, and PyTorch/Triton.

## Technical Details
gpt-oss-120b has 117B total parameters, about 5.1B active parameters per token, 128 experts, 4 active experts per token, and 128k context. gpt-oss-20b has 21B total parameters, about 3.6B active parameters per token, 32 experts, 4 active experts per token, and 128k context. Reasoning effort changes output length and latency, so it is a serving parameter, not just a prompt preference.

## Connections
- [[vllm-project]] is one disclosed serving path.
- [[ttft-tpot-itl]] and [[goodput-slo]] are necessary for reasoning-effort capacity planning.
