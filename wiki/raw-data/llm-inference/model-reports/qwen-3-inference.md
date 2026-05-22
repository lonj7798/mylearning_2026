<!-- scope: inference-relevant facts from Qwen3 model reports and model cards
     see-also: vllm-project, sglang-project
-->

# Qwen3 Inference
- **Core Insight:** Qwen3 combines dense and MoE open models with long-context support and explicit thinking/non-thinking inference modes.
- **Guideline:** For Qwen3 serving, expose reasoning-mode controls and verify the exact model card for context length, active experts, and chat template.
- **Authors:** Qwen Team, Alibaba
- **Year:** 2025
- **URL:** https://qwenlm.github.io/blog/qwen3/
- **Relevant topics:** dense transformer, MoE, GQA, long context, reasoning mode, chat template

## Abstract
Qwen3 is a family of open-weight language models including dense and mixture-of-experts variants. The reports emphasize strong multilingual and reasoning performance, long-context capability, and a user-controllable thinking mode. Inference behavior depends on model size, dense versus MoE architecture, and chat-template settings.

## Key Contributions
- Provides both dense and sparse MoE serving targets in one model family.
- Uses attention variants such as GQA in released configurations to reduce KV cost.
- Supports long-context inference, increasing prefill and KV-cache pressure.
- Discloses recommended chat templates and thinking controls that affect latency and output length.

## Key Figures/Tables to Study
- Qwen3 blog/model tables: dense vs MoE sizes, active parameters, and context length.
- Hugging Face model cards: exact runtime requirements, chat templates, and generation examples.

## Technical Details
MoE Qwen3 variants reduce active parameters per token but still require memory for all expert weights unless sharded or offloaded. Thinking mode can intentionally generate longer hidden reasoning traces or visible reasoning content depending on template behavior, so benchmark output lengths must be controlled. Long-context runs should measure prefill throughput separately from decode TPOT.

## Connections
- [[vllm-project]] and [[sglang-project]] both serve Qwen-family models.
- [[ttft-tpot-itl]] is needed because reasoning mode changes latency profile.
