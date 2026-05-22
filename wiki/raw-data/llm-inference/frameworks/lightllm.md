<!-- scope: LightLLM serving framework, router, and token/KV cache architecture
     deps: [[continuous-batching]]
     see-also: [[vllm]], [[sglang]]
-->

# LightLLM
- **Core Insight:** LightLLM is a Python-first LLM serving framework built around an efficient router and token-level KV-cache management.
- **Guideline:** Study LightLLM for token-attention and router design ideas, but verify current model and hardware support before choosing it for production.
- **Authors:** ModelTC / LightLLM contributors
- **Year:** 2023-present
- **URL:** https://github.com/ModelTC/lightllm and https://lightllm-en.readthedocs.io/en/latest/
- **Relevant topics:** token attention, router, continuous batching, KV memory manager, OpenAI API, multi-level cache

## Abstract
LightLLM is an inference and serving framework for large language models. Its docs emphasize an efficient router that manages token usage precisely with Token Attention. The codebase includes OpenAI/TGI-compatible server APIs, core request objects, router components, model inference workers, and KV-cache memory managers.

## Key Contributions
- Token Attention design aims to manage token-level KV usage accurately and avoid out-of-memory states.
- Router layer schedules incoming requests and tracks running token state.
- Provides multiple API surfaces, including OpenAI-compatible and TGI-like server modules.
- Includes specialized KV-cache memory managers for normal, quantized, CPU/offload, and model-specific layouts.
- Supports deployment/tuning docs for router and API server arguments.

## Key Figures/Tables to Study
- Efficient Router docs: token-state diagram and OOM prevention explanation.
- `lightllm/server/router`: routing, scheduling, and request lifecycle.
- `lightllm/common/kv_cache_mem_manager`: cache allocation and specialized managers.
- `lightllm/server/api_openai.py` and `api_server.py`: public serving entrypoints.

## Technical Details
Public APIs:
- Launch through LightLLM server commands documented in the repo/docs.
- API modules include OpenAI-compatible, LightLLM-native, and TGI-compatible paths.
- Server arguments control model path, ports, parallelism, max tokens, and cache-related behavior.

Scheduler/cache approach:
- Router admits and tracks requests with explicit token accounting.
- Token Attention treats tokens as cache-management units rather than coarse per-sequence allocations.
- KV memory managers allocate/free token cache slots and include variants for quantized or model-specific cache formats.
- CPU cache/offload modules support larger effective memory at extra transfer cost.

Relevant code/docs:
- GitHub: https://github.com/ModelTC/lightllm
- Docs: https://lightllm-en.readthedocs.io/en/latest/
- Efficient router: https://lightllm-en.readthedocs.io/en/latest/framework/router.html
- KV managers: https://github.com/ModelTC/lightllm/tree/main/lightllm/common/kv_cache_mem_manager
- Server source: https://github.com/ModelTC/lightllm/tree/main/lightllm/server

Strengths:
- Clear focus on serving memory safety and token-level cache accounting.
- Python codebase is relatively approachable for reading router/cache behavior.
- Offers useful comparison point to block/page and radix-tree cache designs.

Limitations:
- Smaller ecosystem and adoption than vLLM, SGLang, TensorRT-LLM, or TGI.
- Documentation is less comprehensive for some advanced internals.
- Production suitability depends on exact model/backend support and operational requirements.

## Connections
- Token Attention contrasts with [[vllm-kv-cache-manager]] and [[sglang-radixattention]].
- Router behavior connects to [[continuous-batching]] and [[sglang-scheduler]].
- API compatibility places it near [[hf-tgi]] and [[llama-cpp-server]].
