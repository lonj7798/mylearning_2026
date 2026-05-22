<!-- scope: inference-relevant facts from DeepSeek-V3 technical report
     see-also: deepseek-r1-inference, sglang-project
-->

# DeepSeek-V3 Inference
- **Core Insight:** DeepSeek-V3 pairs large-scale MoE with Multi-head Latent Attention to reduce KV-cache cost at frontier scale.
- **Guideline:** Model DeepSeek-V3 serving as sparse expert routing plus MLA KV compression; dense-model KV formulas overestimate the wrong components.
- **Authors:** DeepSeek-AI
- **Year:** 2024
- **URL:** https://arxiv.org/abs/2412.19437
- **Relevant topics:** MoE, MLA, FP8, expert parallelism, long context

## Abstract
DeepSeek-V3 is a large mixture-of-experts decoder model using Multi-head Latent Attention (MLA) and DeepSeekMoE. The report emphasizes efficient training and inference through sparse activation, FP8 training/inference support, load balancing, and long-context capability. MLA compresses key/value representation compared with standard attention KV caches.

## Key Contributions
- Uses MoE to activate only a subset of parameters per token.
- Uses MLA to reduce KV-cache footprint and memory bandwidth pressure.
- Discusses FP8-oriented efficiency in the model/system stack.
- Provides a major open reference for expert-parallel serving design.

## Key Figures/Tables to Study
- Architecture table: total parameters, activated parameters, layers, experts, and context length.
- MLA description: latent KV representation and implications for cache storage.
- System sections: expert routing, load balance, and precision choices.

## Technical Details
DeepSeek-V3 serving requires routing tokens to selected experts while keeping attention state efficient. Expert parallelism and all-to-all communication can dominate distributed inference, especially at high batch sizes. MLA changes KV-cache layout: instead of caching full per-head K/V tensors as in MHA/GQA, the runtime can cache compressed latent states plus position-related components.

## Connections
- [[deepseek-r1-inference]] builds reasoning behavior on the V3-style base.
- [[sglang-project]] and [[vllm-project]] are common open runtimes for DeepSeek-family serving.
