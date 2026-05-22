<!-- scope: inference-relevant facts from Mixtral model cards and report
     see-also: vllm-project, sglang-project
-->

# Mixtral Inference
- **Core Insight:** Mixtral made sparse MoE inference mainstream by activating only a small number of experts per token while retaining many total parameters.
- **Guideline:** Benchmark Mixtral with expert-parallel communication in mind; active FLOPs are low, but all expert weights and routing overhead still matter.
- **Authors:** Mistral AI
- **Year:** 2023
- **URL:** https://mistral.ai/news/mixtral-of-experts/
- **Relevant topics:** sparse MoE, sliding window attention, GQA, expert routing, open weights

## Abstract
Mixtral 8x7B is a sparse mixture-of-experts decoder model that selects two experts per token from eight feed-forward experts. It uses the Mistral-family attention design and supports long-context inference relative to earlier open models. Mixtral 8x22B later extended the family with larger experts and stronger quality.

## Key Contributions
- Popularized open sparse-MoE serving at practical quality levels.
- Activates only a subset of FFN experts for each token.
- Requires runtimes to handle expert routing and memory placement efficiently.
- Demonstrates that parameter count and active compute must be reported separately.

## Key Figures/Tables to Study
- Mistral announcement/model card: expert count, active experts, context, and license.
- Runtime benchmark notes: tensor/expert parallel behavior for Mixtral serving.

## Technical Details
Mixtral's attention KV cache is not sparse in the same way as the FFN compute: every generated token still extends the attention cache. The MoE layers reduce active compute but can increase memory pressure because all experts must be resident or efficiently sharded. Small batches may underutilize experts; large batches can hit all-to-all or memory bandwidth limits.

## Connections
- [[deepseek-v3-inference]] is a later MoE+MLA design.
- [[vllm-project]] and [[sglang-project]] include MoE serving support.
