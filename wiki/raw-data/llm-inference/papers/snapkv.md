<!-- scope: SnapKV paper on prompt-observation-based KV-cache compression
     deps: [[h2o]]
     see-also: [[attention-sinks]], [[quest-kv]], [[infllm]]
-->

# SnapKV: LLM Knows What You Are Looking for Before Generation
- **Core Insight:** Attention patterns observed near the end of prefill can predict which prompt KV positions each head will need during generation.
- **Guideline:** For long prompts, compress KV after prefill using head-specific important positions rather than carrying the full prompt cache into decode.
- **Authors:** Yuhong Li, Yingbing Huang, Bowen Yang, Bharat Venkitesh, Acyr Locatelli, Hanchen Ye, Tianle Cai, Patrick Lewis, Deming Chen
- **Year:** 2024
- **URL:** https://arxiv.org/abs/2404.14469
- **Relevant topics:** KV-cache compression, long context, attention observation, head-specific selection, training-free inference

## Abstract
SnapKV addresses the growing KV-cache cost for long input contexts. It observes that during generation, each attention head tends to focus on specific prompt features, and that this pattern can be inferred from an observation window during prefill. SnapKV clusters/selects important KV positions per head and discards less useful prompt KV entries, reducing memory and improving decode speed without fine-tuning.

## Key Contributions
- Introduces a training-free KV compression method for long-context prompts.
- Uses an observation window at the end of the prompt to estimate future attention needs.
- Selects clustered important KV positions per attention head.
- Maintains comparable quality on long-context tasks while reducing memory.
- Reports faster generation and higher memory efficiency under long inputs.

## Key Figures/Tables to Study
- Attention-feature stability evidence across generation.
- SnapKV pipeline: observation, selection/clustering, compressed cache.
- LongBench or long-context task results.
- Memory and generation-speed comparisons against baselines.

## Technical Details
After prefill, SnapKV examines attention from a short observation window to earlier prompt tokens. It uses these attention patterns to select a compact set of prompt KV positions, often separately per head. Decode then attends to the compressed prompt cache plus generated-token cache.

The method targets prompt-side KV cache after prefill, so it directly reduces memory and bandwidth during decode. It is complementary to paged allocation: selected KV entries still need layout and scheduling support in the serving engine.

## Connections
- [[h2o]] is also attention-based eviction but focuses on heavy hitters during generation.
- [[quest-kv]] selects cache pages based on the current query.
- [[attention-sinks]] explains why some fixed early positions may need preservation.

## Notes
SnapKV is prompt-compression oriented: it reduces the KV carried from prefill into decode.
