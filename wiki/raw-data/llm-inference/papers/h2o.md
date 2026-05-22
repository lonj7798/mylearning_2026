<!-- scope: Heavy-Hitter Oracle paper for KV-cache eviction
     deps: transformer-attention
     see-also: [[attention-sinks]], [[snapkv]], [[quest-kv]]
-->

# H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models
- **Core Insight:** A small subset of heavy-hitter tokens accounts for most attention value, so KV cache can be pruned by retaining recent tokens plus historically important tokens.
- **Guideline:** Use attention-derived eviction when KV memory is the bottleneck and approximate cache retention is acceptable.
- **Authors:** Zhenyu Zhang, Ying Sheng, Tianyi Zhou, Tianlong Chen, Lianmin Zheng, Ruisi Cai, Zhao Song, Yuandong Tian, Christopher Re, Clark Barrett, Zhangyang Wang, Beidi Chen
- **Year:** 2023
- **URL:** https://arxiv.org/abs/2306.14048
- **Relevant topics:** KV-cache eviction, heavy hitters, attention sparsity, memory reduction, long generation

## Abstract
H2O observes that during generation, not all prior tokens contribute equally to attention. A small heavy-hitter set receives most attention mass and should be retained, while many other KV entries can be evicted. The method keeps heavy hitters and recent tokens, reducing KV-cache footprint and improving throughput across inference systems.

## Key Contributions
- Identifies heavy-hitter behavior in attention during LLM generation.
- Proposes a KV-cache eviction policy retaining heavy hitters and recent tokens.
- Provides theoretical framing for submodular heavy-hitter selection.
- Implements H2O on multiple inference systems.
- Reports large throughput improvements under reduced cache budgets.

## Key Figures/Tables to Study
- Attention mass concentration plots.
- Heavy-hitter plus recent-token cache illustration.
- Accuracy/throughput tradeoff under different cache ratios.
- System comparisons on OPT model sizes.

## Technical Details
The full KV cache grows linearly with generated sequence length and batch size. H2O maintains a bounded cache by scoring tokens according to accumulated attention contribution. Tokens with high contribution remain in cache; a recent window is also preserved to maintain local coherence.

This is an approximate inference method. It changes the attention context seen by the model, so evaluation must include downstream quality as well as throughput. It is most relevant when serving long generations or many concurrent requests under tight KV-memory budgets.

## Connections
- [[attention-sinks]] keeps positional sink tokens; H2O keeps attention heavy hitters.
- [[snapkv]] computes compression decisions from prompt-observation behavior.
- [[quest-kv]] makes KV selection query-aware at inference time.

## Notes
Quality evaluation is essential because H2O changes the exact attention context, unlike pure allocation methods.
