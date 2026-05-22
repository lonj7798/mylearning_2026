<!-- scope: Lilian Weng's overview blog on large transformer model inference and quantization
     deps: [[llm-int8]], [[gptq]], [[awq]]
     see-also: [[hf-quantization-fundamentals]], [[dettmers-llm-int8-blog]]
-->

# Lilian Weng — Large Transformer Model Inference Optimization
- **Core Insight:** Weng's survey post is a structured walkthrough of all four major inference-cost reduction families — quantization, pruning, distillation, sparse attention — placing quantization alongside its sibling techniques rather than treating it in isolation.
- **Guideline:** Use the post as the canonical reading-list bibliography when onboarding to LLM inference optimization; the quantization section is a compact taxonomy of PTQ vs QAT, then 4-bit and 8-bit methods.
- **Authors:** Lilian Weng
- **Year:** 2023 (post dated 2023-01-10)
- **URL:** https://lilianweng.github.io/posts/2023-01-10-inference-optimization/
- **Relevant topics:** PTQ vs QAT, GPTQ, AWQ, SmoothQuant, distillation, sparsity, FlashAttention

## Summary
Lilian Weng's blog post "Large Transformer Model Inference Optimization" is one of the most-linked educational surveys on transformer inference efficiency. The post organizes optimization techniques into a clear taxonomy: (1) distillation, (2) quantization, (3) pruning, (4) sparsity / architectural changes. The quantization section walks through PTQ vs QAT, then surveys outlier-aware methods (LLM.int8, SmoothQuant, GPTQ, AWQ) with intuitive explanations and reference figures from the original papers. The post is widely used as the "first reading" for engineers approaching LLM efficiency for the first time; it complements rather than replaces the underlying papers, and its bibliography is excellent.

## Key Points
- Categorizes inference optimizations into 4 families; quantization is one of them.
- Quantization sub-taxonomy: PTQ (no retrain) vs QAT (retrain with fake-quant), then symmetric vs asymmetric, per-tensor vs per-channel.
- Covers LLM.int8, SmoothQuant, GPTQ, AWQ explicitly; mentions QLoRA and BitNet briefly.
- The companion attention-optimization section (FlashAttention, multi-query attention) is read alongside.
- Mathematical notation is clean and unified across methods.

## Technical Details

### Taxonomy structure
The post sequences the optimization families by decreasing invasiveness:
1. **Distillation** — train a smaller student model.
2. **Quantization** — reduce numerical precision of the existing model.
3. **Pruning** — remove redundant weights or attention heads.
4. **Sparsity / architectural** — sparse attention, mixture-of-experts.

### Quantization sub-taxonomy table (reproduced from the post)
| Axis | Choices | Notes |
|------|---------|-------|
| When | PTQ vs QAT | PTQ = post-train; QAT = re-train with fake-quant |
| Granularity | per-tensor / per-channel / per-group | finer = more accurate, more overhead |
| Sign | symmetric vs asymmetric | symmetric simpler; asymmetric better for ReLU-like dists |
| Range | absmax vs percentile | absmax for weights, percentile for activations |
| Dtype | INT8 / INT4 / FP8 | trade-off space |

### Methods walkthrough
- **LLM.int8()**: outlier carve-out into FP16, INT8 for the rest.
- **SmoothQuant**: migrate activation outliers into weights via per-channel scaling.
- **GPTQ**: Hessian-aware OBS-style column quantization.
- **AWQ**: protect salient channels via activation-magnitude scaling.

### Why the post is useful
- Unifies notation: a single set of symbols across methods.
- Each method explained with the original paper's headline figure.
- Compact bibliography — every section ends with the relevant primary sources.

## Connections
- [[llm-int8]] / [[smoothquant]] / [[gptq]] / [[awq]] — methods explained in the post.
- [[hf-quantization-fundamentals]] — adjacent educational series with more code.
- [[dettmers-llm-int8-blog]] — deeper dive on the outlier intuition Weng summarises.
- [[sebastian-raschka-quant]] — alternative practitioner blog with similar role.
