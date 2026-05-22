<!-- scope: HuggingFace FP8 training deep-dive blog
     deps: [[fp8-e4m3]], [[fp8-e5m2]]
     see-also: [[transformer-engine-blog]], [[deepseek-v3-fp8]]
-->

# HuggingFace — Scaling FP8 Training to 1000+ GPUs Deep-Dive
- **Core Insight:** HF's FP8 training deep-dive walks through the full per-tensor scaling regime — amax tracking, delayed-scaling histograms, recipe choices, and the empirical loss-curve impact — for production-scale FP8 pretraining runs using nanotron + Transformer Engine.
- **Guideline:** When debugging FP8 training divergence, look at amax history plots first; non-stationary or bimodal amax distributions are the leading indicator of an unstable run.
- **Authors:** Ferdinand Mom, Phuc Nguyen, Haojun Zhao, Leandro von Werra (HuggingFace research team)
- **Year:** 2024 (initial); updated 2025
- **URL:** https://huggingface.co/blog/nouamanetazi/scaling-fp8-training
- **Relevant topics:** FP8 pretraining, delayed scaling, amax dynamics, nanotron, loss curves

## Summary
HuggingFace's FP8 deep-dive is one of the few public write-ups of large-scale FP8 pretraining covering the full empirical loop — not just the recipe but what the per-tensor amax distributions actually look like during a real run. The post documents experiments at the 8B-parameter scale on H100 clusters using `nanotron` + Transformer Engine, including delayed-scaling tuning, comparison of `current scaling` vs `delayed scaling` recipes, attention FP8 behaviour, and the loss-curve match against BF16 baselines. Practical guidance includes amax-history depth, which layers to keep in BF16 (final layernorm, embedding, lm_head), how to handle tensor-parallel amax synchronization, and divergence diagnostics. The conclusion is that FP8 at 8B converges identically to BF16 with ~30-40% throughput gain.

## Key Points
- Recipe comparison: `delayed scaling` (default) vs `current scaling` (one-step).
- Amax history depth sweep: 16 vs 128 vs 1024 — larger reduces variance, slower to adapt.
- Layers kept in BF16: embedding, final LN, lm_head; everything else FP8.
- TP amax all-reduce mandatory; without it ranks desync and loss diverges by ~step 1000.
- FP8 attention requires careful KV scale tracking; cuDNN handles this since 9.0.
- Throughput gain at 8B: ~35% over BF16 on H100 SXM5.

## Technical Details

### Experimental setup
- Model: 8B-parameter Llama-style transformer.
- Hardware: 128 H100 SXM5.
- Framework: nanotron + Transformer Engine 1.x.
- Recipe: DelayedScaling, hybrid format (E4M3 fwd, E5M2 bwd).

### Recipe parameters tested
| Knob | Tested values | Default chosen |
|------|---------------|----------------|
| `amax_history_len` | 16, 128, 1024 | 1024 |
| `amax_compute_algo` | max, most_recent | max |
| `margin` | 0, 1, 2 | 0 |
| `fp8_format` | E4M3, E5M2, HYBRID | HYBRID |

### Layers exempt from FP8
- Token embedding (small, accuracy-critical).
- Final RMSNorm/LayerNorm.
- LM head (output projection back to vocab).
- Optional: first N transformer layers (stability at warmup).

### Amax dynamics
- Healthy run: amax distribution narrows quickly in the first ~500 steps, then stabilizes.
- Unhealthy run: bimodal amax — some tensors stuck at scale-saturation, others at scale-underflow.
- Fix: increase `margin` to 1 if saturating; decrease `amax_history_len` if too slow to adapt.

### Tensor-parallel coordination
- Each TP rank computes a local amax over its weight slice.
- Without all-reduce, ranks diverge: rank A scales by 100, rank B by 50 → matmul produces inconsistent results.
- TE's `reduce_amax=True` (default in Megatron / nanotron integrations) all-reduces inside the TP group.

### Loss curve results
- BF16 baseline loss at step 50k: 2.15.
- FP8 HYBRID delayed scaling: 2.16 (within noise).
- FP8 with `amax_history_len=16`: 2.18 (slightly worse — too noisy scale).
- FP8 with TP amax sync disabled: diverges at step ~1000.

### Throughput
| Recipe | Tokens/sec/GPU | Speedup |
|--------|----------------|---------|
| BF16 | 3000 | 1.0× |
| FP8 HYBRID | 4050 | 1.35× |
| FP8 + FP8 KV attention | 4250 | 1.42× |

### Divergence checklist (from the post)
1. Check amax history plots per layer.
2. Verify TP amax all-reduce is on.
3. Confirm embedding + final LN are in BF16.
4. Inspect gradient norm — if NaN, drop to BF16 for that layer.

## Connections
- [[transformer-engine-blog]] — recipe definitions and APIs.
- [[fp8-e4m3]] / [[fp8-e5m2]] — operand formats.
- [[deepseek-v3-fp8]] — frontier-scale FP8 training comparison.
- [[megatron-fp8]] — adjacent framework with same FP8 stack.
