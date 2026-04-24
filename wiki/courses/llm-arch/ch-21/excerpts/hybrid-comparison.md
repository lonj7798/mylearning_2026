# Excerpt: Hybrid vs. Pure Architecture Comparison

<!-- source: [[jamba|report]], [[mamba|paper]], [[mamba-2|paper]] -->

## The Three-Way Comparison

The most important result in the Jamba report is not any single benchmark number — it is the demonstration that a hybrid architecture can match pure-Transformer quality while fundamentally changing the memory scaling curve. This excerpt compiles the comparison data.

## Benchmark Quality

### Standard LM Benchmarks (vs. Mixtral 8x7B, similar active params)

| Benchmark | Jamba (12B active) | Mixtral (12.9B active) | Delta |
|-----------|--------------------|----------------------|-------|
| HellaSwag (10-shot) | 87.1% | 86.7% | +0.4 |
| WinoGrande (5-shot) | 82.5% | 81.2% | +1.3 |
| ARC-Easy | 73.5% | 74.2% | -0.7 |
| MMLU (5-shot) | 67.4% | 70.6% | **-3.2** |
| GSM8K (3-shot CoT) | 59.9% | 60.4% | -0.5 |
| HumanEval (pass@1) | 29.3% | 34.8% | **-5.5** |

**Pattern:** Jamba wins on commonsense/world-knowledge tasks (HellaSwag, WinoGrande) where long-range contextual understanding matters. It loses on knowledge-intensive (MMLU) and code generation (HumanEval) tasks. The MMLU gap (-3.2) and HumanEval gap (-5.5) are the most notable deficits.

**Hypothesis for the gaps:** MMLU tests factual knowledge recall from training data — this relies heavily on the FFN layers acting as key-value memories ([[ch-08]]). With MoE routing, each token only accesses 2 of 16 experts, which may reduce effective knowledge capacity compared to Mixtral's architecture at the same active parameter count. HumanEval requires precise syntactic generation where exact token-level patterns matter — potentially a weakness of SSM layers that compress sequential context.

### Pure Mamba vs. Jamba (Ablation Data)

The ablation models provide controlled comparisons at equal total compute:

| Metric | Pure Mamba (32L) | Jamba (4 attn + 28 SSM) | Pure Transformer (32L) |
|--------|-----------------|------------------------|----------------------|
| Perplexity (held-out) | Competitive | Competitive | Baseline |
| Few-shot format adherence | Poor | Strong | Strong |
| Needle-in-haystack (128K) | Degraded | Excellent | Excellent |
| Throughput (single GPU) | Highest | High | Lowest |
| KV cache (256K) | 0 GB | 4 GB | 128 GB |

The pattern is clear: pure Mamba wins on efficiency but fails on precise retrieval. Pure Transformer wins on quality but fails on memory. Jamba captures the quality of the Transformer with the efficiency profile closer to Mamba.

## Efficiency Metrics

### Throughput

Single A100 80GB, batch size 16, 8K context:

| Model | Throughput (relative) |
|-------|-----------------------|
| Mixtral 8x7B | 1.0x |
| Llama-2 7B | ~1.0x |
| **Jamba** | **3.0x** |

The 3x throughput advantage comes from two sources:
1. **Reduced KV cache memory** → larger batch sizes fit in GPU memory → higher throughput
2. **SSM layers have no KV cache loading** → less memory bandwidth consumed per decoding step → 28 of 32 layers are faster during autoregressive generation

### Memory at Scale

At 256K context on a single 80GB GPU:

| Model | KV Cache | Remaining for weights+activations | Feasible? |
|-------|----------|-----------------------------------|-----------|
| Llama-2 7B | 128 GB | -48 GB | No |
| Mistral 7B | 32 GB | 48 GB | Marginal |
| Mixtral 8x7B | 32 GB | 48 GB | No (weights too large) |
| **Jamba** | **4 GB** | **76 GB** | **Yes** |

Jamba's 4 GB KV cache at 256K context leaves 76 GB for everything else — comfortably accommodating the 12B active parameters in FP16 (~24 GB) with room for activations and batching.

## The Mamba-2 Comparison

Mamba-2 ([[mamba-2|paper]]) was published after Jamba (mid-2024 vs. early 2024). It introduces State Space Duality (SSD) — a theoretical framework showing SSMs and linear attention are mathematically dual — and achieves 2-8x speedup over Mamba-1 while maintaining quality.

**How Mamba-2 changes the comparison:**

| Dimension | Mamba-1 (used in Jamba) | Mamba-2 |
|-----------|------------------------|---------|
| Core algorithm | Selective scan | SSD (dual quadratic/recurrent) |
| Speed | Baseline | 2-8x faster |
| State transition | Diagonal A | Scalar-identity A (simpler) |
| Multi-head | No | Yes (like multi-head attention) |
| Quality | Competitive | Competitive (slightly better) |

Mamba-2's improvements are complementary to Jamba's hybrid approach. A hypothetical "Jamba-2" could replace the Mamba-1 layers with Mamba-2 layers, gaining 2-8x speedup on the SSM layers without changing the attention layers or MoE structure. The hybrid architecture is agnostic to which SSM variant is used — it composes with SSM improvements.

However, Mamba-2 still inherits the fundamental state-bottleneck limitation: even with the SSD framework and multi-head SSM, the fixed-size state cannot perform arbitrary-position retrieval as well as attention. The duality with linear attention means Mamba-2 shares linear attention's weaknesses (which [[ch-15]] covers). The need for attention layers in a hybrid remains.

## What Each Architecture Is Best For

| Use Case | Best Architecture | Why |
|----------|-------------------|-----|
| Short context (< 8K), maximum quality | Pure Transformer | No memory constraint; mature tooling |
| Long context (128K+), single GPU | Hybrid (Jamba) | KV cache reduction is necessary |
| Maximum throughput, quality-tolerant | Pure SSM (Mamba-2) | Zero KV cache, fastest decoding |
| Large-scale reasoning (100B+) | Transformer + MLA (DeepSeek) | MLA gives cache reduction within proven paradigm |
| Resource-constrained deployment | Hybrid (Jamba) | Single GPU, high throughput, long context |

The hybrid is not universally best — it is best when the deployment constraint requires long context on limited hardware and quality must remain competitive with pure Transformers.
