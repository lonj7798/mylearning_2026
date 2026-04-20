<!-- scope: Jamba technical report
     deps: [[ch-01]], [[ch-02]]
     see-also: [[gemma-3]], [[mixtral]]
-->

# Jamba: A Hybrid Transformer-Mamba Language Model — Technical Report
- **Core Insight:** SSM + attention hybrid (1:7 ratio) gets linear-time efficiency for most layers while preserving precise retrieval where needed.
- **Guideline:** Match layer type to its role -- SSM for long-range flow, attention for precise recall.

- **Organization:** AI21 Labs
- **Year:** 2024
- **URL:** https://arxiv.org/abs/2403.19887
- **Relevant chapters:** Hybrid architectures, state space models (Mamba), Transformer-SSM integration, MoE, memory-efficient inference, long context

## Abstract
We present Jamba, a novel base large language model based on a hybrid Transformer-Mamba mixture-of-experts (MoE) architecture. Specifically, Jamba interleaves blocks of Transformer and Mamba layers, enjoying the benefits of both model families. MoE is added in some of these layers to increase model capacity while keeping active parameter usage manageable. This flexible architecture allows resource- and objective-specific configurations. The resulting model fits in a single 80GB GPU and achieves state-of-the-art performance on standard language model benchmarks and long-context evaluations, supporting up to 256K tokens context length with high throughput and small memory footprint compared to vanilla Transformers.

## Architecture Summary

| Component | Value |
|-----------|-------|
| Total Parameters | 52B |
| Active Parameters | 12B |
| Architecture | Hybrid Transformer-Mamba + MoE |
| Total Jamba Blocks | 4 |
| Layers per Block | 8 (32 total) |
| Attention-to-Mamba Ratio | 1:7 (4 attention layers, 28 Mamba layers) |
| Total Attention Layers | 4 |
| Total Mamba Layers | 28 |
| MoE Experts | 16 per MoE layer |
| Active Experts per Token | 2 |
| MoE Frequency | Every 2 layers (alternating MLPs) |
| Context Length | 256K tokens |
| Vocabulary Size | 64K tokens (BPE) |

- **Attention:** Grouped-Query Attention (GQA)
- **Activation function:** SwiGLU
- **Normalization:** RMSNorm (applied in Mamba layers for stability at scale)
- **Positional encoding:** None explicit — Mamba's recurrent structure provides implicit positional information; ablations show RoPE adds no benefit

## Key Architectural Innovations

1. **Hybrid Transformer-Mamba architecture** — first large-scale model to interleave Transformer attention layers with Mamba state space model (SSM) layers within the same model. The 1:7 attention-to-Mamba ratio means only 4 out of 32 layers use attention, dramatically reducing KV cache while maintaining quality.
2. **Extreme KV cache reduction** — only 4 attention layers (out of 32) maintain a KV cache. At 256K context with 16-bit precision: Jamba uses 4GB vs. 128GB for Llama-2-7B and 32GB for Mixtral. This is a 32x reduction vs. standard Transformers.
3. **MoE on alternating layers** — applies MoE every 2nd layer (replacing the MLP), with 16 experts and top-2 routing. This increases total capacity (52B) while keeping active parameters low (12B).
4. **Mamba layers for sequential processing** — Mamba (Structured State Space Model) processes sequences in linear time O(n) rather than quadratic O(n^2) for attention, enabling efficient processing of extremely long contexts.
5. **RMSNorm stabilization for Mamba** — discovered that applying RMSNorm within Mamba layers prevents training loss spikes that occur when scaling Mamba to large model sizes. This is a key finding for hybrid architectures.
6. **Emergent in-context learning** — even with only 4 attention layers (1:7 ratio), the hybrid model exhibits successful in-context learning via emergent induction heads. Pure Mamba models struggle with format adherence; the hybrid approach resolves this.

## Design Decisions and Tradeoffs

- **1:7 attention-to-Mamba ratio:** Aggressively minimizes attention layers. Ablations show that pure Mamba models are strong but fail at in-context learning tasks; even a small fraction of attention layers fixes this. The 1:7 ratio is the sweet spot — more attention would increase KV cache without proportional quality gains.
- **No explicit positional encoding:** Mamba's recurrent structure provides implicit position information. Ablations confirm that adding RoPE to the attention layers neither helps nor hurts, so it was omitted for simplicity.
- **MoE on every other layer (not every layer):** Applying MoE to every layer would increase total parameters but also increase routing overhead. Alternating dense and MoE layers provides a balance.
- **Single GPU deployment:** The architecture was explicitly designed to fit in a single 80GB GPU (unlike larger MoE models that require multi-GPU setups), making it practical for broader deployment.
- **Mamba's limitation on recall:** Pure Mamba can struggle with precise information retrieval from long contexts. The attention layers compensate by providing exact-match recall capabilities, creating a complementary partnership.

## Training Details

- **Hardware:** NVIDIA H100 GPUs
- **Framework:** In-house with FSDP, tensor parallelism, sequence parallelism, expert parallelism
- **Dataset:** Proprietary text data (web, books, code) with March 2024 cutoff; quality filters and deduplication applied
- **Scale:** Ablation models tested up to 250B tokens; production model trained beyond this
- **License:** Apache 2.0

## Performance Highlights

**Standard benchmarks (vs. Mixtral 8x7B):**

| Benchmark | Jamba (12B active) | Mixtral (12.9B active) |
|-----------|--------------------|----------------------|
| HellaSwag (10-shot) | 87.1% | 86.7% |
| WinoGrande (5-shot) | 82.5% | 81.2% |
| MMLU (5-shot) | 67.4% | 70.6% |
| GSM8K (3-shot CoT) | 59.9% | 60.4% |
| HumanEval (pass@1) | 29.3% | 34.8% |

**Throughput (single A100 80GB, 8K context):**
- Jamba processes batch size 16+ with 3x throughput vs. Mixtral

**Long context (4x A100 GPUs, 128K tokens):**
- 3x throughput vs. Mixtral
- 2x context support vs. Mixtral, 7x vs. Llama-2-70B on single GPU

**KV cache comparison (256K context, 16-bit):**
| Model | KV Cache |
|-------|----------|
| Llama-2 7B | 128 GB |
| Mistral 7B | 32 GB |
| Mixtral 12.9B | 32 GB |
| **Jamba 12B** | **4 GB** |

- Needle-in-a-haystack: excellent performance up to 256K tokens
- Key finding: hybrid models outperform both pure Transformer and pure Mamba architectures at equivalent scale
