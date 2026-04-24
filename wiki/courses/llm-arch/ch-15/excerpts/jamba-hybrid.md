# Excerpt: Jamba Hybrid Architecture

Source: [[jamba|report]] — AI21 Labs, "Jamba: A Hybrid Transformer-Mamba Language Model" (2024)

---

## The Hybrid Thesis

Jamba is the first large-scale production model to validate the hypothesis that SSMs and attention are complementary, not competing. The architecture is not a 50/50 split — it is aggressively SSM-dominant, using attention only where it is strictly necessary.

> We present Jamba, a novel base large language model based on a hybrid Transformer-Mamba mixture-of-experts (MoE) architecture. Specifically, Jamba interleaves blocks of Transformer and Mamba layers, enjoying the benefits of both model families.

Jamba's architecture interleaves Mamba and Transformer attention layers at a 1:7 ratio: 4 attention layers and 28 Mamba layers across 32 total layers. MoE is applied on alternating layers (16 experts, top-2 routing). The model is organized into 4 blocks of 8 layers each, with exactly one attention layer per block (the first layer of each block).

| Component | Value |
|-----------|-------|
| Total parameters | 52B |
| Active parameters | 12B |
| Attention layers | 4 (of 32) |
| Mamba layers | 28 (of 32) |
| Context length | 256K tokens |
| MoE | 16 experts, top-2, every other layer |

## Why Not Pure Mamba?

The paper's ablations reveal a specific failure mode of pure Mamba models:

> Pure Mamba models are strong but fail at in-context learning tasks; even a small fraction of attention layers fixes this. The 1:7 ratio is the sweet spot — more attention would increase KV cache without proportional quality gains.

The problem is precise: Mamba's fixed-size state cannot guarantee exact retrieval of arbitrary tokens from the context. In-context learning requires the model to identify and reproduce specific patterns from the prompt — format adherence, few-shot examples, structured outputs. The attention layers provide explicit token-to-token matching that the SSM layers cannot.

With only 4 attention layers, Jamba fully recovers in-context learning capability. The attention layers function as "retrieval checkpoints" — the Mamba layers handle efficient sequential processing between them, and the attention layers ensure precise information can be recalled when needed.

## The KV Cache Advantage

Because only 4 layers maintain a KV cache:

> At 256K context with 16-bit precision: Jamba uses 4GB vs. 128GB for Llama-2-7B and 32GB for Mixtral. This is a 32x reduction vs. standard Transformers.

| Model | KV Cache (256K context) |
|-------|------------------------|
| Llama-2 7B | 128 GB |
| Mistral 7B | 32 GB |
| Mixtral 12.9B | 32 GB |
| **Jamba 12B active** | **4 GB** |

The 28 Mamba layers contribute zero to the KV cache — their state is a fixed-size vector (~256 KB total) regardless of sequence length. This is what makes 256K context feasible on a single 80GB GPU.

## Training Stability: The RMSNorm Discovery

> Discovered that applying RMSNorm within Mamba layers prevents training loss spikes that occur when scaling Mamba to large model sizes. This is a key finding for hybrid architectures.

The original Mamba paper trained at 3B scale without stability issues. Jamba found that scaling beyond this causes training loss spikes in the Mamba layers. The fix — adding RMSNorm within the Mamba block — parallels the Pre-LN vs Post-LN evolution in Transformers. Any deep recurrence that compounds activations across many steps needs explicit normalization to prevent magnitude drift.

## No Positional Encoding

> Mamba's recurrent structure provides implicit positional information; ablations show RoPE adds no benefit.

Jamba uses no RoPE, no sinusoidal encoding, no ALiBi. The Mamba layers' recurrent state evolution naturally encodes positional information — the state at position $t$ is the result of $t$ sequential applications of the transition dynamics, inherently different from the state at position $t'$.

This is a practical simplification: no need to worry about RoPE frequency bands, context extension, or NTK-aware interpolation for the Mamba layers. The 4 attention layers could use RoPE, but ablations showed no benefit.

## Throughput

> Jamba processes batch size 16+ with 3x throughput vs. Mixtral. At 128K context: 3x throughput vs. Mixtral.

The throughput advantage comes from two sources:
1. **Reduced KV cache** — less memory bandwidth consumed loading cached values during autoregressive decoding
2. **Linear-time Mamba layers** — 28 of 32 layers scale linearly with sequence length instead of quadratically

## Benchmark Results: Competitive Quality

The quality results demonstrate that the hybrid architecture does not sacrifice capability for efficiency:

| Benchmark | Jamba (12B active) | Mixtral (12.9B active) |
|-----------|--------------------|----------------------|
| HellaSwag (10-shot) | 87.1% | 86.7% |
| WinoGrande (5-shot) | 82.5% | 81.2% |
| MMLU (5-shot) | 67.4% | 70.6% |
| GSM8K (3-shot CoT) | 59.9% | 60.4% |

Jamba matches or slightly trails Mixtral on most benchmarks while using fewer active parameters (12B vs 12.9B), running at 3x throughput, and using 8x less KV cache. The slight MMLU deficit may reflect the reduced number of attention layers — MMLU heavily tests knowledge retrieval, which is attention's strength.

Critically, Jamba achieves excellent needle-in-a-haystack performance up to 256K tokens. The 4 attention layers are sufficient for precise retrieval even at extreme context lengths, because the Mamba layers between them effectively propagate relevant information through the state to the next attention checkpoint.

---

## Design Principle

The Jamba architecture embodies a clear design principle: **match layer type to its computational role**. SSM layers handle the bulk of sequential processing where linear-time scaling matters. Attention layers handle the minority of computation where precise token retrieval is needed. This is not a compromise — it is a principled allocation of computational resources based on what each mechanism does best.
