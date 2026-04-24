<!-- scope: Qwen3.5 technical report
     deps: [[ch-01]], [[ch-02]], [[qwen-3]]
     see-also: [[qwen-3]], [[qwen-3-6]], [[deepseek-v3]]
-->

# Qwen3.5 Technical Report
- **Core Insight:** Replacing most attention layers with Gated DeltaNet (linear attention) in a hybrid MoE architecture enables million-token contexts at a fraction of the KV-cache cost of full attention.
- **Guideline:** Hybrid linear-attention + sparse-MoE is the emerging recipe for scaling context length without proportionally scaling inference cost; the 3:1 DeltaNet-to-attention ratio is a concrete design point to study.

- **Organization:** Qwen Team, Alibaba Cloud
- **Year:** 2026 (February 16 flagship; March 1 small models)
- **URL:** https://arxiv.org/abs/2604.15804 (Qwen3.5-Omni report)
- **Relevant chapters:** Hybrid attention architectures, Gated DeltaNet / linear attention, fine-grained MoE scaling, multimodal early fusion, long-context inference

## Abstract
Qwen3.5 is the successor to Qwen3, introducing a fundamentally new hybrid architecture that replaces most standard attention layers with Gated Delta Networks (GDN) — a linear-attention variant — while retaining sparse Mixture-of-Experts routing. The flagship model, Qwen3.5-397B-A17B, activates only 17B of its 397B total parameters per token, supports 262K native context (extensible to ~1M via YaRN RoPE scaling), and is natively multimodal through early fusion training on text, image, and video tokens. Compared to Qwen3, the series doubles the expert count (128 → 512), expands language coverage from 119 to 201 languages, and replaces the pure-attention backbone with the hybrid DeltaNet design. All open-weight models are released under Apache 2.0.

## Architecture Summary

**Hybrid Layer Layout (3:1 pattern):**
The backbone alternates Gated DeltaNet and Gated Attention sublayers in a fixed 3:1 ratio. For the 397B flagship (60 layers): `15 × (3 × (Gated DeltaNet → MoE) → 1 × (Gated Attention → MoE))`. Three out of every four sublayers use efficient linear attention; only every fourth uses full quadratic attention.

**Gated DeltaNet (GDN):**
- Produces q, k, v plus two gates (α, β) via linear projections and lightweight convolutions with normalization.
- Replaces standard attention with a fast-weight delta-rule update: the correction term `(v_t − S_{t-1}^T k_t)` is the prediction error — the difference between the new value and what the recurrent state currently predicts for this key. Instead of blindly accumulating, DeltaNet corrects existing memory, yielding strong in-context retrieval without quadratic cost.
- Eliminates per-token KV-cache I/O for 75% of layers, dramatically improving generation throughput and serving concurrency for long contexts.

**Dense Models:**

| Model | Params | Hidden Dim | Layers | GDN Heads (V / QK) | GA Heads (Q / KV) | Head Dim | FFN Dim | Context |
|-------|--------|-----------|--------|--------------------|--------------------|----------|---------|---------|
| Qwen3.5-0.8B | 0.8B | — | — | — | — | — | — | 262K |
| Qwen3.5-2B | 2B | — | — | — | — | — | — | 262K |
| Qwen3.5-4B | 4B | 2,560 | 32 | 32 / 16 | 16 / 4 | 128 / 256 | 9,216 | 262K |
| Qwen3.5-9B | 9B | 4,096 | 32 | 32 / 16 | 16 / 4 | 128 / 256 | 12,288 | 262K |
| Qwen3.5-27B | 27B | 5,120 | 64 | 48 / 16 | 24 / 4 | 128 / 256 | 17,408 | 262K |

**MoE Models:**

| Model | Total Params | Active Params | Layers | GDN Heads (V / QK) | GA Heads (Q / KV) | Experts (Total / Routed+Shared) | Expert FFN Dim | Context |
|-------|-------------|---------------|--------|--------------------|--------------------|--------------------------------|----------------|---------|
| Qwen3.5-35B-A3B | 35B | 3B | 40 | 32 / 16 | 16 / 2 | 256 / 8+1 | 512 | 262K |
| Qwen3.5-122B-A10B | 122B | 10B | — | — | — | — | — | 262K |
| Qwen3.5-397B-A17B | 397B | 17B | 60 | 64 / 16 | — | 512 / 10+1 | — | 262K |

- **Vocabulary size:** 248,320 tokens (expanded from Qwen3's 151,669)
- **Activation function:** SwiGLU
- **Positional encoding:** RoPE with YaRN scaling (262K native → ~1.01M extended)
- **Normalization:** RMSNorm pre-normalization
- **MoE routing:** Fine-grained expert segmentation; 256 experts (small models) or 512 experts (flagship) with 8–10 routed + 1 shared expert per token

## Key Architectural Innovations

1. **Gated DeltaNet hybrid backbone** — replaces 75% of attention layers with linear-attention GDN blocks. The 3:1 GDN-to-attention ratio preserves the expressive power of full attention (every 4th layer) while eliminating quadratic KV-cache cost for most layers. This is the first major open-weight model family to adopt DeltaNet at scale.
2. **512 fine-grained MoE experts (flagship)** — doubles Qwen3's 128-expert count. The flagship uses 512 total experts with 10 routed + 1 shared active per token, reintroducing shared experts (which Qwen3 had dropped). Smaller MoE variants use 256 experts with 8 routed + 1 shared.
3. **Native multimodal early fusion** — text, image, and video tokens are fused during pre-training (not bolted on post-hoc), achieving cross-generational parity with dedicated vision-language models like Qwen3-VL.
4. **201-language multilingual expansion** — up from Qwen3's 119 languages, with an expanded vocabulary (248,320 tokens) to handle broader linguistic coverage.
5. **Million-token context** — 262K native context via YaRN RoPE scaling, extensible to ~1.01M tokens. The GDN backbone makes this practical by reducing per-token memory cost for long sequences.
6. **Scalable RL across million-agent environments** — reinforcement learning training scaled across massive agent environments with progressively complex task distributions, plus asynchronous RL frameworks supporting massive-scale agent scaffolds.
7. **Near-100% multimodal training efficiency** — training infrastructure achieves near-parity with text-only training efficiency even when processing multimodal tokens.

## Design Decisions and Tradeoffs

- **GDN vs full attention:** The delta-rule memory correction in GDN provides strong in-context retrieval, but linear attention fundamentally trades off fine-grained token-pair attention patterns. Retaining full attention every 4th layer is the hedge — enough global attention to handle tasks requiring precise token interactions, while GDN handles the bulk of sequence processing cheaply.
- **Shared experts return:** Qwen3 dropped shared experts; Qwen3.5 brings back 1 shared expert per MoE layer. This likely reflects empirical evidence that a shared expert for common-knowledge routing improves sample efficiency without meaningful inference cost.
- **512 experts at flagship scale:** More experts means finer-grained specialization but increases routing complexity and memory for expert parameters. The 10+1 active (of 512) ratio means only ~2% of experts fire per token.
- **Vocabulary expansion (151K → 248K):** The 64% vocabulary expansion supports 201 languages but increases embedding table size. The tradeoff favors global deployment coverage.
- **Text-first vs multimodal variants:** The open-weight text models and the Omni multimodal model share the same backbone but differ in training data mix. The Omni variant adds ARIA (Adaptive Rate Interleave Alignment) for streaming speech synthesis.

## Training Details

- **Pre-training data:** Trillions of multimodal tokens (exact count not disclosed; predecessor Qwen3 used 36T text tokens)
- **Languages:** 201 languages and dialects (up from 119 in Qwen3)
- **Pre-training curriculum:** Multi-stage, following the Qwen3 pattern of general → reasoning → long-context stages
- **Post-training:** Scaled RL across million-agent environments with progressively complex task distributions
- **Distillation:** Smaller models trained via distillation from flagship, maintaining the Qwen3 approach

**Qwen3.5-Omni additions:**
- Hybrid Attention MoE framework for both Thinker and Talker components
- ARIA (Adaptive Rate Interleave Alignment) for stable streaming speech synthesis
- Temporal alignment: contiguous position numbering across modalities to prevent positional conflicts

## Performance Highlights

**Qwen3.5-397B-A17B:**

| Benchmark | Score |
|-----------|-------|
| AIME'26 | 91.3% |
| MMLU-Redux | 94.9% |
| MMLU-Pro | 87.8% |

**Qwen3.5-27B:**

| Benchmark | Score |
|-----------|-------|
| AIME'26 | 92.7% |
| LiveCodeBench v6 | 80.7% |
| SWE-bench Pro | 51.2% |

**Qwen3.5-9B:**

| Benchmark | Score |
|-----------|-------|
| MMLU-Pro | 82.5% |

- The GDN hybrid architecture achieves comparable quality to full-attention models while enabling ~19x decoding throughput improvement.
- 27B dense model is competitive with (and sometimes exceeds) the 397B MoE flagship on reasoning benchmarks, demonstrating strong distillation and dense-model training.
- All open-weight models released under Apache 2.0.

## What Changed from Qwen3

| Dimension | Qwen3 | Qwen3.5 |
|-----------|-------|---------|
| Attention backbone | Full GQA (all layers) | Hybrid: 75% Gated DeltaNet + 25% Gated Attention |
| Max experts (flagship) | 128 routed, 0 shared | 512 total, 10 routed + 1 shared |
| Active experts | 8 of 128 | 10+1 of 512 (flagship) or 8+1 of 256 (smaller) |
| Vocabulary size | 151,669 | 248,320 |
| Languages | 119 | 201 |
| Native context | 128K | 262K (extensible to ~1.01M) |
| Multimodal | Separate Qwen3-VL model | Native early fusion in base model |
| Flagship params | 235B total / 22B active | 397B total / 17B active |
| Positional encoding | RoPE | RoPE + YaRN scaling |
| Shared experts | No | Yes (1 per layer) |
