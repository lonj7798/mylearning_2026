<!-- scope: OLMo 2 technical report
     deps: [[ch-01]], [[ch-02]]
     see-also: [[llama-3]], [[phi-4]]
-->

# OLMo 2: 2 OLMo 2 Furious — Technical Report
- **Core Insight:** Two-stage pre-training (web data then curated data) plus model souping produces better results than either stage alone.
- **Guideline:** Training curriculum matters; save high-quality data for the final stage.

- **Organization:** Allen Institute for AI (AI2)
- **Year:** 2024-2025
- **URL:** https://arxiv.org/abs/2501.00656
- **Relevant chapters:** Fully open models, training stability, two-stage training, curriculum learning, reproducible research

## Abstract
OLMo 2 is a family of fully open dense autoregressive language models at 7B, 13B, and 32B scales. All artifacts are released including model weights, full training data, training code and recipes, training logs, and thousands of intermediate checkpoints. OLMo 2 base models sit at the Pareto frontier of performance to training compute, often matching or outperforming comparable open-source models while using fewer FLOPs. OLMo 2 32B is the first fully open model to outperform GPT-3.5-Turbo and GPT-4o mini on a suite of popular multi-skill academic benchmarks.

## Architecture Summary

| Component | 7B | 13B | 32B |
|-----------|-----|------|------|
| Hidden Dimension | 4,096 | 5,120 | 5,120* |
| Layers | 32 | 40 | 64* |
| Attention Heads | 32 | 40 | 40* |
| KV Heads | 32 (MHA) | 40 (MHA) | 8 (GQA) |
| Max Position Embeddings | 4,096 | 4,096 | 4,096 |
| Training Tokens | 4T | 5T | 6T |
| Training FLOPs | 1.8e23 | 4.6e23 | — |

*32B architecture values estimated from scaling patterns; exact configs in model card.

- **Architecture:** Decoder-only Transformer
- **Attention:** MHA for 7B/13B; GQA for 32B (switched for scaling efficiency)
- **Normalization:** RMSNorm (switched from nonparametric LayerNorm in OLMo 1)
- **Positional encoding:** RoPE (replaced absolute positional embeddings from OLMo 1)
- **Training stability:** QK-Norm, Z-loss regularization, improved initialization preserving activation/gradient scale across layers

## Key Architectural Innovations

1. **Two-stage pre-training with curriculum learning** — Stage 1 trains on the broad OLMo-Mix-1124 dataset (~3.9T tokens for 7B). Stage 2 introduces Dolmino Mix 1124, a carefully curated high-quality data mix, during the annealing phase. This late-stage curriculum significantly improves downstream task performance.
2. **Training stability improvements over OLMo 1:**
   - Switched from nonparametric LayerNorm to RMSNorm
   - Added QK-Norm for stable attention scores at scale
   - Replaced absolute positional embeddings with RoPE
   - Added Z-loss regularization to prevent logit explosion
   - Improved initialization to preserve activation and gradient scale across layers
3. **Model souping** — trains multiple variants with different annealing mixes (50B and 300B token variants), then merges the best checkpoints via weight averaging ("model souping") for the final model.
4. **Fully open release** — every artifact is public: weights, training data (OLMo-Mix-1124, Dolmino-Mix-1124), code, recipes, logs, and thousands of intermediate checkpoints. This level of openness is unmatched by other frontier models.
5. **GQA for 32B scale** — switched from MHA (7B/13B) to Grouped-Query Attention for the 32B model to manage KV cache at scale.

## Design Decisions and Tradeoffs

- **Dense over MoE:** Deliberately chose a dense architecture for simplicity and reproducibility, even though MoE would be more efficient at the 32B scale. The goal is to serve as a reference implementation for the research community.
- **Two-stage over single-stage training:** The curriculum approach adds complexity but significantly improves performance. The Dolmino Mix introduced in Stage 2 focuses on educational, math, academic, and instruction-following content.
- **Model souping over single best checkpoint:** Training multiple annealing variants and merging is more expensive but produces a more robust final model than selecting a single checkpoint.
- **Full openness over competitive advantage:** Releasing all artifacts (data, code, intermediate checkpoints) reduces competitive moat but maximizes scientific value. This is a deliberate philosophical choice by AI2.
- **Conservative context length (4K):** Unlike models pushing to 128K+, OLMo 2 keeps a modest 4K context, focusing resources on data quality and training stability rather than context extension.
- **QK-Norm for stability:** Adds a small compute overhead but prevents the attention score instability that can cause loss spikes during training, which is critical for long training runs.

## Training Details

**Stage 1 — Initial Pre-training:**
- Dataset: OLMo-Mix-1124 (~3.9T tokens)
- 7B: ~1 epoch, 13B: ~1.2 epochs, 32B: ~1.5 epochs

**Stage 2 — Annealing with Dolmino Mix:**
- Dataset: Dolmino-Mix-1124 (843B tokens)
- Composition: 50% high-quality filtered documents + academic, math, educational, Q&A, instruction (synthetic and human)
- Multiple variants trained (50B and 300B token mixes), merged via model souping

**Post-training:**
- SFT + DPO + PPO with preference mix dataset
- Built on Tulu 3 training recipes

**Infrastructure (32B):**
- 160 nodes x 8 NVIDIA H100 GPUs (1,280 GPUs total)
- Google Cloud Engine AI Hypercomputer
- GPUDirect-TCPXO interconnect
- >1,800 tokens/sec/GPU (~38% MFU)

**Data cutoff:** December 2023

## Performance Highlights

- **OLMo 2 32B:** First fully open model to outperform GPT-3.5-Turbo and GPT-4o mini on a multi-skill benchmark suite.
- **OLMo 2 7B:** Outperforms Llama-3.1-8B despite lower total training FLOPs.
- **OLMo 2 13B:** Outperforms Qwen 2.5 7B.
- All models sit at the Pareto frontier of performance vs. training compute among open models.
- The OLMES evaluation framework (20 benchmarks) provides standardized assessment across knowledge recall, commonsense, general, and mathematical reasoning.
- Intermediate checkpoints enable the research community to study training dynamics, emergent capabilities, and scaling behavior.
