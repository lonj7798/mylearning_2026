<!-- scope: DeepSeek-V3 technical report
     deps: [[ch-01]], [[ch-02]]
     see-also: [[deepseek-v2]], [[deepseek-r1]]
-->

# DeepSeek-V3 Technical Report
- **Core Insight:** Auxiliary-loss-free load balancing via bias terms is more stable than loss-based balancing for MoE.
- **Guideline:** When routing is unstable, add adaptive bias rather than gradient penalties.

- **Organization:** DeepSeek AI
- **Year:** 2024
- **URL:** https://arxiv.org/abs/2412.19437
- **Relevant chapters:** MoE scaling, auxiliary-loss-free load balancing, multi-token prediction, FP8 training, efficient distributed training

## Abstract
We present DeepSeek-V3, a strong Mixture-of-Experts (MoE) language model with 671B total parameters with 37B activated for each token. To achieve efficient inference and cost-effective training, DeepSeek-V3 adopts Multi-head Latent Attention (MLA) and DeepSeekMoE architectures, which were thoroughly validated in DeepSeek-V2. Furthermore, DeepSeek-V3 pioneers an auxiliary-loss-free strategy for load balancing and sets a multi-token prediction training objective for stronger performance. It is pre-trained on 14.8 trillion diverse and high-quality tokens, followed by Supervised Fine-Tuning and Reinforcement Learning stages to fully harness its capabilities. The entire training process requires only 2.788M H800 GPU hours. Throughout the entire training process, we did not experience any irrecoverable loss spikes or perform any rollbacks.

## Architecture Summary

| Component | Value |
|-----------|-------|
| Total Parameters | 671B |
| Active Parameters per Token | 37B (~5.5% activation) |
| Transformer Layers | 61 |
| Hidden Dimension | 7,168 |
| Attention Heads | 128 |
| KV Compression Dim (d_c) | 512 |
| Routed Experts | 256 per layer |
| Shared Experts | 1 per layer |
| Active Routed Experts (top-K) | 8 per token |
| Context Length | 128K tokens (extended from 4K) |
| Vocabulary Size | 129,280 tokens |

## Key Architectural Innovations

1. **Auxiliary-loss-free load balancing** — replaces traditional auxiliary losses (which degrade model quality) with dynamic bias terms that adjust routing. A bias is decreased if an expert is overloaded and increased if underloaded, controlled by a speed parameter gamma. This achieves balanced expert utilization without any loss penalty, complemented by a sequence-wise auxiliary loss with an extremely small coefficient.
2. **Multi-Token Prediction (MTP)** — trains the model to predict multiple sequential tokens at each position, using D sequential MTP modules that maintain a complete causal chain. This improves training signal density and pre-training performance, weighted by factor lambda in the combined loss.
3. **FP8 mixed precision training** — uses tile-wise (1x128) quantization for activations and block-wise (128x128) for weights, with fine-grained scaling factors along the inner GEMM dimension. Doubles computational speed compared to BF16 while maintaining <0.25% relative loss error. High-precision accumulation every 128 elements prevents numerical drift.
4. **DualPipe algorithm** — overlaps computation and all-to-all communication, achieving near-zero communication overhead during expert-parallel training.
5. **No token dropping** — unlike DeepSeek-V2, the auxiliary-loss-free balancing strategy eliminates the need to drop tokens during training or inference.

## Design Decisions and Tradeoffs

- **256 routed experts (vs. 160 in V2):** More experts provide finer-grained specialization but increase routing complexity and communication overhead. Mitigated by node-limited routing (max 4 nodes per token).
- **Auxiliary-loss-free over auxiliary loss:** Traditional load-balancing losses directly penalize the main training objective. The bias-based approach decouples balancing from training quality, but requires careful tuning of the speed parameter gamma.
- **MTP as training objective only:** MTP modules are used during training but can be optionally discarded during inference. When used during inference, enables speculative decoding for 1.8x speedup.
- **FP8 over BF16:** Halves compute cost but requires careful quantization design (fine-grained scaling, high-precision accumulation) to prevent quality degradation. Embeddings, output heads, gating, normalization, and attention remain in FP32/FP16.
- **No tensor parallelism:** Eliminated tensor parallelism entirely by combining pipeline parallelism (16-way), expert parallelism (64-way across 8 nodes), and ZeRO-1 data parallelism. Simplifies implementation and reduces communication.
- **Distillation from R1:** Post-training incorporates distillation from DeepSeek-R1 reasoning models, transferring chain-of-thought reasoning capabilities to V3.

## Training Details

- **Pre-training data:** 14.8 trillion diverse, high-quality tokens
- **Pre-training compute:** 2.664M H800 GPU-hours (~3.7 days per trillion tokens on 2,048 H800s)
- **Context extension:** 119K H800 GPU-hours (two stages: 32K then 128K)
- **Post-training:** ~5K H800 GPU-hours (SFT + RL)
- **Total compute:** 2.788M H800 GPU-hours
- **Training stability:** Zero irrecoverable loss spikes, zero rollbacks throughout entire training
- **Parallelism:** 16-way Pipeline Parallelism, 64-way Expert Parallelism, ZeRO-1 Data Parallelism

**Memory optimizations:**
- RMSNorm and MLA up-projection recomputation (avoid storing intermediate activations)
- EMA parameters stored on CPU
- Shared embedding/output head between MTP and main model

**Post-training:**
- Supervised Fine-Tuning (reasoning + non-reasoning data, with R1 distillation)
- Reinforcement Learning via GRPO with rule-based and model-based reward models

## Performance Highlights

| Benchmark | DeepSeek-V3 | Comparison |
|-----------|-------------|------------|
| MMLU | 88.5% | Competitive with GPT-4o |
| MMLU-Pro | 75.9% | — |
| GPQA Diamond | 59.1% | — |
| MATH-500 | State-of-the-art (non-long-CoT) | Outperforms o1-preview |
| LiveCodeBench | Top performer | Best among open models |
| Chinese SimpleQA | #1 | Surpasses all closed-source models |

- Outperforms all open-source models and achieves performance comparable to GPT-4o and Claude-3.5-Sonnet.
- Achieves this at a fraction of the training cost: 2.788M H800 GPU-hours total (estimated $5.6M at $2/GPU-hour).
- Remarkable training stability: no loss spikes or rollbacks, a significant achievement at this scale.
