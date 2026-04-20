<!-- scope: DeepSeek-V2 technical report
     deps: [[ch-01]], [[ch-02]]
     see-also: [[deepseek-v3]], [[deepseek-r1]], [[mixtral]]
-->

# DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model — Technical Report
- **Core Insight:** Multi-head Latent Attention compresses KV cache 93% by sharing a low-rank latent across heads.
- **Guideline:** KV cache size is an architectural choice, not a fixed cost of attention.

- **Organization:** DeepSeek AI
- **Year:** 2024
- **URL:** https://arxiv.org/abs/2405.04434
- **Relevant chapters:** Multi-head Latent Attention, Mixture-of-Experts, KV cache compression, efficient inference

## Abstract
We present DeepSeek-V2, a strong Mixture-of-Experts (MoE) language model characterized by economical training and efficient inference. It comprises 236B total parameters, of which 21B are activated for each token, and supports a context length of 128K tokens. DeepSeek-V2 adopts innovative architectures including Multi-head Latent Attention (MLA) for efficient inference and DeepSeekMoE for economical training. Compared with DeepSeek 67B, DeepSeek-V2 achieves significantly stronger performance, and meanwhile saves 42.5% of training costs, reduces the KV cache by 93.3%, and boosts the maximum generation throughput to 5.76 times.

## Architecture Summary

| Component | Value |
|-----------|-------|
| Total Parameters | 236B |
| Active Parameters | 21B per token |
| Transformer Layers | 60 |
| Hidden Dimension | 5,120 |
| Attention Heads | 128 |
| Per-Head Dimension | 128 |
| Context Length | 128K tokens |
| Vocabulary Size | 100K tokens (BBPE) |

**MLA Configuration:**
| Component | Value |
|-----------|-------|
| KV Compression Dim (d_c) | 512 |
| Query Compression Dim (d_c') | 1,536 |
| Decoupled RoPE Head Dim (d_h^R) | 64 |

**DeepSeekMoE Configuration:**
| Component | Value |
|-----------|-------|
| Routed Experts per Layer | 160 |
| Shared Experts per Layer | 2 |
| Active Routed Experts per Token | 6 |
| Expert Intermediate Dim | 1,536 |
| Device-Limited Routing (M) | 3 devices |

## Key Architectural Innovations

1. **Multi-head Latent Attention (MLA)** — replaces standard MHA with low-rank key-value compression. Keys and values are jointly compressed into a latent vector through down-projection, then reconstructed via up-projection. KV cache stores only the compressed latent plus a small decoupled RoPE component, reducing KV cache by 93.3% compared to standard MHA. Achieves performance superior to MHA while requiring cache equivalent to GQA with ~2.25 groups.
2. **Decoupled RoPE in MLA** — since compressing KV into latent vectors conflicts with position-sensitive RoPE, separate position-aware queries and keys are used alongside the compressed representations. This maintains positional encoding compatibility while preserving compression benefits.
3. **DeepSeekMoE with fine-grained experts** — uses 160 small experts (dim 1,536) rather than fewer large experts, with 2 shared experts always active. Fine-grained segmentation provides more flexible expert combinations and better specialization.
4. **Device-limited routing** — restricts each token to experts on at most 3 devices (out of 8), limiting communication overhead while maintaining routing diversity.
5. **Token-dropping strategy** — drops tokens with lowest affinity scores when devices exceed their compute budget, with ~10% of training sequences exempt from dropping to preserve full information.
6. **Group Relative Policy Optimization (GRPO)** — introduced for RL alignment, using group-based advantage estimation to eliminate the need for a separate critic model.

## Design Decisions and Tradeoffs

- **MLA vs. GQA/MQA:** MLA provides stronger performance than both GQA and MQA while achieving comparable or better KV cache compression. The tradeoff is increased architectural complexity and non-standard inference kernels.
- **Fine-grained experts (160 x 1,536):** More smaller experts give finer routing granularity but increase the routing overhead and complexity. Compared to standard 8-expert MoE designs, this provides dramatically more expert combinations.
- **Device-limited routing:** Balances communication cost against routing flexibility. Limiting to 3 devices per token reduces all-to-all communication but may prevent optimal expert assignment.
- **Three load-balancing losses:** Expert-level (alpha=0.003), device-level (alpha=0.05), and communication-level (alpha=0.02) auxiliary losses ensure balanced computation, though auxiliary losses can slightly degrade model quality.
- **Shared experts:** 2 always-active shared experts capture common knowledge, reducing the burden on routed experts to learn redundant representations.

## Training Details

- **Pretraining data:** 8.1 trillion tokens (Chinese ~12% more than English)
- **Optimizer:** AdamW (beta1=0.9, beta2=0.95, weight_decay=0.1)
- **Max learning rate:** 2.4e-4
- **LR schedule:** Warmup over 2K steps, decay by 0.316x at 60% and 90% of training
- **Batch size:** Scaled from 2,304 to 9,216 over first 225B tokens
- **Sequence length:** 4K (extended to 128K post-training via YaRN, scale=40)
- **Infrastructure:** NVIDIA H800 GPUs with NVLink/NVSwitch intra-node, InfiniBand inter-node
- **Parallelism:** 16-way zero-bubble pipeline parallelism, 8-way expert parallelism, ZeRO-1 data parallelism
- **Training cost:** 172.8K GPU-hours per trillion tokens (42.5% less than DeepSeek 67B)

**Alignment:**
- SFT: 1.5M instances (1.2M helpfulness, 0.3M safety), 2 epochs, LR=5e-6
- RL: GRPO with group-based baseline estimation

## Performance Highlights

| Benchmark | DeepSeek-V2 | DeepSeek 67B |
|-----------|-------------|--------------|
| MMLU (5-shot) | 78.5% | 71.3% |
| BBH (3-shot) | 78.9% | 68.7% |
| HumanEval (0-shot) | 48.8% | 45.1% |
| MATH (4-shot) | 43.6% | 18.7% |
| GSM8K (8-shot) | 79.2% | 63.4% |
| C-Eval (5-shot) | 81.7% | 65.2% |

**Efficiency gains vs. DeepSeek 67B:**
- Training cost: 42.5% reduction
- KV cache: 93.3% reduction
- Max generation throughput: 5.76x increase
- Generation throughput: >50K tokens/sec (single 8-GPU node)
