<!-- scope: MLA vs GQA quantitative comparison, parent: [[ch-07]] -->

# MLA vs GQA: A Quantitative Comparison

Multi-Head Latent Attention (MLA) and Grouped-Query Attention (GQA) represent two fundamentally different strategies for KV cache compression. GQA reduces the **number** of cached heads. MLA reduces the **dimensionality** of what gets cached. This excerpt compares them quantitatively across cache size, compute cost, quality, and implementation complexity.

---

## Cache Size: The Primary Metric

For a model with $H$ query heads, head dimension $d_k$, and $L$ layers, the KV cache per token is:

### MHA Baseline

$$\text{KV}_{\text{MHA}} = 2 \times H \times d_k \times L \times \text{precision}$$

For a 70B-class model ($H = 64$, $d_k = 128$, $L = 80$, FP16):

$$\text{KV}_{\text{MHA}} = 2 \times 64 \times 128 \times 80 \times 2 = 26{,}214{,}400 \text{ bytes} \approx 25 \text{ MB/token}$$

### GQA ($G = 8$)

$$\text{KV}_{\text{GQA}} = 2 \times G \times d_k \times L \times \text{precision}$$

$$= 2 \times 8 \times 128 \times 80 \times 2 = 3{,}276{,}800 \text{ bytes} \approx 3.1 \text{ MB/token}$$

**Compression ratio vs MHA:** $H / G = 8\times$

### MLA (DeepSeek-V2 config, [[deepseek-v2|report]])

$$\text{KV}_{\text{MLA}} = (d_c + d_h^R) \times L \times \text{precision}$$

$$= (512 + 64) \times 60 \times 2 = 69{,}120 \text{ bytes} \approx 67.5 \text{ KB/token}$$

Note: DeepSeek-V2 has 60 layers, $d_c = 512$ (KV compression dim), $d_h^R = 64$ (decoupled RoPE dim).

**Compression ratio vs MHA (DeepSeek 67B baseline):** $\sim 15\times$ reduction, or 93.3%.

### Normalized Comparison at Same Scale

To compare fairly, consider a hypothetical 70B dense model ($H = 64$, $d_k = 128$, $L = 80$) with different attention mechanisms:

| Mechanism | Cache per token per layer | Cache per token (80L) | vs MHA |
|-----------|--------------------------|----------------------|--------|
| MHA ($G = 64$) | 32,768 bytes | 2,621,440 bytes | 1.0x |
| GQA ($G = 8$) | 4,096 bytes | 327,680 bytes | 8x smaller |
| GQA ($G = 4$) | 2,048 bytes | 163,840 bytes | 16x smaller |
| MQA ($G = 1$) | 512 bytes | 40,960 bytes | 64x smaller |
| MLA ($d_c = 512$, $d_h^R = 64$) | 1,152 bytes | 92,160 bytes | 28.4x smaller |
| MLA ($d_c = 256$, $d_h^R = 64$) | 640 bytes | 51,200 bytes | 51.2x smaller |

**Key observation:** MLA's compression ratio depends on $d_c$, which is an architectural hyperparameter. With $d_c = 512$, MLA compresses ~28x (between GQA-4 and MQA). With $d_c = 256$, it approaches MQA-level compression. The crucial difference is that MLA's compression is learned and adaptive, while GQA/MQA's is a fixed structural constraint.

---

## Quality Comparison

### DeepSeek-V2 Ablation Results

The DeepSeek-V2 report ([[deepseek-v2|report]]) includes ablations comparing MLA, GQA, and MQA at equivalent cache budgets. The results are striking:

| Attention Type | KV Cache (per token, per layer) | MMLU | BBH | HumanEval |
|---------------|--------------------------------|------|-----|-----------|
| MHA (baseline) | 32,768 bytes | Reference | Reference | Reference |
| MQA | 512 bytes | Degraded | Degraded | Degraded |
| GQA (G=variable) | ~cache-matched to MLA | Close to MHA | Close to MHA | Close to MHA |
| MLA ($d_c = 512$) | 1,152 bytes | **Slightly above MHA** | **Slightly above MHA** | **Slightly above MHA** |

MLA at 1,152 bytes per token per layer outperforms both GQA and MQA at similar or larger cache budgets. The hypothesis: the low-rank bottleneck acts as a regularizer, forcing the model to learn more structured, compressible KV representations.

### The GQA Paper's Quality Data ([[gqa|paper]])

Ainslie et al. showed that GQA-8 (with uptraining) achieves quality "close to MHA" on summarization, translation, and QA benchmarks. The gap is small (< 0.5% on most benchmarks) but consistently present. MQA shows larger degradation, especially on tasks requiring diverse attention patterns.

### Raschka's Practical Assessment ([[raschka-attention-variants|blog]])

Raschka notes that MLA "works best at ~100B+ parameters; smaller models often benefit more from GQA." At smaller scales, the low-rank compression may lose too much information, and the implementation complexity is harder to justify.

---

## Compute Cost

### Training

| Component | GQA | MLA |
|-----------|-----|-----|
| KV projections | $2 \times d \times G \times d_k$ (reduced from MHA) | $d \times d_c$ (down) + $d_c \times H \times d_k \times 2$ (up, K+V) |
| Extra projections | None | Query compression: $d \times d_c'$ (down) + $d_c' \times H \times d_k$ (up) |
| Total matmuls | Standard (4 projections) | 6+ projections per layer |

MLA adds significant compute: the down-projection into the latent, plus up-projections to reconstruct K and V. However, the DeepSeek-V2 training was still 42.5% cheaper than the 67B MHA baseline because the MoE architecture (not MLA) drove compute savings.

### Inference (Decoding)

During autoregressive decoding, MLA reconstructs K and V from the cached latent $c_t$ at each step. This adds two matrix multiplications ($W_{UK}$ and $W_{UV}$) per layer per cached token. However, since decoding is memory-bandwidth-bound, the reduced cache loading (28x fewer bytes from HBM) dominates: MLA achieves 5.76x higher generation throughput despite the extra compute.

---

## Implementation Complexity

| Factor | GQA | MLA |
|--------|-----|-----|
| Custom CUDA kernels | Not required | Required for fused up-projection + attention |
| RoPE compatibility | Native (applied to K directly) | Requires decoupled RoPE (separate cached component) |
| Serving stack support | Universal (vLLM, TGI, TensorRT-LLM, llama.cpp) | Limited (custom DeepSeek kernels, partial vLLM) |
| Uptraining from MHA | Yes, 5% compute ([[gqa|paper]]) | No established recipe |
| Quantization compatibility | Standard | Latent quantization is a research question |

GQA's implementation simplicity is a genuine competitive advantage. As Raschka ([[raschka-attention-variants|blog]]) notes, "locally-run models with GQA often achieve better tok/sec throughput than architecturally superior alternatives because of better tooling support."

---

## Decision Framework

**Use GQA when:**
- Model size < 100B parameters
- You need broad serving stack compatibility (vLLM, llama.cpp, etc.)
- You're converting an existing MHA checkpoint (uptraining recipe available)
- Implementation simplicity is a priority
- Standard configuration: $G = H/8$

**Use MLA when:**
- Model size >= 100B parameters
- You can invest in custom inference kernels
- Maximum KV cache compression is critical (e.g., very long context)
- Quality preservation at extreme compression ratios is needed
- You're training from scratch (no uptraining recipe for MLA conversion)

**The industry trajectory:** As of early 2026, MLA adoption is accelerating at the frontier (DeepSeek V3, Kimi K2, GLM-5, Mistral Large 3), while GQA remains dominant in the 7B-70B range. The convergence point may shift as inference tooling for MLA matures.

---

## References

- [[deepseek-v2|DeepSeek AI, "DeepSeek-V2" (2024) (report)]]
- [[gqa|Ainslie et al., "GQA" (2023) (paper)]]
- [[mqa|Shazeer, "Fast Transformer Decoding" (2019) (paper)]]
- [[raschka-attention-variants|Raschka, "A Visual Guide to Attention Variants" (2026) (blog)]]
