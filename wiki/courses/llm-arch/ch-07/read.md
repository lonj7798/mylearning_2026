# Chapter 7: Attention Variants

<!-- scope: MHA → MQA → GQA → MLA, Flash Attention, sliding window — the memory/quality/speed trilemma
     deps: [[ch-02]]
     see-also: [[ch-25]], [[ch-19]], [[ch-20]]
-->

## Overview

The attention mechanism from [[ch-02]] is elegant but expensive. Standard multi-head attention (MHA) stores separate key and value vectors for every head at every position, creating a KV cache that grows linearly with sequence length, model width, and head count. At 70B parameters with a 4K context, that cache alone consumes tens of gigabytes per request — and autoregressive decoding is *memory-bandwidth-bound*, meaning the GPU spends most of its time loading those cached tensors rather than doing arithmetic.

This chapter covers six attention variants that attack different points in the memory/quality/speed trilemma. MQA and GQA reduce the *number* of KV heads. MLA compresses *what* gets cached into a low-rank latent. Flash Attention eliminates unnecessary memory traffic by restructuring *how* attention is computed on hardware. Sliding window attention bounds *which* tokens participate. Together, these techniques form the attention design space that every modern LLM navigates — and understanding the tradeoffs between them is essential for anyone designing or evaluating model architectures.

The progression is not merely historical. It reflects a deepening understanding of where the real bottlenecks lie: not in FLOPs, but in bytes moved between levels of the GPU memory hierarchy.

---

## 1. Multi-Head Attention (MHA): The Baseline

Standard MHA from "Attention Is All You Need" projects the input into $H$ independent query, key, and value heads:

$$Q_h = XW_Q^h, \quad K_h = XW_K^h, \quad V_h = XW_V^h \quad \text{for } h = 1, \ldots, H$$

Each head computes attention independently, and results are concatenated and projected:

$$\text{head}_h = \text{softmax}\!\left(\frac{Q_h K_h^\top}{\sqrt{d_k}}\right) V_h, \qquad \text{MHA}(X) = \text{Concat}(\text{head}_1, \ldots, \text{head}_H) \, W_O$$

**Parameter count for KV projections:** Each of $W_K$ and $W_V$ has shape $(d_\text{model},\; H \cdot d_k)$. For a 70B-class model ($d_\text{model} = 8192$, $H = 64$, $d_k = 128$): each projection is $8192 \times 8192 = 67$M parameters, so KV projections total ~134M per layer.

**KV cache during inference:** At each decoding step, you store keys and values for every head at every past position. Per token, per layer:

$$\text{KV bytes} = 2 \times H \times d_k \times \text{precision}$$

<div style="background:#1a1a2e; border-radius:12px; padding:24px; margin:20px 0;">
<div style="color:#e0e0e0; font-size:14px; margin-bottom:16px; font-family:sans-serif; font-weight:bold;">KV Cache Memory: Llama 2 70B at 4K Context (MHA Hypothetical vs Actual GQA)</div>
<table style="width:100%; border-collapse:collapse; color:#e0e0e0; font-size:13px;">
<thead>
<tr style="border-bottom:2px solid #e94560;">
<th style="text-align:left; padding:8px;">Config</th>
<th style="text-align:right; padding:8px;">KV Heads</th>
<th style="text-align:right; padding:8px;">Per-Token KV</th>
<th style="text-align:right; padding:8px;">Cache @ 4K seq</th>
<th style="text-align:right; padding:8px;">Cache @ 32K seq</th>
</tr>
</thead>
<tbody>
<tr style="border-bottom:1px solid #333;">
<td style="padding:8px; color:#e94560; font-weight:bold;">MHA (64 KV heads)</td>
<td style="text-align:right; padding:8px;">64</td>
<td style="text-align:right; padding:8px;">32 KB</td>
<td style="text-align:right; padding:8px;">10.0 GB</td>
<td style="text-align:right; padding:8px;">80.0 GB</td>
</tr>
<tr style="border-bottom:1px solid #333;">
<td style="padding:8px; color:#4ecdc4; font-weight:bold;">GQA (8 KV heads)</td>
<td style="text-align:right; padding:8px;">8</td>
<td style="text-align:right; padding:8px;">4 KB</td>
<td style="text-align:right; padding:8px;">1.25 GB</td>
<td style="text-align:right; padding:8px;">10.0 GB</td>
</tr>
<tr style="border-bottom:1px solid #333;">
<td style="padding:8px; color:#ffd93d; font-weight:bold;">MQA (1 KV head)</td>
<td style="text-align:right; padding:8px;">1</td>
<td style="text-align:right; padding:8px;">0.5 KB</td>
<td style="text-align:right; padding:8px;">0.16 GB</td>
<td style="text-align:right; padding:8px;">1.25 GB</td>
</tr>
</tbody>
</table>
<div style="color:#888; font-size:11px; margin-top:12px;">
Calculation: 2 (K+V) x KV_heads x d_k (128) x layers (80) x seq_len x 2 bytes (FP16). E.g., MHA: 2 x 64 x 128 x 80 x 4096 x 2 = 10.7 GB.
</div>
</div>

At 32K context with full MHA, a *single request* would consume 80 GB of KV cache — exceeding the memory of an A100. This is why every 70B+ model uses a KV-reduction strategy.

---

## 2. Multi-Query Attention (MQA, [[mqa|paper]]): One KV Head to Rule Them All

Shazeer (2019) made a deceptively simple observation: autoregressive decoding is **memory-bandwidth-bound, not compute-bound**. At each decoding step, the arithmetic (one matrix multiply per layer) is tiny, but the model must load the entire KV cache from HBM. The bottleneck is bytes moved, not FLOPs computed.

MQA collapses all $H$ key heads and $H$ value heads into a single shared pair:

$$K = XW_K, \quad V = XW_V \qquad \text{(shared across all query heads)}$$

where $W_K, W_V \in \mathbb{R}^{d_\text{model} \times d_k}$ — reduced from $(d_\text{model},\; H \cdot d_k)$.

<div style="background:#1a1a2e; border-radius:12px; padding:24px; margin:20px 0;">
<div style="color:#e0e0e0; font-size:14px; margin-bottom:20px; font-family:sans-serif; font-weight:bold;">Head Layout: MHA vs MQA vs GQA</div>
<div style="display:flex; gap:32px; flex-wrap:wrap; justify-content:center;">
<!-- MHA -->
<div style="text-align:center;">
<div style="color:#e94560; font-weight:bold; margin-bottom:10px; font-size:13px;">MHA (H=8)</div>
<div style="display:flex; gap:3px; margin-bottom:4px;">
<div style="width:28px; height:28px; background:#e94560; border-radius:4px; display:flex; align-items:center; justify-content:center; color:#fff; font-size:10px; font-weight:bold;">Q1</div>
<div style="width:28px; height:28px; background:#e94560; border-radius:4px; display:flex; align-items:center; justify-content:center; color:#fff; font-size:10px; font-weight:bold;">Q2</div>
<div style="width:28px; height:28px; background:#e94560; border-radius:4px; display:flex; align-items:center; justify-content:center; color:#fff; font-size:10px; font-weight:bold;">Q3</div>
<div style="width:28px; height:28px; background:#e94560; border-radius:4px; display:flex; align-items:center; justify-content:center; color:#fff; font-size:10px; font-weight:bold;">Q4</div>
<div style="width:28px; height:28px; background:#e94560; border-radius:4px; display:flex; align-items:center; justify-content:center; color:#fff; font-size:10px; font-weight:bold;">Q5</div>
<div style="width:28px; height:28px; background:#e94560; border-radius:4px; display:flex; align-items:center; justify-content:center; color:#fff; font-size:10px; font-weight:bold;">Q6</div>
<div style="width:28px; height:28px; background:#e94560; border-radius:4px; display:flex; align-items:center; justify-content:center; color:#fff; font-size:10px; font-weight:bold;">Q7</div>
<div style="width:28px; height:28px; background:#e94560; border-radius:4px; display:flex; align-items:center; justify-content:center; color:#fff; font-size:10px; font-weight:bold;">Q8</div>
</div>
<div style="display:flex; gap:3px; margin-bottom:2px;">
<div style="width:28px; height:28px; background:#0f3460; border-radius:4px; display:flex; align-items:center; justify-content:center; color:#4ecdc4; font-size:10px; font-weight:bold;">K1</div>
<div style="width:28px; height:28px; background:#0f3460; border-radius:4px; display:flex; align-items:center; justify-content:center; color:#4ecdc4; font-size:10px; font-weight:bold;">K2</div>
<div style="width:28px; height:28px; background:#0f3460; border-radius:4px; display:flex; align-items:center; justify-content:center; color:#4ecdc4; font-size:10px; font-weight:bold;">K3</div>
<div style="width:28px; height:28px; background:#0f3460; border-radius:4px; display:flex; align-items:center; justify-content:center; color:#4ecdc4; font-size:10px; font-weight:bold;">K4</div>
<div style="width:28px; height:28px; background:#0f3460; border-radius:4px; display:flex; align-items:center; justify-content:center; color:#4ecdc4; font-size:10px; font-weight:bold;">K5</div>
<div style="width:28px; height:28px; background:#0f3460; border-radius:4px; display:flex; align-items:center; justify-content:center; color:#4ecdc4; font-size:10px; font-weight:bold;">K6</div>
<div style="width:28px; height:28px; background:#0f3460; border-radius:4px; display:flex; align-items:center; justify-content:center; color:#4ecdc4; font-size:10px; font-weight:bold;">K7</div>
<div style="width:28px; height:28px; background:#0f3460; border-radius:4px; display:flex; align-items:center; justify-content:center; color:#4ecdc4; font-size:10px; font-weight:bold;">K8</div>
</div>
<div style="display:flex; gap:3px;">
<div style="width:28px; height:28px; background:#0f3460; border-radius:4px; display:flex; align-items:center; justify-content:center; color:#ffd93d; font-size:10px; font-weight:bold;">V1</div>
<div style="width:28px; height:28px; background:#0f3460; border-radius:4px; display:flex; align-items:center; justify-content:center; color:#ffd93d; font-size:10px; font-weight:bold;">V2</div>
<div style="width:28px; height:28px; background:#0f3460; border-radius:4px; display:flex; align-items:center; justify-content:center; color:#ffd93d; font-size:10px; font-weight:bold;">V3</div>
<div style="width:28px; height:28px; background:#0f3460; border-radius:4px; display:flex; align-items:center; justify-content:center; color:#ffd93d; font-size:10px; font-weight:bold;">V4</div>
<div style="width:28px; height:28px; background:#0f3460; border-radius:4px; display:flex; align-items:center; justify-content:center; color:#ffd93d; font-size:10px; font-weight:bold;">V5</div>
<div style="width:28px; height:28px; background:#0f3460; border-radius:4px; display:flex; align-items:center; justify-content:center; color:#ffd93d; font-size:10px; font-weight:bold;">V6</div>
<div style="width:28px; height:28px; background:#0f3460; border-radius:4px; display:flex; align-items:center; justify-content:center; color:#ffd93d; font-size:10px; font-weight:bold;">V7</div>
<div style="width:28px; height:28px; background:#0f3460; border-radius:4px; display:flex; align-items:center; justify-content:center; color:#ffd93d; font-size:10px; font-weight:bold;">V8</div>
</div>
<div style="color:#888; font-size:11px; margin-top:6px;">8 Q + 8 K + 8 V = 24 heads</div>
</div>
<!-- MQA -->
<div style="text-align:center;">
<div style="color:#ffd93d; font-weight:bold; margin-bottom:10px; font-size:13px;">MQA (H=8, 1 KV)</div>
<div style="display:flex; gap:3px; margin-bottom:4px;">
<div style="width:28px; height:28px; background:#e94560; border-radius:4px; display:flex; align-items:center; justify-content:center; color:#fff; font-size:10px; font-weight:bold;">Q1</div>
<div style="width:28px; height:28px; background:#e94560; border-radius:4px; display:flex; align-items:center; justify-content:center; color:#fff; font-size:10px; font-weight:bold;">Q2</div>
<div style="width:28px; height:28px; background:#e94560; border-radius:4px; display:flex; align-items:center; justify-content:center; color:#fff; font-size:10px; font-weight:bold;">Q3</div>
<div style="width:28px; height:28px; background:#e94560; border-radius:4px; display:flex; align-items:center; justify-content:center; color:#fff; font-size:10px; font-weight:bold;">Q4</div>
<div style="width:28px; height:28px; background:#e94560; border-radius:4px; display:flex; align-items:center; justify-content:center; color:#fff; font-size:10px; font-weight:bold;">Q5</div>
<div style="width:28px; height:28px; background:#e94560; border-radius:4px; display:flex; align-items:center; justify-content:center; color:#fff; font-size:10px; font-weight:bold;">Q6</div>
<div style="width:28px; height:28px; background:#e94560; border-radius:4px; display:flex; align-items:center; justify-content:center; color:#fff; font-size:10px; font-weight:bold;">Q7</div>
<div style="width:28px; height:28px; background:#e94560; border-radius:4px; display:flex; align-items:center; justify-content:center; color:#fff; font-size:10px; font-weight:bold;">Q8</div>
</div>
<div style="display:flex; gap:3px; justify-content:center; margin-bottom:2px;">
<div style="width:230px; height:28px; background:#0f3460; border-radius:4px; display:flex; align-items:center; justify-content:center; color:#4ecdc4; font-size:10px; font-weight:bold;">K (shared)</div>
</div>
<div style="display:flex; gap:3px; justify-content:center;">
<div style="width:230px; height:28px; background:#0f3460; border-radius:4px; display:flex; align-items:center; justify-content:center; color:#ffd93d; font-size:10px; font-weight:bold;">V (shared)</div>
</div>
<div style="color:#888; font-size:11px; margin-top:6px;">8 Q + 1 K + 1 V = 10 heads</div>
</div>
<!-- GQA -->
<div style="text-align:center;">
<div style="color:#4ecdc4; font-weight:bold; margin-bottom:10px; font-size:13px;">GQA (H=8, G=2)</div>
<div style="display:flex; gap:3px; margin-bottom:4px;">
<div style="width:28px; height:28px; background:#e94560; border-radius:4px; display:flex; align-items:center; justify-content:center; color:#fff; font-size:10px; font-weight:bold;">Q1</div>
<div style="width:28px; height:28px; background:#e94560; border-radius:4px; display:flex; align-items:center; justify-content:center; color:#fff; font-size:10px; font-weight:bold;">Q2</div>
<div style="width:28px; height:28px; background:#e94560; border-radius:4px; display:flex; align-items:center; justify-content:center; color:#fff; font-size:10px; font-weight:bold;">Q3</div>
<div style="width:28px; height:28px; background:#e94560; border-radius:4px; display:flex; align-items:center; justify-content:center; color:#fff; font-size:10px; font-weight:bold;">Q4</div>
<div style="width:28px; height:28px; background:#e94560; border-radius:4px; display:flex; align-items:center; justify-content:center; color:#fff; font-size:10px; font-weight:bold;">Q5</div>
<div style="width:28px; height:28px; background:#e94560; border-radius:4px; display:flex; align-items:center; justify-content:center; color:#fff; font-size:10px; font-weight:bold;">Q6</div>
<div style="width:28px; height:28px; background:#e94560; border-radius:4px; display:flex; align-items:center; justify-content:center; color:#fff; font-size:10px; font-weight:bold;">Q7</div>
<div style="width:28px; height:28px; background:#e94560; border-radius:4px; display:flex; align-items:center; justify-content:center; color:#fff; font-size:10px; font-weight:bold;">Q8</div>
</div>
<div style="display:flex; gap:3px; justify-content:center; margin-bottom:2px;">
<div style="width:113px; height:28px; background:#0f3460; border-radius:4px; display:flex; align-items:center; justify-content:center; color:#4ecdc4; font-size:10px; font-weight:bold;">K group 1</div>
<div style="width:113px; height:28px; background:#0f3460; border-radius:4px; display:flex; align-items:center; justify-content:center; color:#4ecdc4; font-size:10px; font-weight:bold;">K group 2</div>
</div>
<div style="display:flex; gap:3px; justify-content:center;">
<div style="width:113px; height:28px; background:#0f3460; border-radius:4px; display:flex; align-items:center; justify-content:center; color:#ffd93d; font-size:10px; font-weight:bold;">V group 1</div>
<div style="width:113px; height:28px; background:#0f3460; border-radius:4px; display:flex; align-items:center; justify-content:center; color:#ffd93d; font-size:10px; font-weight:bold;">V group 2</div>
</div>
<div style="color:#888; font-size:11px; margin-top:6px;">8 Q + 2 K + 2 V = 12 heads</div>
</div>
</div>
</div>

**The speedup is nearly proportional to the cache reduction** because decoding is bandwidth-bound. If you reduce the KV cache by $H\times$, you reduce the bytes loaded per step by approximately $H\times$, which translates almost directly to $H\times$ faster decoding.

**The quality cost is real but modest.** Sharing K and V across all query heads reduces the model's ability to form diverse attention patterns — different heads can no longer attend to different representations of the same position. The paper acknowledges "only minor quality degradation," but the degradation is task-dependent and becomes more noticeable at scale. This gap is what motivated GQA.

**Training caveat:** MQA provides no speedup during training. Training parallelizes across sequence length (all positions computed simultaneously via teacher forcing), so the KV cache size is irrelevant. MQA is purely an inference optimization. Models must also be trained from scratch with MQA — you cannot trivially convert an existing MHA checkpoint.

---

## 3. Grouped-Query Attention (GQA, [[gqa|paper]]): The Industry Standard

Ainslie et al. (2023) observed that MQA's all-or-nothing sharing was more aggressive than necessary. GQA introduces $G$ key-value groups, where each group of $H/G$ query heads shares one key and one value head:

$$\text{MHA: } G = H \qquad \text{GQA: } 1 < G < H \qquad \text{MQA: } G = 1$$

The KV cache reduction factor is $H/G$ compared to MHA. For Llama 2 70B ($H = 64$, $G = 8$): an 8x reduction.

### The Uptraining Recipe

A key practical contribution: you do not need to retrain from scratch. The paper demonstrates converting existing MHA checkpoints to GQA using only **5% of the original pre-training compute**:

1. For each GQA group, **mean-pool** the original MHA key/value heads within that group
2. Continue pre-training (uptraining) for a fraction of the original compute

Mean-pooling outperforms both random initialization and selecting a single head because it preserves the maximum information from the original checkpoint. This recipe made GQA practical for organizations with existing MHA models — you spend 5% more compute to get dramatically cheaper inference forever.

### Why GQA Won

GQA has become the default attention configuration in modern LLMs: Llama 2 ([[llama-2|report]])/3, Mistral, Gemma, Mixtral, Qwen. The reasons are pragmatic:

1. **Quality close to MHA.** Multiple KV groups retain enough representational diversity to avoid MQA's quality degradation on hard tasks.
2. **Speed close to MQA.** With $G = 8$ and $H = 64$, you get an 8x cache reduction — most of MQA's benefit.
3. **Implementation simplicity.** GQA requires no custom CUDA kernels, no novel training recipes (beyond uptraining), and integrates cleanly with existing serving stacks (vLLM, TensorRT-LLM, etc.).
4. **Inference stack maturity.** As Raschka ([[raschka-attention-variants|blog]]) notes, locally-run models with GQA often achieve better tok/sec throughput than architecturally superior alternatives because of better tooling support.

The standard configuration has converged on approximately $G = H/8$: Llama 2 70B uses 8 KV heads for 64 query heads, Mistral 7B uses 8 KV heads for 32 query heads.

---

## 4. Multi-Head Latent Attention (MLA): Low-Rank KV Compression

DeepSeek-V2 ([[deepseek-v2|report]]) (2024) introduced a fundamentally different approach. Instead of reducing the *number* of KV heads (GQA) or sharing them (MQA), MLA compresses the *content* of the KV cache into a low-rank latent vector.

### The Core Mechanism

Keys and values are jointly compressed through a learned down-projection into a latent vector $c_t$, then reconstructed via up-projection at inference time:

$$c_t = W_{\text{DKV}} \, x_t \qquad \in \mathbb{R}^{d_c}$$

$$k_t^{(h)} = W_{UK}^{(h)} \, c_t, \qquad v_t^{(h)} = W_{UV}^{(h)} \, c_t$$

where $d_c \ll H \cdot d_k$. The KV cache stores only $c_t$ (the compressed latent) instead of all key and value vectors.

<div style="background:#1a1a2e; border-radius:12px; padding:24px; margin:20px 0;">
<div style="color:#e0e0e0; font-size:14px; margin-bottom:16px; font-family:sans-serif; font-weight:bold;">MLA: Low-Rank KV Compression</div>
<div style="display:flex; flex-direction:column; align-items:center; gap:12px;">
<!-- Input -->
<div style="background:#16213e; padding:10px 40px; border-radius:8px; color:#e0e0e0; font-size:13px; font-weight:bold;">
x_t (hidden state, d=5120)
</div>
<div style="color:#e94560; font-size:18px;">&#8595;</div>
<div style="color:#888; font-size:11px;">W_DKV down-projection</div>
<!-- Latent -->
<div style="background:#e94560; padding:12px 30px; border-radius:8px; color:#fff; font-size:13px; font-weight:bold; border:2px solid #ff6b81;">
c_t (latent, d_c=512) &larr; CACHED
</div>
<div style="display:flex; gap:40px; margin-top:4px;">
<div style="display:flex; flex-direction:column; align-items:center; gap:6px;">
<div style="color:#e94560; font-size:18px;">&#8595;</div>
<div style="color:#888; font-size:11px;">W_UK up-project</div>
<div style="background:#0f3460; padding:8px 20px; border-radius:8px; color:#4ecdc4; font-size:12px; font-weight:bold;">K heads (128 x 128)</div>
</div>
<div style="display:flex; flex-direction:column; align-items:center; gap:6px;">
<div style="color:#e94560; font-size:18px;">&#8595;</div>
<div style="color:#888; font-size:11px;">W_UV up-project</div>
<div style="background:#0f3460; padding:8px 20px; border-radius:8px; color:#ffd93d; font-size:12px; font-weight:bold;">V heads (128 x 128)</div>
</div>
</div>
</div>
<div style="color:#888; font-size:11px; margin-top:16px; text-align:center;">
Standard MHA caches H x d_k = 128 x 128 = 16,384 dims per K/V. MLA caches d_c = 512 dims.<br>
Compression ratio: 16,384 / 512 = 32x per tensor. With decoupled RoPE component (+64 dims): ~93.3% reduction.
</div>
</div>

### The RoPE Compatibility Problem

There is an important subtlety. Rotary position embeddings (RoPE) are applied directly to key vectors and are position-dependent. But compressing keys into a position-agnostic latent $c_t$ would destroy positional information — the up-projection cannot recover where in the sequence a token appeared.

DeepSeek's solution: **decoupled RoPE**. Separate the position-sensitive component into a small additional key/value pair ($d_h^R = 64$ dimensions) that is cached alongside the latent. The total cache per token per layer becomes:

$$d_c + d_h^R = 512 + 64 = 576 \text{ dims}$$

Compare to MHA's $2 \times H \times d_k = 2 \times 128 \times 128 = 32{,}768$ dims. That's a **93.3% reduction**.

### DeepSeek-V2 Results

| Metric | vs. DeepSeek 67B (MHA) |
|--------|----------------------|
| KV cache | **93.3% reduction** |
| Generation throughput | **5.76x increase** |
| Training cost | 42.5% reduction |
| Quality (MMLU) | 78.5% vs 71.3% (+7.2 points) |

The quality result is remarkable: MLA does not merely match MHA — it slightly *outperforms* it. The hypothesis is that the low-rank bottleneck acts as a regularizer, forcing the model to learn more structured representations. The DeepSeek-V2 ablations showed MLA outperforming both GQA and MQA at equivalent cache budgets.

### When to Use MLA vs GQA

MLA's advantage scales with model size. Raschka's survey ([[raschka-kv-cache|blog]]) notes that MLA "works best at ~100B+ parameters; smaller models often benefit more from GQA." The reasons:

- The architectural complexity (custom inference kernels, decoupled RoPE) adds fixed overhead that is only justified at scale
- Smaller models have less redundancy in their KV representations, so the low-rank compression loses more information
- Inference tooling support for MLA is still maturing compared to GQA's ubiquitous support

As of 2026, MLA has been adopted by DeepSeek V3, Kimi K2, GLM-5, and Mistral Large 3 — all 100B+ models.

---

## 5. Flash Attention ([[flash-attention|paper]]): IO-Aware Exact Attention

Flash Attention (Dao et al., 2022) attacks a different bottleneck entirely. MQA/GQA/MLA reduce the *size* of tensors stored. Flash Attention reduces unnecessary *movement* of tensors between memory levels.

### Why Attention Is Memory-Bound

The critical hardware insight: modern GPUs have a severe imbalance between compute throughput and memory bandwidth.

<div style="background:#1a1a2e; border-radius:12px; padding:24px; margin:20px 0;">
<div style="color:#e0e0e0; font-size:14px; margin-bottom:16px; font-family:sans-serif; font-weight:bold;">GPU Memory Hierarchy (A100)</div>
<div style="display:flex; gap:24px; align-items:flex-end; justify-content:center; flex-wrap:wrap;">
<div style="text-align:center;">
<div style="background:#e94560; width:80px; height:200px; border-radius:8px 8px 0 0; display:flex; flex-direction:column; justify-content:center; align-items:center; margin:0 auto;">
<div style="color:#fff; font-weight:bold; font-size:24px;">19</div>
<div style="color:#fff; font-size:11px;">TB/s</div>
</div>
<div style="background:#16213e; padding:8px; border-radius:0 0 8px 8px; width:80px; margin:0 auto;">
<div style="color:#4ecdc4; font-weight:bold; font-size:12px;">SRAM</div>
<div style="color:#888; font-size:10px;">192KB/SM</div>
<div style="color:#888; font-size:10px;">~20MB total</div>
</div>
</div>
<div style="text-align:center;">
<div style="background:#0f3460; width:80px; height:32px; border-radius:8px 8px 0 0; display:flex; flex-direction:column; justify-content:center; align-items:center; margin:168px auto 0;">
<div style="color:#ffd93d; font-weight:bold; font-size:16px;">2.0</div>
<div style="color:#ffd93d; font-size:11px;">TB/s</div>
</div>
<div style="background:#16213e; padding:8px; border-radius:0 0 8px 8px; width:80px; margin:0 auto;">
<div style="color:#ffd93d; font-weight:bold; font-size:12px;">HBM</div>
<div style="color:#888; font-size:10px;">80 GB</div>
<div style="color:#888; font-size:10px;">(main VRAM)</div>
</div>
</div>
</div>
<div style="color:#888; font-size:11px; margin-top:16px; text-align:center;">
SRAM is ~10x faster than HBM but ~4000x smaller. Standard attention materializes the full N x N matrix in HBM — wasteful.
</div>
</div>

Standard attention performs these steps, each requiring a full round-trip to HBM:

1. Load Q, K from HBM, compute $S = QK^\top$, **write S to HBM** (N x N matrix)
2. Load S from HBM, compute softmax, **write P to HBM** (N x N matrix)
3. Load P, V from HBM, compute output $O = PV$, write O to HBM

The N x N matrices S and P are the problem. They consume $O(N^2)$ memory and require $O(N^2)$ HBM accesses. For sequence length 4K with FP16: the attention matrix alone is 4096 x 4096 x 2 bytes = 32 MB *per head per layer*. At seq length 16K, it's 512 MB per head per layer.

### Tiling: The Core Algorithm

Flash Attention never materializes the full N x N attention matrix. Instead, it divides Q, K, V into blocks that fit in SRAM and computes attention *tile by tile*:

<div style="background:#1a1a2e; border-radius:12px; padding:24px; margin:20px 0;">
<div style="color:#e0e0e0; font-size:14px; margin-bottom:16px; font-family:sans-serif; font-weight:bold;">Flash Attention Tiling</div>
<div style="display:flex; gap:24px; align-items:flex-start; justify-content:center; flex-wrap:wrap;">
<!-- Standard -->
<div style="text-align:center;">
<div style="color:#e94560; font-weight:bold; font-size:12px; margin-bottom:8px;">Standard: Full N x N in HBM</div>
<div style="width:120px; height:120px; background:linear-gradient(135deg, #e94560 0%, #0f3460 100%); border-radius:4px; border:2px solid #e94560; margin:0 auto; display:flex; align-items:center; justify-content:center;">
<div style="color:#fff; font-size:11px; font-weight:bold;">S = QK<sup>T</sup><br>N x N<br>in HBM</div>
</div>
<div style="color:#e94560; font-size:11px; margin-top:4px;">O(N^2) HBM accesses</div>
</div>
<!-- Flash -->
<div style="text-align:center;">
<div style="color:#4ecdc4; font-weight:bold; font-size:12px; margin-bottom:8px;">Flash: Tiles in SRAM</div>
<div style="width:120px; height:120px; background:#16213e; border-radius:4px; border:2px solid #4ecdc4; margin:0 auto; display:grid; grid-template-columns:repeat(4,1fr); grid-template-rows:repeat(4,1fr); gap:2px; padding:3px;">
<div style="background:#4ecdc4; border-radius:2px; display:flex; align-items:center; justify-content:center; font-size:8px; color:#1a1a2e; font-weight:bold;">B1</div>
<div style="background:#0f3460; border-radius:2px;"></div>
<div style="background:#0f3460; border-radius:2px;"></div>
<div style="background:#0f3460; border-radius:2px;"></div>
<div style="background:#0f3460; border-radius:2px;"></div>
<div style="background:#0f3460; border-radius:2px;"></div>
<div style="background:#0f3460; border-radius:2px;"></div>
<div style="background:#0f3460; border-radius:2px;"></div>
<div style="background:#0f3460; border-radius:2px;"></div>
<div style="background:#0f3460; border-radius:2px;"></div>
<div style="background:#0f3460; border-radius:2px;"></div>
<div style="background:#0f3460; border-radius:2px;"></div>
<div style="background:#0f3460; border-radius:2px;"></div>
<div style="background:#0f3460; border-radius:2px;"></div>
<div style="background:#0f3460; border-radius:2px;"></div>
<div style="background:#0f3460; border-radius:2px;"></div>
</div>
<div style="color:#4ecdc4; font-size:11px; margin-top:4px;">O(N^2 d^2/M) HBM accesses</div>
</div>
</div>
<div style="color:#888; font-size:11px; margin-top:12px; text-align:center;">
Only one block lives in SRAM at a time. Results are accumulated with online softmax.<br>
Block size B = ceil(M / (4d)), where M = SRAM capacity, d = head dimension.
</div>
</div>

### The Online Softmax Trick

The mathematical challenge: softmax requires a global normalization denominator $\sum_j e^{s_j}$ across the entire key sequence. How can you compute this tile-by-tile without seeing all scores at once?

The solution (Milakov & Gimelshein, 2018) maintains two running statistics per query row:

- $m$: the running maximum score (for numerical stability)
- $\ell$: the running sum of exponentials

When processing a new block of keys, you update:

$$m_{\text{new}} = \max(m_{\text{old}}, m_{\text{block}})$$
$$\ell_{\text{new}} = \ell_{\text{old}} \cdot e^{m_{\text{old}} - m_{\text{new}}} + \ell_{\text{block}} \cdot e^{m_{\text{block}} - m_{\text{new}}}$$

The output accumulator is similarly rescaled. When all blocks have been processed, the result is **mathematically identical** to standard attention — no approximation whatsoever.

### The Backward Pass: Recomputation > Storage

During backpropagation, standard attention requires the stored $N \times N$ attention matrix $P$. Flash Attention instead **recomputes** $P$ from Q, K, V blocks on-the-fly during the backward pass. This trades extra FLOPs for massive memory savings.

This is counterintuitive: recomputation is *faster* than storage-and-retrieval because the recomputation happens in fast SRAM, while loading stored matrices from HBM is slow. The bottleneck was never arithmetic — it was memory IO.

### Flash Attention 2 ([[flash-attention-2|paper]]): Closing the Gap to GEMM

FlashAttention-1 reached only 25-40% of theoretical max FLOPs/s. Dao (2023) identified three sources of inefficiency:

1. **Too many non-matmul FLOPs** (softmax rescaling, masking) — restructured to minimize these, keeping tensor cores busy with matrix multiplications
2. **Parallelism limited to batch x heads** — FA-2 also parallelizes across the sequence dimension, improving GPU occupancy
3. **Suboptimal warp communication** — FA-2 partitions Q across warps (sharing K/V) instead of splitting K/V across warps, reducing shared memory traffic

The result: **50-73% of theoretical max FLOPs/s on A100** (2x faster than FA-1), achieving 225 TFLOPs/s per A100 (72% model FLOPs utilization). For causal attention, FA-2 also skips entire masked-out blocks, saving roughly 50% of computation for the causal triangle.

### Why Flash Attention Is Not "Approximate Attention"

This distinction matters. Methods like Linformer, Performer, and Linear Attention *approximate* the attention matrix with lower-rank or kernel-based substitutes, trading quality for speed. Flash Attention computes the **exact same result** as standard attention — it's an algorithmic optimization of memory access patterns, not a mathematical approximation. The $O(N^2)$ FLOP count is unchanged; what changes is the $O(N^2 d^2/M)$ HBM access count, which is substantially less than the standard $O(N^2)$ when $M \gg d^2$.

**Practical impact:**
| Benchmark | Speedup |
|-----------|---------|
| BERT-large (seq 512) | 15% wall-clock |
| GPT-2 (seq 1K) | 3x wall-clock |
| Long-range arena (1K-4K) | 2.4x wall-clock |
| Path-X (seq 16K) | First Transformer to achieve >chance |
| Memory reduction | O(N) instead of O(N^2) |

Flash Attention is now the default in every major training and inference framework. If you are running attention without Flash Attention (or its successors), you are leaving 2-3x performance on the table for free.

---

## 6. Sliding Window Attention (SWA): Bounded Memory

Mistral 7B ([[mistral-7b|report]]) (2023) introduced sliding window attention to the mainstream. Each attention layer attends only to the most recent $W$ tokens:

$$\text{Attention}_\ell(t) = \text{softmax}\!\left(\frac{q_t K_{[t-W:t]}^\top}{\sqrt{d_k}}\right) V_{[t-W:t]}$$

With a window size $W = 4096$ and $L = 32$ layers, the theoretical effective receptive field spans $W \times L = 131{,}072$ tokens — information from early tokens propagates through the residual stream across layers, even though no single layer sees beyond its window.

### Rolling Buffer KV Cache

The key implementation insight: use a **fixed-size circular buffer** of size $W$:

$$\text{cache\_index}(t) = t \mod W$$

Position $t$ overwrites position $t - W$ in the buffer. This caps KV cache memory at a constant regardless of sequence length — a property no other attention variant provides. For Mistral 7B at 32K context: 8x cache memory reduction versus storing all positions.

### The Hybrid Approach

Modern architectures rarely use pure SWA. Instead, they combine local (sliding window) and global (full) attention layers:

- **Gemma 3:** 5:1 ratio of local to global layers, with $W = 1024$. Ablations showed minimal quality impact.
- **OLMo 3:** Similar hybrid approach.
- **Pairing with GQA:** SWA bounds *which* tokens are cached; GQA reduces the cache *per token*. Mistral 7B uses both (SWA $W = 4096$, GQA with 8 KV heads), achieving a multiplicative reduction.

**The tradeoff:** SWA sacrifices guaranteed global attention within a single layer. For tasks requiring precise retrieval of information from early in a long context (needle-in-a-haystack), SWA relies entirely on information propagation through the residual stream across layers. This is why the hybrid approach — mixing SWA layers with occasional global attention layers — has become standard.

---

## Core Insights from the Literature

### Insight 1: Decoding is bandwidth-bound, not compute-bound
**Paper:** Shazeer, "Fast Transformer Decoding: One Write-Head is All You Need" ([[mqa|paper]])

This single observation — that the bottleneck during autoregressive decoding is loading KV tensors from HBM, not the matrix arithmetic — reframed all subsequent attention optimization work. Before MQA, the community was focused on reducing FLOPs (approximate attention methods like Linformer, Performer). Shazeer showed the right target was *bytes moved*. This insight equally motivates GQA, MLA, Flash Attention, and [[ch-25]]'s PagedAttention. **Guideline:** When optimizing inference, profile memory bandwidth utilization before touching the compute. If your GPU's arithmetic units are idle waiting for data, no amount of FLOP reduction helps.

### Insight 2: You can convert existing MHA models to GQA cheaply
**Paper:** Ainslie et al., "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints" ([[gqa|paper]])

The uptraining recipe — mean-pool KV heads, continue training for 5% of original compute — was the practical breakthrough that made GQA ubiquitous. Without it, organizations would need to retrain their models from scratch to benefit from KV-cache reduction. The mean-pooling initialization is notably better than both random init and single-head selection because it preserves maximal information from the original heads. **Guideline:** For existing MHA models, default to GQA with $G = H/8$ groups. The 5% compute cost is trivially justified by the permanent inference savings.

### Insight 3: Attention's bottleneck is memory IO, and the fix is *exact*, not approximate
**Paper:** Dao et al., "FlashAttention" ([[flash-attention|paper]])

A decade of approximate attention research (Linformer, Performer, BigBird, Longformer) tried to reduce attention's $O(N^2)$ complexity by trading quality for speed. Flash Attention showed that the *real* bottleneck was memory IO between HBM and SRAM, and that tiling + online softmax eliminates unnecessary memory traffic while computing the exact same result. The approximate methods often failed to achieve wall-clock speedups despite lower theoretical FLOP counts, precisely because they didn't account for memory hierarchy. **Guideline:** Always use Flash Attention (or successors) for both training and inference. Never materialize the full $N \times N$ attention matrix in HBM. The algorithmic savings are free — exact results, less memory, faster execution.

### Insight 4: KV cache size is a design parameter, not a fixed cost
**Paper:** DeepSeek AI, "DeepSeek-V2" ([[deepseek-v2|report]])

MLA demonstrated that you can compress the KV cache by 93% while *improving* quality over MHA. The low-rank bottleneck acts as a regularizer, forcing more structured representations. This reframes the design space: the KV cache need not be proportional to $H \times d_k$ — you can learn a compressed representation whose dimensionality is an independent architectural hyperparameter. Combined with GQA and SWA, this means the memory cost of attention is almost entirely under the architect's control. **Guideline:** For 100B+ models, evaluate MLA against GQA — the cache savings may justify the implementation complexity. For smaller models, GQA remains the pragmatic choice due to tooling maturity.

### Insight 5: Implementation maturity can outweigh architectural superiority
**Paper/Source:** Raschka, "A Visual Guide to Attention Variants" ([[raschka-attention-variants|blog]])

GQA persists as the dominant choice despite theoretically superior alternatives (MLA, hybrid architectures) because inference tooling — vLLM, TensorRT-LLM, llama.cpp — is deeply optimized for it. Several 2025 releases deliberately maintained GQA over MLA for this reason. The MiniMax-M2 retreat from linear attention is a related cautionary tale: linear attention degraded multi-turn and reasoning quality enough to justify returning to quadratic attention despite the efficiency gains. **Guideline:** Evaluate attention variants not just on paper metrics but on end-to-end tok/sec in your target serving stack. Architectural novelty that can't be efficiently served is academic.

---

## Key Takeaways

1. **Autoregressive decoding is memory-bandwidth-bound.** The entire field of attention variants flows from this fact. Reducing bytes moved per decoding step matters more than reducing FLOPs.

2. **MQA/GQA/MLA form a compression spectrum for KV heads.** MQA (1 KV head) is maximum compression with quality cost. GQA ($G$ groups) is the industry standard sweet spot. MLA (low-rank latent) achieves 93% compression with quality preservation at 100B+ scale.

3. **Flash Attention is algorithmic, not approximate.** It computes exact attention by tiling for SRAM, reducing HBM accesses from $O(N^2)$ to $O(N^2 d^2/M)$ while cutting memory from $O(N^2)$ to $O(N)$. The key enabler is the online softmax trick.

4. **Sliding window attention bounds memory at the cost of per-layer global context.** The hybrid approach (SWA + occasional global layers) has become standard for balancing efficiency with retrieval quality.

5. **The five techniques compose multiplicatively.** A model can use GQA (fewer KV heads) + SWA (fewer cached positions) + Flash Attention (fewer HBM accesses) simultaneously. Mistral 7B demonstrates this with GQA-8 + SWA-4096 + Flash Attention.

6. **Choose attention variant based on serving constraints, not training quality.** All these variants provide no benefit during training. The choice is entirely about inference cost, which is where you spend the most compute over the model's lifetime.

7. **Tooling maturity is a legitimate architectural consideration.** GQA's dominance is partly because it's the best-supported variant in production serving stacks. MLA is architecturally superior at scale but requires custom kernels.

---

## References

- [[mqa|Shazeer, "Fast Transformer Decoding: One Write-Head is All You Need" (2019) (paper)]] — MQA
- [[gqa|Ainslie et al., "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints" (2023) (paper)]] — GQA and uptraining
- [[flash-attention|Dao et al., "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness" (2022) (paper)]] — Flash Attention
- [[flash-attention-2|Dao, "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning" (2023) (paper)]] — FA-2
- [[deepseek-v2|DeepSeek AI, "DeepSeek-V2: A Strong, Economical, and Efficient MoE Language Model" (2024) (report)]] — MLA
- [[llama-2|Touvron et al., "Llama 2: Open Foundation and Fine-Tuned Chat Models" (2023) (report)]] — GQA validation at 70B
- [[mistral-7b|Jiang et al., "Mistral 7B" (2023) (report)]] — sliding window attention + rolling buffer
- [[raschka-attention-variants|Raschka, "A Visual Guide to Attention Variants in Modern LLMs" (2026) (blog)]] — taxonomy and comparative analysis
- [[flash-attention-explained|Gordic, "ELI5: Flash Attention" (2023) (blog)]] — Flash Attention deep dive with GPU memory hierarchy
- [[raschka-kv-cache|Raschka, "Understanding and Coding the KV Cache in LLMs from Scratch" (2025) (blog)]] — KV cache implementation
- Milakov & Gimelshein, "Online Normalizer Calculation for Softmax" (2018) — online softmax enabling Flash Attention tiling
