---
chapter: ch-15
course: model-quantization
phase: read
excerpt_of: "KVQuant: Towards 10 Million Context Length LLM Inference with KV Cache Quantization"
source_url: https://arxiv.org/abs/2401.18079
created_at: "2026-05-21"
---

# Excerpt: KVQuant — pre-RoPE K + non-uniform + dense-and-sparse

**Authors:** Coleman Hooper, Sehoon Kim, Hiva Mohammadzadeh, Michael W. Mahoney, Yakun Sophia Shao, Kurt Keutzer, Amir Gholami (Berkeley)
**Year:** 2024
**Venue:** NeurIPS 2024
**URL:** https://arxiv.org/abs/2401.18079
**Raw-data source:** [[raw-data/kvquant]]

---

## What KVQuant adds to KIVI

KIVI gets you to INT2 with the K/V axis asymmetry. KVQuant pushes to **sub-2-bit average** by stacking four techniques:

1. **Pre-RoPE K quantization** — quantize un-rotated K; RoPE applied on dequant.
2. **Non-uniform per-channel codes** — k-means codebook per K channel.
3. **Dense-and-sparse decomposition** — 1% outliers in FP16, rest in low-bit dense.
4. **Q-norm quantization** — the Query is also quantized.

---

## 1. Pre-RoPE K quantization

Standard RoPE: `K_t = RoPE_t(W_k · x_t)`. The rotation applies token-dependent angles to pairs of channels `(i, i + d/2)`, **entangling** them.

KVQuant's observation: post-RoPE K's per-channel outlier structure is destroyed by the token-dependent rotation. Pre-RoPE K has a stationary per-channel distribution → quantize that, apply RoPE on dequant.

```python
def kvquant_store_k(W_k, x_t, kv_cache, t):
    k_pre_rope = W_k @ x_t
    # Quantize pre-RoPE K with per-channel non-uniform code
    kv_cache.store_pre_rope_k(quantize_non_uniform(k_pre_rope))
    # Note: no RoPE here

def kvquant_attention(Q, kv_cache):
    K_post_rope = []
    for t_past in range(len(kv_cache)):
        k_pre = kv_cache.dequant_pre_rope_k(t_past)
        k_post = apply_rope(k_pre, position=t_past)
        K_post_rope.append(k_post)
    return softmax(Q @ stack(K_post_rope).T / sqrt(d))
```

**Cost**: one extra dequant + RoPE step per attention call, fully fused into the attention kernel. ~10% attention-time overhead at long context.

**Benefit**: post-RoPE per-channel quant loses ~1 ppl at INT2 vs pre-RoPE. The figure-2 KVQuant visualisation shows: pre-RoPE K has neat vertical channel bands (perfect for per-channel quant); post-RoPE K is mixed (per-channel quant misses).

---

## 2. Non-uniform per-channel codes

For each K channel `c`, compute calibration histogram across many tokens. Fit a k-means codebook of size `2^B` (B = 2 or 3) restricted to that channel. Store the codebook (negligible amortised cost) and per-element 2/3-bit index.

```python
def kvquant_calibrate_K_codebook(K_calibration, B=2):
    # K_calibration: (n_calib_tokens, n_heads, head_dim)
    codebooks = []
    for c in range(n_heads * head_dim):
        samples = K_calibration[:, c]               # (n_calib_tokens,)
        codebook = kmeans_1d(samples, k=2**B)
        codebooks.append(codebook)
    return codebooks
```

Equivalent to [[squeezellm]]'s non-uniform weight quant applied to KV. **V uses per-token uniform quant** (consistent with KIVI's V is token-wise finding).

---

## 3. Dense-and-sparse decomposition

Identify the top **1%** (by absolute value) of pre-RoPE K and per-token V elements per layer. Store them in a **sparse FP16** buffer (`index + value` tuples). Dense path uses the non-uniform code at 2/3-bit.

```
K[t, h, c] = top_1_percent_sparse_FP16 ∪ dense_2bit_indexed
```

At attention time:

```
attn_logits = Q · dense_K^⊤ + Q · sparse_K^⊤
```

Dense path uses the low-bit dequant + attention dot-product (cheap). Sparse path is a small FP16 SpMV (cheap at 1% density).

Borrowed from [[spqr]] (which did this for weights). The principle: outliers are signal not noise, preserve them precisely while compressing the rest aggressively.

---

## 4. Q-norm quantization

The Query is also quantized so `QK^⊤` happens in low-bit arithmetic. Per-token symmetric INT4 or INT8. (The Query isn't cached — quantized fresh each step. Mostly bandwidth-neutral; the gain is allowing tensor-core INT4 GEMM for the attention dot-product.)

---

## Effective bit budget

At 2-bit dense:
- Dense: 2 bits/element.
- Sparse: 1% × (24 bits index + 16 bits FP16) ≈ 0.4 bits/element amortised.
- **Total ≈ 2.4 bits/element**.

At 3-bit dense: ~3.4 bits/element.

---

## The numbers

LLaMA-7B Wikitext-2:

| Method | KV bits | ppl | Δppl |
|--------|---------|-----|------|
| FP16 | 16 | 5.47 | — |
| KIVI | 4 | 5.49 | +0.02 |
| **KVQuant** | **4** | **5.48** | **+0.01** |
| KIVI | 2 | 7.0 | +1.53 |
| **KVQuant** | **3** | **5.55** | **+0.08** |
| KVQuant | 2 | 6.04 | +0.57 |

KVQuant at 3-bit is essentially lossless (+0.08). KVQuant at 2-bit beats KIVI at 2-bit by ~1 ppl (the pre-RoPE trick + non-uniform + dense-and-sparse stacking).

---

## Context-length math (the headline)

LLaMA-7B FP16 KV: `2 × 32 layers × 32 heads × 128 dim × 2 bytes = 512 KB / token`.
1M tokens = 512 GB. Impossible on one A100.

LLaMA-7B at KVQuant 2-bit: 64 KB / token → 64 GB for 1M tokens → **fits in one A100-80GB**.

Across 8 A100s: 10M-token contexts. This is the paper's title claim — enabled entirely by the KV-cache quant scheme.

---

## Pitfalls

- **Pre-RoPE K storage doubles attention-time RoPE work.** Every dequant must apply RoPE per-position. Worth it at INT2 (1+ ppl recovery); marginal at INT4 (run KIVI's simpler post-RoPE).
- **K codebook per channel is per-layer.** Don't share across layers — per-layer activation distribution differs. ~5 KB / layer overhead.
- **Sparse path's 1% needs careful calibration.** Below 0.5% loses 0.2 ppl; above 2% throughput tanks (sparse SpMV gets expensive).
- **Q-norm quant adds a calibration step.** Per-token absmax is fine; static calibration is overkill.
- **Pre-RoPE works only if you have RoPE.** For ALiBi or absolute-position models the pre/post distinction doesn't apply; quantize the K you have.

---

## Connections

- [[excerpts/kivi]] — direct predecessor with the K/V asymmetry.
- [[excerpts/gear]] — quant + low-rank residual + sparse, complementary approach.
- [[ch-11]] — [[squeezellm]] (non-uniform weight quant) and [[spqr]] (dense-and-sparse for weights) are the algorithmic ancestors.
- [[ch-19]] / [[turboquant]] — the data-oblivious successor that eliminates the per-channel calibration via random rotation.
