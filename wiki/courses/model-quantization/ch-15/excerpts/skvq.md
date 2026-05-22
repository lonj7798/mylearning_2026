---
chapter: ch-15
course: model-quantization
phase: read
excerpt_of: "SKVQ: Sliding-window Key and Value Cache Quantization for Large Language Models"
source_url: https://arxiv.org/abs/2405.06219
created_at: "2026-05-21"
---

# Excerpt: SKVQ — sliding window + channel reorder for million-token contexts

**Authors:** Haojie Duanmu, Zhihang Yuan, Xiuhong Li, Jiangfei Duan, Xingcheng Zhang, Dahua Lin
**Year:** 2024
**URL:** https://arxiv.org/abs/2405.06219
**Raw-data source:** [[raw-data/skvq]]

---

## The long-context observation

Attention queries decay sharply across token distance. The most-recent tokens carry the bulk of attention mass — a sliding window of recent tokens deserves high precision while older tokens can be compressed aggressively.

SKVQ exploits this with **three combined techniques**:

1. **Sliding-window precision schedule** — last W tokens at FP16/INT4, older history at INT2 K / INT1.5 V.
2. **Channel reorder** — permute channels so adjacent channels have similar magnitude, then group quant fits cleanly.
3. **Clipped dynamic group quantization** — per-group scale clipped at learned percentile to suppress outliers.

---

## Sliding-window precision schedule

Let W be the window size (e.g. 128). For a token at distance d from the current step:

| Distance | Precision |
|----------|-----------|
| d < W (recent) | FP16 or INT4 |
| d ≥ W (history) | INT2 K / INT1.5 V |

At each decode step the boundary slides forward by 1 — the now-too-old (d = W) token gets re-quantized down to low-bit and appended to the compressed history buffer.

```python
class SKVQ_Cache:
    def __init__(self, window_size=128):
        self.W = window_size
        self.recent_K_fp = []     # last W tokens FP16
        self.recent_V_fp = []
        self.history_K_int2 = []  # older tokens INT2 K
        self.history_V_int15 = [] # older tokens INT1.5 V

    def append(self, k, v):
        self.recent_K_fp.append(k)
        self.recent_V_fp.append(v)
        if len(self.recent_K_fp) > self.W:
            evicted_K = self.recent_K_fp.pop(0)
            evicted_V = self.recent_V_fp.pop(0)
            self.history_K_int2.append(quantize_int2(evicted_K))
            self.history_V_int15.append(quantize_int1p5(evicted_V))
```

The boundary slide costs one quantization per decode step — amortised negligible.

---

## Channel reorder

For K (and similarly V): from calibration, compute per-channel max magnitude. Sort channels in descending order. Permute the channel axis of K (and the matching weights `W_k, W_q` to preserve attention math) so similar-magnitude channels are adjacent.

After reorder, group-of-32 per-group quant fits well because the channels in each group have similar magnitudes.

```python
def calibrate_channel_perm(K_calibration):
    # K_calibration: (n_tokens, n_heads, head_dim)
    per_channel_max = K_calibration.abs().max(dim=0).values    # (H, D)
    perm = per_channel_max.argsort(descending=True, dim=-1)    # per-head perm
    return perm

# Absorb permutation into weights — zero runtime cost
def fold_perm_into_weights(W_k, W_q, perm):
    # W_k: (H*D, hidden), W_q: (H*D, hidden)
    W_k_permuted = W_k[perm]
    W_q_permuted = W_q[perm]      # same perm — preserves QK^⊤
    return W_k_permuted, W_q_permuted
```

This is the same trick [[duquant]] (ch-14) uses on activations — zero runtime cost, all benefit. Improves group-of-32 quant by 0.3–0.5 ppl.

---

## Clipped dynamic group quantization

Per group g of size 32, compute the absmax `m_g`. Clip at the c-th percentile of `{m_g}`:

```
m̂_g = min(m_g, percentile_c({m_g}))
scale_g = m̂_g / (2^{b-1} − 1)
```

c=99 typical — kills the single-group outliers that would otherwise dominate the scale.

The clipped values that exceed `m̂_g` are simply truncated to ±max — they take a hit but the **other 99% of groups** keep tight precision.

---

## V at 1.5 bits

1.5 bits = pack pairs of V elements into a single 3-bit code from an 8-entry codebook (vector quant of pairs). Codebook fit per-token via small k-means.

```
V[t, c], V[t, c+1] → 3-bit index into 8-entry codebook
                     → effectively 1.5 bits per element
```

Why 1.5 bits works for V: V is averaged by attention weights → its individual precision matters less than its aggregate contribution. Pair-wise VQ captures the joint distribution of adjacent channels well.

---

## Bit budgets

```
K: 2 bits + FP16 group scale per 32  → 2.5 bits/element
V: 1.5 bits + FP16 codebook per group → 1.75 bits/element
Recent window (W=128): FP16, contributes ≈ 0 to amortised cost for T >> W
```

Average: ~2 bits/element across K and V over long contexts.

---

## Throughput numbers

LLaMA-7B at 1M context: KV at FP16 = 250 GB (impossible on 80 GB). KV at SKVQ-2/1.5 = ~30 GB, **fits**. **Decoding speedup 7×** because each attention call reads 8× less HBM.

LongBench tasks within 1–2 points of FP16 at full compression.

---

## Pitfalls

- **Boundary discontinuity** at t = T - W. The token transitioning from FP16 to INT2 K loses precision suddenly; some attention spikes can occur. If observed, use a graded precision schedule (FP16 → INT4 → INT2) over multiple boundaries.
- **Channel reorder must be applied to W_q and W_k consistently** to preserve the QK^⊤ math. Forgetting to update W_q corrupts attention.
- **Per-token codebook for V at 1.5 bits** adds per-token k-means cost — amortised negligible for long contexts but expensive for short ones; SKVQ shines only above 32K tokens.
- **Window size W trades quality vs memory.** W=128 is a good default; W=512 helps short-attn tasks but eats memory.
- **Re-thresholding clip percentile** under distribution shift. The c=99 percentile is calibration-fit; long-tail prompts may need re-calibration.

---

## Connections

- [[excerpts/kivi]] — the axis-asymmetry recipe SKVQ builds on for the history compression.
- [[excerpts/kvquant]] — alternative sub-2-bit path; KVQuant emphasises codebook + sparse, SKVQ emphasises recency + reorder.
- [[ch-14]] — [[duquant]] is the channel-reorder ancestor for activations.
- StreamingLLM (architecture, not quant) is the recency-bias predecessor.
