---
chapter: ch-15
course: model-quantization
phase: read
excerpt_of: "KIVI: A Tuning-Free Asymmetric 2-bit Quantization for KV Cache"
source_url: https://arxiv.org/abs/2402.02750
created_at: "2026-05-21"
---

# Excerpt: KIVI — per-channel K + per-token V at INT2

**Authors:** Zirui Liu, Jiayi Yuan, Hongye Jin, Shaochen Zhong, Zhaozhuo Xu, Vladimir Braverman, Beidi Chen, Xia Hu
**Year:** 2024
**Venue:** ICML 2024
**URL:** https://arxiv.org/abs/2402.02750
**Raw-data source:** [[raw-data/kivi]]

---

## The load-bearing empirical finding

The KIVI paper's central contribution is a **distribution study** of K and V tensors across LLaMA-2, Falcon, Mistral. The finding:

- **K cache** has **channel-wise outliers** — a few specific channels are 10–100× larger than the rest, and *the same channels every token*.
- **V cache** has **token-wise variation** — no consistent channel-outlier pattern, but specific tokens dominate.

This is the chapter's "Figure 2" moment — a heatmap showing K(token, channel) magnitudes with persistent vertical bands (channel outliers) and V(token, channel) magnitudes with no consistent vertical structure.

---

## The causal story

`K_t = W_k · x_t`. The input `x_t` is the residual stream, which carries the LLM.int8() outlier features — a small number of consistent channels of magnitude 10–100× the bulk. `W_k` is a learned projection; empirically it preserves the channel structure of its input.

→ Across all tokens t, the *same* channels of K are large. By construction of how transformers work.

V is structurally similar. **But** V is used at attention time as a weighted average: `out_q = Σ_t a_{q,t} V_t`. The attention weights `a_{q,t}` re-mix the channel pattern across tokens; over the softmax average, channel-wise outliers wash out. What dominates V's contribution is **which token** is being attended, not which channel → token-wise variation dominates.

---

## The recipe

For a KV tensor of shape `(n_tokens, n_heads, head_dim)`:

- **K cache**: per-channel quantization, group size 32 along token axis. Each channel `c` has its own scale `s_K[c]` per chunk of 32 tokens.
- **V cache**: per-token quantization, group size 32 along channel axis. Each token `t` has its own scale `s_V[t]` per chunk of 32 channels.

Both at INT2, both tuning-free (no calibration / fine-tuning).

---

## The streaming implementation

Decode-time KV cache grows by one token per step. KIVI maintains:

- A small **FP16 residual buffer** of the last `< g = 32` tokens.
- A **quantized INT2 packed past** for older tokens.

When the residual buffer fills, the chunk is quantized using its own statistics and appended.

```python
class KIVI_Cache:
    def __init__(self, g=32):
        self.g = g
        self.K_int2 = []          # list of (g, H, D) INT2 chunks
        self.V_int2 = []
        self.K_scales = []        # list of (H, D) FP16 per-channel scales
        self.V_scales = []        # list of (g, H) FP16 per-token scales
        self.K_residual = []      # last <g tokens in FP16
        self.V_residual = []

    def append(self, k, v):
        self.K_residual.append(k)
        self.V_residual.append(v)
        if len(self.K_residual) == self.g:
            self._flush()

    def _flush(self):
        chunk_K = torch.stack(self.K_residual)         # (g, H, D)
        chunk_V = torch.stack(self.V_residual)
        s_K = chunk_K.abs().amax(dim=0)                # (H, D), per-channel
        s_V = chunk_V.abs().amax(dim=-1)               # (g, H), per-token
        self.K_int2.append(quantize_sym_int2(chunk_K, s_K))
        self.V_int2.append(quantize_sym_int2(chunk_V, s_V))
        self.K_scales.append(s_K)
        self.V_scales.append(s_V)
        self.K_residual.clear()
        self.V_residual.clear()
```

---

## Attention math

```
attn_logits = Q · K^⊤ / √d
            = Q · (dequant(K_int2, s_K))^⊤ / √d         # per-channel scale
out         = attn_probs · V
            = attn_probs · dequant(V_int2, s_V)         # per-token scale
```

K dequant: one FP16 multiply per `(channel, chunk)`. V dequant: one FP16 multiply per `(token, chunk_of_channels)`. Both fold into the fused-attention kernel — no materialisation.

---

## Bit budget

INT2 + group 32 + FP16 scale: `2 + 16/32 = 2.5 bits/element`.
INT4 + group 32 + FP16 scale: `4 + 16/32 = 4.5 bits/element`.

The 0.5-bit scale overhead is fixed; the marginal cost of going from INT4 → INT2 is exactly the 2 bits per value.

---

## The numbers

LLaMA-2-7B WikiText-2 (lower = better):

| KV bits | Method | ppl | Δppl |
|---------|--------|-----|------|
| FP16 | — | 5.47 | — |
| INT4 | per-token uniform | 5.65 | +0.18 |
| INT4 | KIVI (per-ch K, per-tok V) | 5.49 | +0.02 |
| INT2 | per-token uniform | 12.4 | +6.9 (collapse) |
| INT2 | per-channel uniform | 8.3 | +2.8 |
| **INT2** | **KIVI** | **7.0** | **+1.5** |

The right axis is worth 5.4 ppl at INT2.

**Throughput**: 2.6× peak memory reduction, 4× larger batch size, **2.35–3.47× decoding throughput** vs FP16 KV.

---

## Ablation — K axis × V axis at INT2

| K axis | V axis | ppl |
|--------|--------|-----|
| per-channel | per-token | **7.0** ← KIVI |
| per-token | per-token | 12.4 |
| per-channel | per-channel | 8.3 |
| per-token | per-channel | 14+ |

K axis is the bigger lever (5.4 ppl). V axis is smaller (1.3 ppl). Both wrong: catastrophic.

---

## Pitfalls

- **Per-token V quant breaks with padded positions.** Mask out padded tokens before computing per-token V scales.
- **Group size 32 is balanced.** Smaller groups (16) → more accurate, more scale overhead. Larger (64, 128) → less overhead, more in-group variance hurts INT2.
- **KIVI's K is post-RoPE.** KVQuant later shows pre-RoPE K quantization gains another ~1 ppl at INT2 — but at the cost of dequant-then-RoPE at attention time. KIVI's post-RoPE choice is the simpler default.
- **GQA changes channel count per head.** For LLaMA-2-70B (8 KV heads), each head has 128 channels — still plenty of channels for KIVI's per-channel K to work. For models with very small per-head channel counts (e.g. < 32), per-channel K may not have enough channels to amortise.
- **Streaming buffer overhead.** The FP16 residual buffer is `< g = 32` tokens × full KV size; at 32 tokens of FP16 it's negligible. Don't be tempted to set g = 8 (more accurate but more residual overhead).

---

## Connections

- [[excerpts/kvquant]] — pre-RoPE K quant + sub-2-bit via non-uniform + dense-and-sparse.
- [[excerpts/gear]] — quant + low-rank residual + sparse, complementary technique.
- [[excerpts/skvq]] — sliding window for million-token contexts.
- [[ch-07]] — [[llm-int8]] is the residual-stream outlier-channel ancestor.
- [[ch-14]] — [[quarot]]'s R4 (K-cache Hadamard after RoPE) is the rotation-based alternative; complements KIVI rather than replacing it.
