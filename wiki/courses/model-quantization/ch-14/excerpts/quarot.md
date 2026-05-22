---
chapter: ch-14
course: model-quantization
phase: read
excerpt_of: "QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs"
source_url: https://arxiv.org/abs/2404.00456
created_at: "2026-05-21"
---

# Excerpt: QuaRot — end-to-end Hadamard rotation for W4A4KV4

**Authors:** Saleh Ashkboos, Amirkeivan Mohtashami, Maximilian L. Croci, Bo Li, Pashmina Cameron, Martin Jaggi, Dan Alistarh, Torsten Hoefler, James Hensman
**Year:** 2024
**URL:** https://arxiv.org/abs/2404.00456
**Raw-data source:** [[raw-data/quarot]]

---

## What QuaRot adds to QuIP

[[quip]] (ch-13) rotates weights only. QuaRot moves the rotation **into the residual stream**, so activations and the KV cache become quantizable too. The trick is **computational invariance**: insert orthogonal Q in matched positions such that the FP outputs are unchanged.

For a residual block `y = x + f(x)` and orthogonal Q:

```
y = Q x + Q · f(Q^⊤ · Q x) = Q (x + f(x))
```

The residual stream is rotated by Q **everywhere** after insertion; the LM head absorbs Q^⊤ at the end. Logits bit-identical in FP.

---

## Folding Q into weights (offline)

For a transformer block with `W_q, W_k, W_v, W_o` (attention) and `W_up, W_gate, W_down` (FFN), the rotation Q folds in:

```
E       ← E Q^⊤                  # embeddings
W_q     ← W_q Q^⊤                # input is Q x → unrotate before projection
W_k     ← W_k Q^⊤
W_v     ← W_v Q^⊤
W_o     ← Q W_o                  # output re-enters rotated residual
W_up    ← W_up Q^⊤
W_gate  ← W_gate Q^⊤
W_down  ← Q W_down
W_lm    ← W_lm Q                 # final unrotation
```

After folding, the residual-stream rotation Q never appears at inference — zero runtime cost for R1.

---

## The four rotation slots

| Slot | Where | Cost | Purpose |
|------|-------|------|---------|
| **R1** | residual stream | offline-folded, free | per-token outliers everywhere |
| **R2** | between V and W_o | online FWHT | per-head V outliers |
| **R3** | between SwiGLU output and W_down | online FWHT | gated-activation spikes |
| **R4** | on K after RoPE | online FWHT (fused with attn) | K-cache outliers |

R2/R3/R4 are *not* foldable because they sit between two operations that compose differently than the residual pattern. They run online as Fast Walsh-Hadamard Transforms, `O(d \log d)` per token, fused into the surrounding kernels.

---

## Why a Hadamard kills outliers

For a vector x with one outlier of magnitude M among d coordinates:

```math
(H_d x)_i \approx \frac{M}{\sqrt{d}} \quad \text{for all } i
```

The outlier energy is spread uniformly across all d coordinates → the post-rotation max coordinate is reduced by ~√d.

| Hidden dim | d | √d reduction |
|---|---|---|
| LLaMA-2-7B residual | 4096 | 64× |
| LLaMA-2-70B residual | 8192 | 90× |
| LLaMA-2-70B FFN intermediate | 28672 | 169× |

A 100× outlier becomes ~1.1× the bulk after rotation in the 70B residual stream. INT4 RTN now works.

---

## The quantization stack on top of rotation

- **Weights**: GPTQ at 4-bit, group size 128 (or per-channel).
- **Activations**: dynamic per-token RTN at 4-bit, symmetric.
- **KV cache**: per-head, per-token at 4-bit (no group along channels).

Each is the simplest possible quantizer at its precision target. The rotation does the heavy lifting; the quantizers fall back to round-to-nearest.

---

## Inference data flow (one block)

```python
def quarot_block_forward(x_rotated):
    # x_rotated is already in the Q-frame (from previous block / embedding)
    h = rmsnorm(x_rotated)                     # rotation commutes with RMS up to scale
    h_q4 = dynamic_quant_4bit(h)               # per-token INT4
    q, k, v = w_qkv_rotated @ h_q4             # W4A4 GEMM
    k = rope_then_R4(k)                        # online FWHT R4 on K
    v = R2_rotation(v)                         # online FWHT R2 on V (per head)
    kv_cache.store_int4(k, v)                  # KV4
    a = attention(q, kv_cache)                 # quantized attention
    o = w_o_rotated @ a                        # W_o has R2^⊤ folded in
    x_rotated = x_rotated + o                  # residual

    # FFN (analogous, with R3 between SwiGLU and W_down)
    return x_rotated
```

The Q rotation never reappears explicitly because it's folded into every weight. R2/R3/R4 are the only online FWHTs.

---

## Numbers — first lossless W4A4KV4

LLaMA-2-70B WikiText-2:

| Method | W | A | KV | ppl | Δppl vs FP16 |
|--------|---|---|----|----|--------------|
| FP16 | 16 | 16 | 16 | 3.32 | — |
| SmoothQuant W4A4 | 4 | 4 | 16 | NaN | collapse |
| OmniQuant W4A4 | 4 | 4 | 16 | 6.11 | +2.79 |
| QuIP (weights only) | 4 | 16 | 16 | 3.42 | +0.10 |
| **QuaRot W4A4KV4** | **4** | **4** | **4** | **3.79** | **+0.47** |

LLaMA-2-7B/13B/70B all retain ≥ 99% of zero-shot accuracy. QuaRot is the first method where the full W4A4KV4 stack works at 70B.

---

## Pitfalls

- **RMSNorm and the rotation don't perfectly commute.** RMSNorm scales by `1/sqrt(mean(x²))`; the rotation preserves the mean-square so it commutes for *unscaled* norms. With learned gain γ, fold γ into adjacent weights before applying Q.
- **`d_in` must be a power of 2** for the FWHT. For irregular dims, block the rotation into power-of-2 chunks; you lose a small amount of incoherence quality.
- **R4 must be applied *after* RoPE.** Pre-RoPE K still has the channel-aligned outlier structure; post-RoPE it's rotated by the position. R4 is applied after RoPE to flatten what remains.
- **The KV cache stores already-R2/R4-rotated K, V.** Subsequent attention reads in the rotated frame and the dot-product math works out without explicit unrotation.
- **Don't combine QuaRot with outlier sidecars** (QUIK-style). The rotation already kills the outliers; the sidecar wastes INT8 channels on a now-flat distribution.

---

## Connections

- [[excerpts/quip]] (ch-13) — algorithmic ancestor; QuIP rotates weights only, QuaRot extends to activations + KV.
- [[excerpts/spinquant]] — same insertion graph but learns Q via Cayley parametrization.
- [[ch-15]] — KIVI / KVQuant are the alternative KV-quant paths; QuaRot's R4 makes per-token KV4 RTN work.
- [[ch-19]] — Marlin / Machete kernels for the W4 weight GEMM; QuaRot bolts on top.
