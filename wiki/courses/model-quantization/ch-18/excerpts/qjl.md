---
chapter: ch-18
course: model-quantization
phase: read
excerpt_of: "QJL: 1-Bit Quantized JL Transform for KV Cache Quantization with Zero Overhead (Zandieh, Daliri, Han, Google Research 2024)"
source_url: https://arxiv.org/abs/2406.03482
created_at: "2026-05-21"
---

# Excerpt: QJL — 1-bit JL sketch with asymmetric estimator

**Authors:** Amir Zandieh, Majid Daliri, Insu Han (Google Research)
**Year:** 2024 (June)
**URL:** https://arxiv.org/abs/2406.03482
**Raw-data source:** [[raw-data/qjl]]

---

## The earliest paper in the data-oblivious trilogy

[[qjl]] (June 2024) → [[polarquant]] (Feb 2025) → [[turboquant]] (April 2025, ICLR 2026). QJL is where the idea starts: **eliminate per-block metadata by exploiting sign-bit JL sketches + an asymmetric inner-product estimator**.

---

## The sketch

For a vector `v ∈ R^d` and a fixed random JL projection matrix `Π ∈ R^{m × d}` (entries iid `±1/√m` or Gaussian):

- **Stored** (cached side): `s(v) = sign(Π · v) ∈ {-1, +1}^m`. **One bit per output coordinate. No scale. No zero-point.**
- **Live** (query side): `Π · u` computed full-precision on the query vector `u`.

`Π` is fixed at startup and shared globally — no per-vector, no per-block state.

---

## The asymmetric inner-product estimator

The thing being preserved is the inner product `⟨u, v⟩` — what attention's `softmax(QK^T)` actually consumes.

```
⟨u, v⟩ ≈ α · ⟨s(v), Π · u⟩ = α · Σ_j s(v)_j · (Π · u)_j
```

where `α` is a known scaling constant depending on `d`, `m`, and the JL distribution.

### Why asymmetric

If both sides were sign-only — `⟨sign(Π·u), sign(Π·v)⟩` — the estimator would be **biased**. By the classic identity:

```
E[sign(Π·u)_j · sign(Π·v)_j] = (2/π) · arcsin(⟨u, v⟩ / (‖u‖·‖v‖))
```

That gives the angle (up to a `2/π` factor), not the inner product. A symmetric sign-only sketch reconstructs the angle but loses the magnitude.

The asymmetric construction — sign on one side, full-precision on the other — undoes the `arcsin` nonlinearity and yields an **unbiased** estimator with minimal additional variance.

This is the load-bearing trick. It's what makes "1 bit per JL output" actually work for attention scoring.

---

## Zero metadata overhead

Traditional KIVI / KVQuant store, per block of `B` elements:

- `B · k` bits for quantized values (`k` = bit-width).
- 1 scale (FP16, 16 bits) + 1 zero-point (FP16, 16 bits) = 32 bits per block.
- → `32/B` extra bits per element.

For `B = 32` and `k = 3`, the overhead alone adds **1 bit/element** — meaning "3-bit" KV is really ~4-bit effective.

QJL stores `m` sign bits per vector and *nothing else*. The scaling constant `α` is fixed at startup (depends only on dimensions), not per-block. Effective rate: pure 1 bit per JL-output coordinate, with `m ≈ d/4` to `d/2` giving "3-bit equivalent" effective rate per input coordinate.

---

## Bit budget

| Sketch width m | Effective bits / input coord | Memory reduction |
|----------------|------------------------------|-------------------|
| m = d/4 | ~0.25 bit | ~16× over FP16 |
| m = d/2 | ~0.5 bit | ~8× over FP16 |
| m = d | ~1 bit | ~5× over FP16 (paper headline "3-bit equivalent") |

**Result:** > 5× KV memory reduction at 3-bit equivalent with **no measured accuracy loss** on NLP benchmarks.

---

## CUDA kernel

The released kernel fuses:

1. JL projection of K.
2. Sign quantization.
3. Packing into bytes.
4. At decode time, the unpacked `±1` signs are multiplied by `Π · q` for the current query.

This avoids materializing the dequantized K cache and saves both memory bandwidth and SM time. Beats unquantized FP16 attention on long contexts.

---

## What QJL targets and what it doesn't

**Targets:** inner-product preservation — attention-score reconstruction. The asymmetric estimator is unbiased for `⟨u, v⟩`.

**Doesn't target:** vector reconstruction. If you need `v ≈ decode(s(v))` accurate in MSE, QJL is the wrong tool — you only have `m` sign bits, which is information-theoretically insufficient to reconstruct `v` itself.

This is why [[polarquant]] (reconstruction) and [[turboquant]] (both, via two-stage pipeline) exist — different goals, different sketches.

---

## Connections

- [[polarquant]] / [[excerpts/polarquant]] — sister paper from the same group; uses polar coords instead of JL signs. Targets reconstruction.
- [[turboquant]] / [[excerpts/turboquant]] — direct successor; subsumes QJL as **Stage 2** of a two-stage pipeline (scalar quant + 1-bit QJL residual).
- [[kivi]] / [[kvquant]] / ch-15 — per-channel/per-token baselines that pay the per-block metadata QJL eliminates.
- [[rate-distortion-theory]] / ch-01 — sign-only is the 1-bit limit of any scalar quantizer; QJL gets away with it via the asymmetric estimator and high-dimensional concentration.
- [[ch-18]] — parent synthesis.
