---
chapter: ch-18
course: model-quantization
phase: read
excerpt_of: "TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate (Zandieh, Daliri, Hadian, Mirrokni, ICLR 2026)"
source_url: https://arxiv.org/abs/2504.19874
created_at: "2026-05-21"
---

# Excerpt: TurboQuant — two-stage pipeline matching rate-distortion bound

**Authors:** Amir Zandieh, Majid Daliri, Majid Hadian, Vahab Mirrokni (Google Research)
**Year:** 2025 (Apr arxiv); ICLR 2026
**URL:** https://arxiv.org/abs/2504.19874
**Raw-data source:** [[raw-data/turboquant]]

---

## The synthesis paper

[[qjl]] (Jun 2024) → [[polarquant]] (Feb 2025) → **TurboQuant (Apr 2025, ICLR 2026)**. Random-rotation + per-coordinate optimal scalar quantizer + 1-bit QJL transform on the residual gives a *data-oblivious* KV-cache quantizer that **hits the rate-distortion bound up to a small constant** — no calibration, no normalization overhead, online-ready.

---

## The two-stage pipeline

For a KV vector `v ∈ R^d`:

### Stage 1 — random rotation + per-coordinate scalar quant

`v' = R · v` where `R` is a random orthogonal matrix (Hadamard-based, fixed at session start, **not data-dependent**).

After rotation, the per-coordinate magnitudes follow a tightly concentrated Beta distribution — the same high-dimensional measure-concentration argument as PolarQuant. This lets one *fixed* scalar quantizer be near-optimal across all coordinates.

Apply the analytically optimal scalar quantizer for that Beta to each coordinate. **No per-block scale/zero-point is needed** — the rotation made it unnecessary. This is the "zero memory overhead" claim.

### Stage 2 — 1-bit QJL on the residual

Compute the residual: `e = v' − Q(v')`.

Apply a JL sketch `Π` of width `m`, store only the sign of each component: `sign(Π · e) ∈ {-1, +1}^m`.

This residual is what restores **unbiasedness for inner-product estimates** — critical for attention-score reconstruction. Stage 1 alone is biased for inner products (the scalar quant error has nonzero mean in general); Stage 2 corrects this via the asymmetric estimator from [[qjl]].

---

## The inner-product estimator

For two vectors `u, v` with TurboQuant representations:

```
⟨u, v⟩ ≈ ⟨Q(u'), Q(v')⟩ + α · ⟨sign(Π·e_u), Π·e_v⟩
        ─────── Stage 1 ────── + ──────── Stage 2 correction ────────
```

The QJL leg uses the **asymmetric estimator** — one side quantized to ±1, the other side a standard (unquantized) JL projection. The asymmetric construction is what makes the estimator unbiased with minimal additional distortion.

---

## Hitting the rate-distortion bound

Standard Shannon rate-distortion theory ([[rate-distortion-theory]], ch-01) gives a lower bound `D ≥ σ² · 2^(-2R)` for Gaussian sources at rate `R` bits per coordinate.

TurboQuant's theoretical contribution: **the two-stage random-rotation + scalar + QJL pipeline matches this bound up to a constant factor that does not grow with dimension or bit-width.**

This is the strongest possible guarantee for a fixed-rate quantizer. You cannot do better than a constant-factor improvement over TurboQuant within the data-oblivious class.

---

## Bit budget

| Component | Bits per channel | Notes |
|-----------|------------------|-------|
| Stage 1 scalar quant | ~3 | per-coord, uniform on Beta CDF |
| Stage 2 QJL residual | ~0.5 | m ≈ d/2; one sign-bit per JL output |
| **Total** | **~3.5** | quality-neutral |
| Per-block scale/zero | **0** | the whole point |

- **3.5 bits/channel = quality-neutral** (matches FP16 on long-context evals).
- **2.5 bits/channel = marginal degradation** (small drop on RULER / NIAH).

---

## Empirical headlines

- **3-bit KV with no measurable quality loss** on long-context benchmarks (LongBench, NIAH, RULER, ZeroSCROLLS, L-Eval) on Gemma and Mistral.
- **6× KV memory reduction.**
- **H100 CUDA kernel: up to 8× speedup over FP32 keys at 4-bit**, ~2× over FP16 at 4-bit — demonstrating the rotation + per-coordinate quant pipeline is GEMM-friendly.

---

## Why it beats per-channel KV quant

KIVI / KVQuant store per-channel or per-token scales in FP16, which adds 1–2 bits/element of overhead. TurboQuant eliminates this by rotating the data into a coordinate frame where the distribution is *analytically known*, so the quantizer's grid is fixed in advance. **The rotation is the "free" operation that makes data-oblivious work.**

This is the conceptual unification: KIVI's per-channel scales were trying to absorb the channel-localized outlier structure of K. TurboQuant rotates that structure away, so the *same* fixed quantizer fits all coordinates.

---

## Hyperparameters

| Knob | Typical | Notes |
|------|---------|-------|
| Rotation | Random Hadamard | fixed at startup; not learned |
| Scalar quant levels | 8–16 (3–4 bit) | per-channel uniform on Beta CDF |
| QJL sketch width m | ~d/2 | controls Stage-2 distortion |
| Bit budget | 2.5–3.5 bit/channel | quality-neutral at 3.5, marginal at 2.5 |
| Block size | **none** | fully per-channel; no per-block normalization |

---

## Implementation reference

The pseudocode you should be able to write from the paper:

```python
# Initialization (once, at session start)
R = random_hadamard(d)                # rotation
Q = optimal_scalar_quant_for_beta(d)  # Stage 1 quantizer
Pi = random_jl(m, d)                  # Stage 2 sketch matrix
alpha = jl_scaling_constant(d, m)

# Per-vector store (during prefill / decode)
def store_kv(v):
    v_prime = R @ v
    v_q = Q.quantize(v_prime)
    e = v_prime - Q.dequantize(v_q)
    s = sign(Pi @ e)
    return (v_q, s)  # NO per-block scale

# Inner-product estimation (during attention scoring)
def estimate_inner_product(u, stored_v):
    v_q, s = stored_v
    u_prime = R @ u
    stage1 = u_prime @ Q.dequantize(v_q)
    e_v_proj = Pi @ (u_prime - Q.dequantize(v_q))  # asymmetric: real on this side
    stage2 = alpha * (s * e_v_proj).sum()
    return stage1 + stage2
```

---

## Connections

- [[qjl]] / [[excerpts/qjl]] — Stage 2 IS the QJL transform; TurboQuant subsumes QJL into a two-stage pipeline.
- [[polarquant]] / [[excerpts/polarquant]] — same group's prior KV-quant paper; PolarQuant uses polar coords + angle quant, TurboQuant generalizes via the random-rotation + scalar-quant frame and adds the QJL residual.
- [[rate-distortion-theory]] / ch-01 — the theoretical floor TurboQuant approaches; the "near-optimal" claim is against this bound.
- [[uniform-quantization-noise]] / ch-01 — Bennett high-resolution analysis; justifies per-coordinate scalar quant in Stage 1.
- [[quarot]] / [[spinquant]] / ch-14 — both use rotations to flatten distributions, but for weight/activation PTQ; TurboQuant ports the rotation idea to KV cache and pairs it with a JL residual instead of round-to-nearest.
- [[kivi]] / [[kvquant]] / [[gear]] / ch-15 — competitor KV-cache PTQ methods that TurboQuant supersedes at ≤ 3-bit.
- [[ch-18]] — parent synthesis.
- [[ch-22]] — capstone candidate; this is one of the recommended reproduction targets.
