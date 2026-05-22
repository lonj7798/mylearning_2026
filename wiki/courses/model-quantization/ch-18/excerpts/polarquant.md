---
chapter: ch-18
course: model-quantization
phase: read
excerpt_of: "PolarQuant: Quantizing KV Caches with Polar Transformation (Han, Kacham, Karbasi, Mirrokni, Zandieh, AISTATS 2026)"
source_url: https://arxiv.org/abs/2502.02617
created_at: "2026-05-21"
---

# Excerpt: PolarQuant — polar coords with closed-form angle distribution

**Authors:** Insu Han, Praneeth Kacham, Amin Karbasi, Vahab Mirrokni, Amir Zandieh (Google Research / Yale)
**Year:** 2025 (Feb arxiv); AISTATS 2026
**URL:** https://arxiv.org/abs/2502.02617
**Raw-data source:** [[raw-data/polarquant]]

---

## The middle paper in the trilogy

[[qjl]] (Jun 2024) → **PolarQuant (Feb 2025)** → [[turboquant]] (Apr 2025, ICLR 2026). QJL targeted inner-product preservation; PolarQuant targets *reconstruction*.

---

## The recursive pair-wise polar transform

For `v ∈ R^d`:

1. **Random preconditioning:** `v' = R · v` (R = random orthogonal, typically randomized Hadamard).
2. **Pair coordinates:** `(v'_1, v'_2), (v'_3, v'_4), …, (v'_{d-1}, v'_d)`.
3. **Convert each pair to polar:** `r_i = √(v'_{2i-1}² + v'_{2i}²)`, `θ_i = atan2(v'_{2i}, v'_{2i-1})`.
4. **Recurse on the radii:** treat `(r_1, r_2, …)` as a new vector and repeat the pair-wise polar conversion until a single global radius and a tree of angles remain.

`O(d log d)`, fully parallelizable.

After the recursion: one global radius `r_top` + a tree of angles `{θ_i}`. Quantize the angles, store one FP16 radius. Done.

---

## The closed-form angle distribution — the central theorem

The load-bearing claim: **after random orthogonal preconditioning, the angle distribution is closed-form (a specific Beta-style density that depends only on `d`).**

### Why

Without preconditioning, the angles `θ_i` of arbitrary KV vectors have a distribution that depends on the input — so a fixed quantizer would be far from optimal.

After multiplying by random orthogonal `R`, the rotated vector `v'` looks **isotropic** — its direction is uniform on the unit sphere. This is the high-dimensional measure-concentration argument: in high `d`, rotating a generic vector by a random `R` produces a vector whose distribution depends only on `‖v‖` and `d`, not on the specific entries of `v`.

For isotropic `v'`, each pair-wise polar angle `θ_i` follows a closed-form Beta-style density that depends only on `d`, not on the input data. This is the "data-oblivious" hook: one quantizer fits all inputs.

### Quantizer construction

The optimal scalar quantizer for the closed-form angle density comes from [[lloyd-max-quantizer]] applied to that specific Beta — precomputable, no online calibration.

---

## Bit budget

- **Per angle:** ~k bits (typical k = 3–4).
- **Per radius:** at the top of the recursion tree, one global FP16 radius is stored.
- **Overhead:** zero per-block scale/zero-point — only the angle codes + one radius.
- **Net effective rate:** < 4 bits/element at **4.2× compression** vs FP16.

---

## Comparison to per-block PTQ

| Method | Per-block overhead | Calibration | Best bit-width |
|--------|--------------------|-------------|----------------|
| KIVI | 1 scale + 1 zero-point per block (FP16) | yes | 2-bit (per-channel K, per-token V) |
| KVQuant | dense-and-sparse split, FP16 outliers | yes | sub-2-bit |
| **PolarQuant** | **none** (1 global radius per vector) | **no** | **< 4 bits → 4.2×** |

**Result:** > 4.2× KV-cache compression with the **best long-context quality scores** vs prior SOTA at the same bit-rate.

---

## Why PolarQuant matters for the chapter

It's the *reconstruction-targeted* member of the trilogy. QJL targets inner products (one-sided estimator); PolarQuant targets the V cache where you sometimes need accurate retrieval, not just attention-score-weighted readout. TurboQuant combines both via the two-stage pipeline.

PolarQuant is also the cleanest *proof of the data-oblivious thesis*: the explicit closed-form angle density makes the "no calibration needed" claim concrete. Read this paper for the *theoretical* statement of why rotation enables data-oblivious quant, then read [[qjl]] and [[turboquant]] for the inner-product engineering.

---

## Connections

- [[qjl]] / [[excerpts/qjl]] — same group; QJL targets inner-product preservation via sign-bit JL; PolarQuant targets reconstruction via polar coords. Complementary preconditioning ideas.
- [[turboquant]] / [[excerpts/turboquant]] — direct successor that generalizes random-rotation + scalar-quant into a two-stage framework with QJL residual.
- [[quarot]] / [[spinquant]] / ch-14 — rotation-based weight/activation PTQ; same insight (rotation flattens distribution) applied to a different tensor.
- [[kivi]] / [[kvquant]] / ch-15 — the per-channel / per-token baselines PolarQuant supersedes via overhead elimination.
- [[rate-distortion-theory]] / ch-01 — the optimal scalar quantizer for the closed-form angle distribution comes from Lloyd-Max applied to that specific density.
- [[lloyd-max-quantizer]] / ch-03 — the algorithm used to derive the optimal angle quantizer.
- [[ch-18]] — parent synthesis.
