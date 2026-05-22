<!-- scope: Online vector quantization with near-optimal distortion rate; KV-cache PTQ at 2.5-3.5 bits via random rotation + scalar quant + 1-bit QJL residual
     deps: [[qjl]], [[polarquant]], [[rate-distortion-theory]], [[uniform-quantization-noise]]
     see-also: [[kivi]], [[kvquant]], [[gear]], [[quarot]]
-->

# TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate
- **Core Insight:** Random-rotation + per-coordinate optimal scalar quantizer + 1-bit QJL transform on the residual gives a *data-oblivious* KV-cache quantizer that hits the rate-distortion bound up to a small constant — no calibration, no normalization overhead, online-ready, 3-bit KV with zero quality loss.
- **Guideline:** When you need training-free KV-cache compression to ≤ 3 bits at serving time, prefer TurboQuant over KIVI/KVQuant-style per-channel schemes: same or better quality, no calibration set, no per-block scale/zero-point overhead, and the rotation runs fully online.
- **Authors:** Amir Zandieh, Majid Daliri, Majid Hadian, Vahab Mirrokni (Google Research)
- **Year:** 2025 (arxiv); ICLR 2026
- **URL:** https://arxiv.org/abs/2504.19874
- **Relevant topics:** KV-cache quantization, vector quantization, rate-distortion, Johnson-Lindenstrauss, polar transform, data-oblivious, online quantization

## Abstract
Vector quantization, a problem rooted in Shannon's source coding theory, aims to quantize high-dimensional Euclidean vectors while minimizing distortion in their geometric structure. TurboQuant addresses both mean-squared error (MSE) and inner-product distortion, overcoming the failure of existing methods to reach optimal distortion rates. The algorithms are **data-oblivious** — no calibration set, suitable for online applications — and achieve near-optimal distortion rates within a small constant factor across all bit-widths and dimensions. For KV cache quantization, TurboQuant achieves absolute quality neutrality at 3.5 bits per channel and only marginal degradation at 2.5 bits per channel.

## Key Contributions
- A **two-stage** quantizer: a primary scalar quantizer that minimizes MSE per coordinate after random rotation, followed by a **1-bit Quantized Johnson-Lindenstrauss (QJL) transform on the residual** to recover inner-product unbiasedness.
- A theoretical proof that the algorithm matches the rate-distortion lower bound (Gaussian source) up to a constant factor that does not grow with dimension or bit-width.
- A **data-oblivious** construction — the rotation, scalar quantizer, and JL sketch are fixed without seeing any data; this is what makes it online-feasible for streaming KV caches.
- Empirical KV-cache results: 3-bit KV with **no measurable quality loss** on long-context benchmarks (LongBench, NIAH, RULER, ZeroSCROLLS, L-Eval) on Gemma and Mistral; 6× KV memory reduction.
- An H100 CUDA kernel achieving up to **8× speedup** over FP32 keys at 4-bit, demonstrating the rotation + per-coordinate quant pipeline is GEMM-friendly.

## Key Figures/Tables to Study
- The two-stage pipeline diagram: rotation → scalar quant → QJL residual. The cleanest one-image summary of the algorithm.
- The rate-distortion curve plot comparing TurboQuant to product quantization, RabbiQ, and per-block scalar baselines — shows TurboQuant tracks the optimal curve while baselines diverge at low bit-rates.
- The long-context evaluation table on Gemma and Mistral — the headline "3-bit with no quality loss" claim.
- H100 throughput table — 8× over FP32, ~2× over FP16 at 4-bit.

## Technical Details

### Algorithm pipeline
For a KV vector `v ∈ R^d`:

1. **Random rotation**: `v' = R · v` where `R` is a random orthogonal matrix (Hadamard-based, fixed at session start, **not data-dependent**). This induces a tightly concentrated Beta distribution on the per-coordinate magnitudes — the high-dimensional measure-concentration argument that lets one *fixed* scalar quantizer be near-optimal across all coordinates.
2. **Per-coordinate scalar quant**: apply the analytically optimal scalar quantizer for that Beta distribution to each coordinate of `v'`. Because the rotation makes coordinates near-independent and the distribution is known in closed form, no per-block scale/zero-point is needed — this is where the "zero memory overhead" claim comes from.
3. **Residual capture via 1-bit QJL** (Stage 2, only when inner-product unbiasedness is required): compute the residual `e = v' − Q(v')`, apply a JL sketch `Π` of width `m`, and store only the sign of each component: `sign(Π · e) ∈ {-1, +1}^m`. This residual is what restores unbiasedness for inner-product estimates (critical for attention-score reconstruction in KV cache).

### Inner-product estimator
For two vectors `u, v` with quantized representations `(Q(u'), sign(Π·e_u))` and likewise for `v`:
`⟨u, v⟩ ≈ ⟨Q(u'), Q(v')⟩ + α · ⟨sign(Π·e_u), Π·e_v⟩`
where the QJL leg uses the asymmetric estimator from the [[qjl]] paper — one side quantized to ±1, the other side a standard (unquantized) JL projection. The asymmetric construction is what makes the estimator unbiased with minimal additional distortion.

### Bit budget
- 3.5 bits/channel = ~3 bits for the scalar quant + ~0.5 bit/channel for the QJL residual sketch (one sign-bit per JL output dimension, with JL width `m ≈ d/2`).
- 2.5 bits/channel = drop the JL sketch width or compress the scalar code; marginal quality drop.
- **No per-block scale/zero-point stored** — this is the "zero overhead" advantage over KIVI/KVQuant which need 1–2 extra bits/number for scale & zero-point.

### Why it beats per-channel KV quant
KIVI/KVQuant store per-channel or per-token scales in FP16, which adds 1–2 bits/element of overhead. TurboQuant eliminates this by rotating the data into a coordinate frame where the distribution is *analytically known*, so the quantizer's grid is fixed in advance. The rotation is the "free" operation that makes data-oblivious work.

### Hyperparameters
| Knob | Typical | Notes |
|------|---------|-------|
| Rotation | Random Hadamard | fixed at startup; not learned |
| Scalar quant levels | 8–16 (3–4 bit) | per-channel uniform on Beta CDF |
| QJL sketch width m | ~d/2 | controls Stage-2 distortion |
| Bit budget | 2.5–3.5 bit/channel | quality-neutral at 3.5, marginal at 2.5 |
| Block size | none — fully per-channel | no per-block normalization needed |

## Connections
- [[qjl]] — Stage 2 IS the QJL transform; TurboQuant subsumes QJL into a two-stage pipeline.
- [[polarquant]] — same group's prior KV-quant paper; PolarQuant uses polar coords + angle quant, TurboQuant generalizes via the random-rotation + scalar-quant frame and adds the QJL residual.
- [[rate-distortion-theory]] — the theoretical floor TurboQuant approaches; the "near-optimal" claim is against this bound.
- [[uniform-quantization-noise]] — the per-coordinate scalar quant is justified via the same Bennett high-resolution analysis.
- [[quarot]] / [[spinquant]] — both also use rotations to flatten distributions, but for *weight/activation* PTQ; TurboQuant ports the rotation idea to *KV cache* and pairs it with a JL residual instead of round-to-nearest.
- [[kivi]] / [[kvquant]] / [[gear]] — competitor KV-cache PTQ methods that TurboQuant supersedes at ≤3-bit; useful read for understanding the per-channel-K / per-token-V baseline TurboQuant beats.

## Notes
This paper is part of the Zandieh / Mirrokni lineage: [[qjl]] (2024) → [[polarquant]] (Feb 2025) → TurboQuant (Apr 2025, ICLR 2026). Read them in that order for full theoretical context. The "data-oblivious" framing is the conceptual through-line: no calibration set, no learned parameters, no per-block normalization — everything is fixed at startup and works online.
