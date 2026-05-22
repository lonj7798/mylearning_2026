<!-- scope: KV-cache quantization via polar-coordinate transform on randomly preconditioned vectors; angle quantization with analytically known distribution, no per-block scale
     deps: [[qjl]], [[rate-distortion-theory]]
     see-also: [[turboquant]], [[kivi]], [[kvquant]]
-->

# PolarQuant: Quantizing KV Caches with Polar Transformation
- **Core Insight:** After random preconditioning, the angles in the polar representation of KV vectors follow a tightly concentrated, **analytically computable** distribution, which means a single fixed quantizer for the angles eliminates the per-block scale/zero-point overhead that traditional KV-cache PTQ pays.
- **Guideline:** Use polar-coordinate quant when you want KV-cache compression > 4× without calibration data, especially for long-context regimes where per-block normalization metadata becomes a meaningful fraction of the stored cache.
- **Authors:** Insu Han, Praneeth Kacham, Amin Karbasi, Vahab Mirrokni, Amir Zandieh (Google Research / Yale)
- **Year:** 2025 (arxiv Feb); AISTATS 2026
- **URL:** https://arxiv.org/abs/2502.02617
- **Relevant topics:** KV-cache quantization, polar coordinates, random preconditioning, data-oblivious quant, long-context

## Abstract
Large language models require significant memory to store Key-Value (KV) embeddings in the KV cache, especially when handling long-range contexts. Quantization of these KV embeddings is a common technique to reduce memory consumption. PolarQuant employs **random preconditioning + polar transformation**: it transforms KV embeddings into polar coordinates using an efficient recursive algorithm and then quantizes the resulting angles. The key insight is that, after random preconditioning, the angles in the polar representation exhibit a **tightly bounded and highly concentrated distribution with an analytically computable form**. This eliminates the need for explicit normalization (zero-point + scale), a step that traditional methods require which introduces significant memory overhead because quantization parameters must be stored in full precision per data block. PolarQuant bypasses normalization, enabling substantial memory savings. The long-context evaluation shows PolarQuant compresses the KV cache by over **4.2×** while achieving the best quality scores compared to SOTA methods.

## Key Contributions
- A **recursive pair-wise polar transformation**: process the `d`-dimensional vector by recursively converting pairs of coordinates `(x, y) → (r, θ)`, eventually yielding one radius and a vector of angles.
- A proof that, after random orthogonal preconditioning, the **angle distribution is closed-form** (a specific Beta-style density that depends only on `d`) — so the optimal angle quantizer is computable in advance without seeing data.
- Elimination of the **per-block scale/zero-point overhead** that KIVI/KVQuant pay (those add 1–2 bits/number); PolarQuant stores only the quantized angle indices plus one global radius.
- > 4.2× KV-cache compression with the best long-context quality vs prior PTQ KV quant methods at the same bit-rate.

## Key Figures/Tables to Study
- The recursive polar-transform diagram: how `d`-dim vector gets folded pair-wise into `(r, θ_1, θ_2, …)`.
- The closed-form angle-distribution plot after random preconditioning — the empirical histogram matches the analytic Beta perfectly, which is the whole basis for "no normalization needed."
- The long-context quality vs compression curve vs KIVI / KVQuant.

## Technical Details

### Pair-wise polar transform
For a vector `v ∈ R^d` after preconditioning `v' = R · v`:
1. Pair coordinates: `(v'_1, v'_2), (v'_3, v'_4), …, (v'_{d-1}, v'_d)`.
2. For each pair convert to polar: `r_i = sqrt(v'_{2i-1}² + v'_{2i}²)`, `θ_i = atan2(v'_{2i}, v'_{2i-1})`.
3. Recurse on the radii: treat `(r_1, r_2, …)` as a new vector and repeat the pair-wise polar conversion until a single global radius and a tree of angles remain.

The recursion is `O(d log d)` and is fully parallelizable.

### Why random preconditioning matters
Without preconditioning, the angles `θ_i` of arbitrary KV vectors have a distribution that depends on the input — so a fixed quantizer would be far from optimal. After multiplying by a random orthogonal `R` (typically a randomized Hadamard), the rotated vector `v'` looks isotropic, which makes each angle `θ_i` follow a closed-form density that depends only on dimension `d`, not on the input data. This is the "data-oblivious" hook: one quantizer fits all inputs.

### Bit budget
- Per angle: ~ k bits (typical k = 3–4).
- Per radius: at the top of the recursion tree, one global FP16 radius is stored.
- Overhead: zero per-block scale/zero-point — only the angle codes + one radius.
- Net effective rate: < 4 bits/element at 4.2× compression vs FP16.

### Comparison to per-block PTQ
| Method | Per-block overhead | Calibration | Best bit-width |
|--------|--------------------|-------------|----------------|
| KIVI | 1 scale + 1 zero-point per block (FP16) | yes | 2-bit (with per-channel K, per-token V) |
| KVQuant | dense-and-sparse split, FP16 outliers | yes | sub-2-bit |
| **PolarQuant** | **none** (1 global radius per vector) | **no** | < 4 bits → 4.2× |

## Connections
- [[qjl]] — same Zandieh+Daliri+Han lineage; QJL targets inner-product preservation via sign-bit JL; PolarQuant targets reconstruction via polar coords. They are complementary preconditioning ideas.
- [[turboquant]] — direct successor that generalizes random-rotation + scalar-quant into a two-stage framework with QJL residual, hitting the rate-distortion bound and pushing to 2.5 bits.
- [[quarot]] / [[spinquant]] — rotation-based weight/activation PTQ; same insight (rotation flattens distribution) applied to a different tensor.
- [[kivi]] / [[kvquant]] — the per-channel/per-token baselines that PolarQuant supersedes via overhead elimination.
- [[rate-distortion-theory]] — the optimal scalar quantizer for the closed-form angle distribution comes from Lloyd-Max applied to that specific density.

## Notes
PolarQuant is the **middle** paper of the Zandieh / Mirrokni KV-cache trilogy: [[qjl]] (Jun 2024) → PolarQuant (Feb 2025) → [[turboquant]] (Apr 2025, ICLR 2026). Read in order for the cleanest conceptual arc.
