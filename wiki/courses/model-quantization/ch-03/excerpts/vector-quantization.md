---
chapter: ch-03
course: model-quantization
phase: read
excerpt_of: "Linde-Buzo-Gray Vector Quantization (Linde, Buzo, Gray 1980)"
source_url: https://ieeexplore.ieee.org/document/1094577
created_at: "2026-05-21"
raw_data_source: [[raw-data/vector-quantization]]
---

# Excerpt: LBG — vector quantization that closes the scalar gap

**Authors:** Yoseph Linde, Andrés Buzo, Robert M. Gray.
**Year:** 1980.
**Venue:** IEEE Transactions on Communications, January 1980.
**URL:** see source_url.

---

## The one-box objective

Given training vectors `{x_t} ⊂ ℝ^d` and target codebook size `N`:

```math
\min_C \; D(C) \,=\, \frac{1}{T} \sum_t \min_k \|x_t - c_k\|^2,
\qquad C = \{c_1, \ldots, c_N\} \subset \mathbb{R}^d
```

---

## Two necessary conditions (generalized Lloyd)

```math
S_k \,=\, \{ x : \|x - c_k\| \le \|x - c_j\| \text{ for all } j \}  \qquad \text{(Voronoi cell)}
```

```math
c_k \,=\, \frac{1}{|S_k \cap X|} \sum_{x_t \in S_k} x_t  \qquad \text{(centroid)}
```

This is exactly **`k`-means** with squared-error distortion in `ℝ^d`.

---

## LBG splitting algorithm

```
C ← {(1/T) Σ x_t}                                       # single centroid (mean)
while |C| < N:
    C ← C ∪ {c + ε · u : c ∈ C}                         # double via perturb (u random unit vector)
    repeat:
        assign each x_t to nearest c_k                  # NN step
        update each c_k = mean(assigned x_t)            # centroid step
    until D change < threshold
```

The splitting init avoids the bad local minima that plague random `k`-means init.

---

## Rate-distortion at dimension `d`

For dimension-`d` VQ with codebook size `N = 2^{Rd}`, high-rate Gaussian distortion satisfies

```math
D_{\text{VQ}}(R, d) \,\approx\, \sigma^2 \cdot G_d \cdot 2^{-2R}
```

where `G_d → 1` as `d → ∞` (Shannon bound) and `G_1 = π√3 / 2 ≈ 2.72` (scalar Lloyd-Max).

**`G_d` values:** `G_1 ≈ 2.72` (1.53 dB loss), `G_2 ≈ 1.16` (0.65 dB), `G_4 ≈ 1.07` (0.30 dB), `G_∞ = 1` (= R(D) bound).

This is the structural reason additive vector quantization (AQLM, ch-14) reaches sub-2-bit at quality where scalar INT/FP4 collapse — going from `d = 1` to `d = 8` recovers ~1.3 dB.

---

## Cost

- **Encoding:** `O(N · d)` per vector (brute-force nearest neighbour).
- **Storage:** `N · d` floats for codebook.
- **Bit-rate:** `log₂ N` bits per `d`-dim vector = `(log₂ N) / d` bits per scalar.

Brute-force encoding becomes infeasible for large `N · d`. Tree-structured VQ, multi-stage VQ (residual), and product quantization ([[excerpts/product-quantization]]) trade some distortion for log / linear cost.

---

## Operational use in LLM quant

- Group `d = 8` weights together → 256-vector codebook → 1 bit / weight.
- Run LBG on the empirical layer weight distribution (or on rotated weights post-[[quarot]]) to learn the codebook.
- Encoding cost amortized over all vectors in the layer.
- **AQLM uses additive VQ** — sum of `M` codewords from `M` small codebooks — to get further compression with manageable encoding cost.

---

## Connections

- [[excerpts/lloyd-max-quantizer]] — LBG = scalar Lloyd in `d` dimensions.
- [[excerpts/rate-distortion-theory]] — VQ asymptotically achieves the `R(D)` bound; scalar codes are stuck 1.53 dB above.
- [[excerpts/product-quantization]] — Jégou's PQ factorizes the LBG codebook for tractable encoding.
- [[squeezellm]] — uses non-uniform LUT (essentially scalar VQ) weighted by sensitivity.
- [[aqlm]] — additive VQ for sub-2-bit LLM weights, direct LBG descendant (ch-14).
- [[ch-03]] — parent synthesis.
