<!-- chapter: ch-18
     track: frontier-2025-2026
     title: Data-Oblivious KV + 2026 KV Frontier
     sources: [[qjl]], [[polarquant]], [[turboquant]], [[kvtc]], [[adaptive-kv-cache-quant]], [[kv-cache-survey]], [[kv-cache-compression-survey-2025]]
     figures: figures/rotation-concentration.html
-->

# Chapter 18 — Data-Oblivious KV + 2026 KV Frontier

> **Core insight.** Calibration-based KV-cache quantization (KIVI / KVQuant / GEAR, ch-15) had one structural cost it could never escape: every block needed its own FP16 scale and zero-point. At 2-bit, that *metadata* eats 1+ bits per element — so "2-bit KV" was really ~3.5-bit effective. The 2024–2026 data-oblivious trio (QJL → PolarQuant → TurboQuant) eliminates this by exploiting **high-dimensional measure concentration after a random rotation**: once the KV vector is rotated, its coordinates' distribution becomes *analytically known and identical across all vectors*, so a single fixed quantizer works everywhere with **zero per-block metadata**. TurboQuant proves the resulting two-stage rotation + scalar + 1-bit QJL residual matches the rate-distortion bound up to a constant.
>
> **Guideline.** For new long-context KV-cache compression projects in 2026, default to TurboQuant (data-oblivious, no calibration, 3-bit at quality parity). Use KVTC for *persistent* / reusable cache storage where you can afford a one-shot calibration pass. Use adaptive-KV for edge / on-device where the bit budget should follow per-token importance. KIVI/KVQuant (ch-15) remain the right baseline to *understand* — but no longer the right *default*.

---

## Why this chapter exists

KV-cache memory dominates long-context inference. At 128K context, a 7B model's KV cache can be larger than the model weights themselves. Bandwidth-bound decode means every byte of KV you don't have to read from HBM is a token-per-second you gain.

Ch-15 covered the calibration-based 2024 wave (KIVI per-channel-K + per-token-V; KVQuant non-uniform + dense-and-sparse outliers; GEAR with low-rank residual). These work — but they share a hidden tax. Every block carries metadata: 1 FP16 scale (16 bits) + 1 FP16 zero-point (16 bits) = 32 bits per block. At block size 32 with 3-bit quant, that's **1 extra bit per element** of overhead. "3-bit" is really 4-bit effective; "2-bit" is really 3.5-bit effective.

The 2024–2026 papers from the Zandieh / Mirrokni group remove this tax entirely. Their conceptual through-line is **the data-oblivious thesis**: *eliminate calibration via high-dimensional measure concentration after rotation*. Once you rotate, the per-coordinate distribution is identical across all inputs and known in closed form, so one fixed quantizer works for every vector with no stored scales.

Three things to walk away with:

1. The asymmetric inner-product estimator from [[qjl]] — why sign-only on one side of a dot product (and full-precision on the other) is unbiased, while symmetric sign-only is biased.
2. The closed-form angle distribution from [[polarquant]] — the proof that rotation + polar coordinates makes the quantizer data-independent.
3. The two-stage TurboQuant pipeline — random rotation + per-coordinate scalar + 1-bit QJL residual on the leftover — and *why* it matches the rate-distortion bound up to a constant.

Then the 2026 extensions ([[kvtc]] transform coding for reusable caches, [[adaptive-kv-cache-quant]] per-token bit allocation) — orthogonal axes that compose with rotation-based quant.

---

## 1. The hidden cost of calibration-based KV quant

Let `B` = block size, `k` = bits per element, `s` = bits per scale (FP16 = 16). KIVI / KVQuant / GEAR all pay:

```
bits_per_element_effective = k + (s_scale + s_zero) / B
                           = k + 32 / B
```

| Method | k | B | Effective bits |
|--------|---|---|----------------|
| KIVI 2-bit | 2 | 32 | 2 + 1 = **3.0** |
| KIVI 4-bit | 4 | 32 | 4 + 1 = **5.0** |
| KVQuant 4-bit | 4 | 32 | 4 + 1 = **5.0** |
| KVQuant 3-bit (sub-2 V) | 3 | 32 | 3 + 1 = **4.0** |

So "3-bit KV" with a 32-element block stores ~33 % more bits per element than the label suggests. At long context, this metadata is significant memory.

The data-oblivious approach gets rid of the `32/B` term entirely. *No per-block scale, no per-block zero-point.* One global rotation matrix (fixed at startup, shared across all KV vectors) and a fixed quantizer per coordinate.

---

## 2. The data-oblivious thesis

The intuition lives in high-dimensional probability. Take any vector `v ∈ R^d`. Multiply it by a random orthogonal `R` (e.g. a randomized Hadamard). The rotated vector `v' = R·v` has the same L2 norm (rotations preserve length), but its **per-coordinate distribution** is *concentrated*: most coordinates of `v'` are close to `‖v‖/√d`. As `d` grows, the per-coordinate distribution converges to a *known, fixed* density — a property of `d` only, not of `v`.

This means:

- The optimal scalar quantizer for `v'_i` depends only on `d`, not on `v`.
- One quantizer works for all vectors. No calibration, no per-block scale, no zero-point.

This is the *whole trick*. Everything in this chapter is an instantiation of it: QJL uses the rotation to make sign-bit JL sketches behave nicely; PolarQuant uses the rotation to make polar angles concentrate on a Beta distribution; TurboQuant uses the rotation to make per-coordinate scalar quant near-optimal.

Note the contrast with [[quarot]] / [[spinquant]] (ch-14): those papers used the same rotation trick on *weights and activations* for PTQ. The 2024–2026 KV trio ports the trick to *KV caches* and pairs it with sketches / polar transforms / scalar quant instead of round-to-nearest.

---

## 3. QJL — 1-bit JL sketch with asymmetric estimator

[[qjl]] (Zandieh, Daliri, Han, Google Research, June 2024) is the earliest paper in the trilogy.

### 3.1 The sketch

For each KV vector `v ∈ R^d`:

- **Stored** (cached side): `s(v) = sign(Π · v) ∈ {-1, +1}^m`. **One bit per output coordinate. No scale. No zero-point.**
- **Live** (query side): `Π · u` computed full-precision on the query vector `u`.

`Π ∈ R^{m × d}` is a fixed random JL projection (entries iid `±1/√m` or Gaussian), shared globally.

### 3.2 The asymmetric inner-product estimator

The inner product is what matters in attention — `softmax(QK^T)` only needs accurate dot products, not accurate reconstructions of K itself.

```
⟨u, v⟩ ≈ α · ⟨s(v), Π · u⟩ = α · Σ_j s(v)_j · (Π · u)_j
```

`α` is a known scaling constant depending on `d`, `m`, and the JL distribution — fixed at startup.

**Why asymmetric?** A symmetric `sign(Π·u) · sign(Π·v)` estimator is *biased* — by the well-known `2/π · arcsin(cos·)` identity, it gives the angle, not the inner product. The asymmetric version — sign-only on one side, full-real on the other — recovers an unbiased estimator with minimal variance.

This is the load-bearing trick: it's *why* you can store the K cache at one bit per output coordinate without losing the attention-score accuracy.

### 3.3 Bit budget

Compared to KIVI/KVQuant which pay 32-bit-per-block overhead:

| Sketch width m | Effective bits / input coord | Memory vs FP16 |
|----------------|------------------------------|----------------|
| m = d/4 | ~0.25 bit | ~16× |
| m = d/2 | ~0.5 bit | ~8× |
| m = d | ~1 bit | ~5× (paper headline "3-bit equivalent quality") |

At 3-bit equivalent quality, QJL achieves **> 5× KV memory reduction** with **zero quality loss** on NLP benchmarks. The released CUDA kernel fuses JL projection + sign quantization + packing + the asymmetric inner-product computation into one op — beats unquantized FP16 attention on long contexts.

---

## 4. PolarQuant — polar coords with closed-form angle distribution

[[polarquant]] (Han, Kacham, Karbasi, Mirrokni, Zandieh, Feb 2025 → AISTATS 2026) is the middle paper. It targets *reconstruction* (not just inner-product), which makes it useful for cases where the V cache also needs accurate retrieval.

### 4.1 Recursive pair-wise polar transform

For `v ∈ R^d`:
1. Apply random orthogonal preconditioning: `v' = R · v`.
2. Pair coordinates: `(v'₁, v'₂), (v'₃, v'₄), …`.
3. For each pair, convert to polar: `r_i = √(v'_{2i-1}² + v'_{2i}²)`, `θ_i = atan2(v'_{2i}, v'_{2i-1})`.
4. Recurse on the radii: treat `(r₁, r₂, …)` as a new vector, repeat the pair-wise polar conversion until a single global radius and a tree of angles remain.

`O(d log d)`, fully parallelizable.

### 4.2 The closed-form angle distribution

The paper's central theorem: **after random orthogonal preconditioning, the angles `θ_i` follow a closed-form Beta-style density that depends only on `d`.**

Without preconditioning, the angles' distribution depends on the input data. After preconditioning, the rotated vector `v'` looks isotropic, which makes each angle follow a fixed density. That fixed density is what lets a *single* angle quantizer be near-optimal across all inputs — no calibration.

### 4.3 Bit budget

- Per angle: ~k bits (typical k = 3–4).
- Per radius: at the top of the recursion tree, one global FP16 radius is stored.
- **Overhead: zero per-block scale/zero-point**. Only the angle codes + one radius.

| Method | Per-block overhead | Calibration | Best bit-width |
|--------|--------------------|-------------|----------------|
| KIVI | 1 scale + 1 zero-point per block (FP16) | yes | 2-bit (per-channel K, per-token V) |
| KVQuant | dense-and-sparse split, FP16 outliers | yes | sub-2-bit |
| **PolarQuant** | **none** (1 global radius per vector) | **no** | **< 4 bits → 4.2× compression** |

Result: > 4.2× KV-cache compression with the **best long-context quality** among PTQ KV methods at the same rate.

---

## 5. TurboQuant — the unified rate-distortion-optimal pipeline

[[turboquant]] (Zandieh, Daliri, Hadian, Mirrokni, April 2025 → ICLR 2026) is the synthesis. It takes the random-rotation idea (QJL, PolarQuant) and generalizes it into a **two-stage quantizer that matches the rate-distortion lower bound up to a constant factor**.

### 5.1 The two-stage pipeline

For each KV vector `v ∈ R^d`:

**Stage 1 — random rotation + per-coordinate scalar quant.**
- `v' = R · v` where `R` is a fixed random orthogonal (Hadamard-based, shared globally, *not* data-dependent).
- After rotation, the per-coordinate magnitudes follow a tightly concentrated Beta distribution — the same high-dimensional concentration argument as PolarQuant.
- Apply the analytically optimal scalar quantizer for that Beta to each coordinate. *No per-block scale needed* — the rotation made it unnecessary.

**Stage 2 — 1-bit QJL on the residual.**
- Compute the residual: `e = v' − Q(v')`.
- Apply a JL sketch `Π` of width `m`, store only the sign: `sign(Π · e) ∈ {-1, +1}^m`.
- This residual is what restores *inner-product unbiasedness* — critical for attention-score reconstruction.

### 5.2 The inner-product estimator

For two vectors `u, v` with TurboQuant representations:

```
⟨u, v⟩ ≈ ⟨Q(u'), Q(v')⟩ + α · ⟨sign(Π·e_u), Π·e_v⟩
```

Stage-1 term: dot-product of the scalar-quantized rotated vectors. Stage-2 term: the QJL residual correction with the asymmetric estimator from [[qjl]].

### 5.3 Why it hits the rate-distortion bound

Standard Shannon rate-distortion theory ([[rate-distortion-theory]], ch-01) gives a lower bound `D ≥ σ² · 2^(-2R)` for Gaussian sources. Any quantizer at rate `R` bits per coordinate pays distortion `D` at least that large.

TurboQuant's theoretical contribution is proving that **the two-stage random-rotation + scalar + QJL pipeline matches this bound up to a constant factor that does not grow with dimension or bit-width**. That's the strongest possible guarantee for a fixed-rate quantizer: you cannot do better than a constant-factor improvement.

In practice this means:

- **3.5 bits/channel = quality-neutral** (matches FP16 on long-context evals).
- **2.5 bits/channel = marginal degradation** (small drop on RULER / NIAH).

### 5.4 Bit budget

| Component | Bits | Notes |
|-----------|------|-------|
| Stage 1 scalar quant | ~3 bits | per-coord, uniform on Beta CDF |
| Stage 2 QJL residual | ~0.5 bit | m ≈ d/2; one sign-bit per JL output dim |
| **Total** | **~3.5 bits/channel** | quality-neutral on Gemma + Mistral |
| Per-block scale/zero | **0** | the whole point |

### 5.5 Empirical headlines

- **3-bit KV with no measurable quality loss** on long-context benchmarks (LongBench, NIAH, RULER, ZeroSCROLLS, L-Eval) on Gemma + Mistral.
- **6× KV memory reduction.**
- **H100 CUDA kernel: up to 8× speedup over FP32 keys at 4-bit**, demonstrating the rotation + per-coordinate quant pipeline is GEMM-friendly.

### 5.6 Hyperparameters

| Knob | Typical | Notes |
|------|---------|-------|
| Rotation | Random Hadamard | fixed at startup; not learned |
| Scalar quant levels | 8–16 (3–4 bit) | per-channel uniform on Beta CDF |
| QJL sketch width m | ~d/2 | controls Stage-2 distortion |
| Bit budget | 2.5–3.5 bit/channel | quality-neutral at 3.5 |
| Block size | **none** | fully per-channel; no per-block normalization |

---

## 6. KVTC — transform coding for reusable caches

[[kvtc]] (Staniszewski, Lancucki, ICLR 2026) targets a different problem: **persistent / reusable KV-cache storage** for shared-prefix workloads (iterative code editing, multi-turn chat with a long system prompt).

### 6.1 The codec pipeline

Borrows directly from media compression:

1. Collect a short calibration sample of KV-cache tensors.
2. Fit a PCA-like transform to decorrelate feature dimensions.
3. Quantize transformed coefficients with adaptive precision.
4. **Entropy-code** the quantized coefficients for compact storage.
5. Decode when the cache is reused.

### 6.2 Deployment boundary

KVTC is *not* a drop-in replacement for live low-bit attention kernels like KIVI or TurboQuant. It's the right tool when:

- Large prefixes repeat across requests (chat history, system prompt).
- Caches need to be stored compactly *between* turns or requests (offload to CPU / disk).
- The compression budget can amortize a one-shot decode cost.

Reports **up to 20× compression** while maintaining reasoning + long-context accuracy on AIME25, GSM8K, LiveCodeBench, LongBench, MATH-500, MMLU, Qasper, RULER. Higher compression in selected use cases.

### 6.3 Why this matters for the chapter

KVTC is the *classical-compression* angle on KV caches: decorrelate first, quantize second, entropy-code last. It connects the rate-distortion thread of TurboQuant to the engineering practice of media codecs. Both share the insight that quantization is the *middle* step, not the only step.

---

## 7. Adaptive KV — per-token bit allocation

[[adaptive-kv-cache-quant]] (Boroujeni et al., CVPR 2026) takes an orthogonal axis: **token-dependent bit-width**.

### 7.1 The controller

A lightweight policy selects per-token precision from `{2-bit, 4-bit, 8-bit, FP16}` during decoding. Inputs:

- token frequency,
- token quality score,
- attention variance,
- entropy / uncertainty signal.

These features estimate whether a token's cached K/V will matter enough to justify extra bits.

### 7.2 Precision menu

| Precision | Intended use |
|-----------|--------------|
| 2-bit | low-impact tokens |
| 4-bit | default compressed cache |
| 8-bit | moderately important tokens |
| FP16 | highly important tokens |

### 7.3 Where this fits

Tested at on-device / edge scale (SmolLM-135M / 360M / 1.7B). Not a universal KV quantizer, but a useful pointer: 2026 KV compression moved from "pick one bit-width" to "*allocate* bits dynamically." This is orthogonal to TurboQuant's data-oblivious framing — they compose.

---

## 8. The 2025–2026 survey context

Two surveys frame the broader landscape:

[[kv-cache-survey]] (Shi et al., COLM 2024): **three-axis taxonomy** — architectural (pre-training: MQA/GQA, MLA), quantization (deployment: KIVI/KVQuant/GEAR/this chapter), eviction (inference: H2O, StreamingLLM, Quest). The optimal recipe combines all three.

[[kv-cache-compression-survey-2025]]: same three axes, with the 2025 update — *quantization-aware eviction policies* matter. Naive composition of quant + evict can compound errors; the 2025 fix is to evict tokens whose quant error is small (sacrifice precision-redundant tokens, keep precision-critical ones).

### Example budget for 7B / 1M context on 80 GB GPU

| Axis | Choice | Reduction |
|------|--------|-----------|
| Architectural | GQA-8 | 8× |
| Quantization | TurboQuant 3-bit | ~5× |
| Eviction | StreamingLLM (4K sink + 4K recent) | variable |
| **Cumulative** | | **>100×** vs FP16 KV |

### K vs V asymmetry — still holds

The K vs V asymmetry from [[kivi]] / [[kvquant]] (ch-15) still applies in 2026: K has channel-aligned outliers (RoPE-induced), V is roughly Gaussian per token. But TurboQuant's rotation absorbs this — after the random Hadamard, both K and V coordinates concentrate on the same Beta distribution. *That's another reason rotation-based methods work so well.* The asymmetry was an artifact of working in the raw coordinate frame; rotating it away makes one quantizer fit both.

---

## 9. Practitioner's cheat-sheet

```python
# TurboQuant-style data-oblivious KV cache (pseudocode, kernel-level)
class TurboQuantKVCache:
    def __init__(self, head_dim, m):
        # All fixed at startup — no per-vector state
        self.R = random_hadamard(head_dim)                    # rotation
        self.Q = optimal_scalar_quantizer_for_beta(head_dim)  # Stage 1
        self.Pi = random_jl(m, head_dim)                      # Stage 2 sketch
        self.alpha = jl_scaling_constant(head_dim, m)

    def store(self, v):
        v_rot = self.R @ v
        v_q   = self.Q.quant(v_rot)              # scalar quant per coord
        e     = v_rot - self.Q.dequant(v_q)      # residual
        s     = sign(self.Pi @ e)                # 1-bit sketch
        return (v_q, s)                          # NO per-block scale stored

    def attention_score(self, q, k_cached):
        v_q, s = k_cached
        q_rot = self.R @ q
        score_stage1 = q_rot @ self.Q.dequant(v_q)
        score_stage2 = self.alpha * (s * (self.Pi @ (q_rot - self.Q.dequant(v_q)))).sum()
        return score_stage1 + score_stage2
```

```python
# Adaptive KV — per-token bit allocation (sketch)
def choose_precision(token_features):
    # features: frequency, attention_variance, entropy, quality_score
    if features.attention_variance > THRESH_HIGH:
        return FP16
    elif features.entropy < THRESH_LOW:
        return 2  # low-impact token
    elif features.quality_score > THRESH_MED:
        return 8
    else:
        return 4  # default
```

---

## Common pitfalls

- **Confusing JL "1-bit per JL output coord" with "1-bit per input coord":** the bit budget depends on `m`, the JL sketch width. m = d gives "3-bit equivalent" quality at 1 bit per JL output ≈ 1 bit per input coord; m = d/2 gives ~0.5 bit per input coord.
- **Storing per-block metadata anyway:** the whole point of the data-oblivious approach is to eliminate the scale + zero-point overhead. If your implementation still stores them, you're not getting the bit-budget win.
- **Using PolarQuant for inner-product-only workloads:** PolarQuant targets reconstruction; QJL/TurboQuant target inner products. Attention uses inner products — prefer QJL/TurboQuant.
- **Symmetric sign-only inner product:** biased by the `2/π · arcsin` identity. Always use the asymmetric form (sign on one side, real on the other).
- **Forgetting the rotation matters:** without random rotation, the per-coordinate distribution depends on data → fixed quantizer is wrong → you're back to needing calibration. The rotation is the *load-bearing* step.
- **Applying KVTC to live decode:** KVTC's entropy-coding decode is too slow for the live attention path. Use it for *persistent* / reusable cache storage only.
- **Naive composition with eviction:** quant + eviction can compound errors. Use quant-aware eviction (evict tokens whose quant error is small) per the 2025 survey guidance.

---

## Connections and what's next

- **[[kivi]] / [[kvquant]] / ch-15** — the calibration-based 2024 wave this chapter supersedes. Read those first to understand *what tax* the data-oblivious approach is eliminating.
- **[[rate-distortion-theory]] / ch-01** — the rate-distortion lower bound that TurboQuant matches up to a constant. The theoretical floor that defines what "near-optimal" means.
- **[[quarot]] / [[spinquant]] / ch-14** — the *weight/activation* rotation lineage. Same root insight (rotation flattens distributions) ported to a different tensor.
- **[[uniform-quantization-noise]] / ch-01** — Bennett's analysis; justifies the per-coordinate scalar quant in Stage 1 of TurboQuant.
- **[[kv-cache-compression-survey-2025]]** — the broader compression landscape (quant + evict + architectural).
- **ch-19** — production kernels and serving stacks; how do TurboQuant / KIVI / KVQuant actually run in vLLM / TRT-LLM?
- **ch-22 capstone** — TurboQuant is one of the recommended reproduction targets; the algorithm in §5 is the implementation specification.

## Further reading

- [[qjl]] / [[polarquant]] / [[turboquant]] — read in **chronological order** (June 2024 → Feb 2025 → April 2025) for the cleanest conceptual arc.
- [[kv-cache-compression-survey-2025]] — landscape view; three-axis taxonomy.
- [[kvtc]] — the transform-coding angle.
- [[adaptive-kv-cache-quant]] — token-level bit allocation.

## Companion visualization

**[figures/rotation-concentration.html](figures/rotation-concentration.html)** — interactive viz of high-dimensional measure concentration: take a random `v ∈ R^d`, apply a random Hadamard, and plot the per-coordinate distribution of `v` vs `R·v`. Sliders for `d` (8 → 2048) show how the rotated distribution sharpens onto the analytically known density — the visual proof of why data-oblivious works.
