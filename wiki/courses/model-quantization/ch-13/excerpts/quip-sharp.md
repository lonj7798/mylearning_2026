---
chapter: ch-13
course: model-quantization
phase: read
excerpt_of: "QuIP#: Even Better LLM Quantization with Hadamard Incoherence and Lattice Codebooks"
source_url: https://arxiv.org/abs/2402.04396
created_at: "2026-05-21"
---

# Excerpt: QuIP# — randomized Hadamard + E₈ lattice + fine-tune

**Authors:** Albert Tseng, Jerry Chee, Qingyao Sun, Volodymyr Kuleshov, Christopher De Sa (Cornell)
**Year:** 2024
**Venue:** ICML 2024
**URL:** https://arxiv.org/abs/2402.04396
**Raw-data source:** [[raw-data/quip-sharp]]

---

## What changes from QuIP

QuIP's three weak spots:

1. The random orthogonal U, V cost `O(d²)` to apply — fine for offline weight quantization, painful at inference.
2. Scalar round-to-nearest is provably suboptimal for the post-rotation Gaussian distribution.
3. There is no compensation step for the residual rounding error.

QuIP# fixes all three:

1. **Randomized Hadamard Transform (RHT)** — same `μ = O(\log d)` incoherence guarantee, `O(d \log d)` cost via FWHT.
2. **E₈ lattice codebook (E₈P variant)** — vector quantization on the densest 8-dim sphere packing; ~0.3 bits better than scalar RTN at the same MSE.
3. **Fine-tuning pass** — 1 epoch of cross-entropy distillation updating the codeword assignments and per-layer scales.

---

## Randomized Hadamard Transform

Let `S ∈ {±1}^d` be a uniform random sign vector and `H_d` the d×d Sylvester Hadamard (recursively, `H_1 = [1]`, `H_{2k} = [[H_k, H_k], [H_k, -H_k]]`).

```math
R = \frac{1}{\sqrt{d}} \cdot \mathrm{diag}(S) \cdot H_d
```

R is orthogonal (`R^⊤ R = I`) and achieves the same `μ = O(\log d)` incoherence as a uniformly-sampled orthogonal. The matmul `R x` decomposes into:

```
x' = S ⊙ x          # element-wise sign flip
x'' = H_d x'        # Fast Walsh-Hadamard Transform, O(d log d)
R x = x'' / √d
```

The FWHT is a butterfly — same structure as the radix-2 FFT but with real ±1 twiddles. Trivially vectorisable; the GPU kernel sustains > 1 TFLOP/s of effective bandwidth on H100.

QuIP# uses independent RHTs on each side: `R_left = (1/√d_out) · diag(S₁) · H_{d_out}`, `R_right = (1/√d_in) · diag(S₂) · H_{d_in}`. Pre/post-process `W ← R_left^⊤ W R_right`.

---

## E₈ lattice — why it's optimal

The E₈ lattice in ℝ⁸ achieves the **densest sphere packing in 8 dimensions** (Viazovska 2017, Fields Medal). For an isotropic Gaussian source, lattice quantization with E₈ achieves rate-distortion within ~0.1 bit of the Shannon lower bound — better than scalar (round-to-nearest), product quantization (PQ), or random vector codes at the same bit budget.

After RHT-incoherence, post-rotation weights are approximately i.i.d. Gaussian (concentration of measure → spherical symmetry). So E₈ is the matching codebook.

**The E₈P codebook** (the variant QuIP# actually uses): start from the half-integer-shifted E₈ lattice points; keep those within a ball whose radius gives exactly 256 codewords with high symmetry. Encode each 8-dim weight vector as the index of its nearest E₈P codepoint. 8 bits per 8-vector = **1 bit/weight nominal**; combined with a sign bit and a per-block FP16 scale, the amortised rate is **2 bits/weight**.

---

## Encoding (nearest-codepoint search)

For each rotated weight group `v ∈ ℝ⁸`, find the nearest E₈ lattice point. The E₈ lattice has the closed-form nearest-point algorithm (Conway & Sloane Ch. 20):

```python
def e8_nearest(v):
    # E8 = D8 ∪ (D8 + (½,½,…,½))
    g0 = nearest_d8(v)
    g1 = nearest_d8(v - 0.5) + 0.5
    return g0 if dist(v, g0) <= dist(v, g1) else g1

def nearest_d8(v):
    # D8 = even-sum-of-coords sublattice of Z^8
    r = np.round(v)
    if int(r.sum()) % 2 == 0:
        return r
    # if odd-sum, flip the coordinate with largest rounding residual
    i = np.argmax(np.abs(v - r))
    r[i] += np.sign(v[i] - r[i])
    return r
```

Cost: O(d) per 8-vector. Negligible compared to the FWHT.

---

## Decoding (LUT)

The 256 E₈P codewords are stored as an `8 × 256` FP16 lookup table per layer (~ 4 KB). Decoding: index → 8-vector → multiply by per-block scale.

Inference GEMV pipeline:

```
x_rot = R_right^⊤ x                     # FWHT, O(d log d)
for each 8-row block of W_q:
    w_8 = LUT[index_byte] * scale        # 8 FP16s
    accumulate w_8 @ x_rot_block        # 8 MACs
y = R_left @ accumulator                # FWHT, O(d log d)
```

The LUT is L1-cache-resident. The bottleneck shifts from HBM-bandwidth (W reads) to ALU throughput (the LUT decode + 8-way dot). On H100, QuIP# at 2-bit runs at ~85% of the FP16 GEMV throughput on memory-bound batch sizes (which are the relevant regime for serving).

---

## Fine-tuning the compressed codes

After PTQ, run ~1 epoch of cross-entropy distillation from the FP teacher on 256 C4 sequences (small, fast). Update:

- **Per-layer scales** (continuous, differentiable).
- **Hadamard sign vectors S₁, S₂** (small per-element perturbations, treated as continuous embedding).
- **Codeword indices** for the worst-error 8-vectors (discrete; local search over the K=8 nearest E₈ codewords).

Loss: token-level KL divergence between FP and quantized logits on the calibration corpus.

Recovers ~0.2–0.5 ppl on LLaMA-2-70B at 2-bit. The discrete-index update is structurally the same idea that [[pv-tuning]] later formalises as a proper alternating P/V step.

---

## The numbers

LLaMA-2-70B WikiText-2 (lower = better):

| Method | Bits | ppl | Δppl |
|--------|------|-----|------|
| FP16 | 16 | 3.32 | — |
| GPTQ | 4 | 3.43 | +0.11 |
| AWQ | 4 | 3.41 | +0.09 |
| QuIP | 2 | 4.40 | +1.08 |
| AQLM | 2 | 3.83 | +0.51 |
| **QuIP#** | **2** | **3.81** | **+0.49** |
| QuIP# | 3 | 3.41 | +0.09 |

QuIP# at 2-bit ≈ GPTQ at 4-bit on accuracy; 2× the memory savings. QuIP# at 3-bit ≈ FP16 within noise.

---

## Pitfalls

- **Block size 8 is locked by E₈.** If `d_in % 8 != 0`, pad. (Most LLMs have `d_in` multiples of 64.)
- **The codebook is per-layer.** Don't share across layers — the post-rotation scale and Gaussianisation are layer-specific.
- **FWHT requires power-of-2 dim.** For d=6144 (LLaMA-2-13B FFN intermediate), block into chunks of 2048; you lose a small amount of incoherence quality but keep the speed.
- **Fine-tune is not optional at 2-bit.** Without it, 2-bit QuIP# loses ~0.5 ppl more vs the headline numbers.
- **Encoding is one-shot.** Once codes are written, treat them as immutable; re-running incoherence with a fresh seed produces a different (incompatible) checkpoint.

---

## Connections

- [[excerpts/quip]] — parent paper with the LDLQ algorithm and µ-incoherence theorem.
- [[ch-14]] — AQLM, VPTQ, GPTVQ as alternative sub-2-bit codebook families; PV-Tuning as the principled discrete-fine-tune replacement for STE.
- [[ch-08]] — GPTQ as the scalar-rounding baseline QuIP# replaces.
- Sphere-packing reference: Conway & Sloane, *Sphere Packings, Lattices and Groups* (Springer 1999), Ch. 4 (E₈) and Ch. 20 (nearest-point algorithms).
