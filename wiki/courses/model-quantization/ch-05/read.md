<!-- chapter: ch-05
     phase: classical-bridge
     title: Classical PTQ Playbook + OBS to GPTQ Lineage
     sources: [[quantization-mapping]], [[data-free-quantization]], [[brecq]], [[obs-obd]], [[obc]], [[qdrop]], [[hawq]]
     forward: [[gptq]] (ch-08), [[smoothquant]] (ch-09), [[awq]] (ch-09)
-->

# Chapter 5 — Classical PTQ Playbook + OBS to GPTQ Lineage

> **Core insight.** Every "modern" LLM PTQ algorithm — GPTQ, SmoothQuant, AWQ, OmniQuant, SpQR — is a billion-parameter re-implementation of three classical ideas: (1) Krishnamoorthi's taxonomy of where to put a scale, (2) the Optimal Brain Surgeon second-order weight-edit formula, and (3) BRECQ's block-wise reconstruction objective. There is essentially no new mathematics after 2022 in production PTQ; only better numerical engineering at LLM scale.
>
> **Guideline.** When confronted with a new "post-training quantization" paper for LLMs, immediately ask three questions. (1) Where does it put the scale — per-tensor, per-channel, per-group, per-token? (2) Does it use OBS-style closed-form error compensation, gradient-based block reconstruction, or just round-to-nearest? (3) Is the calibration objective per-layer, per-block, per-stage, or per-network? Almost every paper occupies a unique point in that 3-axis cube; the cube comes from this chapter.

---

## Why this chapter exists

Ch-04 already gave you the **QAT lineage** ([[brecq]], [[adaround]], [[lsq]]) — methods that need gradient flow. This chapter covers the **PTQ lineage**, where you have one shot: a trained FP model, a tiny calibration set (~128 sequences), and no backprop through the full network. That constraint is what every LLM-era paper inherits.

The story has four threads that converge in 2022 on GPTQ:

1. **Krishnamoorthi 2018** (the playbook) — codifies the 4-axis design space (symmetric vs asymmetric × per-tensor vs per-channel × weight vs activation × PTQ vs QAT). Every later paper picks a point in this cube.
2. **Nagel 2019 / DFQ** (the equivalent-transformation seed) — shows that *rescaling weights across consecutive layers* is a free knob; [[smoothquant]] resurrects this idea for activations in 2022.
3. **OBS 1993 → OBC 2022** (the Hessian thread) — the second-order error-compensation rule that GPTQ ports to LLM scale.
4. **BRECQ 2021 + QDrop 2022** (the block reconstruction thread) — the optimisation grain that [[omniquant]] later adopts for LLMs.

Walk through them in order and the GPTQ paper reads as a one-page engineering note rather than a breakthrough. That's the point: by the end of this chapter you should be able to *derive* GPTQ from OBC + a Cholesky reformulation.

---

## 1. The Krishnamoorthi cube — the four orthogonal axes

From [[quantization-mapping]], the canonical taxonomy. Memorise it; every PTQ paper inhabits one cell.

| Axis | Options | What the choice buys / costs |
|---|---|---|
| **Scale placement (weights)** | per-tensor / per-channel / per-group | per-tensor: 1 scalar per matrix, cheapest, ~3-bit accuracy loss on transformers; per-channel: 1 scalar per output row, free at runtime (fold into next op); per-group (G=128): 1 scalar per 128 contiguous input dims per row, ~0.05 bit overhead, standard for W4 |
| **Scale placement (activations)** | per-tensor / per-token / per-channel | per-tensor: GEMM-friendly, breaks at LLM scale (see [[llm-int8]] ch-07); per-token: 1 scalar per row of X, dynamic, recomputed every forward; per-channel: 1 scalar per column, requires extra rescale at GEMM output |
| **Symmetry** | symmetric (Z=0) / asymmetric (Z ∈ ℤ) | symmetric: no zero-point math, slight wastage on one-sided distributions; asymmetric: extra Z term in GEMM, exact 0.0 representation |
| **Training regime** | PTQ / QAT | PTQ: one calibration pass; QAT: insert fake-quant + retrain ~10% of original schedule |

The affine map (asymmetric):

```
q = clamp(round(x / S) + Z,  Q_min, Q_max)        # real → integer
x̂ = S · (q − Z)                                    # integer → real
```

`S ∈ ℝ₊` is the **scale**; `Z ∈ ℤ` is the **zero-point** chosen so real 0.0 maps exactly to integer Z (critical for ReLU and padding correctness). Symmetric is `Z = 0` for signed INT8 (range −128..127) or `Z = 2^{k−1}` for unsigned.

**Practical pitfall.** The Krishnamoorthi production sweet spot for CNNs was *per-channel symmetric weights + per-tensor asymmetric activations*. For LLMs, the per-tensor activation cell is empty — emergent outliers ([[llm-int8]]) destroy it past ~6.7B. Every modern LLM activation quantizer is either per-token or per-channel, and the rest of this course documents the migration.

### The bias and requantize trick

Bias is stored as **int32** with `S_bias = S_w · S_x` and never separately calibrated — it ride-shares on the matmul accumulator scale. The output gets rescaled to the next layer's input scale via:

```
M = S_w · S_x / S_y ≈ M_0 · 2^{-n},   M_0 ∈ [0.5, 1) stored as int32
```

This formula is the engineering core of every integer-only pipeline. We cover it properly in [[ch-06]].

---

## 2. DFQ: equivalent transformations without data — the SmoothQuant seed

Nagel et al. 2019 ([[data-free-quantization]]) noticed something cute about ReLU networks: for any positive `S` and consecutive layers `W_i → ReLU → W_{i+1}`,

```
W_{i+1} · ReLU(W_i · x + b_i)  =  (W_{i+1} · S⁻¹) · ReLU((S · W_i) · x + S · b_i)
```

i.e. you can absorb a per-channel rescale `S` into the boundary between two layers **without changing the output**. This is *positive-scaling invariance* of ReLU; it generalises to any positive-homogeneous activation.

DFQ exploits this by computing per-output-channel scale factors that equalise the per-channel range across consecutive layers:

```
S^c = √( r_i^c / r_{i+1}^c ),    r_ℓ^c = range of W_ℓ along channel c
```

After CLE (Cross-Layer Equalization), both layers have geometric-mean ranges, minimising the loss when per-tensor scales are then applied. The second trick — **bias correction** — analytically estimates the systematic offset that quantization introduces in each output channel and absorbs it into the bias:

```
ε_c = (W_c − Q(W_c)) · E[x_prev],     b_c ← b_c − ε_c
```

`E[x_prev]` comes from the previous layer's BatchNorm running mean. Zero data needed.

**Why this matters for LLMs.** DFQ requires (a) positive-homogeneous activation (ReLU), and (b) BN for `E[x]`. LLMs satisfy neither — they have GELU/SiLU and LayerNorm. *But the equivalent-transformation idea is exactly what [[smoothquant]] (ch-09) does*: instead of equalising weight ranges, SmoothQuant migrates activation outliers into weights via the per-channel scale `s_j = max(|X_j|)^α / max(|W_j|)^(1−α)`, absorbed into the previous LayerNorm. Same idea, different invariance argument.

---

## 3. The OBS thread — second-order weight editing

The single most-implemented idea in modern LLM PTQ is **Hassibi & Stork 1993**'s Optimal Brain Surgeon. The math fits on one slide. From [[obs-obd]]:

**Setup.** A trained model with weights `w ∈ ℝⁿ` at a local minimum of loss `L`. Second-order Taylor expansion around `w`:

```
δL  ≈  gᵀ δw  +  (1/2) δwᵀ H δw   ≈   (1/2) δwᵀ H δw
```

At convergence `g ≈ 0` so only the quadratic term matters. **Goal**: choose `δw` to remove one weight (`w_q + δw_q = 0`) while minimising the loss increase.

**OBS Lagrange solution.** Using Lagrange multipliers on the constraint `e_qᵀ(w + δw) = 0`:

```
δw   =  − (w_q / [H⁻¹]_qq) · H⁻¹_{:, q}        ← the update
δL_q =   w_q² / (2 · [H⁻¹]_qq)                  ← the saliency
```

That's it. Two formulas. The procedure:

1. Compute `H⁻¹` once.
2. For each weight: compute saliency `w_q² / (2 [H⁻¹]_qq)`.
3. Remove the weight with smallest saliency.
4. Apply `δw` to *every other surviving weight*, exactly compensating the first-order effect of the removal.
5. Update `H⁻¹` via Woodbury for the next round.

### From "set to zero" to "round to grid": OBC

Frantar, Singh & Alistarh 2022 ([[obc]]) made a one-line generalisation: **pruning** is "set `w_q` to 0" with `δw_q = −w_q`; **quantization** is "set `w_q` to its nearest grid point" with `δw_q = Q(w_q) − w_q`. The OBS update formula is *the same in both cases*:

```
δL_q     =  (δw_q)² / (2 · [H⁻¹]_qq)
δw_{−q}  = − (δw_q / [H⁻¹]_qq) · H⁻¹_{−q, q}     # update to surviving columns
```

OBC's other contribution was engineering: a **Cholesky factorisation** of `H⁻¹` brings the per-layer cost from naive `O(d⁴)` down to `O(d³)`, making it tractable for BERT-scale layers (d ≈ 768–1024). At LLM scale (d ≈ 4096–16384), Cholesky alone isn't enough — you need the **lazy batched update** that GPTQ adds. We cover that in [[ch-08]].

### What the calibration Hessian actually looks like

For a single linear layer, the load-bearing Hessian is:

```
H  =  (∂² L / ∂W²)|_W   ≈   2 X Xᵀ
```

where `X ∈ ℝ^{d_in × N}` are calibration activations (one minibatch of ~128 sequences × 2048 tokens). H is `d_in × d_in`. Crucial properties:

- **Shared across output rows.** Same H for every row of W → compute once, reuse `d_out` times.
- **Computable with no backward.** Just forward + outer product. This is why GPTQ is *one-shot* PTQ.
- **PSD but often near-singular.** Calibration X has rank ≤ N, so H rank ≤ N. Damp with `H ← H + λI` where `λ = percdamp · mean(diag(H))`. Typical `percdamp = 0.01`. See [[gptq]] in ch-08.

OBC empirical result on BERT-Base 4-bit (Table 4 of the paper): **OBC 84.9 GLUE** vs AdaRound 84.3 vs BRECQ 84.6 vs FP 85.4. Within 0.5 of FP — at BERT scale, without QAT, in a few minutes per layer.

---

## 4. The block-reconstruction thread — BRECQ and QDrop

[[brecq]] is covered in ch-04 as a QAT-flavoured method. Reread it as a *PTQ* method here: BRECQ uses gradient descent on a small calibration set against a per-block reconstruction objective, not against the task loss. The objective:

```
min_{W_k}  E_X  ‖ f_k(X) − f̂_k(X; W_k) ‖²_F
```

where `‖·‖_F` is the Fisher-information-weighted Frobenius norm and `f_k` is the FP block, `f̂_k` is its quantized version. The Fisher diagonal `diag(F) = E[(∂L/∂y)²]` approximates the per-block task Hessian, computed via a single backward of the FP model.

**Why "block" is the sweet spot.** BRECQ tested four reconstruction grains:

| Grain | Per-block accuracy | Compute |
|---|---|---|
| Layer | sub-4-bit fails | cheap |
| **Block** (one residual sub-graph) | **best** | moderate |
| Stage | marginal gain | expensive |
| Network | impractical | very expensive |

The empirical winner is block, because residual connections create error couplings *within* a block that per-layer ignores. This finding directly motivates [[omniquant]] (ch-10), which adopts block-wise reconstruction for LLMs while replacing per-weight rounding (BRECQ's AdaRound inner loop) with *learnable equivalent transformations* (LWC + LET).

### QDrop — the dropout idea that complements BRECQ

[[qdrop]] adds one line to BRECQ's optimisation: at each forward, **randomly disable activation quantization per layer with probability p ≈ 0.5**. The effect is a regulariser that closes the calibration-vs-test distribution gap, decisive at ≤4-bit:

```python
for step in range(N):
    mask = bernoulli(p, size=L)                # NEW: drop quant per layer
    y_fp = f_block_fp(X)
    y_q  = f_block_q(X, dropout=mask)          # masked activation quant
    loss = ||y_fp - y_q||² + λ·reg(V)
    loss.backward()
```

At p = 0.5, each layer sees both regimes (quantized and non-quantized inputs from below) equally; the learned rounding becomes robust to either. Anneal p → 0 over the last 25% so the final rounding decisions are evaluated under the actual deployment forward.

**Empirical effect** on ResNet-18 at 2-bit: AdaRound 52.84% → BRECQ 51.93% → BRECQ+QDrop **54.27%**. The effect grows monotonically as bit-width drops. Several LLM PTQ recipes (notably [[omniquant]]) adopt QDrop-style stochasticity directly.

---

## 5. HAWQ — second-order *per-layer* mixed precision

The third pillar is Hessian-Aware Quantization ([[hawq]]). It answers a different question: given a global bit budget, which layers should get more bits?

The Taylor argument:

```
δL  ≈  (1/2) Σ_ℓ ΔW_ℓᵀ H_ℓ ΔW_ℓ
δL_ℓ  ≤  (1/2) λ_max(H_ℓ) · ‖ΔW_ℓ(b)‖²       ← bound by top eigenvalue
```

Per-layer sensitivity proxy:

```
Ω_ℓ  =  λ_max(H_ℓ) · ‖ΔW_ℓ(b)‖²
```

`λ_max(H_ℓ)` is estimated by **power iteration with Hutchinson** — no need to form H. Hessian-vector products via PyTorch's `torch.autograd.grad(grad·v, params)`. About 50 power steps × 10 minibatches per layer — minutes for ResNet-50, ~30 min for a 175B LLM.

**Bit allocation.** Greedy: start everything at 8-bit; drop the (layer, bit-width) pair with smallest Ω-increment per byte saved until the budget is met. ILP-optimal allocation exists but the greedy is within 0.1% accuracy at 100× the speed.

**Empirical effect.** ResNet-50 on ImageNet: 102 MB FP32 → 12 MB mixed 2/4-bit at 0.8% top-1 drop. BERT-Base via [[q-bert]] (HAWQ for transformers — covered in ch-06): 13× compression at 2.3 GLUE drop with allocation **{embedding: 8, mid-attention: 4, FFN: 3, pooler: 2}**.

**Why uniform 4-bit wins for modern LLMs.** Q-BERT's per-layer sensitivity range was 100×. For 7B+ LLMs, the per-layer Hessian eigenvalues compress dramatically — the loss landscape flattens with scale. Empirically, [[gptq]] gets equivalent quality at uniform 4-bit + `group_size=128` to mixed-precision allocations, with much simpler kernels. HAWQ's *idea* survives as the "which layers to skip" decision in production stacks ([[autogptq]] excludes lm_head and embed_tokens), but the *full mixed-precision allocation* fell out of fashion.

---

## 6. The PTQ taxonomy as one table

The synthesis of this chapter. Every paper from 2018 to 2026 fits in this grid; the column you live in tells you 80% of what the algorithm does.

| Paper / method | Granularity (W / A) | Calibration objective | Optimisation | Reconstruction grain |
|---|---|---|---|---|
| RTN (round-to-nearest) | per-tensor / per-tensor | none | none | per-weight |
| Krishnamoorthi 2018 | per-channel / per-tensor | min/max + KL | none (PTQ) or QAT | per-weight |
| DFQ ([[data-free-quantization]]) | per-tensor / per-tensor + CLE | range-equalise | closed-form | per-pair |
| [[adaround]] (ch-04) | per-channel / — | Hessian (X Xᵀ) | soft rounding | per-layer |
| [[brecq]] (ch-04) | per-channel / per-tensor | Fisher-weighted MSE | gradient | **per-block** |
| QDrop | inherits BRECQ | + Bernoulli mask | gradient | per-block (stochastic) |
| [[obs-obd]] (1993) | per-weight | Hessian | closed-form OBS | per-layer |
| [[obc]] (2022) | per-weight | Hessian + Cholesky | OBS sequential | per-layer |
| **[[gptq]] (ch-08)** | per-group / — | layer-MSE + Cholesky | OBS sequential + lazy batch | per-layer |
| [[hawq]] | per-layer bit-width | λ_max(H_ℓ) | greedy ILP | mixed-bit |

The right-most column is the most important diagnostic. Modern LLM PTQ split into two camps:

- **Per-layer OBS line**: GPTQ → SpQR → QuIP → QuaRot. Closed-form per-column updates, no gradient.
- **Per-block reconstruction line**: BRECQ → OmniQuant → AffineQuant → FlatQuant → SpinQuant. Gradient descent on equivalent transformations, block-grain MSE.

Both lines beat round-to-nearest by 3–10 perplexity points at W4. Neither line beat the other by more than ~0.3 ppl on Llama-2-7B as of 2024.

---

## 7. Practitioner's checklist when reading a new PTQ paper

```text
□ What is the activation scale placement? (per-token, per-channel, per-tensor?)
□ What is the weight scale placement?  (per-group, per-channel?)  group_size?
□ Symmetric or asymmetric weight quant?  (asymmetric ⇒ INT4 zero-point ⇒ ~0.1 bit overhead)
□ Calibration objective: task loss? layer MSE? block MSE? Fisher-weighted?
□ Reconstruction grain: per-weight / per-layer / per-block / per-stage?
□ Optimisation: closed-form (OBS) / gradient (AdaRound, BRECQ) / search (AWQ)?
□ Hessian approximation: full inv-Hessian (OBC), Cholesky factor (GPTQ),
   Fisher diagonal (BRECQ), top eigenvalue (HAWQ), or none (RTN)?
□ Calibration set size and source?  (typical: 128 sequences × 2048 tokens of C4)
□ Outlier handling: ignore (RTN) / isolate (LLM.int8) / migrate (SmoothQuant, AWQ)
   / preserve (SpQR, OWQ) / rotate (QuIP, QuaRot)?
```

Filling in this checklist for any paper takes 10 minutes and tells you whether the paper is genuinely novel or just a tweak.

---

## Connections and what's next

- **Forward to [[gptq]] (ch-08)** — the OBS thread becomes GPTQ via the Cholesky + lazy-batched update.
- **Forward to [[smoothquant]] (ch-09)** — DFQ's equivalent-transformation idea, applied to LLM activations instead of CNN weights.
- **Forward to [[awq]] (ch-09)** — per-channel scale `s_j` driven by activation magnitude, absorbed into adjacent ops (DFQ's invariance argument, again).
- **Forward to [[omniquant]] (ch-10)** — block reconstruction (BRECQ's grain) but with learnable equivalent transformations replacing AdaRound's per-weight rounding.
- **Back to [[adaround]] / [[brecq]] (ch-04)** — the QAT-flavoured cousins that motivate the gradient-based PTQ branch.
- **Back to [[uniform-quantization-noise]] (ch-01)** — Bennett's `σ_q² = Δ²/12` is the noise floor every method here is trying to beat.

## Further reading

- [[quantization-mapping]] — Krishnamoorthi 2018 whitepaper, the canonical PTQ taxonomy.
- [[data-free-quantization]] — Nagel 2019 DFQ.
- [[obs-obd]] — Hassibi & Stork 1993; the two-page paper everything inherits.
- [[obc]] — Frantar 2022; the OBS → OBC → GPTQ pipeline in one paper.
- [[brecq]] — Li 2021; the block-grain argument.
- [[qdrop]] — Wei 2022; the dropout regulariser that complements BRECQ.
- [[hawq]] — Dong 2019; per-layer Hessian sensitivity for mixed precision.
