<!-- chapter: ch-16
     track: 2024-maturation
     title: BitNet Lineage + Microscaling Formats
     sources: [[bitnet]], [[bitnet-b158]], [[bitnet-a48]], [[onebit]], [[era-of-1bit-llms]], [[microscaling-formats]], [[bitnet-models]], [[microsoft-bitnet]]
     figures: (none)
-->

# Chapter 16 — BitNet Lineage + Microscaling Formats

> **Core insight.** Sub-2-bit PTQ does not work below ~1.5 bits because the model never *learned* to be robust to that precision. The BitNet line takes the opposite path: **train from scratch with quantization in the loop**. The result is a new scaling law where 1.58-bit ternary LLMs match FP16 perplexity at ≥ 3B parameters with ~10× less inference memory and ~5× less energy. The companion development is the **OCP MX format family** — a 32-element block + 8-bit shared exponent (E8M0) standard that NVIDIA, AMD, Intel, ARM all agree on, productionised on Blackwell as MXFP4/MXFP6/MXFP8.
>
> **Guideline.** For 1-bit-class LLM training: use BitLinear (sign-quant W + per-token absmax INT8 X + SubLN before each binary matmul, STE backward). Upgrade to BitNet b1.58 (ternary {−1, 0, +1} via per-tensor absmean rounding) once you can train ≥ 3B parameters. For deployment, grab `microsoft/BitNet-b1.58-2B-4T` and bitnet.cpp. For any sub-8-bit numerical format design in 2025, default to the OCP MX layout (32-element blocks, E8M0 scale) and pick the element type (FP4/FP6/FP8/INT8) per use case.

---

## Why this chapter exists

ch-13 / ch-14 reach the 2-bit PTQ frontier with QuIP# / AQLM. Below 2 bits, PTQ collapses — *no* clever rounding, codebook, or rotation gets you to 1 bit on an FP-trained model with usable accuracy. The reason is structural: FP-trained weights occupy a continuous distribution whose information content cannot be compressed below the rate-distortion limit set by their entropy. To go below 2 bits you must change *what gets trained*, not how you compress what already exists.

BitNet (Wang et al. 2023) does exactly that — replaces every `nn.Linear` with a BitLinear layer that binarises weights via `sign(W)` during the forward pass and trains the model from scratch. BitNet b1.58 (Ma et al. 2024) extends to ternary `{−1, 0, +1}` and reaches **FP16 perplexity parity at 3B parameters**. The 2024–2025 follow-ups (a4.8, OneBit, bitnet.cpp, Microsoft's released 2B-4T checkpoint) consolidate this into a working production reality.

The Microscaling formats are the parallel track for **hardware-native low-bit numerics**. Where BitNet is "sub-2-bit weights with INT8 activations", MX is "FP-family sub-8-bit weights *and* activations *and* gradients" with hardware support on Blackwell. MX is the format reference you'll see across DeepSeek V3 (FP8), Llama 3.1 quantization (MXFP4), GPT-OSS (MXFP4), and every 2025 Anthropic-class lab.

Three takeaways:

1. The **BitLinear layer** — sign-quant W + absmax-INT8 X + SubLN + STE — as a drop-in `nn.Linear` replacement.
2. The **b1.58 ternary rule** with per-tensor absmean rounding, why the zero state matters, why log₂(3) ≈ 1.58 bits.
3. The **OCP MX layout** — 32-element block, E8M0 shared exponent, element formats (E2M1 / E2M3 / E3M2 / E4M3 / INT8), and why this is the cross-vendor standard.

Plus the production framings: Microsoft's released 2B-4T checkpoint, bitnet.cpp ternary kernels, OneBit's SVID alternative path to ~1 bit.

---

## 1. BitNet — the BitLinear layer

[[bitnet]] (Wang et al. 2023) is the first scalable 1-bit transformer trained from scratch. The architectural change is to replace every linear projection with a **BitLinear** layer.

### 1.1 Weight binarisation

For each BitLinear with latent FP weight `W ∈ ℝ^{d_out × d_in}`:

```math
\begin{aligned}
\alpha &= \frac{1}{d_{\text{out}} \cdot d_{\text{in}}} \sum_{i,j} W_{i,j}     \quad \text{(scalar mean)} \\
\tilde{W} &= \mathrm{Sign}(W - \alpha)                                          \quad \text{(± 1 binarisation)} \\
\beta &= \frac{1}{d_{\text{out}} \cdot d_{\text{in}}} \|W - \alpha\|_1            \quad \text{(absmean scale)}
\end{aligned}
```

The forward uses `β · W̃` (signs scaled back up). Mean-centering removes any DC component that would otherwise be wasted by the sign function.

### 1.2 Activation quantization (per-token absmax INT8)

```math
\gamma = \max_i |x_i|, \quad \tilde{x} = \mathrm{round}\left(\mathrm{clip}\left(x \cdot \frac{127}{\gamma}, -128, 127\right)\right)
```

Symmetric INT8 per token. Asymmetric uint8 variant for post-ReLU/GELU activations.

### 1.3 SubLN — the stability trick

Insert a LayerNorm immediately before the binary matmul:

```math
y = \beta \cdot \mathrm{SubLN}(\tilde{x}) \cdot \tilde{W}^\top \cdot \frac{\gamma}{127}
```

SubLN normalises the activations to roughly unit variance before they encounter the {±1} weight, keeping dot-product magnitudes well-controlled across depth. Without SubLN, BitNet training is unstable past a few B parameters.

### 1.4 Backward — straight-through estimator

```math
\frac{\partial \mathcal{L}}{\partial W} \approx \frac{\partial \mathcal{L}}{\partial \tilde{W}} \quad \text{(through Sign-STE)}
\qquad
\frac{\partial \mathcal{L}}{\partial x} \approx \frac{\partial \mathcal{L}}{\partial \tilde{x}} \quad \text{(through round-STE)}
```

The optimizer updates the **latent FP weight** W; the binarisation is recomputed each forward. The latent FP weight is kept in BF16/FP32 and never discarded — quantization happens only on the forward path. This is the key difference between BitNet and PTQ: the *training signal* sees the quantization, so the latent weights drift to a configuration that's quantization-robust.

### 1.5 The PyTorch sketch

```python
class BitLinear(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.W = nn.Parameter(torch.randn(d_out, d_in) * (1 / d_in ** 0.5))
        self.subln = RMSNorm(d_in)

    def weight_quant(self, W):
        alpha = W.mean()
        W_centered = W - alpha
        sign_W = torch.sign(W_centered)
        beta = W_centered.abs().mean()
        # STE: forward uses quantized, backward uses identity through sign
        W_q = sign_W * beta
        return W + (W_q - W).detach()

    def act_quant(self, x):
        gamma = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-5)
        x_q = (x * 127 / gamma).round().clamp(-128, 127) * (gamma / 127)
        return x + (x_q - x).detach()

    def forward(self, x):
        x = self.subln(x)
        x_q = self.act_quant(x)
        W_q = self.weight_quant(self.W)
        return F.linear(x_q, W_q)
```

The `x + (x_q - x).detach()` pattern is the standard STE trick: forward returns `x_q`, backward gets gradient through `x` as identity.

### 1.6 Scaling law

BitNet trained from scratch up to 13B parameters exhibits a scaling law **parallel** to FP16 — same slope, modest offset. The offset shrinks with scale (more parameters → more "headroom" for the model to adapt to binary weights). The headline plot (BitNet Fig 4) shows the two curves converging asymptotically.

---

## 2. BitNet b1.58 — ternary weights with absmean rounding

[[bitnet-b158]] (Ma et al. 2024) extends BitNet by quantizing weights to **ternary {−1, 0, +1}** instead of binary {−1, +1}. The third state — zero — costs only log₂(3) ≈ 1.58 bits per weight but recovers the missing expressivity. **Headline**: matches FP16 perplexity from-scratch starting at 3B parameters.

### 2.1 The absmean rounding rule

For each layer's latent FP weight `W ∈ ℝ^{d_out × d_in}`:

```math
\begin{aligned}
\gamma &= \frac{1}{d_{\text{out}} \cdot d_{\text{in}}} \sum_{i,j} |W_{i,j}|        \quad \text{(absmean, scalar per-tensor)} \\
\tilde{W} &= \mathrm{RoundClip}\left(\frac{W}{\gamma + \epsilon}, -1, +1\right) \\
\text{where } \mathrm{RoundClip}(x, a, b) &= \max(a, \min(b, \mathrm{round}(x)))
\end{aligned}
```

The result `W̃_{i,j} ∈ {−1, 0, +1}`. The per-tensor scale γ is the absmean — provably the minimum-MSE 1-level magnitude under absolute error, and a tight choice for the ternary code.

Forward: `y = γ · (W̃ · x̃)` where `x̃` is the INT8-quantized activation (same per-token absmax as BitNet).

```python
def weight_quant_b158(W, eps=1e-5):
    scale = W.abs().mean().clamp(min=eps)
    # round to nearest in {-1, 0, +1}, then divide back
    W_q = (W / scale).round().clamp(-1, 1)
    return W_q * scale
```

Backward: STE through RoundClip.

### 2.2 Why log₂(3) = 1.58

A ternary cell has 3 states. Information-theoretic minimum storage = log₂(3) ≈ 1.585 bits.

**Practical packing**: pack 5 ternary weights into 8 bits because `3^5 = 243 ≤ 256`. Resulting density: **1.6 bits/weight**.

```
Base-3 encoding:   weight_byte = w_0 + 3·w_1 + 9·w_2 + 27·w_3 + 81·w_4
                   each w_i ∈ {0, 1, 2} (mapped from {−1, 0, +1})
                   8 bits / 5 weights = 1.6 bits/weight
```

5 of 256 byte values are unused; trivial overhead.

### 2.3 Why the zero matters

Pure binary `{−1, +1}` cannot express "this connection is off". Empirically, real LLM weights cluster heavily near zero (Gaussian prior). Binarizing zero to ±1 forces every weight to contribute, which costs both accuracy and effective capacity.

Ternary `{−1, 0, +1}` lets the model express "off" at the cost of 0.58 bits/weight. Empirically this is the difference between b1.58 reaching FP16-parity at 3B and pure BitNet trailing by a few ppl indefinitely.

### 2.4 Scaling behavior (the headline)

| Params | b1.58 ppl | FP16 ppl | Gap |
|--------|-----------|----------|-----|
| 700M | +1.5 | baseline | b1.58 worse |
| 1.3B | +1.0 | baseline | b1.58 worse |
| **3B** | **same** | baseline | **parity** |
| 13B | slightly better | baseline | b1.58 ≥ FP16 |
| 70B (extrapolated) | better | baseline | b1.58 > FP16 |

The "crossover at 3B" claim is the central empirical result. At ≥ 3B, the discrete weight space acts as regularization — the model's effective capacity is unchanged because the latent FP weights have plenty of redundancy to encode within the ternary codebook.

### 2.5 Inference

- Weight stored as ternary, 5-per-byte base-3 packing.
- Activation INT8 per-token.
- Matmul: integer multiply-accumulate with output rescale by `γ · γ_x / 127`.
- **Zero-skipping**: any 0-weight contributes nothing → can be skipped in custom kernels, gaining ~30% sparsity speedup empirically.

```
For w ∈ {-1, 0, +1}:
    w = +1:  y += x
    w = -1:  y -= x
    w = 0:   skip
```

No multiplier needed. This is what `bitnet.cpp` exploits.

---

## 3. BitNet a4.8 — 4-bit activations

[[bitnet-a48]] (Wang et al. 2024) extends b1.58 with **4-bit activations**. The naive A4 fails because certain intermediate tensors (especially FFN gate/down inputs) contain outlier channels that crush INT4 dynamic range. The fix is a **hybrid quantization-sparsification** strategy.

### 3.1 Absmean activation scaling

Standard absmax: `scale = max(|x|) / 7`. BitNet a4.8 uses **absmean variant**:

```
scale = mean(|x|) · k       (k = learned per-layer multiplier)
```

Absmean is more robust to single-token outliers than absmax — important when the underlying weights are ternary and have no precision headroom to absorb activation quant error.

### 3.2 Hybrid routing per tensor position

| Position | Quant |
|----------|-------|
| Attention Q, K, V inputs | INT4 absmean |
| FFN gate/down inputs | INT8 + sparsification (top-K) |
| Output projection inputs | INT4 absmean |
| KV cache | 3-bit per token |

### 3.3 Sparsification (FFN intermediate)

```math
\tilde{h}_i = \begin{cases} h_i & \text{if } |h_i| \ge \tau_k(|h|) \\ 0 & \text{else} \end{cases}
```

where τ_k is the K-th largest absolute value. Sparsity ratio ~45% (so ~55% of activations live). Multiply-accumulate skips zero entries.

The FFN intermediate is the most outlier-prone position in any LLM; sparsifying past the top-K kills the outlier tail without crushing precision.

### 3.4 Result

a4.8 matches b1.58 quality at the same training cost, with **INT4 tensor-core kernels** for faster inference and ~55% of parameters active per token. Net: ~2× decode throughput vs b1.58 at the same memory footprint.

---

## 4. OneBit — true 1-bit weights via SVID

[[onebit]] (Xu et al. NeurIPS 2024) takes a different route to 1 bit. Instead of ternary `{−1, 0, +1}` with one global scale, factor each weight matrix as **sign × per-row scale × per-column scale**:

```math
W \approx \mathrm{diag}(a) \cdot S \cdot \mathrm{diag}(b)
```

- `S ∈ {−1, +1}^{m × n}`: binary sign matrix, 1 bit per element.
- `a ∈ ℝ^m`: per-row scale (FP16).
- `b ∈ ℝ^n`: per-column scale (FP16).

Effective bits/weight = `1 + 16/n + 16/m ≈ 1` for large `m, n`.

This is **Sign-Value-Independent Decomposition (SVID)** — disentangles the sign (1-bit) from the magnitude (two FP vectors), preserving the rank-1 dominant magnitude structure that single-scale binarisation loses.

### 4.1 Decomposition initialisation

```
b_j = (1/m) Σ_i |W_{ij}|                # per-column mean
a_i = (1/n) Σ_j |W_{ij}| / b_j           # per-row mean
S_{ij} = sign(W_{ij})
```

Then refine via alternating SVD-style update: re-derive a, b given S and vice versa, minimising `||W − diag(a) S diag(b)||_F`.

Without this init, randomly assigned S leads to massive initial loss that STE fine-tuning cannot recover.

### 4.2 Inference math

```
y = W x ≈ diag(a) · (S · (diag(b) · x))
```

1. `x' = diag(b) · x` — n FP multiplies.
2. `y' = S · x'` — each `y'_i = Σ_j S_{ij} · x'_j`. With S ∈ {−1, +1}, each MAC is a sign-flipped add. Implementable as XNOR + popcount on bit-packed S.
3. `y = diag(a) · y'` — m FP multiplies.

Total: O(mn) sign-MACs + O(m+n) FP multiplies (amortised negligible).

### 4.3 OneBit vs b1.58

| | OneBit | BitNet b1.58 |
|--|--------|--------------|
| States | {−1, +1} | {−1, 0, +1} |
| Bits/weight | ~1 | log₂(3) ≈ 1.58 |
| Scale | per-row + per-column (rank-1) | per-tensor |
| Training | QAT fine-tune of FP base | scratch |
| Quality | 81% of FP | 100% at 3B+ |

OneBit reaches **81%+ of non-quantized FP performance** at 1 bit/weight, dramatically below prior 1-bit baselines (PB-LLM, BiLLM) which sit at ~70%. The trade-off vs b1.58: OneBit is reachable by *fine-tuning an existing FP LLM* — no need to retrain from scratch — at the cost of trailing FP by ~20%.

---

## 5. The Era of 1-bit LLMs — three-axis scaling law

[[era-of-1bit-llms]] consolidates the thesis. The implication of BitNet b1.58's parity result: the **FP16 baseline that the rest of the field uses is sub-optimal**. The cost-optimal frontier on the `(parameters × tokens × bits/weight)` volume sits at sub-2-bit weights.

Informal scaling fit:

```math
\mathrm{Loss}(N, D, b) \approx A \cdot N^{-\alpha} + B \cdot D^{-\beta} + C(b)
```

with `C(b)` plateauing for `b ≥ 1.58` — every bit beyond ternary is wasted at fixed N, D. Compute cost scales linearly in `b` for HBM-bound regimes (decode), making sub-2-bit strictly cheaper at iso-quality.

### 5.1 Fine-tune-to-1.58 (lambda schedule)

The community follow-up (HuggingFace blog) shows you can convert an existing FP16 LLM to 1.58-bit via gradual quantization:

```python
def lambda_quant_step(t, T, w):
    lam = min(2 * t / T, 1.0)
    w_quant = w + lam * (weight_quant_b158(w) - w).detach()
    return w_quant
```

`λ_t` ramps from 0 to 1 over the first half of training, then stays at 1. Warms the model up to ternary precision rather than abruptly switching at step 0.

### 5.2 Hardware implication

A ternary MAC `y += w · x` with `w ∈ {−1, 0, +1}` reduces to:
- `w = +1`: `y += x`
- `w = −1`: `y −= x`
- `w = 0`: skip

**No multiplier needed.** Lookup-table-based MACs in custom silicon project 10–70× energy reduction vs FP16 (preliminary estimates, BitNet white paper).

---

## 6. Microsoft's BitNet model releases + bitnet.cpp

[[bitnet-models]] catalogues the actual released checkpoints. The "Era of 1-bit LLMs" thesis is now a working production reality.

### 6.1 Released checkpoints

| Org / Repo | Model | Bits | Params | Training tokens | Notes |
|------------|-------|------|--------|-----------------|-------|
| `microsoft/BitNet-b1.58-2B-4T` | BitNet b1.58 | 1.58 | 2B | 4T | **Official reference**; from scratch |
| `microsoft/BitNet-b1.58-2B-4T-gguf` | same | 1.58 | 2B | 4T | gguf for bitnet.cpp |
| `1bitLLM/bitnet_b1_58-large` | BitNet b1.58 | 1.58 | 700M | ~100B | Community proof |
| `1bitLLM/bitnet_b1_58-3B` | BitNet b1.58 | 1.58 | 3B | ~100B | Community larger |
| `HF1BitLLM/Llama3-8B-1.58-100B-tokens` | Llama-3-8B → 1.58 | 1.58 | 8B | +100B continued | Post-hoc converted |
| `tiiuae/Falcon-E-*` | Falcon-E | 1.58 | 1B–10B | varied | TII ternary family |

### 6.2 bitnet.cpp inference framework

Forked from llama.cpp; reuses the ggml tensor library + gguf format, replaces the integer/float matmul kernels with **ternary-specialised** ones.

**Kernel idea**: weights stored as 5 ternary per byte (base-3 packing). At matmul time, decode the byte → 5 ternary signs, then accumulate `±x_i` or skip-zero. No multiply needed.

**Lookup-table optimization**: precompute the `3^5 = 243`-entry LUT for 5-element ternary blocks → integer-only inner loop.

**Platforms & speedups**:
- x86 (AVX2 + AVX-VNNI): **2.37–6.17×** vs llama.cpp q2_K.
- ARM (NEON): **1.37–5.07×**, optimized for Apple Silicon.
- GPU (May 2025): CUDA + Metal kernels.
- NPU: Qualcomm + Intel NPU backends in development.

**Energy**: **55–82% reduction** vs equivalent FP16 inference.

### 6.3 Quality claims

On the standard 0-shot harness, BitNet-b1.58-2B-4T is **within 1-2 points** of Llama-3.2-1B and Qwen2.5-1.5B at FP16 across MMLU / HellaSwag / Winogrande / ARC / GSM8K — at the same parameter count but **~10× smaller memory footprint** and **~5× lower energy**.

The 2B-4T release is the moment "Era of 1-bit LLMs" stops being a paper claim and becomes a deployable artefact.

---

## 7. Microscaling formats — the OCP MX standard

[[microscaling-formats]] (Rouhani et al. 2023) defines the **MX format family**: a shared 8-bit power-of-two exponent per 32-element block, combined with narrow (4–8 bit) element representations.

This is the production-grade alternative to BitNet's ternary path: instead of pushing weights to 1.58 bits with the rest of the stack in INT8, the MX family pushes everything (weights, activations, gradients) to sub-8-bit FP/INT with shared-exponent blocks.

### 7.1 The MX block structure

A block of 32 elements `(v_1, ..., v_32)` is stored as:

```
X (shared scale, 8 bits): an E8M0 exponent representing 2^X
(d_1, ..., d_32) (per-element value, k bits each)
```

Decoded value: `v_i = 2^X · d_i`.

Shared scale is **power-of-two only** (no mantissa), so dequantization is a **shift, not a multiply** — cheap in hardware.

### 7.2 Element formats

| Format | Bits | Composition | Representable | Bits/element (with scale) |
|--------|------|-------------|---------------|----------------------------|
| **E2M1 (MXFP4)** | 4 | 1 sign + 2 exp + 1 mantissa | ± {0, 0.5, 1, 1.5, 2, 3, 4, 6} | 4.25 |
| **E2M3 (MXFP6)** | 6 | 1 sign + 2 exp + 3 mantissa | 16 distinct positive vals | 6.25 |
| **E3M2 (MXFP6 alt)** | 6 | 1 sign + 3 exp + 2 mantissa | wider range, coarser | 6.25 |
| **E4M3 (MXFP8)** | 8 | 1 sign + 4 exp + 3 mantissa | same as OFP8 E4M3 | 8.25 |
| **MXINT8** | 8 | twos-complement integer | −127..127 | 8.25 |

### 7.3 Bit accounting

For 32-element block:

```
MXFP4:  8-bit scale + 32 × 4-bit values = 136 bits / 32 elements = 4.25 bits/element
MXFP6:  8 + 32 × 6 = 200 / 32 = 6.25 bits/element
MXFP8:  8 + 32 × 8 = 264 / 32 = 8.25 bits/element
```

**Scale overhead is always 0.25 bits/element** — the "free" cost of microscaling.

### 7.4 Why 32-element blocks

- Small enough to fit one tensor-core fragment (warp-level cooperation on H100/Blackwell).
- Small enough to track local outliers (per-channel variation captured by per-block scale).
- Large enough that the 8-bit scale is amortised to < 1 bit/element.

The 32 is not arbitrary — it matches the tensor-core fragment size, so MX dequant is essentially free in the GEMM pipeline.

### 7.5 Hardware support

**Blackwell** (NVIDIA) and follow-on **MI3xx** (AMD) ship native MX support: tensor-core fragments load 32-element blocks with the shared E8M0 scale and produce FP32 accumulators directly.

**NVFP4** is a Blackwell extension with FP8 block scale + FP32 tensor scale (a 2-level hierarchy on top of MX) — closer to DeepSeek V3's fine-grained scaling philosophy. NVFP4 sits alongside MX as a parallel option for inference deployment.

### 7.6 Training vs inference

MX supports both: forward (weights × activations in MX), backward (gradients in MX), optimizer state often kept FP32. Microsoft's MX paper trains generative LMs at MXFP6 / MXFP4 weights+activations+gradients at FP32-parity downstream accuracy.

This is the bridge to **ch-17** (low-precision pretraining) where MXFP4/NVFP4 are the production formats for native sub-FP8 training.

---

## 8. BitNet vs MX — two paths to sub-8-bit

| | BitNet b1.58 | Microscaling MX |
|--|--------------|-----------------|
| Weight bits | 1.58 (ternary) | 4–8 (FP/INT element) |
| Activation bits | 8 (a8) or 4 (a4.8) | 4–8 |
| Scale unit | per-tensor scalar | per-block (32 elements) |
| Hardware | custom kernels (bitnet.cpp); no native HW | Blackwell native + AMD MI3xx |
| Training | from scratch, BitLinear | from scratch, MX everywhere |
| Inference | integer add/sub, no multipliers | tensor-core GEMM with MX dequant |
| Energy claim | 55–82% reduction | 30–50% reduction |
| Quality crossover | parity at 3B | parity at any scale (FP32-equivalent benchmark) |

**When to pick which:**
- **BitNet** if you control the silicon (NPU, custom ASIC) and want minimum-energy inference.
- **MX** if you target Blackwell / MI3xx and want production training + inference paths.

Both are pretrain-from-scratch routes; neither is a PTQ. PTQ tops out at ~2 bits (ch-14); from-scratch with BitLinear or MX gets you below 2 bits or to sub-FP8 training.

---

## 9. Pitfalls

- **BitLinear's latent FP weight is permanent.** You cannot discard it after training and "switch to ternary only"; the optimizer needs the FP weight to update. For deployment, take the ternary forward output as the final artefact — the FP weight is irrelevant.
- **SubLN placement is critical.** It must be *immediately before* the binary matmul, not buried somewhere else in the block. Misplaced SubLN destabilises training at ~1B+.
- **STE assumes the gradient direction is correct.** For Sign() with mean-centering, this holds approximately. For other discrete quantizers (lattice, codebook) STE can be very biased — use PV-Tuning (ch-14) instead.
- **b1.58 LR is ~5× the FP16 baseline LR.** Sign(·) flattens the loss surface; bigger steps help. Don't reuse FP16 LR schedule directly.
- **b1.58 doesn't reach parity below 3B.** If you're training a 700M model expect a 1.5 ppl gap. b1.58 is an asymptotic-parity claim.
- **OneBit's SVID must be re-initialised per matrix.** Shared a, b vectors across layers destroy the per-layer magnitude structure.
- **MX block size 32 is fixed by the spec.** Don't experiment with block 16 or 64 — your hardware path falls off the native MX support.
- **E8M0 scale has no mantissa.** This means dequant is a shift; it also means the representable scale set is `2^k` for integer k. For activations with non-power-of-2 dynamic range, you lose 0.5 bits of effective precision — accepted as the cost of free dequant.
- **MX inference on Ampere needs emulation.** Native MX is Blackwell+ only; emulated MX on H100 is 2–3× slower than native FP8.
- **bitnet.cpp model conversion is one-way.** Once you've converted to gguf-ternary you cannot continue training; treat it as a deployment artefact.

---

## Connections and what's next

- **[[bnn]] / [[xnor-net]] / ch-04** — the original 1-bit weight + activation nets; BitNet is the LLM-scale realisation of the same idea with the additional STE + SubLN + scratch-training tricks.
- **[[straight-through-estimator]] / ch-04** — STE is the backward used in BitLinear; the discrete-fine-tune problem is ultimately resolved by [[pv-tuning]] (ch-14) for codebook methods, but for sign-quant STE is still the default.
- **[[lsq]] / ch-04** — learned step-size QAT; the "learnable scale" intuition surfaces in BitNet a4.8's absmean activation scaling (k is a learned per-layer multiplier).
- **[[fp8-formats-paper]] / ch-02** — FP8 cousins of MX; FP8 E4M3 is the MXFP8 element format. DeepSeek V3 FP8 native training (ch-17) uses fine-grained scaling that's structurally similar to MX with different block sizes.
- **[[bitnet-models]]** — the production-checkpoint side of this chapter; covered in §6.
- **[[nvfp4-training]] / [[mxfp4-pretraining]] / ch-17** — the natural follow-up: native pretraining at MXFP4/NVFP4 on Blackwell.

## Further reading

- [[bitnet]] — Wang et al. 2023, the original 1-bit transformer paper.
- [[bitnet-b158]] — Ma et al. 2024, ternary weights + scaling-law parity.
- [[bitnet-a48]] — Wang et al. 2024, 4-bit activations.
- [[onebit]] — Xu et al. NeurIPS 2024, SVID decomposition for ~1-bit weights.
- [[era-of-1bit-llms]] — survey-style consolidation of the 1.58-bit thesis.
- [[microscaling-formats]] — Rouhani et al. 2023, the OCP MX format paper.
- [[bitnet-models]] — Microsoft + community released checkpoints; bitnet.cpp framework.
- [[microsoft-bitnet]] — lab summary; Furu Wei, Hongyu Wang, Shuming Ma.

## Excerpts

- [excerpts/bitnet.md](excerpts/bitnet.md) — the BitLinear layer in full, sign quantization, SubLN, STE.
- [excerpts/bitnet-b158.md](excerpts/bitnet-b158.md) — ternary absmean rule, log₂(3) packing, scaling-law table.
- [excerpts/onebit.md](excerpts/onebit.md) — SVID decomposition, sign matrix + per-row/col scales.
- [excerpts/microscaling-formats.md](excerpts/microscaling-formats.md) — OCP MX block layout, element format catalogue, hardware support.
- [excerpts/bitnet-models.md](excerpts/bitnet-models.md) — released checkpoints, bitnet.cpp framework, speedup + energy numbers.
