<!-- chapter: ch-12
     phase: 2023-refinements
     title: QLoRA + NF4 — 4-bit Fine-Tuning at Consumer Scale
     sources: [[qlora]], [[nf4]], [[qa-lora]], [[loftq]], [[llm-qat]], [[lq-lora]], [[peqa]], [[bitsandbytes-nf4]]
-->

# Chapter 12 — QLoRA + NF4: 4-bit Fine-Tuning at Consumer Scale

> **Core insight.** Fine-tuning a 65B model on a single 48 GB GPU is possible if (a) the frozen base is stored in an information-theoretically near-optimal 4-bit code for the Gaussian weight prior (NF4), (b) the per-block scales are themselves quantized to FP8 to recover ~0.4 bits/weight (double quantization), (c) optimizer states are paged through CPU via NVIDIA unified memory, and (d) gradients flow through a fused dequant kernel into a BF16 LoRA adapter that is the only thing trained. Same final quality as 16-bit full fine-tune; 16× less GPU.
>
> **Guideline.** For SFT / instruction-tuning under tight GPU budgets, use QLoRA with `bnb_4bit_quant_type="nf4"`, `bnb_4bit_use_double_quant=True`, `bnb_4bit_compute_dtype=torch.bfloat16`, paged AdamW (`optim="paged_adamw_32bit"`), and LoRA `r=64` on **all** linear layers (q, k, v, o, gate, up, down) — not just attention. Effective bits ≈ 4.127. Fits LLaMA-65B fine-tuning into a single 48 GB GPU in ~24 hr.

---

## Why this chapter exists

[[ch-08]] and [[ch-09]] showed how to *infer* with 4-bit weights at near-FP quality. [[ch-11]] pushed further to 3-bit with sensitivity-aware codebooks. None of those address **fine-tuning** — they assume you have a well-trained model and only need to deploy it cheaply. For training, the conventional path was: load FP16 weights (140 GB for LLaMA-65B) + FP32 gradients + AdamW states (~560 GB) → ≥10 A100s.

QLoRA ([[qlora]], NeurIPS 2023) compressed all of that into a single 48 GB consumer-class GPU. It is the single most influential paper for open-source LLM adoption in 2023–2024: every HuggingFace 4-bit fine-tune since uses this recipe. The Guanaco family (7B / 13B / 33B / 65B) fine-tuned in 24 GPU-hours reached 99.3% of ChatGPT's Vicuna benchmark.

The recipe stacks four innovations:

1. **NF4** ([[nf4]]) — a 4-bit quantile code optimal for the Gaussian weight prior. The format itself.
2. **Double Quantization** — quantize the per-block scales (FP32) down to FP8 with a per-group outer scale. Saves ~0.37 bits/weight.
3. **Paged Optimizers** — page AdamW states between GPU and CPU memory via NVIDIA UVM, absorbing the gradient-spike memory peaks that otherwise OOM.
4. **Frozen 4-bit base + BF16 LoRA** — gradients flow through on-the-fly dequantization into the BF16 LoRA adapter; the base never leaves 4-bit memory.

This chapter walks all four, then surveys the alternatives that pushed the quant-aware-finetune frontier: **QA-LoRA** (merge-friendly), **LoftQ** (joint init), **LLM-QAT** (full QAT, data-free), **LQ-LoRA** (low-rank-plus-quantized decomposition with ILP bit allocation), and **PEQA** (scale-only PEFT).

---

## 1. NF4 — the 4-bit code for Gaussian weights

LLM weights, after per-block absmax normalization, are **well modelled by a zero-mean unit-variance normal**. The MSE-optimal 4-bit code for this prior is approximately the Lloyd-Max quantizer for N(0,1) ([[lloyd-max-quantizer]], ch-03). Dettmers took a simpler near-optimal route: place the 16 reconstruction levels at **equal quantiles of the standard normal CDF**, restricted to the symmetric range so |max| = 1 matches per-block absmax normalization.

### The 16 NF4 values (from QLoRA Table 14, bitsandbytes source)

```
[-1.0,
 -0.6961928,
 -0.5250730,
 -0.39491748,
 -0.28444138,
 -0.18477343,
 -0.09105475,
  0.0,
  0.07958029,
  0.16093020,
  0.24611230,
  0.33791524,
  0.44070983,
  0.56261432,
  0.72295684,
  1.0]
```

8 negative + zero + 7 positive = 16 levels. Asymmetric (one extra negative) because the quantile placement around zero isn't perfectly symmetric.

### Construction

1. Compute the symmetric N(0,1) quantile function `Q(p) = √2 · erf⁻¹(2p - 1)`.
2. Take 8 quantile points on the positive side (shifted so the extreme = 1.0).
3. Mirror across zero for the negative side; include 0.0.
4. Normalize so |max| = 1 (matches per-block absmax).

### Why NF4 beats INT4 on LLM weights

INT4 reconstruction values: uniform spacing in `[-1, +1]` → `0, ±1/7, ±2/7, ..., ±1`. **Wastes resolution at the tails** (low Gaussian mass) and **is too coarse near zero** (high Gaussian mass).

NF4 spacing: **dense near 0, sparse at the tails** — matches the Gish-Pierce `p^{1/3}` optimal density up to the symmetric normalization constraint ([[information-theoretic-bounds]], ch-01).

Empirical gain: ~0.3–0.5 PPL on LLaMA-class models at 4-bit; larger gain at lower bit-widths.

### Comparison with FP4 E2M1

| | NF4 | FP4 E2M1 |
|---|---|---|
| Levels | 16, quantile-spaced | 16, log-spaced |
| Best for | unimodal Gaussian weights | log-magnitude data |
| Hardware | software dequant only | Blackwell native |
| LLM weight PPL | best | second |

NF4 has no native tensor-core support (non-uniform LUT prevents direct integer multiply). The dequant runs as a 16-entry LUT lookup → BF16 → tensor core. Modern Marlin/Machete kernels handle NF4 the same way.

> **Pitfall.** NF4's Gaussian assumption is **not universal**. Some early transformer layers and RMSNorm gain parameters are heavy-tailed; INT4 with per-group scale may match or exceed NF4 there. NF4 is also **not used for activations** (post-GeLU/SiLU is heavy-tailed positive, not Gaussian).

---

## 2. Per-block quantization and the bit budget

NF4 is stored block-wise:

```
block size B = 64 weights
per-block scale s = max(|w_b|)  (FP32 or FP16)
per-weight: 4-bit NF4 index q_i,  reconstruct as  w_i ≈ s · NF4[q_i]
```

**Naïve bit budget:**

```
4 (NF4 index) + 32 / 64 (FP32 scale per 64 weights) = 4.5 bits/weight
```

That 0.5-bit overhead is the obvious place to compress.

---

## 3. Double Quantization — saving the scale overhead

Per-block FP32 scales `{s_0, s_1, ...}` are themselves quantized:

- Group every **256 block-scales** together.
- Quantize them to **FP8** with one FP32 outer scale per 256-block group.

### Bit budget with double quant

```
inner overhead:  8 / 64       = 0.125 bits/weight
outer overhead:  32 / (64·256) ≈ 0.002 bits/weight
total:           4 + 0.125 + 0.002 ≈ 4.127 effective bits/weight
```

Savings vs naïve: `4.5 - 4.127 ≈ 0.37` bits/weight. On a 65B model that's ~3 GB of memory recovered for free, at essentially no quality cost.

---

## 4. Paged Optimizer — absorbing gradient spikes

AdamW state for a 65B model: `2 × 65 × 10⁹ × 4 = 520 GB` (two FP32 moments per parameter, 4 bytes each). Even with QLoRA's frozen base, the trainable LoRA adapters + gradient + AdamW state can still spike past 48 GB during backward.

**Paged Optimizer** uses NVIDIA Unified Memory (UVM): optimizer chunks live in CPU DRAM and are paged into GPU HBM on demand. The OS handles the swap; when a backward step transiently spikes memory, instead of OOM-ing, optimizer pages get evicted to CPU. Throughput cost: small, because the paging is amortised over the gradient computation.

Without paging, even LoRA-only fine-tuning of LLaMA-65B at long context will OOM intermittently. With paging, it just runs slower for a few steps and recovers.

> **Pitfall.** Paged optimizer requires CUDA UVM support and a recent driver. If you see severe slowdowns, verify the optimizer chunks aren't pinned in CPU DRAM. The optimizer config is `optim="paged_adamw_32bit"` in HF `TrainingArguments`.

---

## 5. The QLoRA dataflow

For each Linear layer `y = W·x` in the model:

- `W` is frozen 4-bit NF4 (with double-quantized scales).
- A LoRA delta `ΔW = (α/r) · B · A` is added, with `A ∈ R^{r × d_in}` and `B ∈ R^{d_out × r}` in **BF16**, both trainable.

```math
y = \mathrm{dequant}(W) \cdot x + \frac{\alpha}{r} \cdot B (A x)
```

**Forward:** the NF4 weight is dequantised on-the-fly into BF16 inside a fused GEMV kernel (it never materialises as a full BF16 matrix in HBM — it streams through the kernel tile-by-tile). The LoRA path adds a small extra BF16 matmul.

**Backward:** gradients flow through the BF16 dequant op into `A, B`. **`W` never receives gradients.** The 4-bit base is read-only during training.

This is the critical design: training memory is `(NF4 base: ~32 GB for 65B) + (LoRA adapters + AdamW states for adapters only: a few GB)`. The full-precision base never exists.

### Guanaco recipe (the reference config)

| Knob | Value |
|---|---|
| Weight format | NF4 |
| Block size | 64 |
| Double quant | yes (FP8 inner, FP32 outer) |
| Effective bits/weight | ~4.127 |
| **LoRA rank r** | **64** |
| **LoRA target** | **all linear** (q, k, v, o, gate, up, down) |
| LoRA α | 16 |
| Optimizer | paged AdamW 32-bit |
| LR | 2e-4 |
| LR schedule | constant |
| Batch (sequence) | 16 |
| Max seq | 512 (Guanaco) |
| Wall-clock 65B | ~24 hr on 1×A100 48 GB |

**The "all linear" detail matters.** Original LoRA targeted only attention QKV. QLoRA extends to FFN (gate, up, down) and projection (o). With r=64 across all linears, the trainable parameter count is ~0.5% of the model — comparable to a small-rank LoRA on only QKV, but distributed across far more matrices.

---

## 6. Empirical claim — full-FT-equivalent quality

QLoRA's headline result (Table 5 of the paper):

| Model | Full FT MMLU | QLoRA-NF4 MMLU | Δ |
|---|---|---|---|
| LLaMA-7B | 38.4 | 38.5 | +0.1 |
| LLaMA-13B | 47.0 | 46.8 | -0.2 |
| LLaMA-33B | 56.4 | 56.5 | +0.1 |
| LLaMA-65B | 62.7 | 62.7 | 0.0 |

QLoRA matches full 16-bit fine-tuning across MMLU, BBH, and chatbot benchmarks (Vicuna evaluator + human eval). **The 4-bit quantization induces zero downstream quality loss when paired with LoRA fine-tuning.** This is the empirical claim that drove QLoRA's adoption.

The Guanaco family (released alongside the paper) reached 99.3% of ChatGPT's Vicuna benchmark in 24 GPU-hours — at the time, the strongest open chat model.

---

## 7. The production stack: bitsandbytes NF4

[[bitsandbytes-nf4]] is the underlying kernel library.

### Minimal usage

```python
from transformers import BitsAndBytesConfig, AutoModelForCausalLM
import torch

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",          # NF4 (vs the older "fp4")
    bnb_4bit_use_double_quant=True,      # save ~0.4 bits/weight
    bnb_4bit_compute_dtype=torch.bfloat16,
)
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-65b",
    quantization_config=bnb_config,
    device_map="auto",
)
```

### Critical config knobs

| Knob | Recommended | What it does |
|---|---|---|
| `bnb_4bit_quant_type` | `"nf4"` | NF4 codebook (vs `"fp4"` E2M1) |
| `bnb_4bit_use_double_quant` | `True` | quantize the block scales to FP8 |
| `bnb_4bit_compute_dtype` | `torch.bfloat16` | BF16 accumulation (not FP32) |
| `blocksize` | 64 | elements per block-scale |

The `compute_dtype` default in older bitsandbytes versions is FP32 — change it to BF16 for QLoRA. The `quant_type` default in some versions is "fp4" — change it to "nf4".

### Fused GEMV kernel

`gemv_4bit_inference_naive_fp16` (and the BF16 variant): dequantises a tile of NF4 weights into shared memory, then performs a FP16/BF16 GEMV. The fully-materialised BF16 weight matrix never appears in HBM.

---

## 8. The PEFT alternatives — QA-LoRA, LoftQ, LLM-QAT, LQ-LoRA, PEQA

QLoRA opened a flood of follow-up methods, each addressing one of its limitations.

### QA-LoRA — merge-friendly quantized fine-tune

**The QLoRA problem.** After QLoRA training you have INT4 base + BF16 LoRA. To deploy as pure INT4, you must either keep both (FP16 inference cost) or merge LoRA into the base (which forces re-quantization → information loss).

[[qa-lora]] (Xu et al. ICLR 2024) traces this to an **imbalanced DOF problem**: LoRA's `BA` is fully dense per-element, while INT4 base has one scale per group. They don't merge cleanly.

**Fix:** replace `BA` with a **group-wise additive scalar** `δ ∈ R^{d_out × (d_in/G)}` — one trainable scalar per `(output, group)` cell, exactly matching the quantization grid:

```math
y = \mathrm{dequant}(W_q, s + \delta) \cdot x
```

Post-training, set `s_new = s + δ`. The deployed weight is the unchanged INT4 with updated scales. **No re-quantization, no FP16 branch at inference.**

Same total trainable parameter budget as LoRA at a corresponding rank, but the deployed artifact is pure INT4.

### LoftQ — joint quant + LoRA initialization

**The QLoRA problem.** QLoRA initialises LoRA at zero, so the first forward pass already differs from FP by the full quantization error `||W - Q(W)||`. At 2-bit this initial gap is catastrophic and gradient descent can't recover.

[[loftq]] (Li et al. ICLR 2024) jointly solves for `(Q, A, B)` so `Q + BA^⊤ ≈ W`, via alternating minimization:

```
Q^(0) ← Q_b(W),  A^(0) ← 0,  B^(0) ← 0
for t = 1, ..., T (T ≈ 5):
    (B, A) ← TruncSVD_r(W - Q)      # rank-r SVD step
    Q ← Q_b(W - B·A^⊤)              # re-quantize step
```

The initial loss is essentially FP-level instead of starting from the full-quant gap. Downstream QLoRA fine-tuning then converges to higher final accuracy. **Especially important at 2-bit and 2/4-bit mixed precision** where vanilla QLoRA collapses.

### LLM-QAT — data-free full QAT

**The QLoRA problem.** Below ~3 bits PTQ + PEFT saturates. You need real QAT — but LLM pretraining data is unavailable.

[[llm-qat]] (Liu et al. 2023) has the FP teacher **generate its own calibration corpus** by next-token sampling (~100k sequences). The quantized student is then trained with **full-distribution KL distillation**:

```math
\mathcal{L}_{\text{distill}} = \sum_t \mathrm{KL}(p_T(\cdot | x_{<t}) \,\|\, p_S^{\text{quant}}(\cdot | x_{<t}))
```

(Not one-hot cross-entropy — the full output distribution forces matching uncertainty, not just argmax.)

LLM-QAT brings activation quantization (A4, A8) and **KV-cache quant (KV4)** into the same QAT loop, enabling W4A8KV4 and W4A4 configurations that PTQ cannot.

Cost: full QAT (~32–256 GPUs for a 30B model, days). Wall-clock is between PEFT and full pretraining.

### LQ-LoRA — low-rank-plus-quantized joint decomposition

[[lq-lora]] (Guo et al. 2023, revised 2024) generalises QLoRA's stack-of-two into a **joint decomposition**:

```math
W \approx Q + L R^\top, \qquad Q \in \{\text{INT-}b\ \text{grid}\}^{m \times n},\ L \in \mathbb{R}^{m \times r},\ R \in \mathbb{R}^{n \times r}
```

Solved with Fisher-weighted alternating minimization. Plus an **integer linear program** (ILP) that allocates per-layer bit-widths (b_i ∈ {2, 3, 4, 8}) to satisfy a global memory budget.

Result: 2.75-bit-average LLaMA-2-70B fits in 27 GB GPU memory. Early/attention layers typically get more bits than late FFN layers — the ILP picks per layer.

### PEQA — scale-only PEFT

[[peqa]] (Kim et al. NeurIPS 2023) takes the opposite extreme: **only the per-channel quantization scales train**. The integer weight indices are frozen.

Trainable parameter count: `Σ_layer d_out` — for LLaMA-65B, ~5M params vs ~350M for LoRA r=64. **Orders of magnitude smaller**, and the deployed artifact is pure INT-k with no LoRA branch at all.

Surprisingly competitive: scale-only updates capture task-specific dynamic-range shifts well, and the pre-quantized base already covers the bulk of the function.

---

## 9. Comparison — which PEFT to pick

| Method | Trainable params | Inference artifact | Best at |
|---|---|---|---|
| **QLoRA** | LoRA r=64 (~0.5% of model) | NF4 base + BF16 LoRA | general SFT (default) |
| **QA-LoRA** | group-wise δ (~LoRA scale) | pure INT4 (merged) | when deployment must be pure INT4 |
| **LoftQ** | LoRA r + Q init | NF4 base + BF16 LoRA | 2-bit and mixed 2/4-bit |
| **LLM-QAT** | all weights (QAT) | INT-k W + INT-k A + INT-k KV | sub-4-bit activations / KV |
| **LQ-LoRA** | low-rank L, R + Q decomp | INT-b + low-rank | tight memory budgets |
| **PEQA** | per-channel scales only | pure INT-k | the smallest possible artifact |

**Decision rule:** start with QLoRA (default). If you can't afford a BF16 LoRA branch at inference, use QA-LoRA. If you're pushing 2-bit weights, use LoftQ init. If you need activations below A8, use LLM-QAT. If you need extreme memory efficiency at training, use LQ-LoRA + ILP allocation. If the deployment must be pure INT-k with zero adapter weights, use PEQA.

---

## 10. The "fits LLaMA-65B in 48 GB" math, written out

The single concrete claim that made QLoRA famous. Let's verify it.

LLaMA-65B parameter count: 65 × 10⁹.

**FP16 base:** `65e9 · 2 bytes = 130 GB` — already past consumer GPUs.

**NF4 base (with double quant):** `65e9 · 4.127 / 8 bytes ≈ 33.5 GB`. Fits in a single 48 GB A100 (or 40 GB if you push).

**LoRA adapters (r=64 across all linear layers, BF16):**
- ~7 linears per transformer block (q, k, v, o, gate, up, down).
- 80 transformer blocks in LLaMA-65B.
- Per linear: `r · (d_in + d_out) · 2 bytes`. With d ≈ 8192 and r=64: `64 · 16384 · 2 = 2 MB`.
- Total: `80 · 7 · 2 MB ≈ 1.1 GB`.

**AdamW state for adapters (paged, FP32 moments):** `2 · 0.5e9 · 4 ≈ 4 GB` — paged through CPU as needed.

**Activations + gradients + workspace:** ~5–10 GB depending on sequence length.

**Sum (training):** `33.5 + 1.1 + (peak ~10) + paged ≈ 45 GB`. Fits 48 GB with margin.

Without QLoRA's stack: the FP16 base alone (130 GB) doesn't fit. **The whole recipe is the difference between "needs 4 A100s" and "needs 1 A100".**

---

## Connections and what's next

- **[[qlora]]** — full extract; the four-piece recipe.
- **[[nf4]]** — full extract; the 4-bit quantile code; ch-02's [[nf4]] page is the format reference.
- **[[qa-lora]]** — merge-friendly group-wise adapter; pure INT4 deployment.
- **[[loftq]]** — joint quant + LoRA init via alternating minimization; critical at 2-bit.
- **[[llm-qat]]** — data-free QAT via teacher self-generation; sub-4-bit activations / KV.
- **[[lq-lora]]** — joint low-rank + quantized decomposition with ILP bit allocation.
- **[[peqa]]** — scale-only PEFT; smallest possible trainable param count.
- **[[bitsandbytes-nf4]]** — production kernel library.
- **[[spqr]]** — same author's (Dettmers) sibling weight-only PTQ; near-lossless 4-bit inference.
- **[[lloyd-max-quantizer]]** (ch-03) — NF4's theoretical ancestor; Lloyd-Max for N(0,1).
- **[[information-theoretic-bounds]]** (ch-01) — Gish-Pierce p^(1/3) optimal density that NF4 approximates.
- **[[ch-14]] / [[aqlm]]** — sub-2-bit successor; pushes the QLoRA-style memory frontier further.
- **[[ch-19]]** — production deployment kernels; Marlin / Machete consume QLoRA checkpoints.

## Further reading

- Read QLoRA §3.1 (NormalFloat derivation) and §3.2 (Double Quantization) end-to-end — they're the conceptual core.
- The Guanaco evaluation (paper Table 6) is a useful exercise in chatbot benchmark interpretation; Dettmers later wrote critical follow-ups on Vicuna eval reliability.
- bitsandbytes source `csrc/kernels.cu` for `kQuantizeBlockwiseNF4` and `gemv_4bit_inference_naive_fp16` — the actual kernel reading reveals exactly how the dequant fuses with the GEMV tile-by-tile.
