---
chapter: ch-14
course: model-quantization
phase: read
excerpt_of: "QServe: W4A8KV4 Quantization and System Co-design for Efficient LLM Serving"
source_url: https://arxiv.org/abs/2405.04532
created_at: "2026-05-21"
---

# Excerpt: QServe — W4A8KV4 on Hopper with register-level dequant

**Authors:** Yujun Lin, Haotian Tang, Shang Yang, Zhekai Zhang, Guangxuan Xiao, Chuang Gan, Song Han (MIT HAN Lab)
**Year:** 2024
**URL:** https://arxiv.org/abs/2405.04532
**Raw-data source:** [[raw-data/qserve]]

---

## The pitch — why W4A8KV4 not W4A4

The naive intuition: lower bits everywhere = better. The reality on Hopper (H100/H200):

- W4 weight path: 4× HBM bandwidth savings, this is the big win.
- A4 activations: only 2× over INT8, but breaks softmax stability and adds dequant cost.
- A8 activations: 1× tensor-core utilisation, no softmax instability.
- KV4: 4× memory + 4× HBM bandwidth on the cache; this is the second big win.

QServe's claim: on Hopper, **W4A8KV4 dominates W4A4 on accuracy at near-identical throughput**, because the A8 vs A4 throughput difference is small but the accuracy gap is large.

---

## The kernel core — progressive group quantization

Naive W4A8: dequantize INT4 → FP16 in registers, feed FP16 into the tensor core. The FP dequant adds 2–3 instructions per element and pushes register pressure. QServe **keeps everything integer until the GEMM accumulator**:

**Stage 1 — per-channel INT8.** Quantize each output channel of W to INT8 with a single per-channel FP16 scale `s_c`. Store as INT8 `W_s`.

**Stage 2 — per-group INT4.** Within each group of g=128 weights along the input axis, quantize the INT8 values to INT4 with a per-group **INT8** scale `s_g` (the scale is itself integer, not FP). Store as INT4 `W_g`.

**At inference**: dequantize INT4 → INT8 entirely in registers via `W_int8 = W_int4 * s_g_int8` (INT8 × INT8 → INT16, very cheap on tensor cores). Feed INT8 operand into INT8 tensor-core GEMM with A8 activation. No FP dequant in critical path.

```
HBM:        INT4 weight + (small) INT8 group scales + (tiny) FP16 channel scales
SMEM:       INT4 weight + INT8 group scales
Registers:  INT8 weight (after INT4 → INT8 register-level dequant)
Tensor core: INT8 × INT8 → INT32 accumulate
Output:     INT32 → FP16 via per-channel scale (once per output element)
```

The whole pipeline avoids FP dequant in the inner loop. Compare to [[marlin-kernel]] which does INT4 → FP16 register dequant.

---

## SmoothAttention

Attention `softmax(QK^⊤/√d) V` is sensitive to KV4 because INT4 K introduces noise that softmax amplifies. QServe applies a SmoothQuant-style learnable per-head scaling:

```
Q' = Q · s,   K' = K / s
```

QK^⊤ is unchanged (`Q' K'^⊤ = Q s · K^⊤ / s = QK^⊤`), but K' has reduced dynamic range → KV4 quantization is gentler. `s` is calibrated to minimise the KL divergence between FP and quantized softmax outputs.

This is structurally the same trick as [[smoothquant]] applied to per-channel weight↔activation; SmoothAttention applies it per-head to Q↔K.

---

## Compute-aware weight reorder

Weight tiles are pre-permuted to match the Tensor Memory Accelerator (TMA) layout on Hopper, eliminating shared-memory bank conflicts during the SMEM → register load. Pure engineering optimisation but worth several percent throughput.

---

## Activation A8 dynamic per-token quant

Per-token absmax to INT8 — same as SmoothQuant but without static calibration scales:

```
scale_t = max_i |x_{t,i}| / 127
x̂_{t,i} = round(x_{t,i} / scale_t)
```

Robust to prompt distribution shift.

---

## KV4 layout

- K: per-head, per-token, INT4 with per-group scale (g=128 along channel).
- V: per-head, per-token, INT4 with per-group scale.

Fused into attention kernel that dequantizes on-the-fly. Borrows the KIVI/KVQuant axis choices (K per-channel-ish, V per-token-ish).

---

## Throughput numbers

| Model | Hardware | Baseline | QServe | Speedup |
|-------|----------|----------|--------|---------|
| Llama-3-8B | H100 | TRT-LLM W8A8 | QServe | **1.2×** |
| Llama-3-8B | L40S | TRT-LLM W8A8 | QServe | **2.4×** |
| Qwen-1.5-72B | A100 | Atom W4A4 | QServe | **3.5×** |

The 3.5× over Atom on Qwen-72B is the headline: Atom's W4A4 hurts the larger model badly via softmax instability; QServe's W4A8 sidesteps that.

---

## Accuracy

LLaMA-2-7B WikiText-2 W4A8KV4:

| Method | ppl | Δppl vs FP16 |
|--------|-----|--------------|
| FP16 | 5.47 | — |
| AWQ W4A16 | 5.62 | +0.15 |
| **QServe W4A8KV4** | **5.62** | **+0.15** |
| Atom W4A4KV4 | 5.93 | +0.46 |

QServe matches AWQ-W4A16 on accuracy while serving INT4 weights *and* INT8 activations *and* INT4 KV.

---

## Atom vs QServe — when to pick which

| | Atom (W4A4KV4) | QServe (W4A8KV4) |
|--|----------------|------------------|
| Target HW | Ampere (A100, A6000) | Hopper (H100, L40S) |
| Activation | INT4 dynamic | INT8 dynamic |
| KV cache | INT4 | INT4 |
| Accuracy | slightly lower (A4 cost) | matches W4A16 |
| Throughput | higher (less compute) | balanced |
| Use case | memory-bound | balanced compute + memory |
| Softmax stability | sensitive on larger models | robust |

---

## Pitfalls

- **W4A8 requires Hopper for the INT8 × INT8 tensor-core path.** On A100 you fall back to FP-dequant; the throughput advantage drops to ~1.5× over W8A8.
- **SmoothAttention scales are per-head learnable but per-layer applied.** Don't share s across heads or layers; per-head calibration is essential.
- **Progressive group quant requires careful scale alignment.** The per-group INT8 scale `s_g` must be representable as INT8 — clip outlier groups during calibration.
- **TMA reorder is layout-coupled to Hopper.** Porting to Ampere requires re-tiling.
- **Doesn't work with rotation methods naively.** QuaRot's R3 (FFN-down) inserts an online FWHT that conflicts with QServe's fused INT4 → INT8 dequant path. The two stacks are usually run separately.

---

## Connections

- [[excerpts/quarot]] / [[excerpts/spinquant]] — the rotation-side of the W4A4 problem; QServe is the deployment-side answer.
- [[ch-15]] — KIVI / KVQuant inform the KV-cache layout QServe inherits.
- [[ch-19]] — [[marlin-kernel]] is the W4A16 ancestor of QServe's dequant pattern; [[machete-kernel]] is the parallel Hopper-native effort.
- [[ch-09]] — [[smoothquant]] is the per-channel migration parent of SmoothAttention.
