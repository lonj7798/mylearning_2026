---
chapter: ch-13
course: model-quantization
phase: read
excerpt_of: "QUIK: Towards End-to-End 4-Bit Inference on Generative Large Language Models"
source_url: https://arxiv.org/abs/2310.09259
created_at: "2026-05-21"
---

# Excerpt: QUIK — W4A4 with INT8 outlier sidecar

**Authors:** Saleh Ashkboos, Ilia Markov, Elias Frantar, Tingxuan Zhong, Xincheng Wang, Jie Ren, Torsten Hoefler, Dan Alistarh (IST Austria, ETH Zurich)
**Year:** 2023
**URL:** https://arxiv.org/abs/2310.09259
**Raw-data source:** [[raw-data/quik]]

---

## Position in the design space

[[quip]] makes 2-bit weight-only PTQ work by rotating the distribution. QUIK is the **other** answer to the outlier problem: **keep the bulk at INT4, route the top ~1% outlier rows/columns through a parallel INT8 path, fuse both into one CUDA kernel**.

QUIK predates QuaRot by half a year and is the first paper to deliver **real end-to-end W4A4 speedup** (3.4× over FP16 on A100) — not just weight-only memory savings.

The two schools are not enemies. QUIK's outlier-sidecar approach persists as the deployment choice when the online Hadamard cost is too high; QuaRot's rotation approach wins on accuracy in the "no special path" regime. Both come from the same IST-Austria lab (Frantar + Alistarh).

---

## Mixed-precision GEMM (the kernel core)

For `Y = X · W^⊤` with `X ∈ ℝ^{B × C_in}`, `W ∈ ℝ^{C_out × C_in}`:

1. From calibration, identify the top-K input channels by max activation magnitude (K ≈ 0.5–1% of C_in, typically 64–128 channels).
2. Identify the top-K output channels by weight sensitivity.
3. Partition `C_in = C_in^{4bit} ∪ C_in^{8bit}`, `C_out = C_out^{4bit} ∪ C_out^{8bit}`.
4. Compute four sub-matmuls and sum:

```
Y_44 = X_4 · W_44^⊤  (W4A4, tensor core)        # majority of FLOPs
Y_48 = X_4 · W_48^⊤  (W8A4)                     # small
Y_84 = X_8 · W_84^⊤  (W4A8)                     # small
Y_88 = X_8 · W_88^⊤  (W8A8, tensor core)        # tiny
Y = Y_44 + Y_48 + Y_84 + Y_88
```

The W4A4 path carries ~99% of the work; the three sidecars add ~3% overhead but recover the accuracy hit from outliers.

The outlier-column choice is **fixed at calibration time** — no runtime branching. The kernel statically knows which columns/rows are wide.

---

## Weight quantization

GPTQ on the W4 partition (group size 128, percdamp 0.01). RTN on the W8 partition. Per-channel weight scales for both.

## Activation quantization

Per-token absmax INT4 on the X_4 slice. Per-token INT8 on the X_8 outlier columns. **Dynamic** — recomputed each forward pass, no static calibration scale.

---

## 2:4 sparsity (optional compounding)

Optionally enforce 2-out-of-4 structured zeros on the W_44 block (NVIDIA Ampere/Hopper sparse tensor cores) for an additional ~2× weight memory savings and ~10% throughput uplift on sparse-supported kernels.

This means three orthogonal compressions stack: INT4 (4×), 2:4 sparsity (2×), outlier sidecar (negligible memory). Total ~8× weight memory reduction vs FP16.

---

## Numbers

LLaMA-2-7B WikiText-2 (lower = better):

| Method | W bits | A bits | KV bits | ppl | Δppl |
|--------|--------|--------|---------|-----|------|
| FP16 | 16 | 16 | 16 | 5.47 | — |
| SmoothQuant W4A4 | 4 | 4 | 16 | NaN | collapse |
| GPTQ-4 + RTN-A4 | 4 | 4 | 16 | NaN | collapse |
| QUIK W4A4 | 4 (+8 outlier) | 4 (+8 outlier) | 16 | 6.05 | +0.58 |
| QUIK W4A4 + 2:4 | 4 + sparse | 4 | 16 | 6.21 | +0.74 |

A100 throughput, LLaMA-2-7B, batch=64, seqlen=2048:

| Method | tok/s | speedup vs FP16 |
|--------|-------|-----------------|
| FP16 | 110 | 1.0× |
| INT8 (SmoothQuant) | 220 | 2.0× |
| QUIK W4A4 | 374 | **3.4×** |
| QUIK W4A4 + 2:4 | 415 | 3.8× |

QUIK is the first method to ship measurable W4A4 end-to-end speedup. SmoothQuant W4A4 collapses to NaN; QUIK works.

---

## Hyperparameters

| Knob | Value |
|------|-------|
| Weight bits | 4 (dense), 8 (outlier rows) |
| Activation bits | 4 (dense), 8 (outlier columns) |
| Outlier fraction | 0.5–1% per axis (~64–128 channels) |
| Group size (W4) | 128 |
| Calibration | 128 × 2048 |
| Sparsity (optional) | 2:4 structured |
| Kernel | custom CUDA, fused W4A4 + W8A8 |

---

## Pitfalls

- **Outlier set is fixed at calibration.** Distribution shift at inference (long prompts, code, multilingual) can change which channels are outliers and degrade quality. QUIK's recipe assumes the calibration distribution is representative.
- **No quant on KV cache.** QUIK is W4A4 only; the KV cache stays FP16. For long contexts, the KV-cache memory dominates — see ch-15 (KIVI / KVQuant) for the missing piece.
- **The 4 sub-matmuls must reduce the same-shape accumulator.** Output rescale happens once, not per sub-matmul, to avoid double rounding.
- **Sub-1% outlier fraction is brittle.** Below 0.5%, accuracy degrades; above 1%, throughput tanks. The sweet spot is narrow.

---

## Connections

- [[excerpts/quip]] — the rotation-based alternative to outlier handling at the same lab.
- [[ch-14]] — [[quarot]] supersedes QUIK in accuracy by removing the need for an outlier path; [[atom]] is QUIK's successor in deployment.
- [[ch-08]] — [[gptq]] is the W4 weight quantizer QUIK uses.
- [[ch-07]] — [[llm-int8]] introduced the outlier-channel idea at INT8; QUIK is the W4A4 instance.
