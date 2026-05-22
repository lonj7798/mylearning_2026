---
chapter: ch-12
course: model-quantization
phase: read
excerpt_of: "QLoRA: Efficient Finetuning of Quantized LLMs"
source_url: https://arxiv.org/abs/2305.14314
created_at: "2026-05-21"
---

# Excerpt: QLoRA — NF4 + DoubleQuant + Paged Opt + LoRA

**Authors:** Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, Luke Zettlemoyer
**Year:** 2023 (NeurIPS 2023)
**URL:** https://arxiv.org/abs/2305.14314
**Raw-data source:** [[raw-data/qlora]]

---

## The headline claim

Fine-tune a **65B-parameter LLM on a single 48 GB GPU** while preserving full 16-bit fine-tuning quality. The Guanaco family (7B / 13B / 33B / 65B) fine-tuned this way reaches **99.3% of ChatGPT's Vicuna benchmark in 24 GPU-hours**.

This is the single most influential PEFT result of 2023.

---

## The four-piece stack

1. **NormalFloat 4-bit (NF4)** — a quantile-based 4-bit code optimal for zero-mean unit-variance normal weights.
2. **Double Quantization** — quantize the FP32 block scales themselves to FP8 (per 256-block group). Saves ~0.37 bits/weight.
3. **Paged Optimizers** — NVIDIA UVM keeps AdamW states resident on CPU during pre-emptable peaks.
4. **Frozen-4-bit base + BF16 LoRA** — gradients flow through dequantized base into BF16 LoRA adapters; W is never updated.

---

## NF4 — the 16 values

```
[-1.0,
 -0.6961928, -0.5250730, -0.39491748,
 -0.28444138, -0.18477343, -0.09105475,
  0.0,
  0.07958029, 0.16093020, 0.24611230, 0.33791524,
  0.44070983, 0.56261432, 0.72295684,
  1.0]
```

8 negative + zero + 7 positive. Derived as **equal quantile splits of the standard normal CDF** (adjusted so extremes = ±1 to match per-block absmax normalization).

Empirical: ~0.3–0.5 PPL improvement over INT4 or FP4 E2M1 at the same bit budget on LLaMA-class models. See [[nf4]] for format details.

---

## Block-wise quantization

```
B = 64 weights per block
s = max(|w_b|)   (FP32 per-block scale)
q_i = NF4_round(w_i / s)
recover: w_i ≈ s · NF4[q_i]
```

---

## Double Quantization (the bit-budget magic)

Per-block FP32 scales → group every 256 blocks → quantize 256 scales to FP8 with one FP32 outer scale per group.

```
bits/weight = 4 (NF4) + 8/64 (FP8 scale) + 32/(64·256) (FP32 outer)
            = 4 + 0.125 + 0.002
            ≈ 4.127 effective bits/weight
```

Savings vs naïve FP32 scale: ~0.37 bits/weight. On 65B, that's ~3 GB recovered for free.

---

## Paged Optimizer

AdamW state for 65B = `2 · 65e9 · 4 = 520 GB` — vastly exceeds GPU memory. Even after QLoRA only the adapters need optimizer state, but transient gradient peaks can still OOM.

**Paged Optimizer:** AdamW chunks live in CPU DRAM via NVIDIA Unified Memory; pages swap into GPU HBM on demand. The OS handles eviction during memory peaks. Throughput cost: small.

Without paging, even LoRA-only fine-tuning of LLaMA-65B at long context OOMs intermittently.

---

## QLoRA forward + backward

For each Linear `y = W · x`:

```math
y = \mathrm{dequant}(W_{NF4}) \cdot x + \frac{\alpha}{r} \cdot B (A x)
```

- `W_{NF4}` is frozen 4-bit (with double-quantized scales).
- `A ∈ R^{r × d_in}`, `B ∈ R^{d_out × r}` in **BF16**, both trainable.

**Forward:** the NF4 weight is dequantised on-the-fly into BF16 inside a fused GEMV kernel — the full BF16 weight is never materialised in HBM.

**Backward:** gradients flow through the BF16 dequant op into `A, B`. **W never receives gradients.**

---

## Guanaco recipe (the reference)

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
| Max seq | 512 |
| Wall-clock 65B | ~24 hr on 1×A100 48 GB |

**Critical detail:** LoRA on **all** linear layers (not just attention QKV). With r=64 across all linears, trainable params ≈ 0.5% of model.

---

## Quality claim (Table 5)

| Model | Full FT MMLU | QLoRA-NF4 MMLU | Δ |
|---|---|---|---|
| LLaMA-7B | 38.4 | 38.5 | +0.1 |
| LLaMA-13B | 47.0 | 46.8 | -0.2 |
| LLaMA-33B | 56.4 | 56.5 | +0.1 |
| LLaMA-65B | 62.7 | 62.7 | 0.0 |

**QLoRA matches full 16-bit fine-tuning across MMLU / BBH / chatbot evals.** The 4-bit quantization induces zero downstream quality loss when paired with LoRA fine-tuning.

---

## Connections

- Predecessor: LoRA (Hu et al. 2021).
- Same author's 8-bit predecessor: [[llm-int8]].
- Format: [[nf4]].
- Joint-init successor: [[loftq]].
- Merge-friendly successor: [[qa-lora]].
- Scale-only PEFT alternative: [[peqa]].
- Decomposition successor: [[lq-lora]].
- Full QAT alternative: [[llm-qat]].
- Implementation: [[bitsandbytes-nf4]].
- Sibling from same author: [[spqr]] (inference-only near-lossless 4-bit).
