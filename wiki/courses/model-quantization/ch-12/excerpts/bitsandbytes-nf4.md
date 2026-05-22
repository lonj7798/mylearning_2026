---
chapter: ch-12
course: model-quantization
phase: read
excerpt_of: "bitsandbytes — NF4 / FP4 4-bit Kernels"
source_url: https://github.com/bitsandbytes-foundation/bitsandbytes
created_at: "2026-05-21"
---

# Excerpt: bitsandbytes — production NF4 / FP4 kernel library

**Maintainer:** Tim Dettmers + bitsandbytes-foundation contributors
**Year:** 2023 (4-bit added with QLoRA paper)
**Repo:** https://github.com/bitsandbytes-foundation/bitsandbytes
**Raw-data source:** [[raw-data/bitsandbytes-nf4]]

---

## What it implements

The 4-bit half of bitsandbytes is the production kernel substrate for QLoRA. Two codebooks:
- **NF4** (Normal Float 4-bit, 16 quantiles of N(0, 1)) — default for QLoRA.
- **FP4** (4-bit E2M1 floating-point) — legacy alternative.

Both packed two-per-byte with block-wise scales every 64 elements. Double quantization further quantizes the block scales themselves to 8-bit, recovering ~0.37 bits/parameter.

The forward path is a **fused GEMV kernel** that dequantizes a tile of weights into shared memory, then performs the FP16/BF16 matmul. **The full BF16 weight matrix is never materialised in HBM.**

---

## Repository layout

- **4-bit Python API:** `bitsandbytes/functional.py` → `quantize_4bit`, `dequantize_4bit`.
- **4-bit module:** `bitsandbytes/nn/modules.py` → `Linear4bit`, `Params4bit`.
- **4-bit CUDA kernels:** `bitsandbytes/csrc/kernels.cu` → `kQuantizeBlockwiseNF4`, `gemv_4bit_inference_naive_fp16`.
- **HF integration:** HuggingFace `transformers` `BitsAndBytesConfig` with `load_in_4bit=True`.

---

## NF4 codebook (16 levels)

```
NF4_CODES = [
  -1.0, -0.6961928, -0.5250730, -0.39491748,
  -0.28444138, -0.18477343, -0.09105003, 0.0,
   0.07958029, 0.1609302, 0.24611232, 0.33791524,
   0.44070983, 0.5626170, 0.7229568, 1.0
]
```

Asymmetric: 8 negative + 7 positive + zero. Optimal for weights ≈ N(0, σ²). See [[nf4]] for derivation.

---

## Quantization rule (per 64-element block)

```
s = max(|w|) / 1.0                                  # block scale (FP16)
w_norm = w / s                                      # normalize to [-1, 1]
w_nf4 = argmin_{c ∈ NF4_CODES} |w_norm − c|         # nearest-code search
```

Two NF4 codes pack into one byte.

---

## Double quantization

Block scales `{s_0, s_1, ...}` are themselves quantized to INT8 with a per-256-block super-scale (FP32).

Net savings: 32 bits per block scale → `8 + (32/256) ≈ 8.125` bits per block → effective bit rate drops from ~4.5 bpw (with FP16 scale) to **~4.13 bpw**.

---

## Minimal usage

```python
import torch
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",            # NF4 (not the legacy "fp4")
    bnb_4bit_use_double_quant=True,        # save ~0.4 bits/weight
    bnb_4bit_compute_dtype=torch.bfloat16, # accumulate in BF16
)
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b",
    quantization_config=bnb_config,
    device_map="auto",
)
```

---

## Config / hyperparameters

| Knob | Default | Recommended | What it does |
|---|---|---|---|
| `bnb_4bit_quant_type` | `"fp4"` (legacy) | **`"nf4"`** | codebook choice |
| `bnb_4bit_use_double_quant` | `False` | **`True`** | quantize block scales |
| `bnb_4bit_compute_dtype` | `float32` | **`torch.bfloat16`** | accumulate dtype |
| `blocksize` | 64 | 64 | elements per block-scale |

The defaults are historically wrong for QLoRA — **explicitly set all three** for the recipe Dettmers reports.

---

## Fused GEMV kernel

`gemv_4bit_inference_naive_fp16` (and the BF16 variant):
- Dequantizes a tile of NF4 weights into shared memory.
- Performs FP16/BF16 GEMV against the (small-batch) activation.
- The full FP16/BF16 weight matrix never materialises.

For batch > 1, the path falls back to `bnb.matmul_4bit` which calls `dequantize_4bit` + `torch.matmul`. This is **slower than fused** but correct.

In 2024 newer backends (Marlin, Machete) handle NF4 weights too, with even faster prefill — bitsandbytes' fused GEMV is still the canonical decode kernel.

---

## Key APIs

- `bnb.functional.quantize_4bit(A, blocksize=64, quant_type="nf4")` → packed bytes + scales.
- `bnb.functional.dequantize_4bit(A, quant_state)` → inverse.
- `bnb.nn.Linear4bit(in_features, out_features, bias, compute_dtype, quant_type, quant_storage)` → drop-in 4-bit Linear.
- `class Params4bit(nn.Parameter)` — holds packed 4-bit weight + QuantState.

---

## What QLoRA gets from this

The QLoRA recipe is literally:
1. Wrap every `nn.Linear` in the base model with `bnb.nn.Linear4bit`.
2. Add `peft.LoraConfig(r=64, target_modules="all-linear")` on top.
3. Train with `optim="paged_adamw_32bit"`.

The kernel library handles dequant-on-the-fly. The PEFT library handles LoRA. The HF Trainer handles paging.

---

## Connections

- [[nf4]] — codebook spec.
- [[qlora]] — the fine-tuning recipe on top.
- [[bitsandbytes-int8]] — 8-bit cousin in the same library.
- Production successor kernels: [[marlin-kernel]], [[machete-kernel]] ([[ch-19]]).
