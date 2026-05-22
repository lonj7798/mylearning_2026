---
chapter: ch-07
course: model-quantization
phase: read
excerpt_of: "bitsandbytes — LLM.int8() reference implementation"
source_url: https://github.com/bitsandbytes-foundation/bitsandbytes
created_at: "2026-05-21"
---

# Excerpt: bitsandbytes — LLM.int8() in production

**Authors:** Tim Dettmers et al.
**Year:** 2022 (continuously maintained)
**Raw-data source:** [[raw-data/frameworks/bitsandbytes-int8]]

---

## What it is

The reference implementation of [[llm-int8]]. Drop-in `nn.Linear` replacement (`bnb.nn.Linear8bitLt`) that performs mixed-precision INT8 + FP16 GEMM with per-forward outlier detection. HuggingFace Transformers integrates it via `BitsAndBytesConfig(load_in_8bit=True)`.

---

## Repository layout (load-bearing files)

- `bitsandbytes/functional.py` — Python wrappers for quant/dequant ops.
- `bitsandbytes/csrc/ops.cu` and `csrc/kernels.cu` — the actual CUDA kernels.
- `bitsandbytes/nn/modules.py` — `Linear8bitLt`, `Int8Params`.
- HF integration shim — `BitsAndBytesConfig` in `transformers`.

---

## The quantization rule (vector-wise / row-wise)

```python
# Per-row scale (one scalar per output channel)
s_W = max(|W|, dim=in) / 127
W_int8 = round(W / s_W).clamp(-128, 127)

# Per-row scale of the input (one scalar per token)
s_X = max(|X|, dim=in) / 127
X_int8 = round(X / s_X).clamp(-128, 127)

# Outlier mask: columns with any |X[i,j]| > threshold
mask = (|X|.max(dim=batch) > 6.0)

# Mixed-precision GEMM
Y = (s_X.outer(s_W)) * (X_int8[:, ~mask] @ W_int8[:, ~mask].T) \
  + X[:, mask] @ W[:, mask].T          # FP16 fallback
```

This is the algorithm. The CUDA kernels (`gemm_mixed_8bit_lt`) overlap the INT8 and FP16 paths on separate streams; the algorithmic content is the same.

---

## Key APIs

```python
# Top-level INT8 matmul with outlier extraction
bnb.matmul(A, B, out=None, state=MatmulLtState())

# Drop-in nn.Linear replacement
bnb.nn.Linear8bitLt(in_features, out_features, has_fp16_weights=False, threshold=6.0)

# Block-wise INT8 quantization (used for optimizer states)
bnb.functional.quantize_blockwise(A, code, blocksize=4096)

# 8-bit optimizer states using block-wise quantization
bnb.optim.Adam8bit(...)
bnb.optim.AdamW8bit(...)
```

---

## Hyperparameter knobs

| Knob | Default | Notes |
|---|---|---|
| `threshold` | 6.0 | outlier magnitude cutoff (the α from the paper) |
| `has_fp16_weights` | False | keep FP16 master weight in memory (only set True for QAT) |
| `index` | None | per-tensor override for outlier index (rarely needed) |
| `memory_efficient_backward` | False | re-quantize backward; slower but lower memory |

---

## Usage in HuggingFace Transformers

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0,
    llm_int8_skip_modules=["lm_head"],   # don't quantize the output head
)

model = AutoModelForCausalLM.from_pretrained("facebook/opt-13b", quantization_config=bnb_config)
```

`load_in_8bit=True` walks the model graph and replaces every `nn.Linear` with `Linear8bitLt`. The `Int8Params` wrapper packs INT8 weight + FP16 scales + outlier index into one PyTorch Parameter so `model.to(device)` moves everything atomically.

---

## Empirical: memory + accuracy

| Model | FP16 memory | bnb INT8 memory | Zero-shot acc Δ |
|---|---|---|---|
| OPT-6.7B | 13 GB | 7 GB | < 0.1 |
| OPT-13B | 26 GB | 13 GB | < 0.1 |
| OPT-30B | 60 GB | 30 GB | < 0.2 |
| OPT-175B | 350 GB | 180 GB | < 0.2 |

~50% memory savings at essentially zero accuracy cost. Fits OPT-13B on one A100-24GB, OPT-30B on one A100-40GB, OPT-175B on a single 8×A100 node.

---

## When to use bnb-INT8 vs alternatives in 2026

| Situation | Best choice |
|---|---|
| Quick deployment, ≤ 70B model, any GPU | **bnb-INT8** (`load_in_8bit=True`) |
| Production serving, throughput-critical | GPTQ-W4 + [[marlin-kernel]] (ch-08, ch-19) |
| ≤ 7B model, memory-critical | NF4 ([[bitsandbytes-nf4]], ch-12) |
| Fine-tuning while quantized | QLoRA on NF4 (ch-12) |
| Sub-4-bit | AQLM, QuIP# (ch-13, ch-14) |
| Edge / mobile | llama.cpp k-quants (ch-19) |

bnb-INT8 sits in the "just works, no calibration, drop-in" niche. It's not the fastest or smallest; it's the most foolproof.

---

## Common pitfalls

- **Inference with `bnb.nn.Linear8bitLt` and very short prompts.** The outlier-detection kernel has CPU-GPU round-trip overhead that dominates at <8 tokens; latency is FP16-worse there.
- **Training with `has_fp16_weights=False`.** Backward will be incorrect. For QAT, set it True (costs memory but preserves master copies).
- **Mixing with `torch.compile()`.** The outlier mask is a data-dependent control flow; `torch.compile()` graph-breaks. Use eager mode or compile around the bnb modules.
- **`load_in_8bit=True` + LoRA without `prepare_model_for_kbit_training`.** Adapter layers will be in fp16 but the base in int8; gradient flow can be wrong. Use the HF PEFT prepare helper.

---

## What it does NOT do

- **Does not quantize weights below 8-bit.** That's NF4 / FP4 / GPTQ / AWQ.
- **Does not quantize embeddings or `lm_head` by default.** These layers stay FP16 (sensitive).
- **Does not support all hardware.** CUDA-centric; ROCm and CPU paths exist but are slower.
- **Does not handle weight-only INT8.** It's INT8-A and INT8-W jointly. For W4A16, use AutoGPTQ + Marlin.

---

## Connections

- [[excerpts/llm-int8]] — paper this library implements.
- [[ch-07]] — parent synthesis.
- [[ch-12]] — [[bitsandbytes-nf4]] is the 4-bit cousin in the same repo.
- [[ch-08]] — [[autogptq]] is the W4 weight-only sibling library, often combined.
