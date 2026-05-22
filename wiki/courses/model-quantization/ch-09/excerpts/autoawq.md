---
chapter: ch-09
course: model-quantization
phase: read
excerpt_of: "AutoAWQ — production implementation of Activation-aware Weight Quantization"
source_url: https://github.com/casper-hansen/AutoAWQ
created_at: "2026-05-21"
---

# Excerpt: AutoAWQ — the de-facto open AWQ implementation

**Maintainer:** Casper Hansen + AWQ paper contributors
**Year:** 2023 (released after AWQ paper)
**Repo:** https://github.com/casper-hansen/AutoAWQ
**Raw-data source:** [[raw-data/autoawq]]

---

## What it implements

AutoAWQ is the production realisation of the [[awq]] algorithm. The full pipeline:

1. Load FP16 base via HF transformers.
2. Run calibration forward to collect per-channel `mean(|X_j|)`.
3. Per layer, grid-search α ∈ [0, 1] on 20 points; pick α minimising layer-output MSE.
4. Apply per-channel scaling; absorb `s` into preceding LayerNorm/Linear.
5. INT4 group-wise asymmetric quantize (group_size=128) the scaled weight.
6. Pack INT4 + FP16 scales + INT4 zero-points into AWQ checkpoint format.
7. Serve via fused GEMM / GEMV / Marlin kernel.

---

## Repository layout (the things to actually read)

- `awq/quantize/quantizer.py` — `AwqQuantizer` class (orchestration).
- `awq/quantize/scale.py` — `auto_scale_block()` and `apply_scale()` (the α search).
- `awq/modules/linear/gemm.py`, `awq/modules/linear/gemv.py` — Python kernel wrappers.
- `awq_ext/quantization/gemm_cuda_gen.cu`, `awq_ext/quantization/gemv_cuda.cu` — CUDA sources.
- `awq/models/*.py` — per-architecture HF adapters (llama, mistral, mixtral, qwen2, ...).

---

## Minimal usage

```python
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

model_id = "meta-llama/Llama-3-8B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoAWQForCausalLM.from_pretrained(model_id)

quant_config = {
    "w_bit": 4,
    "q_group_size": 128,
    "version": "GEMM",       # GEMM | GEMV | Marlin
    "zero_point": True,
}
model.quantize(tokenizer, quant_config=quant_config, n_samples=128)
model.save_quantized("llama-3-8b-awq")

# Load for inference
model = AutoAWQForCausalLM.from_quantized(
    "llama-3-8b-awq", device_map="auto"
)
```

---

## Kernel selection (the one knob that matters at deploy)

| Version | Best for | Mechanism |
|---|---|---|
| `GEMM` | prefill / batch > 1 | INT4 packed weights + FP16 scales; `mma.sync` tensor cores |
| `GEMV` | batch=1 decode | dequant a tile into shared memory, FP16 vector-matrix |
| `Marlin` | Ampere + Hopper prefill | same family as GPTQ Marlin; usually fastest on H100 |

For chatbot decode (latency-bound, batch=1): `GEMV` or `Marlin`. For batched serving (throughput-bound): `GEMM`. As of 2024, `Marlin` is usually the default if available.

---

## Quantization rule implemented

```python
# Per-channel activation magnitude over calibration
a = X.abs().mean(dim=0)               # [in]

# Grid search α
best_loss = inf
for alpha in linspace(0, 1, 20):
    s = a ** alpha                    # [in]
    W_scaled = W * s
    X_scaled = X / s
    W_q = quantize_group(W_scaled, group_size=128, bits=4, zero_point=True)
    loss = ((W_scaled @ X_scaled.T) - (dequant(W_q) @ X_scaled.T)).pow(2).mean()
    if loss < best_loss:
        best_loss, alpha_star = loss, alpha

# Final stored: W_q (INT4, group=128) + FP16 scales + INT4 zero-points
# s absorbed into previous LayerNorm: γ_new = γ / s_star
```

---

## Config / hyperparameters (defaults)

| Knob | Default | Notes |
|---|---|---|
| `w_bit` | 4 | also 3 |
| `q_group_size` | 128 | per-128-element group scale |
| `version` | `"GEMM"` | also `"GEMV"`, `"Marlin"` |
| `zero_point` | `True` | asymmetric INT4 |
| `n_samples` | 128 | calibration token batches |
| Calibration corpus | Pile / C4 | configurable |

---

## The absorption trick (zero runtime overhead)

The per-channel scale `s` would need a runtime multiply if applied directly. AutoAWQ instead folds it into the **preceding** LayerNorm at quant time:

```python
# Before
y = LayerNorm(x; γ, β); z = Linear(y, W)

# After (algebraically identical):
y_new = LayerNorm(x; γ_new, β_new)   # γ_new = γ / s,  β_new = β / s
z = Linear(y_new, s · W)             # W now stored as INT4 of (s · W)
```

The runtime graph is identical to FP16 in op structure; only the INT4 dequant+matmul replaces FP16 matmul.

---

## What's downstream

- **vLLM** directly consumes AutoAWQ checkpoints via its `QuantConfig`.
- **TensorRT-LLM** ingests AWQ checkpoints in its quant path.
- **HuggingFace `transformers`** supports `from_pretrained(model_id, ...quant="awq")` via `auto-awq` integration.
- **Marlin kernel** (same family as GPTQ-Marlin) is the high-performance backend on Ampere/Hopper.

---

## Connections

- [[awq]] — the paper this implements.
- [[smoothquant]] — same lab's W8A8 predecessor.
- [[autogptq]] — sibling Hessian-based PTQ framework.
- [[marlin-kernel]] — production kernel that AWQ ships through.
- [[vllm-quant]], [[tinychat-and-tensorrt-llm-quant]] — downstream consumers.
