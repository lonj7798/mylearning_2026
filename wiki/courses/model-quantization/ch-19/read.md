<!-- chapter: ch-19
     track: frontier-2025-2026
     title: Production Quant Kernels + Deployment Stacks
     sources: [[marlin-kernel]], [[machete-kernel]], [[tinychat-and-tensorrt-llm-quant]], [[tensorrt-llm-quant]], [[vllm-quant]], [[llama-cpp-ggml]], [[gguf-k-quants]], [[llama-cpp-gguf-releases]], [[torchao]], [[hf-quanto]], [[qserve]]
     figures: figures/kernel-roofline.html
-->

# Chapter 19 — Production Quant Kernels + Deployment Stacks

> **Core insight.** The accuracy of a quantization *algorithm* is necessary but not sufficient. Production speedup is determined by the *kernel* — whether the W4A16 GEMM keeps the tensor cores fed past the critical batch size, whether dequant happens in registers vs in shared memory vs in HBM, whether the kernel uses the right tensor-core ISA for the target hardware (MMA on Ampere, WGMMA + TMA on Hopper, NVFP4 MMA on Blackwell). The same GPTQ checkpoint can run at 4× FP16 throughput (Marlin / Machete) or at *slower than FP16* (a naïve dequant-then-FP16-GEMM kernel) depending on what's behind the `--quantization gptq` flag.
>
> **Guideline.** Match the kernel to the hardware: **Marlin** for Ampere (A100 / A6000), **Machete** for Hopper (H100 / H200), **TensorRT-LLM** for NVIDIA cloud production, **vLLM** for OSS serving, **TinyChat** for single-batch consumer/edge inference, **llama.cpp + gguf k-quants** for CPU / Apple Silicon. Pick the *algorithm* (GPTQ / AWQ / SmoothQuant / FP8 / NVFP4) by accuracy target; pick the *runtime* by hardware. For PyTorch ecosystem integration use **torchao** (CUDA-optimized) or **Quanto** (cross-device portable).

---

## Why this chapter exists

You can quantize a Llama-3-8B model to 4-bit with AWQ in five minutes. Whether that quantized model runs at 30 tokens/s or 300 tokens/s on the same H100 depends entirely on what serving stack you point at it. The 2024–2026 frontier of quantization deployment is *kernel engineering*: warp specialization, async memory copies (`cp.async` on Ampere, TMA on Hopper), tensor-core ISA selection, fused dequant inside the GEMM critical path, layout pre-shuffles that match the tensor-core fragment expectations.

This chapter is the "how it actually runs" chapter. Three things to walk away with:

1. **Marlin and Machete** — the open-source W4A16 GEMMs that achieve near-FP16 throughput at the batch sizes that actually matter (16-128). The kernel-level pipeline (async load + dequant + MMA + double-buffered swap) is the canonical pattern.
2. **TensorRT-LLM and vLLM** — the two production serving stacks. TRT-LLM is NVIDIA's compiled-engine path with full FP8 + NVFP4 + W4A16 + W8A8 coverage; vLLM is the OSS path that auto-dispatches across the same matrix via a uniform `quantization=` flag.
3. **llama.cpp + gguf k-quants** — the CPU / Apple Silicon stack that runs Llama-70B on a 32 GB MacBook. The k-quant ladder (q2_K through q8_0, plus IQ-quants) is the community deployment standard.

Plus the PyTorch ecosystem (torchao, Quanto) and the W4A8KV4 production path (QServe).

---

## 1. Marlin — the Ampere W4A16 GEMM that crossed the critical batch

### 1.1 The critical batch problem

Pre-Marlin W4A16 kernels (ExLlama, AutoGPTQ's CUDA kernel, TinyChat) had a structural flaw: they crushed batch-1 latency (4× FP16) but stopped helping (and often hurt) as soon as the workload became compute-bound rather than memory-bound. Their FLOPS/s collapsed past some "critical batch size" — typically 4 or 8.

This made W4A16 useless for serving. Production workloads run at batch 16-128. Below the critical batch, weight memory bandwidth dominates and any 4× weight compression gives 4× speedup. Above it, the FP16 baseline is faster.

[[marlin-kernel]] (Frantar et al., IST-Austria 2024) closed the gap. **Near-FP16 throughput maintained through batch 16-32**, continuing speedup up to batch ~64 on A100. The kernel-level pipeline is the canonical pattern that subsequent W4A16 kernels (including Machete) follow.

### 1.2 The pipeline

Three stages per CTA, double-buffered:

1. **Stage A — loader warps**: `cp.async` 4-bit weight tile (128 K rows × N cols, packed) and FP16 activation tile from global memory into shared memory buffer *i*.
2. **Stage B — compute warps**: in parallel, dequant the FP16 weights from shared buffer *1−i* via **shift + sign-extend + per-group scale multiply**, then issue tensor-core MMA on FP16 weights × FP16 activations.
3. **Stage C — barrier**: `mbarrier` sync, swap buffers, repeat.

The double buffer means stage A on iteration *k+1* runs in parallel with stage B on iteration *k* — **load latency is hidden by compute**. This is the pattern.

### 1.3 The weight pre-shuffle trick

A static offline reordering of the 4-bit weights so that **after the bit-shifts inside the dequant warp, each tensor-core thread already has its operand in the right register slot.** Eliminates per-call permute instructions.

The pre-shuffle changes the in-memory layout but preserves the bit content — you can take any GPTQ-quantized checkpoint and re-pack it for Marlin offline. vLLM does this automatically when you pass `--quantization gptq_marlin`.

### 1.4 Quantization format consumed

- **GPTQ packed-int4** with **group_size 128** (the de-facto standard since the GPTQ paper, ch-08).
- Per-group scale and zero-point in FP16; weights packed 8 × INT4 per 32-bit word.
- AWQ checkpoints work too (`--quantization awq_marlin`) — same packed layout, different per-channel scale strategy.

### 1.5 Performance

- A100 W4A16 / Llama-70B: up to **2.8× end-to-end speedup** in vLLM.
- Stays on the memory-bandwidth-bound roofline through batch 16, transitions gracefully into compute-bound past 32.
- 2:4 sparsity follow-up: another ~1.6× on top.

### 1.6 Hardware targets

- **Ampere (A100, A6000)**: original target, uses `cp.async`.
- **Hopper (H100)**: Marlin runs but doesn't use TMA/WGMMA — see Machete.

---

## 2. Machete — the Hopper-native rewrite

[[machete-kernel]] (Wilkinson, Neural Magic / Red Hat, October 2024) is the from-scratch redesign for Hopper. The diagnosis: **Marlin leaves ~37% of Hopper's peak compute on the floor** because it uses Ampere's `cp.async` + `mma.sync` instead of Hopper's WGMMA + TMA.

### 2.1 What changed

| Aspect | Marlin (Ampere) | Machete (Hopper) |
|--------|-----------------|-------------------|
| Tensor-core ISA | `mma.sync` (sync) | **WGMMA** (async, larger tiles) |
| Global → shared copy | `cp.async` (per-thread arithmetic) | **TMA** (one descriptor, built-in swizzle) |
| Consumer tile size | Ampere-MMA sized | Hopper-WGMMA wider |
| Layout description | hand-derived | **CUTE layout algebra** |

WGMMA is async, has much larger tile sizes (64×N×16 for FP16), and frees the issuing warp to do other work while the tensor core runs. TMA issues one descriptor to copy a whole tile, with swizzling and out-of-bounds handling built in — Ampere's `cp.async` needed per-thread address arithmetic.

### 2.2 Warp specialization (the canonical Hopper pattern)

Same skeleton as Flash-Attention 3 and DeepSeek-V3's FP8 GEMM:

- **Producer warps**: drive TMA copies of weight + activation tiles into shared memory.
- **Consumer warps**: issue WGMMA against the loaded tiles, accumulate in FP16 registers, write out via `stmatrix`.
- `mbarrier` synchronizes the two groups; double-buffered shared memory hides load latency.

### 2.3 Performance

| Configuration | Speedup vs Marlin |
|---------------|-------------------|
| Llama-70B, 1× H100, geomean | +29% |
| Llama-405B, 4× H100, geomean | +42% |
| Batch ≥ 128 | matches FP16 cuBLAS |

The headline: **W4A16 serving has no throughput cost at batch ≥ 128** — the regime that matters for high-concurrency serving.

### 2.4 vLLM integration

vLLM 0.6.2+ auto-selects Machete over Marlin when CUDA capability ≥ 9.0 (Hopper). Same `--quantization gptq_marlin` / `--quantization awq_marlin` CLI flags trigger Machete on H100. Compatible with all GPTQ / AWQ checkpoints already in the hub — no re-quantization needed.

---

## 3. TensorRT-LLM — the NVIDIA production stack

[[tensorrt-llm-quant]] is NVIDIA's compiled-engine serving framework. The quant matrix:

| Mode | Format | Notes |
|------|--------|-------|
| FP8 | E4M3 (per-tensor or per-row) | Hopper/Blackwell |
| INT8 SmoothQuant | per-channel + per-token | requires calibration |
| INT4 GPTQ | group-128 packed | Marlin-style internal kernel |
| INT4 AWQ | group-128 + activation-aware scale | AWQ kernel |
| **NVFP4** | 16-element FP4 + E4M3 scale + FP32 tensor | **Blackwell only**, TRT-LLM 0.13+ |
| KV cache | INT8 or FP8 (E4M3) | independent flag |

### 3.1 The build pipeline

```
ModelOpt (formerly AMMO) → calibration → quant_config.json + quantized weights
                                        ↓
                                trtllm-build → TensorRT engine (GPU-SKU-specific)
                                        ↓
                                ModelRunner / Triton backend → inference
```

Calibration lives in **ModelOpt** (`https://github.com/NVIDIA/TensorRT-Model-Optimizer`). It produces a `quant_config.json` and re-saves the model with quantized weights. `trtllm-build` then compiles a TensorRT engine targeting a specific GPU SKU (H100 vs B100 etc.) — separate engine build per target hardware.

### 3.2 Config schema (sketch)

```python
from tensorrt_llm.quantization import QuantMode

quant_mode = QuantMode.from_description(
    quantize_weights=True,
    quantize_activations=True,
    per_token=True,
    per_channel=True,
    use_int4_weights=False,
    use_int8_kv_cache=False,
    use_fp8_kv_cache=True,
    use_fp8_qdq=True,
    use_fp8_rowwise=False,
)
```

### 3.3 Kernel selection (per format)

| Format | Backend |
|--------|---------|
| FP8 | cuBLASLt FP8 GEMM (H100 WGMMA path) via Transformer Engine |
| INT8 SmoothQuant | custom CUDA plugin (`smoothQuant.cu`) |
| INT4 AWQ | AWQ GEMM kernel from `awq_ext` |
| INT4 GPTQ | Marlin-style internal kernel |
| NVFP4 | Blackwell tensor-core block-scaled GEMM |

### 3.4 Fusion patterns

- GEMM + bias + activation fused as one TRT layer.
- LayerNorm + GEMM fused via `LayerNormGemm` plugin.
- Multi-head attention fused with KV cache quant via `gptAttentionPlugin`.

### 3.5 NVFP4 path

Engine-build pass quantizes weights to NVFP4 (16-element FP4 blocks + E4M3 scale + FP32 tensor scale); activations cast per-block at runtime. Selective high-precision layers handled the same as the FP8 path. The Blackwell-only tensor-core code path consumes NVFP4 natively.

This is the deployment endpoint of the [[nvfp4-training]] / [[nvfp4-qad]] pipeline (ch-17).

---

## 4. vLLM — the OSS serving stack

[[vllm-quant]] (UC Berkeley + Anyscale + NeuralMagic) is the highest-throughput open-source LLM inference engine, built around PagedAttention. Quantization is a uniform `quantization=` flag that auto-detects checkpoint format.

### 4.1 The unified registry

Located at `vllm/model_executor/layers/quantization/`. Maps name → config class:

| Method | Sources |
|--------|---------|
| `gptq` | GPTQ packed-int4 (legacy kernel) |
| `awq` | AWQ packed-int4 (legacy kernel) |
| `gptq_marlin` | GPTQ → Marlin/Machete (high-throughput) |
| `awq_marlin` | AWQ → Marlin/Machete (high-throughput) |
| `fp8` | FP8 (per-tensor / per-channel) |
| `bitsandbytes` | NF4 / INT8 from bitsandbytes |
| `gguf` | llama.cpp k-quant formats |
| `compressed-tensors` | NeuralMagic unified format |
| `nvfp4` | Blackwell NVFP4 |

### 4.2 Detection flow

```python
# vllm/config.py (simplified)
def _get_quantization_config(model_config, load_config):
    if quantization_arg is not None:
        return QuantConfig.from_args(quantization_arg)
    cfg = hf_config.get("quantization_config", {})
    method = cfg.get("quant_method")
    if method == "gptq":   return GPTQConfig.from_config(cfg)
    if method == "awq":    return AWQConfig.from_config(cfg)
    if method == "fp8":    return Fp8Config.from_config(cfg)
    if method == "compressed-tensors":  return CompressedTensorsConfig.from_config(cfg)
    ...
```

vLLM reads `config.json`'s `quantization_config` field, instantiates the correct `QuantizationConfig` subclass, and routes every Linear layer through the matching `LinearMethodBase`.

### 4.3 KV cache quantization (independent flag)

```bash
vllm serve meta-llama/Llama-3-8B \
  --quantization awq_marlin \
  --kv-cache-dtype fp8_e4m3
```

- **FP8 KV**: per-token scaling, stored as E4M3 (default) or E5M2.
- **INT8 KV**: per-token scaling, stored as INT8.

Both reduce KV memory ~50%; FP8 attention kernels avoid dequant before softmax.

### 4.4 The user-facing API

```python
from vllm import LLM

# Auto-detect from checkpoint
llm = LLM(model="TheBloke/Llama-2-70B-AWQ")

# Explicit
llm = LLM(model="meta-llama/Llama-3-8B", quantization="awq_marlin")

# Full FP8 path
llm = LLM(model="meta-llama/Llama-3-8B-FP8",
          quantization="fp8",
          kv_cache_dtype="fp8")
```

---

## 5. TinyChat — the AWQ reference runtime

[[tinychat-and-tensorrt-llm-quant]] documents MIT Han Lab's reference inference runtime for AWQ-quantized models. Distinctive contributions:

- First W4A16 inference stack that matched the AWQ paper's theoretical 4× speedup on **Jetson / consumer-GPU hardware** — proved 4-bit weight serving was practically deployable on the edge.
- **Lookup-table-based dequant**: each 4-bit weight indexes a per-group LUT in shared memory, avoiding per-element shift + scale on every MMA tile.
- **In-place INT4 weight storage** with on-the-fly cast directly into tensor-core registers.
- **Fused FFN**: combines gate, up, down projections + SwiGLU into one mega-kernel that holds activations in registers across the FFN.
- Reference for the AWQ paper; later forked into vLLM's `awq` backend.

Use TinyChat for **single-batch consumer / edge** inference (Jetson Orin Nano runs Llama-2 7B at ~30 tokens/s). For high-batch serving the Marlin/Machete pipeline dominates.

---

## 6. llama.cpp + gguf k-quants — the CPU / Apple Silicon stack

[[llama-cpp-ggml]] is the C/C++ inference engine that brought local LLM execution to CPUs, Apple Silicon, and consumer GPUs through the **gguf** model file format and the **ggml** tensor library. Its distinctive quantization contribution is the **k-quant family**.

### 6.1 The k-quant family

| Variant | Effective bpw | Use case |
|---------|---------------|----------|
| `q2_K` | 2.6 | extreme size budget |
| `q3_K_S` / `q3_K_M` / `q3_K_L` | 3.4 / 3.6 / 3.9 | small models / edge |
| `q4_K_S` | 4.6 | balanced |
| **`q4_K_M`** | **4.85** | **recommended consumer default** |
| `q5_K_M` | 5.7 | accuracy-leaning |
| `q6_K` | 6.6 | near-FP16 quality |
| `q8_0` | 8.5 | safety baseline |
| `iq4_xs` | 4.25 | codebook-based, more accurate at same size |

### 6.2 Super-block layout (q4_K, 256 weights)

```
1 superblock = 256 weights
            = 16 sub-blocks of 16 weights each
            + 1 FP16 superblock scale
            + 16 per-sub-block scales (each 6-bit, packed)
            + 16 per-sub-block mins (each 6-bit, packed)
```

Total bytes per 256-weight block: 256·4/8 (weights) + 8·6/8·2 (sub-block scales + mins) + 2 (FP16 superblock scale) = **128 + 12 + 2 = 144 bytes / 256 weights = 4.5 bpw**.

### 6.3 Two-level scaling (the load-bearing trick)

```
scale_b = d * unpack_6bit(sub_scales, b)        # FP16 × 6-bit → FP16 sub-scale
min_b   = dmin * unpack_6bit(sub_mins, b)
w[16*b + i] = scale_b * q[16*b + i] − min_b     # INT4 → FP16
```

The FP16 superblock master scale is multiplied by the 6-bit sub-block scale to recover the FP16 sub-block scale. **The scales themselves are quantized against a higher-level scale** — the same two-level idea NVFP4 (ch-17) uses with FP8 block scales + FP32 tensor scale.

### 6.4 _S / _M / _L variants — mixed precision within a model

The suffix denotes how aggressively llama.cpp upgrades sensitive tensors:

- **_S (small)**: all tensors at the base bit-width.
- **_M (medium, default)**: bump `attention.wv`, `attention.wo`, `ffn_down` to q{N+1}_K.
- **_L (large)**: bump even more tensors.

Sensitivity analysis on Llama showed these three layer types empirically dominate the perplexity cost of low-bit quant — a per-tensor-class echo of the [[squeezellm]] (ch-11) sensitivity-aware insight.

### 6.5 IQ-quants (2024 addition)

Importance-weighted quantization using an **imatrix** file computed from calibration text. Uses a learned codebook per super-block. `iq2_XS` / `iq3_XXS` are state-of-the-art at 2-3 bpw for llama.cpp. The imatrix is conceptually a lightweight cousin of GPTQ's Hessian — same idea (weight quant error by activation importance), simpler implementation.

### 6.6 The community release pattern

Curator workflow (e.g. `bartowski`, the dominant 2024+ gguf curator):

1. Download BF16/FP16 weights from the model HF org.
2. Convert to gguf via `convert_hf_to_gguf.py`.
3. Generate imatrix from a 1-2k token calibration corpus.
4. Quantize to 10-12 variants: `Q8_0`, `Q6_K`, `Q5_K_M`, `Q5_K_S`, `Q4_K_M`, `Q4_K_S`, `IQ4_XS`, `Q3_K_M`, `IQ3_M`, `IQ2_M`, `IQ1_M`.
5. Upload all variants to a single HF repo with the imatrix file.
6. Publish PPL comparison table in the README.

The K-quant + IQ-quant ladder is arguably the **most-deployed quantization stack in absolute model count** in the world — every llama.cpp / Ollama / LM Studio user is running one of these variants.

### 6.7 Quality landscape (Llama-3-8B reference)

- BF16: 16 GB.
- Q8_0: 8.5 GB (PPL ≈ baseline + 0.001).
- Q6_K: 6.6 GB (PPL ≈ baseline + 0.005).
- Q5_K_M: 5.7 GB (PPL ≈ baseline + 0.01).
- **Q4_K_M: 4.9 GB (PPL ≈ baseline + 0.03)** — the consumer-default sweet spot.
- IQ4_XS: 4.4 GB (PPL ≈ baseline + 0.05).
- Q3_K_M: 4.0 GB (PPL ≈ baseline + 0.10).
- IQ2_M: 2.7 GB (PPL ≈ baseline + 0.4).
- IQ1_M: 1.9 GB (PPL ≈ baseline + 1.0) — usable for chat but degraded.

---

## 7. PyTorch ecosystem — torchao + Quanto

### 7.1 torchao

[[torchao]] is PyTorch's official native quantization library. Architectural centerpiece is `AffineQuantizedTensor`, a `torch.Tensor` subclass that carries quantized data + scale/zero-point + a layout policy. The API surface is small:

```python
from torchao.quantization import quantize_, int4_weight_only, float8_dynamic_activation_float8_weight

# Inference: W4A16 with group_size=128
quantize_(model, int4_weight_only(group_size=128))

# FP8 dynamic act + FP8 weight
quantize_(model, float8_dynamic_activation_float8_weight())

# FP8 training
from torchao.float8 import Float8Linear  # drop-in nn.Linear replacement
```

Designed to be `torch.compile`-friendly — quant ops are tracable and fusible. INT4 uses CUTLASS `tinygemm`; FP8 uses scaled `_scaled_mm`.

Config matrix:

| Config | Bits (W/A) | Notes |
|--------|-----------|-------|
| `int4_weight_only(group_size=128)` | W4/A16 | inference; CUTLASS tinygemm |
| `int8_weight_only()` | W8/A16 | inference; simple int8 @ bf16 |
| `int8_dynamic_activation_int8_weight()` | W8/A8 | dynamic per-token act quant |
| `float8_dynamic_activation_float8_weight()` | W8/A8 | full FP8 inference |
| `float8_weight_only` | W8/A16 | FP8 weight, BF16 act |

### 7.2 Quanto

[[hf-quanto]] (HuggingFace) prioritizes **cross-device portability** over peak throughput. A `QTensor` subclass holds INT2/INT4/INT8/FP8 weights with per-axis scales; `QLinear` replacements dispatch to vectorized kernels on whichever backend PyTorch supports (CUDA, MPS, CPU, Intel XPU).

```python
from transformers import AutoModelForCausalLM, QuantoConfig

quant_config = QuantoConfig(weights="int4", activations=None)
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3-8B",
    quantization_config=quant_config,
    device_map="auto",
)
```

Trades peak throughput (vs Marlin / bitsandbytes specialized kernels) for cross-device coverage — a quantized Quanto checkpoint runs unchanged on H100, M2 Max, Xeon SPR, or Intel Arc.

---

## 8. QServe — the W4A8KV4 production path

[[qserve]] (Lin et al., MIT HAN Lab 2024) co-designs the quantization scheme and the kernel for **Hopper W4A8KV4** serving. The diagnosis: naïve INT4 dequantization on GPUs incurs 20–90% runtime overhead because dequant happens at the wrong memory hierarchy level (HBM ↔ SMEM).

### 8.1 Progressive group quantization (the key idea)

- **Stage 1 (per-channel INT8):** quantize each output channel of W to INT8 with a single per-channel scale `s_c ∈ FP16`. Store as INT8 `W_s`.
- **Stage 2 (per-group INT4):** within each group of g=128 weights along the input axis, quantize the INT8 values to INT4 with a per-group scale `s_g ∈ INT8` (note: scale is itself an integer, not FP). Store as INT4 `W_g`.
- **At inference:** dequantize INT4 → INT8 entirely **in registers** (multiply by `s_g`, an INT8×INT8 → INT16 op cheap on tensor cores); feed INT8 operand into the INT8 tensor-core GEMM with A8 activation. **No FP dequant in the critical path.**

Why this matters: prior W4A8 (e.g. Marlin) dequantizes INT4 → FP16 in registers and feeds FP16 into the tensor core. The FP dequant adds 2-3 instructions per element and pushes register pressure. QoQ keeps everything integer until the GEMM accumulator.

### 8.2 SmoothAttention

`softmax(QK^T/√d)V` is sensitive to KV4 because INT4 K introduces noise that gets amplified by softmax. QoQ applies a SmoothQuant-style learnable per-head scaling:

```
Q' = Q · s,  K' = K / s
```

`QK^T` is unchanged but `K'` has reduced dynamic range, making K4 quantization gentler. `s` calibrated to minimize softmax KL.

### 8.3 Throughput

- Llama-3-8B: 1.2× over TensorRT-LLM W8A8 on H100, 2.4× on L40S.
- Qwen1.5-72B: 3.5× over Atom W4A4 on A100 (Atom is hurt by softmax instability from A4).

---

## 9. Stack selection guide

```
Hardware       | Best W4A16          | Best W8A8        | Best FP8/FP4    | KV-quant
───────────────|─────────────────────|──────────────────|─────────────────|──────────
Edge / Jetson  | TinyChat (AWQ)      | —                | —               | FP16 only
Single 4090    | vLLM + Marlin       | vLLM SmoothQuant | —               | INT8 / FP8
A100 / OSS     | vLLM + Marlin       | vLLM SmoothQuant | —               | FP8
H100 / OSS     | vLLM + Machete      | vLLM SmoothQuant | vLLM FP8        | FP8
NVIDIA cloud   | TRT-LLM             | TRT-LLM          | TRT-LLM         | FP8 KV
Blackwell      | TRT-LLM (NVFP4)     | TRT-LLM          | TRT-LLM NVFP4   | FP8 / NVFP4
CPU / Apple    | llama.cpp Q4_K_M    | llama.cpp Q8_0   | —               | FP16 only
Cross-device   | Quanto              | Quanto           | Quanto FP8      | —
PyTorch native | torchao int4_only   | torchao W8A8     | torchao FP8     | —
W4A8KV4 / Hopper| QServe             | QServe           | —               | KV4 native
```

---

## 10. Practitioner's cheat-sheet

```python
# vLLM — auto-detect from checkpoint
from vllm import LLM
llm = LLM(model="TheBloke/Llama-2-70B-AWQ")

# vLLM — force Machete on Hopper
llm = LLM(model="meta-llama/Llama-3-8B-AWQ",
          quantization="awq_marlin",       # auto-routes to Machete on H100
          kv_cache_dtype="fp8_e4m3")

# TensorRT-LLM — calibration → build → serve
# Step 1: ModelOpt calibration
import modelopt.torch.quantization as mtq
mtq.quantize(model, mtq.FP8_DEFAULT_CFG, forward_loop=calib_loop)

# Step 2: Build engine
# $ trtllm-build --checkpoint_dir ./calibrated --output_dir ./engine \
#                --use_fp8 --use_fp8_kv_cache --max_batch_size 64

# Step 3: Load + generate
from tensorrt_llm.runtime import ModelRunner
runner = ModelRunner.from_dir("./engine")
outputs = runner.generate(input_ids, max_new_tokens=100)

# torchao — PyTorch-native, torch.compile friendly
from torchao.quantization import quantize_, int4_weight_only
quantize_(model, int4_weight_only(group_size=128))
model = torch.compile(model)

# llama.cpp — CPU / Apple Silicon
# $ llama.cpp/convert_hf_to_gguf.py meta-llama/Llama-3-8B
# $ llama.cpp/llama-quantize Llama-3-8B-F16.gguf Llama-3-8B-Q4_K_M.gguf q4_k_m
# $ llama.cpp/llama-cli -m Llama-3-8B-Q4_K_M.gguf -p "hello"
```

---

## Common pitfalls

- **Using the legacy `gptq` kernel instead of `gptq_marlin`:** legacy AutoGPTQ CUDA kernel collapses past batch 4; Marlin/Machete maintains throughput through batch 64+. Always pass the `_marlin` suffix for serving.
- **Building TRT-LLM engines for the wrong GPU SKU:** engines are SKU-specific (separate build per H100 vs B100). A Hopper-built engine won't run optimally on Blackwell.
- **Mixing Q4_K_M with `_S` thinking it's smaller:** the suffix denotes which *tensors* get upgraded, not overall bit-width. `_M` upgrades critical tensors (wv, wo, ffn_down) so it's slightly larger than `_S` at the same base bit-width.
- **Quantizing the LM head:** for every method in this chapter, the LM head is the most precision-sensitive layer. Always exclude it (`exclude=["lm_head"]` in Quanto, default exclusion in TRT-LLM, etc.).
- **Forgetting `--kv-cache-dtype`:** in vLLM, weight quant and KV quant are *independent* flags. `--quantization awq_marlin` alone leaves KV at FP16; pair with `--kv-cache-dtype fp8_e4m3` for the full memory win.
- **Assuming TinyChat = Marlin = production:** TinyChat is for batch ≤ 8 (consumer / edge); Marlin/Machete is for batch 16-128 (serving). Different kernels for different regimes.
- **Calibrating with the wrong data:** GPTQ + AWQ are sensitive to calibration set distribution. The default 128 sequences from `wikitext` is good for general models but wrong for domain-specific fine-tunes. Calibrate on representative data.
- **Re-quantizing instead of re-packing for Marlin:** any GPTQ-quantized checkpoint can be re-packed for Marlin offline. You don't need to re-run GPTQ.

---

## Connections and what's next

- **[[gptq]] / ch-08** — the algorithm whose 4-bit checkpoint format Marlin / Machete / TRT-LLM consume.
- **[[awq]] / ch-09** — the activation-aware scaling Marlin runs via `awq_marlin`; TinyChat is the original AWQ runtime.
- **[[qlora]] / ch-12** — uses bitsandbytes NF4 weights, served via vLLM's `bitsandbytes` quant config.
- **[[nf4]] / [[gguf-k-quants]] / ch-02** — the format specs production kernels consume.
- **[[deepseek-v3-fp8]] / [[transformer-engine]] / ch-17** — DSV3's FP8 GEMM pattern shares the warp-specialized producer/consumer skeleton with Machete and Flash-Attention 3.
- **[[nvfp4-training]] / [[nvfp4-qad]] / ch-17** — the deployment endpoint of the NVFP4 pipeline is TRT-LLM's NVFP4 engine path.
- **[[kivi]] / [[kvquant]] / ch-15 + [[turboquant]] / ch-18** — KV-cache quant kernels integrated into vLLM's `--kv-cache-dtype` flag and Machete/TRT-LLM's attention plugins.
- **ch-20** — evaluation methodology; how to measure whether a production kernel actually delivers its claimed accuracy.
- **ch-21 lab** — runs all four canonical W4 methods end-to-end through these stacks.

## Further reading

- [[marlin-kernel]] — the canonical Ampere W4A16 kernel; required reading.
- [[machete-kernel]] — Hopper-native successor.
- [[tinychat-and-tensorrt-llm-quant]] — the practitioner-side production-stacks overview.
- [[gguf-k-quants]] — the CPU/Apple deployment format.
- [[qserve]] — W4A8KV4 production path; co-design exemplar.

## Companion visualization

**[figures/kernel-roofline.html](figures/kernel-roofline.html)** — interactive roofline plot for W4A16 GEMM. Sliders for batch size (1 → 256), tensor-core ISA (mma.sync vs WGMMA), and dequant placement (HBM vs SMEM vs register). Shows the critical batch crossover and how Marlin / Machete extend it.
