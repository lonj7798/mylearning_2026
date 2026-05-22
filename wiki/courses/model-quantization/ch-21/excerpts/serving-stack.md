---
chapter: ch-21
course: model-quantization
phase: read
excerpt_of: "vLLM kernel selection per checkpoint format and latency-measurement protocol for the ch-21 lab"
created_at: "2026-05-21"
---

# Excerpt: vLLM Serving Stack for the Lab

**Sources:** [[raw-data/vllm-quant]], [[raw-data/marlin-kernel]], [[raw-data/autogptq]], [[raw-data/autoawq]], [[raw-data/bitsandbytes-nf4]]

---

## Kernel selection per format

vLLM auto-detects checkpoint format via `config.json`'s `quantization_config` field and dispatches to the right CUDA kernel. The override flag is `quantization=<name>`.

| Method checkpoint | vLLM `quantization` flag | Underlying kernel | Hardware |
|-------------------|------------------------|---------------------|----------|
| AutoGPTQ W4 g128 | `gptq_marlin` | Marlin W4A16 GEMM ([[marlin-kernel]]) | Ampere + Hopper |
| AutoGPTQ W4 g128 (legacy) | `gptq` | AutoGPTQ-CUDA / ExLlama | older paths |
| AutoAWQ W4 g128 | `awq_marlin` | Marlin (re-uses Marlin via AWQ→GPTQ-format re-pack) | Ampere + Hopper |
| AutoAWQ W4 g128 (legacy) | `awq` | AWQ-CUDA from TinyChat | fallback |
| QLoRA (NF4 base + LoRA adapter) | `bitsandbytes` + adapter API | bnb fused GEMV | any |
| bitsandbytes-INT8 | `bitsandbytes` | bnb mixed-precision INT8 + FP16 outliers | any |
| FP16 baseline | (no flag) | cuBLAS FP16 GEMM | any |

**Always prefer the Marlin path for GPTQ/AWQ on Ampere/Hopper.** Legacy paths exist for compatibility but are 1.5–3× slower at production batch sizes.

### Loading commands

```python
# AutoGPTQ → Marlin
from vllm import LLM
llm = LLM("out/gptq-w4-g128", quantization="gptq_marlin", dtype="auto")

# AutoAWQ → Marlin
llm = LLM("out/awq-w4-g128", quantization="awq_marlin", dtype="auto")

# bitsandbytes-INT8 (vLLM re-runs bnb at load time)
llm = LLM("meta-llama/Llama-3-8B", quantization="bitsandbytes",
          load_format="bitsandbytes", dtype="bfloat16")

# QLoRA — load NF4 base + attach adapter
llm = LLM("meta-llama/Llama-3-8B", quantization="bitsandbytes",
          load_format="bitsandbytes",
          enable_lora=True, max_loras=1, max_lora_rank=64)
# then route requests with `LoRARequest("alpaca", 1, "out/qlora-r64")`

# FP16 baseline
llm = LLM("meta-llama/Llama-3-8B", dtype="bfloat16")
```

---

## Latency measurement protocol

Per [[ch-20]] §6, report decode latency at batch 1 / 16 / 64. The discipline:

```python
from vllm import LLM, SamplingParams
import time, statistics

def bench(llm, batch_size, n_warm=2, n_meas=10, gen_len=128, prompt="The capital of France is"):
    prompts = [prompt] * batch_size
    sp = SamplingParams(max_tokens=gen_len, temperature=0)
    # Warm
    for _ in range(n_warm):
        llm.generate(prompts, sp)
    # Measure
    samples = []
    for _ in range(n_meas):
        t0 = time.perf_counter()
        outs = llm.generate(prompts, sp)
        t1 = time.perf_counter()
        # Verify all outputs reached gen_len (no early EOS skewing the measurement)
        produced = sum(len(o.outputs[0].token_ids) for o in outs)
        ms_per_token = (t1 - t0) * 1000 * batch_size / produced
        samples.append(ms_per_token)
    return statistics.mean(samples), statistics.stdev(samples)
```

Report `mean ± stdev` in ms/token. Pin `temperature=0` so the output length is deterministic. Verify the produced token count to catch early-EOS skews — a model that hits EOS at token 50 instead of 128 will look 2.5× faster than it really is.

**Peak VRAM**: read `torch.cuda.max_memory_allocated()` after a warm-up + a single full forward at the largest batch. Reset before each method.

**Throughput at batch 16**: total tokens generated / total wall-clock seconds, averaged over 10 trials.

---

## What can go wrong

### Kernel mismatch silently degrades quality

`vllm --quantization gptq` (legacy CUDA kernel) accepts the same checkpoint as `gptq_marlin` but produces *bit-different* outputs for the same prompt because the legacy dequant + GEMM has different numerical behavior. The PPL drift can be ~0.05. Always specify the kernel explicitly when comparing across methods.

### `kv_cache_dtype` interaction

The lab does *not* exercise KV-cache quantization (that's [[ch-22]]). But vLLM defaults to FP16 KV cache; if you accidentally set `kv_cache_dtype="fp8"` you're now testing KV-quant + weight-quant together and the comparison is contaminated. Leave `kv_cache_dtype` at default for ch-21.

### `dtype="auto"` vs explicit

`dtype="auto"` picks BF16 on Hopper, FP16 on Ampere. For cross-hardware reproducibility, set `dtype="bfloat16"` explicitly. The compute precision affects activation magnitudes and therefore latency in subtle ways.

### bitsandbytes runtime re-transform

vLLM's bnb path re-runs bitsandbytes quantization at load time — the base model on disk is still FP16. This means the "load time" includes a quantization step (~30s for 8B INT8). Account for this in your wall-clock report; it's not the "quantize time" (which is ~zero for bnb) but it is a deployment cost.

---

## Verifying the kernel is what you think

After loading, vLLM logs which kernel it picked:

```
INFO ... model_runner.py:1234] Loaded with quantization='gptq_marlin'
INFO ... gptq_marlin.py:567] Using Marlin kernel for layer mlp.up_proj (group_size=128, sym=False)
```

Check the log. A `legacy CUDA fallback` message means the checkpoint shape isn't Marlin-compatible (usually a wrong `group_size` or `sym` mismatch). Re-quantize with Marlin-compatible config rather than accepting the slow fallback.

---

## Connections

- [[ch-21]] §step-by-step — uses these commands.
- [[ch-21]] §Pareto deliverable — the latency / VRAM / throughput row comes from this protocol.
- [[vllm-quant]] — the upstream documentation for the quantization registry.
- [[marlin-kernel]] — the W4A16 kernel that backs the Marlin path.
- [[ch-19]] — production-kernel chapter where these kernels are studied in depth.
