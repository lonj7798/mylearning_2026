<!-- chapter: ch-12
     phase: kernels-runtime
     title: CUDA Graphs + Runtime Optimizations
     sources: [[cuda-graphs-inference]], [[vllm]], [[tensorrt-llm]], [[sglang]]
     back: [[continuous-batching]] (ch-04), [[flashattention]] (ch-11)
     forward: [[vllm-scheduler]] (ch-16), [[sglang-scheduler]] (ch-17), framework internals (ch-18)
-->

# Chapter 12 — CUDA Graphs + Runtime Optimizations

> **Core insight.** A decode step on a 70B LLM launches roughly 800 CUDA kernels (80 layers × ~10 kernels per layer: QKV proj, RoPE, attention, output proj, two FFN matmuls, two layer norms, residual add, sometimes more). Each launch costs ~5–10 µs of CPU-side overhead. **At 10 µs × 800 = 8 ms of pure launch overhead per token** — and a decode step's total budget on H100 is 15–25 ms. Roughly *half* the decode latency is the CPU launching kernels, not the GPU running them. **CUDA Graphs** capture the entire kernel sequence as a static replayable object; one launch replaces 800. The catch: graphs require static shapes, and serving has dynamic batches. The fix is **piecewise capture** — one graph per batch-size bucket. This single optimization typically cuts decode TPOT by 30–50% in production engines.
>
> **Guideline.** Always enable CUDA graphs for decode in production. Capture one graph per power-of-2 batch size up to your maximum (1, 2, 4, 8, 16, 32, ...). Pad small batches up to the nearest bucket. Disable graph capture only for prefill (variable seq length) and during model-init warmup. The same logic applies — at smaller scale — to kernel fusion and NCCL collective fusion.

---

## Why this chapter exists

By chapter 11 you have fast attention (FA2/FA3/FlashDecoding/FlashInfer). The other ~10 ops per layer (QKV, RoPE, residual, layer norm, FFN matmuls, output proj, sometimes RMS norm + SiLU) are also fast individually. **The kernels are not the problem.** The problem is what happens between them.

Three different CPU-side overheads dominate decode latency in 2026:

1. **Kernel launch overhead** (~5–10 µs/launch × 800 kernels = ~8 ms/token). CUDA Graphs fix this.
2. **Tensor allocation overhead** (PyTorch allocator hits, autograd bookkeeping, dispatch). Static memory pools fix this.
3. **NCCL collective overhead** (for tensor-parallel models, ~50 µs per all-reduce × 80 layers = 4 ms/token). NCCL fusion / overlapping fix this.

Plus a fourth, intermittent: **JIT shape compilation** (`torch.compile` retracing when shapes change). Static shape buckets fix this.

This chapter walks through CUDA Graphs first (the biggest single win), then the surrounding runtime optimizations (kernel fusion, NCCL fusion, dynamic-shape JIT), then how vLLM / SGLang / TensorRT-LLM each handle the dynamic-shape problem differently.

---

## 1. Launch overhead — the arithmetic

A kernel launch is not free. The CPU side:

1. PyTorch allocates a CUDA stream slot (`cudaStreamWaitEvent`, ~1 µs).
2. The autograd engine records the operation (if grad-enabled; usually disabled at inference, but still ~1 µs check).
3. PyTorch dispatches through the type/device router (~1 µs).
4. CUDA driver enqueues the kernel onto the stream (`cudaLaunchKernel`, ~3–5 µs).
5. The kernel sits in the stream queue until the GPU is ready.

Total per launch: **5–10 µs on a modern CPU + driver**. The launch is asynchronous — the CPU moves on — but the *cumulative* CPU work is bounded by the number of launches per second.

A Llama-3-70B decode step issues approximately:

| Op per layer | Kernels |
|---|---|
| Pre-attention RMSNorm | 1 |
| QKV projection (or 3 matmuls if fused differently) | 1–3 |
| RoPE | 1 |
| Attention (kernel + KV append) | 1–2 |
| Output projection | 1 |
| Residual add | 1 |
| Pre-FFN RMSNorm | 1 |
| Gate + up projection | 1–2 |
| SiLU/SwiGLU + multiply | 1 |
| Down projection | 1 |
| Residual add | 1 |
| **Total** | **~10–14 kernels** |

× 80 layers = **800–1100 kernels per token**.

At 7 µs/kernel × 1000 kernels = **7 ms per token of pure launch overhead**.

On H100 with FA3 + FlashDecoding, the *GPU compute* for one Llama-3-70B decode token is ~8–12 ms. The launch overhead is comparable to the compute. **Half the decode latency is the CPU launching kernels.**

This is the gap CUDA Graphs close.

---

## 2. CUDA Graphs — capture + replay

The CUDA Graphs API (`cudaGraph`, `cudaGraphExec`, exposed in PyTorch as `torch.cuda.CUDAGraph`) lets you **record** a sequence of GPU operations once and **replay** them with a single launch:

```python
# Setup: allocate static input/output buffers (graph captures pointer values)
static_input = torch.empty(B, S, D, device="cuda")
static_output = torch.empty(B, S, V, device="cuda")

# Warmup pass — autotuning, allocator priming
for _ in range(3):
    static_output.copy_(model(static_input))

# Capture
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    static_output.copy_(model(static_input))

# Replay — one launch instead of N
for token_step in range(N):
    static_input.copy_(real_input_for_this_step)
    g.replay()
    read(static_output)
```

The capture pass observes every kernel issued during the recording block and packages them as a graph object. Replay re-issues all kernels in order with **one `cudaGraphLaunch` call** — typically 1–2 µs.

The savings: 800 launches × 7 µs = 5.6 ms collapses to one launch × 2 µs = 0.002 ms. **A ~5 ms TPOT reduction.** On a 15 ms TPOT baseline, that's 33% off.

### What graphs preserve

- **Kernel sequence and dependencies.** Recorded once.
- **Stream and event ordering.** Recorded.
- **Memory addresses.** Recorded — this is the key constraint (see below).

### What graphs do NOT do

- They don't make individual kernels faster.
- They don't reduce HBM traffic.
- They don't help compute-bound prefill (where launch overhead is a small fraction of GPU time).
- They don't tolerate shape changes.

---

## 3. The static-shape constraint

CUDA graphs capture **pointer addresses, not data**. The captured graph runs against the *exact tensors* it was recorded with. To use the graph with new data, copy the new data into the captured tensors before replay:

```python
static_input.copy_(new_data)  # mutates the recorded buffer
g.replay()                    # runs with the new data
```

This means **shapes cannot change between replays**. If batch size changes from 4 to 5, the recorded graph is invalid:

- Tensor `q` was recorded as `[4, 32, 1, 128]`; now it needs to be `[5, 32, 1, 128]`.
- The kernel block-grid is hardcoded to the recorded shape.
- The graph throws or produces wrong output.

For static batch (offline benchmarking, fixed-batch deployment): use one graph, done.

For dynamic serving: requests arrive and finish continuously, so the running batch size fluctuates. **The graph approach naively doesn't apply.**

---

## 4. Piecewise CUDA graphs — one graph per batch-size bucket

The standard production fix: **capture multiple graphs, one per discrete batch size**. Pad smaller batches up to the nearest captured size.

```python
# At engine init: capture one graph per power-of-2 batch size
graphs = {}
for B in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
    static_input = torch.empty(B, ..., device="cuda")
    static_output = torch.empty(B, ..., device="cuda")
    # warmup
    for _ in range(3):
        static_output.copy_(model(static_input))
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        static_output.copy_(model(static_input))
    graphs[B] = (g, static_input, static_output)

# At serve time: dispatch to the nearest bucket
def decode_step(actual_batch):
    bucket_B = next_power_of_2(actual_batch.size(0))
    g, static_in, static_out = graphs[bucket_B]
    padded = pad_to(actual_batch, bucket_B)
    static_in.copy_(padded)
    g.replay()
    return static_out[:actual_batch.size(0)]
```

### Cost model

- **Capture time**: ~10–60 s at engine init (one capture per bucket × ~5 s each). One-time cost.
- **Memory**: each graph holds static buffers ~ (batch × seq × hidden) bytes. For Llama-3-70B with 9 buckets, ~4 GB of static buffers (acceptable on 80 GB H100).
- **Padding waste**: batch=5 dispatched to bucket=8 runs 60% useful work. Average overhead across uniform random batches: ~25%.
- **Net win**: still 20–30% TPOT improvement even after padding overhead.

The padding overhead is why bucketing matters: too few buckets (just 1, 32, 256) wastes compute; too many buckets (every integer) wastes capture memory. Power-of-2 buckets are the practical sweet spot.

### Shape dimensions that matter

For decode, the dynamic dimensions are:
- **batch size** (number of concurrent decoding requests) — varies per iteration.
- **max KV length** (longest KV cache in the batch) — varies per iteration.

Most engines bucket only over batch size; KV length is handled by padding the attention mask. Some engines (FlashInfer with cooperative scheduling) bucket over both.

---

## 5. Framework-specific approaches

The three production stacks each handle CUDA graphs differently, reflecting different design philosophies.

### vLLM — piecewise CUDA graphs

vLLM (`vllm.compilation.cuda_graph`, `vllm/v1/worker/gpu_model_runner.py`) captures one CUDA graph per batch-size bucket. Defaults:

```text
CUDA_GRAPH_CAPTURE_SIZES = [1, 2, 4] + [8 * i for i in range(1, max_num_seqs // 8 + 1)]
```

At engine init, the worker captures `len(CUDA_GRAPH_CAPTURE_SIZES)` graphs (typically 30+). Each captures the full forward pass at that batch size.

Knobs:
- `--enforce-eager` disables CUDA graphs (debug only; ~30% TPOT regression).
- `--max-num-seqs N` bounds the largest captured graph (memory cost grows linearly).
- `--cuda-graph-sizes` overrides the bucket list.

For prefill, vLLM does *not* use CUDA graphs (variable prompt length defeats bucketing). Prefill runs eagerly with FlashInfer/FA backend.

### TensorRT-LLM — ahead-of-time engine build

TensorRT-LLM goes further. Instead of capturing graphs at runtime, it builds a **TensorRT engine** at deployment time:

```bash
trtllm-build --model llama3-70b --max_batch_size 64 --max_input_len 8192 \
             --max_output_len 2048 --tp_size 4 --quantization fp8
```

This produces a `.engine` file that contains:
- Pre-fused kernels (custom CUDA, hand-tuned per shape regime).
- Pre-selected attention backend (FA3 / FlashDecoding).
- Pre-compiled CUDA graphs internally.
- Pre-tuned NCCL communication patterns.

At serve time, the engine just executes — no JIT compilation, no graph capture overhead. **Lowest production latency**, but the engine is hardware-and-shape-specific. Move to a different GPU model or change `max_batch_size` and you rebuild.

The tradeoff:
- TRT-LLM: lowest latency, slowest iteration, NVIDIA-only.
- vLLM: 90% of TRT-LLM latency, fast iteration, OSS-flexible.

### SGLang — RadixAttention-aware CUDA graphs

SGLang uses CUDA graphs similarly to vLLM (piecewise capture per batch bucket), with one wrinkle: **its scheduler explicitly maintains batch-size stability across iterations**.

Because RadixAttention encourages prefix reuse, SGLang's scheduler can predict batch composition more accurately than naive continuous batching. It schedules requests to keep the active batch on the same bucket as long as possible, minimizing graph switches.

Result: SGLang reports ~5–10% TPOT improvement over vLLM purely from better graph-hit rate.

---

## 6. The other runtime optimizations

CUDA graphs are the biggest single win, but several others stack on top.

### NCCL collective fusion

For tensor-parallel models (TP ≥ 2), every layer ends with an all-reduce of the partial output. Per all-reduce overhead:

- 8-way TP on NVLink-Switch H100: ~30 µs
- 8-way TP on InfiniBand (cross-node): ~150 µs

× 80 layers × 2 all-reduces per layer (attention + FFN) = **5–25 ms of collective overhead per decode token**.

**NCCL fusion** combines successive all-reduces:

- Fuse the attention output all-reduce with the FFN input scatter.
- Use `NCCL_NTHREADS` and `NCCL_BUFFSIZE` tuning.
- Use `--enable-async-allreduce` to overlap with downstream compute.

Combined: ~30% reduction in TP overhead. vLLM exposes this via `--enable-overlap-sched`.

### Kernel fusion

PyTorch dispatches one kernel per op. Many ops can be fused:

- RMSNorm + matmul → single fused kernel (saves 1 launch + 1 HBM round-trip).
- GeLU/SiLU + matmul → single fused kernel.
- Bias add + activation → fused into the preceding matmul.
- Attention + KV append → fused into FlashInfer.

vLLM's `vllm/model_executor/layers/fused_moe` is the FFN-fusion path. TensorRT-LLM fuses aggressively (custom plugins). PyTorch's `torch.compile` does some of this automatically when traceable.

### Static memory pools

Even outside graphs, PyTorch's allocator has overhead. Each layer allocates intermediate tensors; on the next iteration, the same shapes are re-allocated. Static pools:

- Engine pre-allocates all activation buffers once at init.
- Each forward pass uses the same buffer addresses.
- Allocator overhead: zero.

Inside CUDA graphs this is automatic (capture freezes the addresses). Outside graphs, requires manual buffer management. vLLM does this for KV cache (the block manager owns all KV memory); other activations come from a static pool when graphs are used.

### Dynamic-shape JIT (`torch.compile` for inference)

For shape patterns where bucketing is expensive (e.g., prefill with widely varying lengths), `torch.compile` with `dynamic=True` JIT-compiles per shape regime. Less aggressive than CUDA graphs but tolerates shape variance.

vLLM uses `torch.compile` for prefill (variable shape, compile-once-cache-many). Decode uses CUDA graphs (static-shape buckets).

### Stream multiplexing

Multiple CUDA streams let different kernels run concurrently if they're independent. For tensor-parallel models, the NCCL all-reduce can run on a separate stream from the next layer's matmul. Saves the all-reduce latency if there's downstream-independent compute.

This is the `--enable-overlap-sched` path in vLLM.

---

## 7. Putting it all together — the optimized decode step

A 2026-class serving engine's decode step does:

```
1. Scheduler picks batch composition (continuous batching).        [~10 µs CPU]
2. Pad batch to nearest captured bucket.                            [~5 µs CPU]
3. Copy actual data into static input buffer.                       [~50 µs DtoD]
4. cudaGraphLaunch(g_bucket) — one launch.                          [~2 µs CPU + ~10 ms GPU]
5. Read sampled tokens from static output buffer.                   [~50 µs DtoH for small tokens]
6. KV-block manager appends new K, V.                               [~10 µs CPU]
```

Total decode TPOT on Llama-3-70B, H100 8-way TP: ~12–15 ms.

Without CUDA graphs (eager mode): ~22–28 ms (1.7× slower).
Without graphs + NCCL fusion: ~30+ ms (2× slower).
Without graphs + fusion + FlashInfer: ~50+ ms.

The cumulative speedup from the kernel + runtime stack: **3–4× over a naive PyTorch forward pass**.

---

## 8. When NOT to use CUDA graphs

CUDA graphs are not free. Skip them when:

- **Debugging.** Stack traces inside graphs are useless. Use `--enforce-eager`.
- **First training step** (not relevant for serving, but worth knowing).
- **Frequently changing shapes** (e.g., training with variable sequence length and no bucketing).
- **Memory-constrained deployments.** 30 graphs × 4 GB static buffers = 120 GB; on a 24 GB consumer GPU, you may need to disable graphs.
- **Prefill.** Variable prompt length per request defeats bucketing.
- **Speculative decoding draft phase.** Draft model has different shapes per draft length.

---

## 9. Common failure modes

| Symptom | Likely cause |
|---|---|
| Wrong output after batch change | Forgot to copy new data into captured static buffer |
| `cudaErrorIllegalAddress` on replay | Captured a tensor that was freed |
| Capture hangs | Allocator inside the graph; need warmup first |
| Memory explodes at engine init | Too many buckets × too-large static buffers |
| TPOT regresses on next deploy | Bucket list narrowed; padding overhead grew |
| Per-bucket variance | Bucket sizes too sparse; bigger pad waste |

---

## 10. Practitioner's runtime-tuning checklist

```python
# vLLM-style production decode setup (canonical).

llm = LLM(
    model="meta-llama/Llama-3-70B-Instruct",
    tensor_parallel_size=4,
    enforce_eager=False,              # CUDA graphs on — DO THIS
    max_num_seqs=128,                 # bound the largest graph
    max_num_batched_tokens=4096,      # chunk-prefill budget (see ch-05)
    gpu_memory_utilization=0.92,      # leave room for graph buffers
    enable_chunked_prefill=True,      # ch-05
    enable_prefix_caching=True,       # ch-07
    # graph capture sizes — adjust if max_num_seqs is smaller
    # default is reasonable for most workloads
)

# At runtime, monitor:
#   - graph hit rate (% iterations using captured graph)
#   - per-bucket utilization (are some buckets wasted?)
#   - eager-fallback rate (should be < 5%)
```

```bash
# For TensorRT-LLM (production NVIDIA-only):
trtllm-build --model llama3-70b --tp_size 4 --max_batch_size 128 \
             --max_input_len 8192 --max_output_len 2048 \
             --use_paged_kv_cache --kv_cache_type=paged \
             --enable_kv_cache_reuse --quantization=fp8
```

---

## Connections and what's next

- **Back to [[continuous-batching]] (ch-04)** — continuous batching produces the variable batch sizes that piecewise CUDA graphs handle.
- **Back to [[flashattention]] (ch-11)** — FA kernels are the hot path inside the captured graph; FlashInfer is designed to be graph-compatible.
- **Back to [[sarathi-serve]] (ch-05)** — chunked prefill changes the shapes the engine sees; vLLM uses graphs for decode only, eager for prefill.
- **Forward to ch-16 ([[vllm-scheduler]])** — vLLM's piecewise-graph implementation lives in the worker; the scheduler's batch-composition decisions affect graph-hit rate.
- **Forward to ch-17 ([[sglang-scheduler]])** — SGLang's scheduler is designed to maximize graph hits.
- **Forward to ch-18 ([[tensorrt-llm]])** — TRT-LLM's AOT engine build is the most extreme version of this idea.
- **Forward to ch-20** — production model reports (Llama 3, DeepSeek V3) include CUDA-graph configuration as part of the recommended serving recipe.

## Further reading

- [[cuda-graphs-inference]] — synthesis card on CUDA graph semantics + serving use.
- [[vllm]] / [[vllm-scheduler]] — piecewise CUDA graph implementation.
- [[tensorrt-llm]] — AOT engine build with internal graph capture.
- [[sglang]] — graph-friendly scheduling on top of RadixAttention.
- [[flashinfer]] — attention engine designed for graph capture.

## Companion visualization

**[figures/cuda-graph-overhead.html](figures/cuda-graph-overhead.html)** — animated comparison of eager-mode decode (800 small launches stacking up CPU overhead) vs CUDA-graph decode (one launch, GPU saturated). Slider for layer count and per-kernel overhead shows when graphs pay off.
