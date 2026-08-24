# Training OOM Anatomy and the Debugging Loop
<!-- slug: training-oom-failure-modes · type: report · source: https://github.com/stas00/ml-engineering + https://nanotron-ultrascale-playbook.static.hf.space -->

**Core Insight.** Every CUDA OOM has an exact reading: the error message gives you (a) how much was requested, (b) how much is free, and (c) where the traceback points — that locates the *phase*. Identifying phase and dominant consumer before touching any config is the difference between targeted fixing and random tuning.

**Guideline.** Follow the four-step loop: (1) **Read** the OOM message — extract requested bytes, remaining capacity, and traceback phase. (2) **Estimate** peak memory from the formula for that phase. (3) **Smoke-test** at batch=1, seq=128 to confirm model+optimizer fit. (4) **Apply levers** in order of impact: activation checkpointing, ZeRO stage up, batch reduction + gradient accumulation, LoRA, offload.

## Technical Details

**Reading the CUDA OOM message:**
```
RuntimeError: CUDA out of memory. Tried to allocate X GiB
(GPU 0; Y GiB total capacity; Z GiB already allocated;
 W MiB free; PyTorch memory managed V GiB...)
```
- `X GiB requested` — size of the single allocation that failed; this identifies the operation.
- `Z GiB already allocated` — steady-state memory used before the spike.
- `Z + X − Y` = minimum memory reduction needed.
- Traceback tells you the phase: `loss.backward()` → backward phase (activations fully allocated); inside forward → forward phase (activations growing); `optimizer.step()` → optimizer phase (all states live simultaneously).

**Training memory phases and OOM timing:**

| Phase | What is live | Typical OOM cause |
|-------|-------------|-------------------|
| Model load | weights + optimizer states | model too large for N GPUs |
| Forward pass | weights + activations (growing each layer) | long sequence, large batch |
| Forward-backward seam | weights + all activations + logit buffer | vocab logit spike |
| Backward pass | weights + activations (shrinking) + grad buffer | gradient accumulation buffer |
| Optimizer step | weights + grads + optimizer states simultaneously | ZeRO-1/2 insufficient |

**Logit spike identification:** OOM at `lm_head` / `output_projection` / `_get_per_token_logps_and_entropies` → the logit buffer `vocab_size × seq_len × batch × 2` bytes is the culprit. Fix: reduce seq_len or batch, not ZeRO stage.

**The "estimate → smoke → read → lever" loop:**
1. **Estimate** (before any run): compute `16Ψ/N + activations + logit_spike`. If this > GPU capacity, the run cannot start.
2. **Smoke test** (batch=1, seq=128, 1 step): If OOM here, problem is model state memory (weights + optimizer). Apply ZeRO-3 or LoRA.
3. **Read** (profile mid-run with `torch.cuda.memory_summary()` or `torch.cuda.max_memory_allocated()`): pinpoint whether allocation is growing (activation leak) or spiking (logit buffer, all-gather).
4. **Apply levers** in order:

| Lever | Attacks | Cost |
|-------|---------|------|
| Activation checkpointing | activations A | +15–30% compute (recompute fwd) |
| ZeRO-3 / FSDP FULL_SHARD | 16Ψ model states | +1.5× communication vs DDP |
| Reduce micro-batch size + increase grad-accum steps | activations A, logit spike | none on effective batch; lower GPU throughput |
| LoRA (adapter only) | G + O for frozen params | adapter-only gradient signal |
| ZeRO offload (CPU/NVMe) | model states moved to RAM/disk | ×10–100× slower optimizer step |
| Sequence length reduction | activations A (quadratic attn), logit spike | affects context; use packing to maintain throughput |

**Memory fragmentation as a secondary OOM:** PyTorch's caching allocator can hold freed blocks and fail to satisfy a new allocation even when `free_memory > requested`. Diagnostic: `reserved_memory - allocated_memory` is large. Fix: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (PyTorch ≥2.2).

**stas00/ml-engineering guidance on preventing OOM:**
- Use `IPyExperiments` or `torch.cuda.memory_stats()` for line-by-line profiling during development.
- Temporary tensors from softmax and matmul cause "spikes" in the memory timeline — always budget for worst-case peak, not steady-state.
- Gradient accumulation reduces peak activation memory per step (smaller effective micro-batch), but does not reduce model-state memory.

**Training-memory angle:** OOM debugging is not about finding a magic hyperparameter — it is about identifying which of the five memory components (W, G, O, A, L) is overflowing and applying the minimal intervention that resolves the overflow without destroying throughput or training dynamics. The phase label in the traceback is the first filter.

## Citation
Synthesized from: Bekman, S. (2023). ml-engineering: Machine Learning Engineering Open Book. https://github.com/stas00/ml-engineering; HuggingFace Ultra-Scale Playbook https://nanotron-ultrascale-playbook.static.hf.space; MLOps Community "Stop Guessing: CUDA OOM in GRPO Training" https://home.mlops.community/public/blogs/stop-guessing-a-systematic-guide-to-fixing-cuda-out-of-memory-errors-in-grpo-training; HuggingFace Transformers Memory Anatomy https://huggingface.co/docs/transformers/model_memory_anatomy
