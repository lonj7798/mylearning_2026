<!-- chapter: ch-08
     track: scaling
     title: Memory Formulas, the Calculator, and the OOM Debugging Loop
     deps: [[ch-01]], [[ch-02]], [[ch-03]], [[ch-07]]
     sources: [[memory-calculator-notes]], [[training-oom-failure-modes]]
-->

# Chapter 8 — Memory Formulas, the Calculator, and the OOM Debugging Loop

> **Core insight.** Peak GPU memory during training has five additive components — weights (W), gradients (G), optimizer states (O), activations (A), and the logit-buffer spike (L) — and they do not all peak simultaneously. ZeRO-3 divides W+G+O by N (world size); activations and the logit spike are *independent of N* and must be budgeted separately. Every OOM has an exact phase label — forward, backward, or optimizer-step — and the traceback tells you which component just overflowed.

> **Guideline.** Before any training run, compute `M = 16Ψ/N + A_per_layer × L_layers + L_spike + overhead` and compare to GPU capacity with a ≥ 2 GiB safety margin. When the run OOMs, read the CUDA error message to extract the three numbers (requested, free, reserved), identify the phase from the traceback, then apply the minimal lever in order: nodes (for full-FT model states), TP (for activations and MoE), fused-CE (for the logit spike), batch/seq reduction, gradient checkpointing, offload last.

---

## 1. Assembling the Full-FT Per-GPU Formula (ZeRO-3)

Full fine-tuning with ZeRO-3 (or FSDP FULL_SHARD, which is ZeRO-3 under a different name) distributes all three model-state terms across N DP ranks. The per-GPU ledger is:

| Component | Per-GPU formula | Dtype | Notes |
|-----------|----------------|-------|-------|
| Weights | `2Ψ / N` | bf16 | AllGathered transiently; shard held at rest |
| Gradients | `2Ψ / N` | bf16 | ReduceScattered into local shard |
| Adam optimizer | `12Ψ / N` | fp32 | fp32 master `4Ψ` + momentum `4Ψ` + variance `4Ψ` |
| **Model-state subtotal** | **`16Ψ / N`** | | Rule of 16, sharded |
| AllGather buffer (FSDP peak) | `≈ 2 × P_unit` | bf16 | Transient during forward/backward of largest FSDP unit |
| Activations (no checkpointing) | `(sbh/t)(34 + 5as/h) × L` | mixed | s=seq, b=batch, h=hidden, a=heads, t=TP; per layer, times L layers |
| Activations (with checkpointing) | `~2sbh × L` bytes | bf16 | One tensor per layer boundary — the simplified practitioner estimate |
| **Logit-buffer spike** | `vocab × s × b × 2` bytes | bf16 | Materializes at forward-backward seam; NOT reduced by ZeRO, TP, or PP |
| Overhead | `~1–2 GiB` | — | CUDA context, NCCL, allocator fragmentation |

The **baseline mixed-precision Adam identity** that drives the 16Ψ floor ([[zero-memory-optimization]]):

```
2Ψ (fp16/bf16 weights)
+ 2Ψ (fp16/bf16 gradients)
+ 4Ψ (fp32 master params)
+ 4Ψ (fp32 momentum)
+ 4Ψ (fp32 variance)
= 16Ψ bytes per parameter, unreduced
```

ZeRO-3 divides every one of those five terms by N, yielding `16Ψ/N`. The communication tax for this saving is **1.5× vs DDP**: an all-gather before forward (Ψ), reduce-scatter during backward (Ψ), and a second all-gather before backward (Ψ) — total 3Ψ vs DDP's all-reduce of 2Ψ ([[pytorch-fsdp]]).

**Concrete numbers for a 7.5B model at N=64 DP ranks** ([[zero-memory-optimization]]):

| Strategy | Per-GPU model states |
|----------|---------------------|
| DDP (no sharding) | 120 GB |
| ZeRO-1 (optimizer sharded only) | ~31 GB |
| ZeRO-2 (optimizer + grad sharded) | ~17 GB |
| ZeRO-3 (all sharded) | ~1.9 GB |

This is what "ZeRO has the potential to scale beyond 1 Trillion parameters using today's hardware" (Rajbhandari et al., SC 2020) means concretely: the 16Ψ floor becomes 16Ψ/N, making model-state memory a nearly-free resource as you scale N.

**Worked example: 7B model, N=8, s=2048, b=1, h=4096, a=32, t=1, vocab=32K** ([[memory-calculator-notes]]):
- Model states: `16 × 7×10⁹ / 8 ≈ 14 GB`
- Activations (no checkpointing, 32 layers): `(2048 × 1 × 4096 / 1) × (34 + 5 × 32 × 2048 / 4096) × 32 layers ≈ 16 GB`
- Logit spike: `32000 × 2048 × 1 × 2 ≈ 0.13 GB`
- AllGather buffer: `≈ 2 × (7B/32 layers × 2) ≈ 0.44 GB`
- **Estimated total: ~31 GB** — fits on 2× A100-40GB with headroom; tight on one GPU without checkpointing

---

## 2. The Logit-Buffer Spike and Why It Is World-Invariant

The logit spike is a single allocation at the **forward-backward seam**: every forward pass must compute `output_projection(hidden) → logits` of shape `[batch, seq, vocab_size]` in bf16 before the loss can be computed. Its size:

```
vocab_size × seq_len × batch_size × 2 bytes
```

This allocation is:
- **NOT reduced by ZeRO** — ZeRO shards parameter memory, not activations or computation buffers
- **NOT reduced by PP** — every pipeline stage that owns the output layer allocates it fully
- **NOT reduced by DP/FSDP** — each DP replica computes its own logits over its own micro-batch

At modest vocab (32K) and seq (2048), it is only 0.13 GB — negligible. At large vocab (128K+) or large batch/seq, it becomes the **dominant OOM trigger**. For seq=16,384 and vocab=32K: `16384 × 32000 × 2 = 1.05 GB` ([[liger-fused-ce]]). At seq=16K and vocab=128K: `16384 × 131072 × 2 ≈ 4.3 GB` — a single allocation that fires at exactly one point per step.

**Identification signature**: OOM traceback pointing to `lm_head`, `output_projection`, or `_get_per_token_logps_and_entropies` is the logit spike ([[training-oom-failure-modes]]). The fix is **not** more nodes or a higher ZeRO stage — it is fused cross-entropy (Liger [[liger-fused-ce]]) which chunks along the token dimension so the largest temporary allocation is `chunk_size × vocab_size × 2` instead of the full product, or reducing seq/batch.

> **Interactive companion:** [figures/memory-calculator.html](figures/memory-calculator.html) — enter model size, world config (N, TP, PP), and sequence/batch; the calculator outputs all five ledger components side-by-side, highlights the dominant term, and shows how each lever moves the needle.

---

## 3. LoRA Memory: Node-Invariant Activations

LoRA freezes the base model and trains only low-rank adapter matrices of rank r. The optimizer-state and gradient savings are genuine, but the activation budget is **identical** to full fine-tuning at the same sequence length and batch size, and crucially it is **world-invariant**:

| Component | Full-FT (ZeRO-3) | LoRA |
|-----------|-----------------|------|
| Frozen/working weights | `2Ψ / N` | `2Ψ` (all replicas hold the full base) |
| Adapter weights | — | `≈ 2rΨ_target / d` (tiny) |
| Gradients | `2Ψ / N` | only adapter: `≈ 2r × Ψ_target / d` |
| Optimizer states | `12Ψ / N` | only adapter: `≈ 12r × Ψ_target / d` |
| **Activations** | `(sbh/t)(34 + 5as/h) × L` | **same** |
| Logit spike | world-invariant | world-invariant |

The reason activations are unchanged: the frozen base model still performs a full forward pass, and that forward pass must cache activations for the adapter backward. The adapter only injects into a few weight matrices; it does not reduce the number of tensors cached during the forward.

**The world-invariant property** ([[memory-calculator-notes]]): Adding more DP ranks (more nodes, larger world size) helps full-FT because it divides the `16Ψ` model-state term by N. For LoRA, the `16Ψ` term is already tiny (only adapter states, negligible); the dominant budget is activations, which are **not divided by N**. Each replica processes its own tokens — they all hold the same activation peak. Adding nodes to fix a LoRA OOM therefore does nothing useful for the actual overflow component. The correct lever for a LoRA activation OOM is:
1. Reduce batch or sequence length
2. Enable gradient/activation checkpointing
3. Increase TP degree t (which divides the activation formula by t)

This is the key distinction between full-FT and LoRA OOM behavior: **full-FT OOMs scale away with more nodes; LoRA OOMs do not**.

---

## 4. Activation Memory Deep Dive

### 4.1 The Megatron activation formula

With tensor parallelism degree t and sequence parallelism enabled ([[megatron-tp-sp]]):

```
Per-layer activation memory (TP only, no SP):
  sbh(10 + 24/t + 5as/ht)  bytes

Per-layer activation memory (TP + SP):
  (sbh/t)(34 + 5as/h)  bytes
```

The `10sbh` term in the TP-only formula is the layer-norm and dropout activations, which TP alone leaves **replicated** across all t ranks. Sequence parallelism shards these along the sequence dimension, converting `10sbh` into `10sbh/t` — a true t× reduction across the entire layer. The net effect: SP+TP gives a genuine t× reduction in total activation memory over TP alone, with no extra communication bandwidth cost (all-reduce is replaced by AllGather + ReduceScatter, same volume).

### 4.2 Activation checkpointing: the practitioner's formula

Full activation checkpointing (Chen 2016 ([[gradient-checkpointing-chen]])) stores only one activation tensor per layer boundary and recomputes all intra-layer activations during the backward pass. The memory drops from O(L × sbh × factor) to approximately:

```
~2sbh × L  bytes  (simplified practitioner estimate with checkpointing)
```

The compute overhead is **~33%** (one extra forward pass per mini-batch), not 2×. The `sqrt(n)` scheme from Chen 2016 applies more precisely when n is the number of layers: checkpoint every sqrt(L) layers, each segment recomputes locally. PyTorch's `torch.utils.checkpoint.checkpoint()` implements this.

### 4.3 Selective recomputation (Korthikanti 2022)

Full recomputation costs 30–40% extra FLOPs. The insight in [[selective-recompute-korthikanti]] is asymmetric: not all activations are equal.

- **Attention score/softmax matrices** (`s × s` or `s²` terms): large because of quadratic seq growth, but cheap to recompute (matmul + softmax, and FlashAttention already recomputes them in its backward anyway)
- **MLP activations**: smaller but expensive to recompute (large FFN matmuls)

Selective recomputation discards only the attention score region — the `5as²b/ht` term — and retains MLP, LayerNorm, and projection outputs. Result:

- **~5× activation memory reduction**
- **<2% additional FLOPs** (vs 30–40% for full recompute)
- At 530B scale on 2240 A100s: MFU improved from 42.1% (full recompute) to 54.2% (selective recompute + SP) — a 29% throughput improvement

The `70% activation memory reduction at 2.7% compute cost` figure from the Ultra-Scale Playbook ([[ultrascale-playbook]]) refers to this selective scheme applied to GPT-3 175B.

---

## 5. The Allocator Peak at Phase Boundaries

One of the most common OOM surprises: **step 1 succeeds, step 2 OOMs**. This is because the PyTorch CUDA caching allocator has a time structure to memory:

| Phase | What materializes |
|-------|-------------------|
| Model load | Weights (`2Ψ/N`) |
| Step 1 forward | Activations grow layer by layer |
| Step 1 fwd-bwd seam | Logit spike appears, all activations cached simultaneously |
| Step 1 backward | Activations shrink as gradients flow back |
| **End of step 1** | **Adam optimizer states materialize for the first time** (`12Ψ/N`) |
| Step 2 forward | Now weights + full optimizer states + new activations must coexist |

[[ultrascale-playbook]] states this explicitly: "The first training step shows different memory patterns than subsequent steps — optimizer states materialize only after step 1; OOM can appear on step 2 even if step 1 succeeds." The safe practice is to **always profile the peak at step 2**, not step 1.

A secondary cause is **allocator fragmentation**: PyTorch's caching allocator holds freed blocks and can fail to satisfy a new allocation even when `free_memory > requested`. Diagnostic: `reserved_memory - allocated_memory` is large. Fix: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (PyTorch ≥ 2.2) ([[training-oom-failure-modes]]).

The FSDP per-step lifecycle makes this concrete ([[pytorch-fsdp]]): for each forward/backward pass over the largest FSDP unit, an **AllGather buffer** of `≈ 2 × P_unit` bytes is transiently allocated, held for the duration of that unit's compute, then freed. This buffer is the "peak excursion" above the `16Ψ/N` steady state — it must be budgeted explicitly:

```
Peak FSDP memory ≈ 16Ψ/N + 2 × P_largest_unit + activations + logit_spike + overhead
```

---

## 6. The OOM Debugging Loop

Every CUDA OOM produces a message in the form ([[training-oom-failure-modes]]):

```
RuntimeError: CUDA out of memory. Tried to allocate X GiB
(GPU 0; Y GiB total capacity; Z GiB already allocated;
 W MiB free; PyTorch memory managed V GiB...)
```

Extract three numbers: X (requested), Z (already allocated), Y (total). The minimum memory reduction needed is `Z + X − Y`. The traceback locates the **phase**:

| Traceback location | Phase | Dominant consumer | Correct lever |
|-------------------|-------|-------------------|---------------|
| `lm_head` / `output_projection` | Fwd-bwd seam | Logit spike | Fused-CE (Liger), reduce vocab or seq |
| `loss.backward()` or layer backward | Backward pass | All activations cached | Checkpointing, reduce seq/batch |
| `optimizer.step()` | Optimizer phase | Adam states fully live | ZeRO-3 / FSDP, LoRA |
| Inside forward, growing | Forward pass | Activations growing | Reduce seq/batch, TP, checkpointing |
| Model load | Load | Weights + states | ZeRO-3 is essential |

### The six-step debugging loop

```
1. ESTIMATE before the run:
   M = 16Ψ/N + activations(s, b, t) + logit_spike(vocab, s, b) + overhead
   If M > GPU_capacity − 2 GiB safety margin → do not run, adjust first.

2. SMOKE-RUN (batch=1, seq=128, 5 steps):
   If OOM here → model-state memory (weights + optimizer) overflows.
   Apply ZeRO-3 (more nodes) or LoRA.
   If passes → model-state budget is OK; proceed.

3. READ the OOM message:
   Extract X (requested), Z (allocated), Y (total).
   Read traceback → identify phase (fwd / bwd-seam / bwd / opt-step).

4. IDENTIFY the dominant component from phase label (table above).

5. APPLY the lever ordered by impact and cost:
   a. More nodes (ZeRO-3 N↑) — for full-FT model states ONLY
   b. Tensor parallelism (t↑) — for activations and MoE weights
   c. Fused-CE — for logit spike, zero cost
   d. Reduce batch or sequence — for activations + logit spike
   e. Gradient/activation checkpointing — for activations, +15–40% compute
   f. LoRA — if switching from full-FT is acceptable (changes gradient signal)
   g. Offload (CPU/NVMe) — LAST RESORT: 10–100× slower optimizer step

6. FEED back into the calculator:
   After applying a lever, recompute M with the new config.
   Use the measured Z from the next smoke-run as a ground-truth oracle
   to calibrate the formula (discover what the formula missed).
```

**Why offload is last resort**: CPU offload moves optimizer states to RAM (or NVMe), which eliminates the `12Ψ/N` GPU allocation. But every optimizer step now requires a PCIe transfer of `12Ψ/N` bytes in each direction. At Ψ=7B and N=8, that is ~21 GB of PCIe traffic per step at ~32 GB/s → about 0.65 seconds of PCIe transfer per step, vs. <1 ms for an on-GPU optimizer step. The throughput destruction is 100–1000× for large models ([[training-oom-failure-modes]]).

---

## 7. Batch Semantics and MFU

### 7.1 Batch accounting

The **effective batch size** seen by the optimizer at every weight update is:

```
global_batch_size = per_device_batch × grad_accum_steps × DP_replicas
```

Gradient accumulation accumulates gradients over `grad_accum_steps` micro-batches before an optimizer step. Memory impact: the per-step *peak* memory uses `per_device_batch` (one micro-batch at a time), so grad accumulation is a near-free way to increase effective batch without increasing peak activation memory. It does not reduce model-state memory.

Steps per epoch:

```
steps_per_epoch = dataset_tokens / (global_batch_size × seq_len)
```

### 7.2 Model FLOPs Utilization

The **6PD FLOPs/token rule** ([[ultrascale-playbook]], [[transformer-math-101]]):

```
FLOPs per training token ≈ 6 × Ψ
```

(2 for forward matmuls + 2 for backward + 2 for gradient accumulation through weights, all approximate; activation checkpointing adds ~1/3 for recompute, yielding ~8× Ψ in the checkpointed case)

MFU:

```
MFU = (achieved_FLOPs_per_second) / (peak_hardware_FLOPs_per_second)
    = (6Ψ × tokens_per_second) / (GPU_TFLOPS × N_GPUs × 10¹²)
```

A healthy MFU for a well-tuned run on A100s is 40–55%. Below 30%, suspect communication saturation, activation checkpointing overhead, or data-loading bottleneck. The selective-recompute + SP combination on 530B reached 54.2% MFU ([[selective-recompute-korthikanti]]).

---

## 8. Parallelism Levers and What Each Actually Fixes

| Lever | What it divides | Does NOT help |
|-------|----------------|---------------|
| More DP nodes (ZeRO-3) | Model states: `16Ψ → 16Ψ/N` | Activations, logit spike |
| Tensor Parallelism (TP=t) | Activations (with SP: full t×), weight matmuls | Model states by themselves |
| Pipeline Parallelism (PP=p) | Weight memory (each rank holds `Ψ/p` layers) | Per-layer activations; adds bubble |
| Gradient checkpointing | Activations: `A_layers → ~2sbh × L` | Model states, logit spike |
| Fused cross-entropy | Logit spike: eliminates materialization | Anything else |
| LoRA | Gradients + optimizer for frozen params | Activations (world-invariant!) |
| Seq length reduction | Activations (quadratic), logit spike | Model states |
| Offload (CPU/NVMe) | Model states (at extreme throughput cost) | Activations, logit spike |

**Pipeline parallelism bubble**: GPipe and 1F1B share the same bubble fraction `(p-1)/m` (where m = microbatches), but 1F1B caps activation memory at p microbatches in flight vs m for GPipe ([[pipeline-parallelism-1f1b]]). The activation advantage is 1F1B's sole reason to prefer it over GPipe — bubble time is identical.

**Expert parallelism (EP) for MoE**: each GPU holds `E/EP_size` experts. Routing requires two all-to-all collectives per MoE layer (dispatch + combine). The all-to-all buffer size = `tokens × d_model × 2` bytes per rank, scaling with sequence length and batch — a transient peak that must be budgeted on top of the static expert weight allocation ([[deepspeed-moe-ep]]).

---

## Core Insights from the Literature

1. **Five components, not one** ([[memory-calculator-notes]]): Peak GPU memory is the sum of five terms (W, G, O, A, L) that peak at different times during the step. The optimizer states (O) only materialize at the end of step 1. A smoke-run that completes step 1 is not proof that the run will sustain — always validate through step 2.

2. **The logit spike is structurally outside ZeRO** ([[training-oom-failure-modes]], [[liger-fused-ce]]): `vocab × seq × batch × 2` bytes fires at the forward-backward seam and is invisible to any data-parallel or model-state strategy. At 128K+ vocab it becomes the limiting factor before model states. Fused cross-entropy (Liger kernel) is the targeted fix: it chunks the projection + loss into a token-wise loop and never materializes the full tensor, reducing the peak from `B·T·V × 2` to `chunk_size × V × 2`.

3. **LoRA activation invariance** ([[memory-calculator-notes]]): The conventional wisdom that "LoRA uses less memory" is true for optimizer states but false for activations. The frozen base model caches the same activations as full fine-tuning because the adapter backward must flow gradients through the frozen forward pass. Adding DP replicas is the wrong lever for a LoRA OOM; TP and checkpointing are correct.

4. **Selective recompute is Pareto-optimal** ([[selective-recompute-korthikanti]]): The 30–40% compute overhead of full activation checkpointing is not necessary to achieve 5× activation memory reduction. Discarding only the attention score matrices (quadratic in s, cheap to recompute) and retaining MLP activations achieves the same 5× memory saving at <2% compute cost — a 15–20× better compute-per-byte-saved ratio than full recompute.

---

## Key Takeaways

- **The per-GPU formula for full-FT ZeRO-3 is `16Ψ/N + A + L_spike + overhead`**. Model states scale away with N; activations and the logit spike do not.
- **LoRA memory is node-invariant at the activation level**. More nodes fix full-FT OOMs; they do not fix LoRA OOMs. Only TP or seq/batch reduction help.
- **The step-2 trap**: optimizer states materialize at the end of step 1. Always profile through step 2 or the smoke-run will produce a false negative.
- **The OOM traceback is the first diagnostic tool**: `lm_head` → logit spike; `loss.backward()` → activations; `optimizer.step()` → model states.
- **Lever order**: nodes → TP → fused-CE → batch/seq → checkpointing → LoRA → offload (last, 10–100× step-time cost).
- **MFU = 6Ψ × tok/s / (GPU_TFLOPS × N × 10¹²)**. Selective recompute + SP pushes MFU from 42% to 54% at 530B scale.
- **Offload is a last resort** — PCIe bandwidth (~32 GB/s) is 100–1000× slower than HBM (~2 TB/s), making CPU offload a throughput catastrophe for any model where the alternative exists.

---

## References

- Rajbhandari, S. et al. "ZeRO: Memory Optimizations Toward Training Trillion Parameter Models." SC 2020. https://arxiv.org/abs/1910.02054 — [[zero-memory-optimization]]
- Zhao, Y. et al. "PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel." VLDB 2023. https://arxiv.org/abs/2304.11277 — [[pytorch-fsdp]]
- Korthikanti, V. et al. "Reducing Activation Recomputation in Large Transformer Models." MLSys 2023. https://arxiv.org/abs/2205.05198 — [[selective-recompute-korthikanti]], [[megatron-tp-sp]]
- Chen, T. et al. "Training Deep Nets with Sublinear Memory Cost." arXiv:1604.06174, 2016. https://arxiv.org/abs/1604.06174 — [[gradient-checkpointing-chen]]
- Liu, A. et al. "Liger Kernel: Efficient Triton Kernels for LLM Training." arXiv:2410.10989, 2024. https://github.com/linkedin/Liger-Kernel — [[liger-fused-ce]]
- Huang, Y. et al. "GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism." NeurIPS 2019. https://arxiv.org/abs/1811.06965 — [[pipeline-parallelism-1f1b]]
- Rajbhandari, S. et al. "DeepSpeed-MoE." ICML 2022. https://proceedings.mlr.press/v162/rajbhandari22a — [[deepspeed-moe-ep]]
- Penedo, G. et al. "The Ultra-Scale Playbook." HuggingFace, 2025. https://nanotron-ultrascale-playbook.static.hf.space — [[ultrascale-playbook]]
- Bekman, S. "ML Engineering Open Book." https://github.com/stas00/ml-engineering — [[ml-engineering-memory]], [[training-oom-failure-modes]]
- Synthesized formulas and worked examples: [[memory-calculator-notes]]

---

*Next: [[ch-09]] — Capstone: Modeling a 27B MoE Memory Budget End-to-End. You will use the formula from this chapter as the primary worksheet.*
