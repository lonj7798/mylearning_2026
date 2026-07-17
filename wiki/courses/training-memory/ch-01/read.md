<!-- chapter: ch-01
     track: ledger
     title: The Memory Ledger: What Fills a GPU
     deps: []
     sources: [[transformer-math-101]], [[ultrascale-playbook]], [[ml-engineering-memory]]
-->

# Chapter 1 — The Memory Ledger: What Fills a GPU

> **Core insight.** Training memory is not one thing — it is six named residents that each obey exact byte-per-parameter formulas. The static floor for full fine-tuning with AdamW in mixed precision is 16–18 bytes per parameter *before* a single activation is stored; a 27B model therefore requires 432–486 GB of state that cannot be reduced without changing the optimizer, the precision, or the distribution strategy.

> **Guideline.** Build a six-item ledger before every training run: weights (2 B/param bf16) + gradients (2 B/param, trainable only) + Adam states (12 B/param fp32) + activations (formula below) + loss-head logit spike (B·T·V·2 B, transient) + overhead (CUDA context, NCCL, ZeRO gather buffers, several GiB). This sum tells you whether the job fits; the Rule of 16 (items 1–3 sum to 16 B/param) is the fast first filter.

---

## 1. The Six-Item Ledger

Every training step allocates memory across exactly six distinct residents. Understanding each in isolation is prerequisite to understanding why any strategy (LoRA, ZeRO, activation checkpointing, FP8) moves the needle.

### 1.1 Weights — 2 B/param

In bf16 mixed precision, the *working copy* of every parameter occupies 2 bytes. This is the copy used for all forward and backward matrix multiplications. A 27B model → **54 GB** working weights.

There is also an fp32 *master copy* maintained by the optimizer — covered under item 3. The master copy is not a separate ledger line; it is bundled inside the optimizer state allocation.

### 1.2 Gradients — 2 B/param (trainable parameters only)

Gradients are the same dtype as the working weights: bf16, 2 B/param. Crucially, only *trainable* parameters accumulate gradients.

- **Full fine-tune:** all parameters are trainable → 2 B × N_total. 27B → **54 GB** gradients.
- **LoRA (rank *r*):** only the adapter matrices (two low-rank projections per targeted linear layer) are trainable. For a typical LoRA with rank 16 on Q/V projections, the trainable parameter count is < 0.5% of the full model. Gradient memory ≈ 0 at the model scale.

This is the primary mechanism by which LoRA achieves parameter-efficient training — not just fewer weight bytes, but a near-zero gradient allocation and (by extension) near-zero optimizer-state allocation.

[[ml-engineering-memory]] states the rule precisely: "Mixed precision: 6 B/param (2 B bf16 working + 4 B fp32 master)" for weights, and separately "Gradient bytes: 4 B/param (fp32 or mixed-half); 2 B if non-mixed fp16." In modern practice with bf16 gradients the figure is 2 B/param.

### 1.3 Adam Optimizer States — 12 B/param

AdamW maintains three fp32 tensors per trainable parameter:

| Component | Dtype | B/param |
|---|---|---|
| fp32 master weights | fp32 | 4 |
| First moment (momentum, m) | fp32 | 4 |
| Second moment (variance, v) | fp32 | 4 |
| **Total** | | **12** |

27B model full fine-tune → **324 GB** optimizer states.

The fp32 master copy is mandatory even when training in bf16, because the optimizer update rule (`weight -= lr * m / sqrt(v + eps)`) can produce updates whose magnitude is smaller than bf16's quantization step, causing silent gradient underflow. The fp32 copy absorbs small updates without precision loss; only the resulting fp32 value is then downcast to bf16 for the working copy. (See [[ch-02]] for the full precision story from [[mixed-precision-training]].)

The 12 B/param figure comes from the standard AdamW formulation. [[ml-engineering-memory]] itemizes the alternatives:

- BF16 AdamW: 4 B (quantized moments)
- SGD w/ momentum / LION / Adafactor: 4 B
- 8-bit quantized (bitsandbytes): 2 B

These are the optimizer-side levers that exist *within* the static ledger before any distributed strategy is applied.

### 1.4 Activations — The Dynamic Bucket

Activations are the intermediate tensors produced during the forward pass that must be retained to compute gradients during the backward pass. This is the only non-static resident: it scales with batch size and sequence length.

The canonical formula ([[transformer-math-101]], with tensor-parallel degree *t* = 1):

```
m_act = s · b · h · L · (10 + 24 + 5·a·s/h) bytes
      = s · b · h · L · (34 + 5·a·s/h) bytes
```

where:
- s = sequence length
- b = micro-batch size per GPU
- h = hidden dimension
- L = number of layers
- a = number of attention heads

[[ultrascale-playbook]] gives the cleaner form: `m_act = L · seq · bs · h · (34 + 5·n_heads·seq/h)`.

The term `5·a·s/h · s` = `5·a·s²/h` makes the quadratic dependence on sequence length explicit. The (34 + ...) constant covers projections, MLP, and LayerNorm activations; the `5·a·s/h` term is the attention score matrix (`s×s` per head, `a` heads).

**Concrete example** ([[ml-engineering-memory]]): Llama-3-8B, batch=1, seq=32,768:
- Without checkpointing: ~240 GB
- With full gradient checkpointing: ~31 GB

This 8× gap is entirely the attention quadratic term. At batch=1, sequence-length is the only free variable — and it is quadratic, not linear.

> **Interactive companion:** [figures/memory-ledger.html](figures/memory-ledger.html) — a live calculator that fills in the six-item ledger from (N_params, batch, seq, hidden, layers, heads) and shows how each resident scales as you drag the sliders.

### 1.5 The Loss-Head Logit Spike

At the end of every forward pass, the final linear layer projects hidden states to vocabulary logits before computing cross-entropy. This materializes a tensor of shape `[B·T, V]` in fp32:

```
logit bytes = B × T × V × 4 B
```

For a typical setup (seq = 16,384, vocab = 32,000, BF16 stored as FP32 for CE stability):

```
16,384 × 32,000 × 2 = 1.05 GB   (bf16)
16,384 × 32,000 × 4 = 2.10 GB   (fp32 CE inputs)
```

[[liger-fused-ce]] names this "the largest transient memory event in a standard training step for high-vocab models." It is transient — allocated and immediately freed after the CE reduce — but it is the single largest peak in the memory-time trace and the most common trigger for OOM *after* the static memory has been budgeted correctly.

Mitigation (covered in [[ch-02]]): Liger's fused chunked cross-entropy kernel never materializes the full `B·T×V` tensor. It processes tokens in chunks of ≤ 2,048 on CUDA, capping the spike at `2,048 × 32,000 × 2 = 131 MB` regardless of sequence length. The kernel is numerically exact.

### 1.6 Overhead — The Several GiB That Naive Math Never Models

Even a model with zero parameters would consume GPU memory:

- **PyTorch CUDA init:** "When PyTorch uses CUDA for the first time, it may use up 0.5–2 GB of GPU memory" ([[ml-engineering-memory]]) before any model is loaded.
- **NCCL communication buffers:** Each distributed collective (all-reduce, all-gather) requires staging buffers. In large-cluster training these can consume 1–4 GB.
- **ZeRO all-gather transients:** ZeRO-3 / FSDP shards parameters across ranks and all-gathers them per layer during forward. The unsharded shard occupies GPU memory for the duration of that layer's compute — a transient that can equal one layer's full parameter tensor.
- **Kernel workspace memory:** cuBLAS and cuDNN allocate per-kernel workspaces. Large GEMM tiles at batch=1 can allocate 100–500 MB.

The total overhead budget is typically **2–8 GB** on a 80 GB GPU, which is 2.5–10% of capacity. It is invisible to the parameter-count formula and responsible for the class of "runs out with 5 GB to spare on paper" failures.

[[ultrascale-playbook]] notes a related phenomenon: "The first training step shows different memory patterns than subsequent steps — optimizer states materialize only after step 1; OOM can appear on step 2 even if step 1 succeeds." This is the Adam-state materialization spike: Adam allocates its fp32 tensors only when `.step()` is called for the first time.

---

## 2. The Rule of 16: Static States Before Activations

Items 1–3 (working weights + gradients + Adam states) are allocated before any forward pass executes. Summing them:

```
2 B (weights, bf16)
+ 2 B (gradients, bf16)
+ 12 B (Adam states, fp32 master + m + v)
= 16 B/param
```

This is the **Rule of 16**. It is the minimum memory required on *some* device to train a parameter using full-precision AdamW, before a single activation is stored or a single token is processed.

[[ml-engineering-memory]] derives the same floor with slightly different accounting: "6 B/param (2 B bf16 working + 4 B fp32 master) + 4 B gradients + 8 B AdamW optimizer = 18 B/param." The difference (16 vs 18) is whether gradients are counted as bf16 (2 B) or fp32 (4 B). Modern frameworks that accumulate gradients in fp32 for numerical stability before the optimizer step land at 18 B; bf16-gradient training lands at 16 B. Both numbers are widely cited; the chapter uses 16 B as the baseline but 18 B when citing [[ml-engineering-memory]] directly.

**Worked 27B numbers:**

| Component | Formula | 27B result |
|---|---|---|
| Working weights (bf16) | 2 × 27B | 54 GB |
| Gradients (bf16, full FT) | 2 × 27B | 54 GB |
| Adam optimizer states (fp32) | 12 × 27B | 324 GB |
| **Static total (Rule of 16)** | **16 × 27B** | **432 GB** |
| Activations (varies) | formula §1.4 | + varies |
| Loss-head spike | B·T·V | + varies |
| Overhead | fixed | + 2–8 GB |

432 GB of states alone exceed the capacity of 5 × A100-80GB. This is why a 27B full fine-tune *requires* distributed training across at minimum 6–8 GPUs, and typically 16+.

The Rule of 16 is the fast filter: if `16 × N_params > GPU_memory × GPU_count`, either the distributed strategy must shard the states (ZeRO, FSDP) or the trainable parameter count must drop (LoRA) or the optimizer must compress (FP8, Adafactor, 8-bit Adam).

---

## 3. Full Fine-Tune vs LoRA at the Ledger Level

The ledger difference between full fine-tuning and LoRA is not just weight count — it cascades through three of the six items.

| Item | Full fine-tune | LoRA (rank 16, Q/V only) |
|---|---|---|
| Working weights | 2 B × N_total | 2 B × N_total (full model in memory) |
| Gradients | 2 B × N_total | 2 B × N_LoRA ≈ 0 at 27B scale |
| Adam states | 12 B × N_total | 12 B × N_LoRA ≈ 0 at 27B scale |
| Activations | same | same (same forward pass) |
| Logit spike | same | same |
| Overhead | same | same |

The crucial observation: **LoRA still loads the full model into working-weight memory.** A 27B LoRA run still needs 54 GB just for the frozen model weights. What LoRA eliminates is the gradient and optimizer-state burden: instead of 54 + 54 + 324 = 432 GB of dynamic state, it needs 54 GB (frozen weights) + ~200 MB (adapter gradients + Adam states for the 0.1–0.5% trainable params).

This is why LoRA enables single-GPU fine-tuning of otherwise intractable models: the frozen weights are the floor, not the ceiling. With quantization (QLoRA, 4-bit NF4), the 54 GB frozen weight floor drops further, enabling 27B LoRA on 2 × 24 GB consumer GPUs.

---

## 4. Per-GPU vs Total Accounting

The Rule of 16 gives a *total* state size. Distributed training splits this total across GPUs according to the chosen strategy. The key distinction:

**Replicated (vanilla DDP, ZeRO-0):** every GPU holds the full 16 B × N state. Total across the cluster = 16 B × N × GPU_count. Wasteful but simple.

**Partitioned (ZeRO-1/2/3, FSDP):** states are sharded so each GPU holds 16 B × N / GPU_count of states (ideal), plus the all-gather transients during forward. Total state stays at 16 B × N; per-GPU state drops proportionally.

**Key implication:** When you ask "does this model fit on my GPU?" you need to know *which component* is on *which GPU*. ZeRO-3 / FSDP with 8 GPUs makes a 27B full fine-tune (432 GB static states) per-GPU feasible at 432/8 = 54 GB per GPU, fitting inside 80 GB H100 — before activations and overhead. Adding activations at any non-trivial sequence length pushes it back over; gradient checkpointing or sequence parallelism is then required.

[[ultrascale-playbook]] puts this concretely: "As soon as we reach 7B (!), weights and optimizer requirements already starts to add up significantly and exceed the size of a typical GPU memory, e.g. 80GB for a H100 GPU." The 7B threshold at full precision (7 × 16 = 112 GB) confirms that even a single H100-80GB cannot hold a 7B full fine-tune's static states without ZeRO/FSDP sharding.

The activation and logit-spike budgets remain *per-GPU* regardless of the sharding strategy — each GPU processes its own micro-batch, so activations and the logit spike scale with per-GPU batch size and sequence length, not the global batch. This means activation reduction strategies (gradient checkpointing, selective recomputation) remain necessary even after ZeRO fully solves the static-state problem.

---

## Core Insights from the Literature

**1. The static floor is 16–18 B/param — and it is only the starting point** ([[transformer-math-101]], [[ml-engineering-memory]]). Most engineering intuition underestimates training memory by focusing on model weights (2 B/param) and forgetting that Adam triples the optimizer budget alone (12 B/param). The 8× ratio between weights-only and full-Adam-FT is the number to internalize.

**2. Activations are the variable that blows the budget** ([[ultrascale-playbook]], [[transformer-math-101]]). The static floor is constant once the model and optimizer are chosen; activations scale quadratically in sequence length. At seq=32,768, Llama-3-8B's activations (240 GB without checkpointing) dwarf its static states (8 × 16 = 128 GB). Long-sequence training is an activation problem, not an optimizer-state problem.

**3. The logit spike is a separate, invisible peak** ([[liger-fused-ce]]). It does not appear in the 16 B/param formula, it does not appear in activation estimates, and yet it is the largest single-tensor allocation in a standard training step for high-vocab models. Any memory budget that does not include `B·T·V` bytes has missed the actual peak.

**4. LoRA's memory saving is entirely in items 2 and 3, not item 1** (derived from [[ml-engineering-memory]], [[transformer-math-101]]). The frozen weights remain in GPU memory at full size. LoRA saves gradient and optimizer-state allocation for the N_total − N_LoRA frozen parameters. This has an important corollary: LoRA's memory footprint is bounded from below by the inference footprint of the base model, not by the LoRA rank.

---

## Key Takeaways

- Every training step allocates six named residents. The Rule of 16 (2 + 2 + 12 = 16 B/param) is the fast filter for whether a full fine-tune fits.
- A 27B full fine-tune requires 54 + 54 + 324 = 432 GB of static states — before activations, the logit spike, or overhead. This mandates distributed sharding.
- Activations scale quadratically in sequence length via the `5·a·s²/h` attention term. At long context, activation memory dominates the static floor.
- LoRA zeros out items 2 and 3 for frozen parameters but cannot reduce item 1; the inference-memory floor remains.
- The logit spike (`B·T·V × dtype`) is a transient peak not captured by the 16 B/param formula; it is the most common source of OOM after static budgeting looks correct.
- Overhead (CUDA init, NCCL buffers, ZeRO transients) consumes 2–8 GB independent of model size; always subtract it from usable GPU capacity.
- Adam states materialize at the *end of the first optimizer step*, not at model load — OOM on step 2 is a real failure mode.

---

## References

- Quentin Anthony et al. (EleutherAI), "Transformer Math 101," 2023. https://blog.eleuther.ai/transformer-math/
- Guilherme Penedo et al. (HuggingFace / nanotron team), "The Ultra-Scale Playbook: Training LLMs on GPU Clusters," 2025. https://nanotron-ultrascale-playbook.static.hf.space/
- Stas Bekman, "Machine Learning Engineering Open Book," stas00/ml-engineering, ongoing. https://github.com/stas00/ml-engineering
- Tianqi Chen et al., "Training Deep Nets with Sublinear Memory Cost," arXiv:1604.06174, 2016. https://arxiv.org/abs/1604.06174 (gradient checkpointing, [[ch-03]])
- Austin Liu et al., "Liger Kernel: Efficient Triton Kernels for LLM Training," arXiv:2410.10989, 2024. https://github.com/linkedin/Liger-Kernel (logit spike, [[ch-02]])
