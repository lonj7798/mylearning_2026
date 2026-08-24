<!-- chapter: ch-09
     track: capstone
     kind: content
     title: Capstone: Modeling a 27B MoE Memory Budget End-to-End
     deps: [[ch-01]], [[ch-02]], [[ch-03]], [[ch-04]], [[ch-05]], [[ch-06]], [[ch-07]], [[ch-08]]
     sources: [[zero-memory-optimization]], [[deepspeed-moe-ep]], [[transformer-math-101]], [[megatron-tp-sp]], [[pipeline-parallelism-1f1b]], [[selective-recompute-korthikanti]], [[liger-fused-ce]], [[training-oom-failure-modes]], [[pytorch-fsdp]], [[memory-calculator-notes]], [[ultrascale-playbook]]
-->

# Chapter 9 — Capstone: Modeling a 27B MoE Memory Budget End-to-End

> **Core insight.** Every memory constraint in this course is a consequence of a substrate — the A100's 40 GB HBM ceiling, the NVLink bandwidth that caps TP degree, the all-to-all topology that makes naive ZeRO-3 + MoE catastrophic, the linear-attention contract that hard-bans CP. Working through a single concrete model from the six-item ledger to the final per-GPU verdict makes those substrate forces legible and turns the course into one coherent decision tree.

> **Guideline.** Budget in this order: (1) compute the static model-state floor with the Rule of 16; (2) diagnose the MoE topology constraint (EP required, ZeRO-3 expert gather OOMs); (3) pick the only legal attention kernel given the architecture; (4) size activations + the logit spike; (5) assemble the full parallelism plan; (6) verify per-GPU fit. Every OOM you will ever hit on this hardware maps to one of these six steps.

---

## 1. The Model and the Cluster

**Model spec:**
- 27B total parameters (Ψ = 27 × 10⁹)
- 256-expert MoE; most parameters are in expert FFN blocks (typical split: ~3B dense attention + embedding, ~24B expert weights)
- GDN linear-attention blocks — **hard-assert CP = 1** (context parallelism is illegal for this architecture; see §4)
- Vocabulary size V = 248,000 (large multilingual vocab)
- Training sequence length S = 32,768 tokens (32k)
- Cluster: A100-40GB (not H100 — **fp8 is not available**, Transformer Engine runs only on Hopper+)

**Derived geometry** (assume standard MoE layer sizing):
- Hidden dimension h = 4,096
- Attention heads a = 32, head dim = 128
- Number of layers L = 32 (16 attention + 16 MoE blocks, alternating)
- FFN expansion: each expert has d_ff = 4h = 16,384

---

## 2. Step 1: The Six-Item Ledger and the Rule of 16

The [[ch-01]] taxonomy gives six memory residents. The [[ch-08]] calculator formula is:

```
M_total = (16Ψ/N) + A_layers + L_spike + FSDP_allgather_buffer + CUDA_overhead
```

### The Rule of 16

From [[transformer-math-101]] and [[ultrascale-playbook]], mixed-precision Adam decomposes as:

```
bf16 working weights:     2 bytes/param
bf16 gradients:           2 bytes/param
fp32 master copy:         4 bytes/param
fp32 Adam momentum:       4 bytes/param
fp32 Adam variance:       4 bytes/param
─────────────────────────────────────────
Static floor (model states): 16 bytes/param
```

For Ψ = 27B parameters on N DP ranks with ZeRO-3:

```
Model states = 16 × 27×10⁹ / N  bytes
             = 432 GB / N
```

On a single A100-40GB, this is **432 GB** — 10.8× the card capacity. ZeRO-3 (or FSDP FULL_SHARD) is the minimum viable strategy for model states.

**ZeRO scaling check** (from [[zero-memory-optimization]], 7.5B@Nd=64 as reference):
The ZeRO-3 formula is `16Ψ/Nd`. At Nd=64 DP ranks with a 27B model:
```
16 × 27B / 64 = 6.75 GB  of model states per GPU
```
That fits on a 40 GB card with room for activations and the logit spike.

### LoRA vs Full Fine-Tuning on the Static Floor

From [[memory-calculator-notes]]:

> "Key LoRA insight: activations dominate LoRA memory. 'The main memory consumption during LoRA fine-tuning comes from activation gradients in the frozen weights rather than the LoRA parameters.' LoRA optimizer states are negligible; activation budget is identical to full fine-tune at the same s and b."

| Strategy | Model states/GPU | Activations/GPU |
|----------|-----------------|-----------------|
| Full-FT, ZeRO-3, N ranks | 16Ψ/N | identical (same fwd pass) |
| LoRA, frozen base | ~2Ψ (loaded fully) + tiny adapter | **identical** to full-FT |

This is the [[ch-08]] LoRA node-invariance lesson. Adding DP replicas halves the model-state cost under ZeRO-3, but **every replica runs the same forward pass at the same S and b**, so activation memory per GPU does not decrease. Full-FT OOM caused by model states is cured by adding more nodes (more N); LoRA OOM caused by activations is **not** cured by adding more nodes.

---

## 3. Step 2: Why Naive ZeRO-3 OOMs — Expert Parallelism Required

### The Expert Gather Problem

ZeRO-3 shards all parameters across all N ranks. For a dense model, "all-gather before forward" means reconstructing one transformer layer's weights (~hundreds of MB) — acceptable. For a 256-expert MoE with 24B of expert weights, ZeRO-3's all-gather reconstructs all expert parameters simultaneously on every rank before each MoE layer's forward pass. From [[deepspeed-moe-ep]]:

> "Expert parallelism (EP) shards the expert weight matrices across EP_size GPUs so each rank holds only E/EP_size experts; tokens are routed to the correct rank via a pair of all-to-all collectives (dispatch + combine)."

With naive ZeRO-3 (no EP):
- MoE forward: each rank all-gathers the full 24B of expert weights ≈ **48 GB** in bf16 — exceeds the 40 GB card capacity before the layer even runs.
- Peak is `2 × P_unit` (the FSDP all-gather buffer); for 24B of experts the buffer is ~48 GB. OOM is guaranteed.

### Expert Parallelism Fixes This

With EP_size = 8 (one node of 8 A100s):
- Each rank holds 256/8 = **32 experts**
- Expert weight memory per GPU = 24B / 8 = **3 GB** (well within budget)
- Routing: two all-to-all collectives per MoE layer; buffer size = `tokens × d_model × 2 bytes`:
  ```
  At S=32768, b=1, d_model=4096:
  Buffer = 32768 × 4096 × 2 × 2 bytes ≈ 0.5 GB (two collectives)
  ```
  Affordable. Scales with sequence × batch, so watch this at b > 1.

### EP Requires a Hybrid Parallelism Plan

From [[deepspeed-moe-ep]]:

> "DeepSpeed-MoE hybrid parallelism (Rajbhandari 2022): Combines EP + DP for sparse experts, TP for dense layers, and ZeRO for optimizer states — the first framework to train MoE at trillion-parameter scale."

The non-expert layers (attention, embeddings, layer norms) cannot use EP because they are not sparse. They need TP or are replicated under DP. The practical plan is TP_size within a node for dense layers, EP_size across the same node (or across nodes) for expert layers, and outer DP across node groups.

---

## 4. Step 3: Attention Kernel — What GDN Linear Attention Forces

### The CP Hard-Assert

Context parallelism (Ring Attention) works by rotating K/V blocks around a ring of D devices while each device holds 1/D of the sequence. From [[ring-attention]]:

> "Each device owns a contiguous slice of the sequence of length L/N. In each 'round,' each device (a) computes blockwise attention between its local Q slice and the currently held K/V slice, (b) sends its K/V slice to the next device in the ring."

GDN linear-attention computes attention as a recurrent formulation over the sequence — each state depends on all prior states in order, and the output at position t is a function of a running sum of key-value outer products. This recurrence **cannot be blocked and rotated across a ring** without breaking the sequential dependency. The hard-assert CP=1 is an architectural contract, not a choice.

**Legal kernels for GDN linear attention on A100-40GB:**

| Kernel | Legal for GDN? | Reason |
|--------|---------------|--------|
| Standard O(N²) | Yes | No ring needed; but OOMs at 32k |
| FlashAttention (1/2/3) | No | FA assumes standard softmax attention; not compatible with linear-attention formulation |
| xFormers memory-efficient | No | Same assumption: softmax attention block structure |
| Ring/Context Parallel | No | Hard-assert CP=1 bans this |
| SageAttention | No | Quantized approximation designed for softmax attention |
| PagedAttention | No | Inference-only KV-cache management; not a training kernel |
| **GDN-specific linear-attention kernel** | **Yes** | Model ships its own kernel; memory cost is O(N×d) per layer, not O(N²) |

The GDN kernel's memory cost is O(N×d) per layer — the recurrent state (a d×d matrix or d-dim vector, depending on variant). At d=4096, S=32768:

```
Linear-attention state per layer: 4096² × 2 bytes ≈ 32 MB per layer
Across L=16 attention layers: ~512 MB total
```

Compare to vanilla attention at 32k: 32768² × 2 × 32 heads × 16 layers × 2 (S and P) ≈ **4 TB**. The linear-attention architecture is doing the memory work here; the kernel just needs to implement the recurrence correctly without materializing intermediate O(N²) tensors.

**Key consequence for activation memory**: use the GDN kernel's own memory formula. The Megatron `sbh(10 + 24/t + 5as/ht)` formula (from [[megatron-tp-sp]]) is calibrated for softmax attention and the 5as/h·t term (the N² attention score region) **does not apply** here. Use:

```
A_layer ≈ s × b × h × C_linear  (where C_linear is the kernel-specific constant, typically 10-14)
```

Without the quadratic term, the dominant activation memory at 32k is the MLP / FFN activations, not the attention scores.

---

## 5. Step 4: The Logit Spike — 248k Vocab at 32k Sequence

### The Spike Formula

From [[liger-fused-ce]] and [[memory-calculator-notes]], the logit buffer spike occurs at the forward-backward seam:

```
L_spike = vocab_size × seq_len × batch × dtype_bytes
         = 248,000 × 32,768 × 1 × 2  bytes (bf16)
         = 16,273,408,000 bytes
         ≈ 30.4 GB
```

**This is catastrophic.** On a 40 GB A100, the logit spike alone consumes 76% of the card's memory before model weights, activations, or optimizer states are counted.

From [[training-oom-failure-modes]]:

> "Logit spike identification: OOM at `lm_head` / `output_projection` → the logit buffer `vocab_size × seq_len × batch × 2` bytes is the culprit. Fix: reduce seq_len or batch, not ZeRO stage."

ZeRO-3, TP, and PP do **not** reduce the logit spike — it is computed on the rank that holds the lm_head, from a locally-assembled activation, and it is allocated transiently regardless of sharding strategy.

### The Mandatory Fix: Fused Cross-Entropy

From [[liger-fused-ce]]:

> "Standard cross-entropy loss materializes a full `(B·T × V)` logit tensor before computing the loss — a spike that reaches 1+ GB at modest batch/sequence/vocab sizes — and Liger's fused kernel eliminates it by chunking along the token dimension and fusing the linear projection with the loss computation inside a single Triton kernel, reducing peak activation memory by ~80%."

With Liger `LigerFusedLinearCrossEntropyLoss`, the maximum transient allocation is one chunk:

```
Chunk size (typical CUDA): 65,536 tokens
Max transient = 65536 × 248000 × 2 bytes ≈ 32.5 GB
```

That is still 81% of the card. With a smaller chunk size (e.g., 2048 tokens on this constrained hardware):

```
Chunk = 2048 × 248000 × 2 bytes ≈ 1.0 GB
```

**Fused-CE is not optional for this model.** A 248k vocab at 32k sequence without Liger or an equivalent chunked kernel will OOM on any 40 GB card regardless of all other optimizations.

> **Notice.** The logit spike is the one memory item that TP sharding does **not** solve even when lm_head is tensor-parallel. TP shards the weight matrix of lm_head, but the output logits must be gathered for the softmax (or the fused kernel must account for the shard). Ensure the fused-CE kernel is TP-aware, or TP-shard and fuse within each rank (vocabulary parallelism).

---

## 6. Step 5: Activation Memory and Recomputation

### MLP / FFN Activations (Dominant Term)

For the GDN model, activation memory is dominated by the FFN blocks. With TP degree t:

```
A_FFN_per_layer = s × b × h × (C_mlp / t)   per rank
```

For the MoE FFN: each token is routed to one expert (top-1) or two (top-2). Memory is similar to a dense FFN of size d_ff = 16,384 but gated by the routing — in the worst case, all tokens hit the local experts, and the activation peak is:

```
A_MoE_expert_layer ≈ (tokens_local × d_ff × 2 bytes) per EP rank
                   = (32768 / EP_size) × 16384 × 2 bytes
                   = (32768/8) × 16384 × 2 = 4096 × 16384 × 2 ≈ 0.13 GB
```

With 16 MoE layers: **~2 GB** in expert activations (relatively small because EP distributes the tokens).

### Selective Recomputation

From [[selective-recompute-korthikanti]]:

> "Selective activation recomputation: within each transformer layer, identifies which operations have high memory cost but low recompute cost. Attention score computation (softmax, dropout over s×s matrices) is discarded and recomputed; MLP, LayerNorm, and projection outputs are retained. Memory result: 5× reduction in activation memory consumption. Compute result: Execution time overhead from recomputation reduced by over 90% (from ~30–40% to <4%)."

For GDN linear attention, the "attention score computation" is the recurrent state update — cheaper to recompute than the MLP matmuls. Apply selective recomputation: discard intermediate recurrence states within each attention block, retain MLP activations. The net saving is ~5× on the attention activation portion, at <2% compute overhead.

### Activation Estimate (No Recompute, TP=t, SP enabled)

From [[megatron-tp-sp]], with SP:

```
A_layer (non-linear-attn portion) = (s × b × h / t) × 34  bytes
```

At s=32768, b=1, h=4096, t=4 (TP degree within node):

```
A_layer = (32768 × 1 × 4096 / 4) × 34 = 33554432 × 34 ≈ 1.08 GB per layer
```

Across L=32 layers: **~34.6 GB** without recomputation. With 5× selective recompute: **~7 GB**. This fits comfortably alongside model states.

---

## 7. Step 6: Diagnosing a Representative OOM

### Scenario: Full-FT, ZeRO-3 Only, No EP, Batch=1

Using the [[ch-08]] debugging loop:

**Step 1 — Read the OOM message:**
```
RuntimeError: CUDA out of memory. Tried to allocate 48.00 GiB
(GPU 0; 39.59 GiB total capacity; 6.75 GiB already allocated;
 32.59 GiB free; ...)
Traceback: ... inside MoE forward, expert_linear.weight ...
```

**Step 2 — Estimate:**
The request is 48 GB — that is `2 × P_experts = 2 × 24B params × 2 bytes/param = 96 GB`... wait, that is the FSDP all-gather buffer for the full expert weight block. The single allocation of 48 GB is exactly the expert all-gather tensor: reconstruct all 24B of expert params in bf16 = `24×10⁹ × 2 = 48 GB`.

**Step 3 — Identify the lever:**
From [[training-oom-failure-modes]]:

> "Traceback at `lm_head` / `output_projection` → logit spike. Traceback at `loss.backward()` → activation peak. Traceback at `optimizer.step()` → optimizer-state memory."

Here the traceback is inside MoE forward at the expert weight all-gather. Phase: forward pass, expert layer. Cause: ZeRO-3 all-gather reconstructing all expert weights simultaneously.

**Step 4 — Apply the exact lever:**
The lever is **Expert Parallelism** (EP), not ZeRO stage adjustment. ZeRO-3 cannot be avoided for the remaining parameters, but the expert weights must be carved out of the ZeRO-3 shard and managed under EP. DeepSpeed-MoE and NeMo implement this hybrid natively.

---

## 8. The Final Parallelism Plan and Per-GPU Verdict

### Constraints Recap

| Constraint | Source | Forces |
|------------|--------|--------|
| A100 40 GB ceiling | Cluster | ZeRO-3 mandatory; no headroom |
| No fp8 | Volta/Ampere limitation | bf16 master copy required; 16 bytes/param floor |
| 256-expert MoE | Architecture | EP required; naive ZeRO-3 OOMs |
| GDN CP=1 | Architecture hard-assert | No Ring Attention, no CP dimension |
| V=248k, S=32k | Model spec | Fused-CE mandatory (~30 GB spike unfused) |

### Parallelism Assignment

```
world_size = TP × PP × EP × DP
```

- **TP = 4** (within NVLink node, dense attention/embedding layers; 8× is the usual max but 4 leaves room for EP in the node)
- **PP = 1** (pipeline parallelism adds bubble overhead and complicates the MoE all-to-all routing; prefer flat for this cluster size)
- **EP = 8** (one full A100 node of 8 GPUs holds all 256 experts, 32 per GPU)
- **DP = 8** (8 nodes = 64 GPUs total; outer data parallelism with ZeRO-3 sharding optimizer states across the 8-GPU DP groups within each EP domain)

```
Total GPUs: TP × EP × DP = 4 × 8 × 8 = 256 GPUs (32 nodes of 8 A100-40GB)
```

Adjust: if budget is smaller, use PP=2 to halve weight memory per rank, accepting the bubble penalty at (PP-1)/microbatches ≈ 5% with m=20 microbatches.

### Per-GPU Memory Breakdown (DP=8, TP=4, EP=8, PP=1)

```
Component                                    GB/GPU
─────────────────────────────────────────────────────
Dense model states (ZeRO-3, N=DP×TP=32):  16×3B/32     = 1.5  GB
Expert model states (EP=8, no ZeRO-3):    16×24B/8/...
  ↳ Only 1 expert replica per rank: 24B×2B/8           = 6.0  GB  (bf16 params only)
  ↳ Adam states (fp32) on shard:    24B×12B/8/DP_ep    = 4.5  GB  (if ZeRO-3 over DP)
  ↳ Expert subtotal:                                   ≈ 10.5 GB
Activations (selective recompute, TP=4):                ≈ 7.0  GB
All-to-all MoE buffer (tokens×d×2):                    ≈ 0.5  GB
FSDP AllGather buffer peak (transient):                 ≈ 1.5  GB
Logit spike (fused-CE, chunk=2048):                     ≈ 1.0  GB
CUDA context + fragmentation:                           ≈ 1.0  GB
─────────────────────────────────────────────────────
TOTAL:                                                 ≈ 23.0 GB
```

**Verdict: fits on A100-40GB with ~17 GB headroom.** This allows batch_size=2 (doubles activation and logit-chunk costs, adding ~8 GB) while remaining within the 40 GB ceiling.

> **Notice.** If you run LoRA instead of full-FT here, the optimizer state and gradient columns collapse to near-zero, but the activation column and logit-spike column remain identical. The total drops from ~23 GB to ~18 GB. That 5 GB saving comes from ZeRO-3 model states, not from activations. LoRA does not enable a larger sequence or batch on this hardware.

---

## 9. LoRA vs Full-FT: Why 1→2 Nodes Fixed Full-FT but Not LoRA

This is the concrete test of [[ch-08]]'s node-invariance insight. Suppose the original run is on **1 node (8 GPUs, TP=4, DP=2, EP=4)**:

### Full-FT OOM on 1 Node

```
N_DP = 2 (two DP ranks within the node)
Dense model states = 16 × 3B / 2 = 24 GB    ← OOM
```

Adding a second node doubles N_DP to 4, cutting dense model states to 12 GB — the OOM is resolved. This works because **ZeRO-3 divides model states by N_DP** and DP scaling directly attacks the overflowing term.

### LoRA OOM on 1 Node

Same 1-node config, but LoRA: optimizer states and gradients collapse to ~0.1 GB (adapters only). The OOM is in activations:

```
Activations (selective recompute, TP=4, s=32k, b=1) ≈ 7 GB per rank
+ logit spike (unfused, 248k×32k×1×2) ≈ 30 GB      ← OOM
```

Adding a second node: each DP replica processes the **same tokens** at the **same sequence length** with the **same batch size**. Activations per GPU: unchanged at 7 GB. Logit spike per GPU: unchanged at 30 GB. The extra node added zero DP ranks that help the activation / logit problem — it only helped model states, which were already fine for LoRA.

**Resolution for LoRA OOM:** fused-CE (eliminates the 30 GB spike) + selective recomputation (reduces 7 GB to ~1.5 GB) — no additional nodes needed.

---

## 10. Attention Kernel Selection for This Architecture

**GDN linear attention is its own category.** None of the Chapter 6 kernels apply directly:

- **FlashAttention 1/2/3** ([[ch-05]]): implements online softmax + tiling for standard `softmax(QKᵀ/√d)V`. The GDN recurrence has a different mathematical form; FA cannot substitute.
- **xFormers memory-efficient** ([[ch-06]]): CUTLASS FMHA implementation of Rabe & Staats streaming attention — same softmax assumption.
- **Ring Attention** ([[ch-06]], [[ring-attention]]): CP=1 hard-assert makes this illegal at the architecture level. Even if GDN were compatible, the sequential recurrence dependency prevents sequence sharding.
- **SageAttention** ([[ch-06]]): quantized softmax attention; wrong kernel family.
- **PagedAttention** ([[ch-06]]): inference-only KV-cache management; not applicable.
- **PyTorch SDPA** ([[ch-06]]): dispatches to FA/math backends for softmax attention; does not dispatch to custom linear-attention recurrences.

**Legal choice:** the GDN model's own fused CUDA/Triton kernel for its linear-attention recurrence. The kernel's memory profile is O(N×d) per layer (the recurrent state), not O(N²). This is why 32k sequence is tractable on A100 at all — without the linear-attention architecture, 32k would require Ring Attention, which is banned.

---

## 11. One-Screen Memory Memo for This Model

```
MODEL:   27B MoE (3B dense + 24B experts), 256 experts, GDN linear-attn
CLUSTER: A100-40GB, 32 nodes × 8 GPUs = 256 GPUs
PLAN:    TP=4 · EP=8 · DP=8 · PP=1  (world_size = 256)

FLOOR:   16Ψ/N = 16×27B / 64  ≈ 6.75 GB  (model states, ZeRO-3 DP+TP shard)
         Expert weights         ≈ 6.0 GB   (bf16, EP=8)
         Expert Adam states     ≈ 4.5 GB   (fp32, sharded over DP)

ACTIVATIONS:
         Selective recompute, SP, TP=4, s=32k, b=1  ≈ 7 GB
         (5as²/ht attention term = 0: linear attn has no softmax matrix)

SPIKES:
         MoE all-to-all buffer (2×all-to-all, s=32k, d=4096)  ≈ 0.5 GB
         Logit spike UNFUSED: 248k×32k×1×2  ≈ 30.4 GB  ← KILLS 40GB card
         Logit spike FUSED (Liger, chunk=2k): 248k×2k×2 ≈ 1.0 GB

MANDATORY KNOBS:
  [1] EP=8: without it, ZeRO-3 all-gathers 48 GB of expert weights → OOM
  [2] Fused-CE: without it, logit spike = 30 GB → OOM regardless
  [3] CP=0: architecture bans Ring Attn; use GDN's own kernel (O(Nd) memory)
  [4] No fp8: A100, not H100; stay bf16 mixed precision
  [5] Selective recompute: 5× activation saving at <2% FLOPs

FULL-FT PER-GPU ESTIMATE: ~23 GB of 40 GB (leaves room for b=2)
LORA PER-GPU ESTIMATE:    ~18 GB (saves 5 GB in optimizer states only;
                           activations and logit spike unchanged)

SCALING LAW:
  Adding DP nodes → divides model states (helps full-FT OOM)
  Adding DP nodes → does NOT divide activations or logit spike
  LoRA OOM from activations/logit → fix with fused-CE + recompute, not more nodes
```

---

## Core Insights from the Literature

**From [[zero-memory-optimization]] (Rajbhandari 2020):**
ZeRO-3 achieves `16Ψ/N` bytes of model states per GPU, linear in data-parallel world size. The concrete anchor — DDP=120 GB vs ZeRO-3=1.9 GB at 7.5B and Nd=64 — establishes that the reduction is real and large. But the paper's footnote matters: "activations and logit-buffer spike are not touched by ZeRO — those require separate treatment." ZeRO is one leg of the stool, not the whole stool.

**From [[deepspeed-moe-ep]] (Rajbhandari 2022):**
Expert parallelism is not optional for large MoE models on memory-constrained hardware. The all-to-all communication topology EP introduces (two collectives per MoE layer) is a cost that scales with sequence × batch — designers must budget it separately from the static weight memory. At S=32k and large batches, all-to-all buffers can rival expert weight memory.

**From [[selective-recompute-korthikanti]] (Korthikanti 2022):**
Discarding attention score matrices and recomputing them (cheap: just a matmul + recurrence) vs retaining MLP activations (expensive to recompute: large FFN matmuls) achieves 5× activation savings at <2% compute overhead. For the GDN model, the same asymmetry applies: the recurrent state update is cheap to recompute; the expert FFN matmuls are not. Selective recompute should target the attention blocks.

**From [[liger-fused-ce]]:**
The logit spike is the course's most overlooked killer. At large vocab (≥32k) it is a footnote; at 248k it is the dominant OOM trigger, larger than model weights, optimizer states, and activations combined on a 40 GB card. It is invisible in the static 16-byte ledger because it occurs at the forward-backward seam, is transient, and is unaffected by ZeRO. Fused-CE is not a performance optimization at large vocab — it is a correctness requirement for the run to start.

---

## Key Takeaways

- The **Rule of 16** gives `16Ψ/N` for model states under ZeRO-3. At 27B, even 8 GPUs (N=8) yields 54 GB/GPU — too large. You need N≥64 (8 nodes) or supplemental techniques. This is the base sizing constraint that determines the minimum cluster.
- **EP is mandatory for large MoE**: ZeRO-3's all-gather would reconstruct 48 GB of expert weights per rank per layer on A100-40GB. EP shards experts across ranks and replaces the all-gather with two bounded all-to-all collectives.
- **Fused-CE is mandatory at V=248k, S=32k**: the unfused logit buffer = 30 GB, 76% of total card memory. No parallelism strategy avoids this spike unless the lm_head computation is fused.
- **CP=1 collapses the long-sequence toolkit**: Ring Attention is eliminated by the GDN architecture contract. The only legal long-sequence strategy is the GDN kernel's own O(N·d) recurrence. This is why the model's architecture — not the cluster — determines whether 32k is tractable.
- **LoRA node-invariance**: adding DP nodes does not reduce per-GPU activation memory. LoRA OOMs from activations or logit spikes require kernel-side fixes (fused-CE, selective recompute), not more hardware. Full-FT OOMs from model states are solved by more nodes.
- **The per-GPU verdict (TP=4, EP=8, DP=8, PP=1, 256 GPUs):** ~23 GB for full-FT at b=1 — within A100-40GB with room for b=2.

---

## References

- Rajbhandari, S. et al. "ZeRO: Memory Optimizations Toward Training Trillion Parameter Models." SC 2020. https://arxiv.org/abs/1910.02054 — [[zero-memory-optimization]]
- Rajbhandari, S. et al. "DeepSpeed-MoE: Advancing Mixture-of-Experts Inference and Training to Power Next-Generation AI Scale." ICML 2022. https://proceedings.mlr.press/v162/rajbhandari22a — [[deepspeed-moe-ep]]
- Korthikanti, V. et al. "Reducing Activation Recomputation in Large Transformer Models." arXiv:2205.05198, 2022. https://arxiv.org/abs/2205.05198 — [[selective-recompute-korthikanti]]
- Shoeybi, M. et al. "Megatron-LM." arXiv:1909.08053, 2019. https://arxiv.org/abs/1909.08053 — [[megatron-tp-sp]]
- Huang, Y. et al. "GPipe." NeurIPS 2019. https://arxiv.org/abs/1811.06965; Narayanan et al. SC 2021. https://arxiv.org/abs/2104.04473 — [[pipeline-parallelism-1f1b]]
- Liu, A. et al. "Liger Kernel." arXiv:2410.10989, 2024. https://github.com/linkedin/Liger-Kernel — [[liger-fused-ce]]
- Liu, H. et al. "Ring Attention." ICLR 2024. https://arxiv.org/abs/2310.01889 — [[ring-attention]]
- Anthony, Q. et al. "Transformer Math 101." EleutherAI Blog, 2023. https://blog.eleuther.ai/transformer-math/ — [[transformer-math-101]]
- Penedo, G. et al. "The Ultra-Scale Playbook." HuggingFace, 2025. https://nanotron-ultrascale-playbook.static.hf.space/ — [[ultrascale-playbook]]
- Bekman, S. "ML Engineering." https://github.com/stas00/ml-engineering — [[ml-engineering-memory]], [[training-oom-failure-modes]]
- Zhao, Y. et al. "PyTorch FSDP." VLDB 2023. https://arxiv.org/abs/2304.11277 — [[pytorch-fsdp]]

**All prior chapters:** [[ch-01]] (ledger + Rule of 16), [[ch-02]] (optimizer states + fused-CE), [[ch-03]] (activation checkpointing), [[ch-04]] (O(N²) attention memory), [[ch-05]] (FlashAttention), [[ch-06]] (kernel zoo + Ring Attention), [[ch-07]] (parallelism taxonomy), [[ch-08]] (calculator + OOM loop).
