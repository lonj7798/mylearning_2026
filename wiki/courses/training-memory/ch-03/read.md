<!-- chapter: ch-03
     track: ledger
     phase: read
     title: Activations and Gradient Checkpointing
     deps: [[ch-01]], [[ch-02]]
     sources: [[gradient-checkpointing-chen]], [[selective-recompute-korthikanti]]
     created_at: 2026-07-16
-->

# Chapter 3 — Activations and Gradient Checkpointing

> **Core insight.** The forward pass of a Transformer must stash every intermediate tensor that the backward pass will need to multiply through the chain rule — and those tensors scale as O(L·s·b·h) in the static terms plus O(L·a·s²·b/h) in the attention term, making activation memory the dominant, sequence-length-sensitive item in the training ledger. Gradient checkpointing (Chen 2016) converts that O(L) footprint to O(√L) by storing only √L boundary activations and recomputing everything in between during backward — at the cost of exactly one extra forward pass (~33% more compute). Selective recomputation (Korthikanti 2022) refines the tradeoff: keep cheap-to-store activations (MLP, LayerNorm), discard only the large cheap-to-recompute ones (attention score matrices), and recover 5× the memory reduction at <4% recompute overhead.

> **Guideline.** Activations are the one memory ledger item where you buy memory back with compute, not bandwidth. Enable full gradient checkpointing as a baseline to stay within budget; then upgrade to selective recomputation (Megatron's default) whenever throughput matters, gaining most of the memory saving for a tiny compute tax. Pair with sequence parallelism to eliminate the residual activation duplication from tensor parallelism.

---

## 1. What the Backward Pass Must Keep

Every parameter in a neural network is updated by a gradient computed via the chain rule. The chain rule for a composition f(g(x)) requires the value g(x) — not just the final output f — to be available when computing ∂f/∂g. For a Transformer layer with multiple non-linear operations (attention softmax, GELU activations, layer normalization), this means the forward pass must stash the intermediate tensors at each operation boundary so the backward pass can use them as multipliers.

In a standard (no-checkpointing) training run, this produces a memory footprint proportional to the number of layers times the per-layer activation volume:

```
Total activation memory (no recompute) = L * per_layer_activation_bytes
```

For a Transformer with tensor-parallel degree *t*, the per-layer term is ([[transformer-math-101]], [[selective-recompute-korthikanti]]):

```
per_layer_bytes = s * b * h * (10 + 24/t + 5*a*s / (h*t))
```

where:
- `s` = sequence length (tokens)
- `b` = micro-batch size per GPU
- `h` = hidden dimension
- `L` = number of layers
- `a` = number of attention heads
- `t` = tensor-parallel degree (1 = no TP)

The formula has two structurally different terms:

**Linear term** `(10 + 24/t)·s·b·h`: stores MLP activations, LayerNorm outputs, residual connections. Grows linearly with s and b.

**Quadratic term** `5·a·s²·b / (h·t)`: stores the attention score matrix — an `[b, a, s, s]` tensor after softmax and dropout. Grows quadratically with s because every token attends to every other token.

This quadratic attention term is what makes long-context training expensive. At s=8,192 the attention term dominates over the linear term for any realistic h. At s=32,768 it is overwhelming. The formula makes this exact: even with TP=8 (dividing by t inside the attention term), the s² scaling is inescapable without algorithmic help (FlashAttention, covered in [[ch-04]] and [[ch-05]]).

### Activation memory versus the static floor

[[ch-01]] established the static memory floor: for a model with P parameters in mixed-precision AdamW, the floor is 18 bytes/param:
- 2 B: BF16 working weights
- 4 B: FP32 master weights
- 4 B: FP32 gradients
- 8 B: FP32 Adam momentum + variance

Activations sit entirely on top of this floor. For any real sequence length and batch size, activations dominate. As a rough guide: a 7B-parameter model with h=4096, L=32, a=32, s=4096, b=1, t=1 produces activation memory of approximately:

```
32 * 4096 * 1 * 4096 * (10 + 24 + 5*32*4096/4096) = 32 * 4096 * 4096 * (34 + 160)
  ≈ 32 * 4096 * 4096 * 194 ≈ 104 GB
```

Against a static floor of ~7B * 18 = 126 GB, activations at s=4096 already rival the static floor — and with s=16384 the attention term multiplies by 16×.

---

## 2. Gradient Checkpointing — O(√n) Memory for +33% Compute

[[gradient-checkpointing-chen]] (Chen et al., 2016) is the foundational algorithm for trading activation memory for recompute. The key observation: it is not necessary to store *all* intermediate activations. You can store a strategic subset — **checkpoints** — and recompute the in-between activations from scratch during the backward pass, at the moment they are needed.

> **▶ Interactive companion — [`figures/checkpointing.html`](figures/checkpointing.html)**
> *What a checkpoint physically is, and what happens to everything else.* Panel 1 expands one transformer block into its individual saved tensors with real shapes and byte-proportional bars — the checkpoint is **one hidden-state tensor `[B, T, h]` = 33.55 MB**, while the ~1.64 GB of interior tensors it lets you discard includes the `[B, heads, T, T]` attention-probability monster at **1.07 GB** (a 49× ratio). Panel 2 animates forward-store / backward-recompute step by step with a live memory gauge that never exceeds `2√n`. Panel 3 lets you drag `k` across the `k + n/k` curve to see both edges cost `n` and only `√n` minimizes it. Panel 4 shows why "per-block checkpointing" is *not* `k = n`, and Panel 5 maps it onto `torch.utils.checkpoint`'s actual `no_grad`-forward / re-forward mechanics.

### The √n scheme

Divide an n-layer network into segments of size √n. Store exactly one activation tensor per segment boundary (√n tensors total). When the backward pass reaches a segment, recompute that segment's internal activations forward from its stored boundary, then use them for the local gradient computation:

```
Forward pass (store phase):
  For i = 0, sqrt(n), 2*sqrt(n), ..., n:
    x[i] = forward_segment(x[i - sqrt(n)])   # recompute the segment
    checkpoint[i] = x[i]                     # store only the boundary

Backward pass (recompute phase):
  For each segment s (in reverse):
    recompute all activations in s from checkpoint[s.start]
    compute gradients using recomputed activations
    free the recomputed activations
```

Memory cost: O(√n) stored checkpoints + O(√n) activations recomputed in one live segment at a time.

Compute cost: "only the computational cost of an extra forward pass per mini-batch" ([[gradient-checkpointing-chen]]). Not one extra forward pass per layer — one extra forward pass total. The backward pass is computing gradients anyway; the overhead is the segment recomputation, and since there are √n segments each of size √n, the total extra work is n operations — one forward pass. Hence the 33% overhead (forward pass is ~1/3 of total forward+backward FLOPs for a typical Transformer).

### Empirical result

[[gradient-checkpointing-chen]] validated on a 1,000-layer ResNet on ImageNet:
- **Without checkpointing:** 48 GB activation memory
- **With √n checkpointing:** 7 GB activation memory (6.8× reduction)
- **Runtime overhead:** 30% (not 2×, not 100%)

The extreme variant — recursive checkpointing — achieves O(log n) memory at O(n log n) compute cost by applying the scheme recursively. Rarely used in practice because the 33% overhead of √n is already acceptable and the logarithmic scheme complicates implementation.

### Per-layer checkpointing in practice

Modern frameworks simplify the scheme: instead of checkpointing every √n layers, they checkpoint at every layer boundary (store one activation per layer, recompute each layer's internals during backward). This is O(n) checkpoint tensors but each is just the layer input — a much smaller tensor than the full set of internal activations. PyTorch exposes this as `torch.utils.checkpoint.checkpoint()` and HuggingFace exposes `model.gradient_checkpointing_enable()`.

Under full gradient checkpointing, the activation memory formula collapses to:

```
Full recompute activation memory = 2 * s * b * h * L   (bytes)
```

vs. `s·b·h·L·(34 + 5·a·s/h)` without recompute. The `2sbhL` floor is just the layer-input tensors that must be retained as checkpoints; everything else is recomputed on demand.

---

## 3. Selective Recomputation — 5× Memory, <4% Compute Overhead

[[selective-recompute-korthikanti]] (Korthikanti et al., 2022) identifies the inefficiency in full gradient checkpointing: it treats all activations symmetrically, but they are not symmetric:

| Activation type | Memory cost | Recompute cost |
|---|---|---|
| Attention score matrix (s×s per head) | High (quadratic in s) | Low (matmul + softmax) |
| MLP intermediate activations | Medium | High (large FFN matmuls) |
| LayerNorm / residual outputs | Low | Medium |

Full recomputation discards everything and recomputes everything — including the expensive MLP activations. This is wasteful: you pay the full 30–40% compute tax to recover memory you would have gotten much more cheaply by just being selective.

### The asymmetric selection

Selective recomputation keeps only the large cheap-to-recompute activations on the "discard" list, and retains the cheap-to-store activations:

**Discard and recompute during backward:**
- The `[b, a, s, s]` attention score matrix (post-softmax, post-dropout) — this is the entire quadratic term in the formula above

**Retain in memory:**
- MLP activation outputs (expensive to recompute, smaller per element)
- LayerNorm outputs
- Linear projection outputs

This is the correct asymmetry because the attention score matrix is the largest tensor (quadratic s²) but is the cheapest to reconstruct — it is just a matmul between Q and K followed by a softmax. Note that FlashAttention (introduced in [[ch-04]]) already recomputes attention scores in its backward pass as part of its own IO-awareness; selective recomputation in the full-precision attention context exploits the same insight.

### Results at scale

From [[selective-recompute-korthikanti]], validated on a 530B GPT-3-style model on 2,240 NVIDIA A100 GPUs:

- **Activation memory reduction:** 5× (versus full storage)
- **Recompute overhead:** <4% (versus 30–40% for full recomputation)
- **MFU improvement:** 42.1% → 54.2% (+29% throughput) compared to full recomputation

The formula after selective recomputation:

```
Selective recompute activation memory = s * b * h * L * (10 + 24/t)   (bytes)
```

The quadratic `5·a·s²·b/(h·t)` term vanishes entirely — it is the attention matrix, which is now recomputed on demand. What remains is the linear term, which scales favorably with sequence length.

---

## 4. Sequence Parallelism — Eliminating the Last Duplication

Tensor parallelism (TP) splits weight matrices across GPUs, but it leaves LayerNorm and Dropout activations replicated — every GPU in the TP group holds a full copy of those tensors, because they are not directly shardable along the weight dimension. At long sequences, those replicated LayerNorm activations are not negligible.

[[selective-recompute-korthikanti]] introduces **sequence parallelism** as a paired technique: since LayerNorm and Dropout are independent across sequence positions, they can be sharded along the sequence dimension across the TP group. The result:

```
Activation memory before SP  = s * b * h * (10 + 24/t + 5*a*s/(h*t))   per layer
Activation memory after SP   = (s * b * h / t) * (34 + 5*a*s/h)         per layer
```

The entire activation formula is divided by t (tensor-parallel degree), not just the weight-coupled parts. Communication-wise, this replaces the TP all-reduce with an AllGather + ReduceScatter pair — identical bandwidth cost, but the activations are no longer replicated.

Sequence parallelism requires that tensor parallelism communicators already exist (same GPU group), so it adds no new communication topology. It is released in Megatron-LM and NeMo-Megatron as a paired flag with TP.

---

## 5. Where Activations Fit in the Full Ledger

Pulling together [[ch-01]], [[ch-02]], and this chapter:

```
Total GPU memory = static_floor + activation_memory + logit_spike + overheads

static_floor     = 18 bytes/param  (weights 2B + masters 4B + grads 4B + Adam 8B)
                 [with FP8 on H100: 6 bytes/param; see [[ch-02]]]

activation_memory (no recompute) = L * s * b * h * (10 + 24/t + 5*a*s/(h*t))
activation_memory (selective)    = L * s * b * h * (10 + 24/t)
activation_memory (full recompute) = 2 * L * s * b * h

logit_spike      = B*T * V * dtype_bytes   [eliminated by Liger; see [[ch-02]]]
                   e.g., s=16384, V=32000, BF16 → 1.05 GB
```

Activations are the one item in the ledger that the practitioner trades against compute. The static floor is fixed by model size and precision choice (FP32 Adam, BF16 working weights). The logit spike is removed by a kernel swap (Liger, [[ch-02]]). Activations are where the fundamental memory-compute tradeoff lives.

```
            Memory               Compute overhead
─────────────────────────────────────────────────────
No checkpointing      O(L)                +0%
Full checkpointing    O(√L) → O(1/L)*    +33%
Selective recompute   O(L, -quadratic)    +~4%
─────────────────────────────────────────────────────
* Per-layer ckpt variant: O(L) checkpoints of cheap tensors
```

---

## 6. The Activation Memory Cascade: A Mental Model

Think of the backward pass as a worker walking backwards through a factory assembly line. At each station (layer), the worker needs to know exactly what state the station was in during the forward run — the "in-flight" configuration — to compute the gradient. Without checkpointing, the factory photographs every station before moving on (O(n) storage). With √n checkpointing, it photographs only every √n-th station and re-runs each segment forward when it arrives to refresh the intermediate state (O(√n) storage, +1 forward sweep). With selective recomputation, it photographs the cheap-to-store stations and memorizes only that the expensive-but-fast-to-reconstruct measurements (attention scores) can always be re-measured on demand.

The key insight is asymmetry: not all activations cost the same to store or recompute. Selective recomputation exploits this asymmetry perfectly — it discards the largest activations (attention s×s matrices) precisely because those are the cheapest to reconstruct.

---

## Core Insights from the Literature

**1. Memory and compute are fungible for activations (Chen 2016).** [[gradient-checkpointing-chen]] established that activation memory is not a fixed cost — it is a point on a memory/compute Pareto frontier. The O(√n) result is a clean theoretical bound: halving memory costs one extra forward pass, not two. This was the conceptual unlock for training very deep networks.

**2. The quadratic attention term is the right thing to discard (Korthikanti 2022).** [[selective-recompute-korthikanti]]'s selective recomputation is more than an engineering optimization — it is identifying the correct asymmetry. The attention score matrix is large because of O(s²) scaling, yet cheap to reconstruct because it is a matmul + softmax (low arithmetic intensity, amenable to fast recompute). MLP activations are the opposite: smaller per element but expensive (dense GEMMs with large intermediate dimensions). Full checkpointing ignores this asymmetry; selective recomputation exploits it to achieve 5× memory reduction at <4% compute cost.

**3. Sequence parallelism is sequence parallelism of the right activations (Korthikanti 2022).** The 10·s·b·h "non-TP" term in the activation formula (LayerNorm, Dropout) is the residual after tensor parallelism divides everything else. Sequence parallelism is the targeted fix: shard those specific activations along the sequence dimension at zero communication-bandwidth overhead. Without SP, TP alone leaves this term fully replicated across every rank in the TP group.

**4. At long context, activations are the ledger.** The static floor (18 B/param) is fixed. At s=4096 a 7B model already has comparable activation memory. At s=32768 the quadratic attention term makes activations the dominant term by a wide margin. The practical consequence: for any long-context training run, the question "does this fit?" is almost entirely a question about the activation strategy — full recompute vs selective vs no recompute.

---

## Key Takeaways

- **Activation memory scales as O(L·s·b·h) + O(L·a·s²·b/h)** — the quadratic attention term dominates at long sequence lengths and is the primary OOM risk for large-s training.
- **Full gradient checkpointing reduces activation memory from O(L·s·b·h·34) to 2·L·s·b·h** (just storing layer inputs) at the cost of ~33% extra compute. The Chen 2016 theorem gives O(√L) checkpoints for optimal general networks; modern per-layer checkpointing is a practical variant.
- **Empirical validation of Chen 2016:** 1,000-layer ResNet, 48 GB → 7 GB, 30% runtime overhead.
- **Selective recomputation (Korthikanti 2022) achieves 5× activation memory reduction at <4% compute overhead** by discarding only the attention score matrix (large, cheap-to-recompute) and retaining MLP/LayerNorm activations (smaller, expensive-to-recompute).
- **Sequence parallelism** pairs with tensor parallelism to shard the LayerNorm/Dropout activations across the TP group, achieving a true t× activation reduction across the whole layer.
- **530B-parameter validation:** MFU improved 42.1% → 54.2% (29% faster) with selective recompute + sequence parallelism vs full recomputation on 2,240 A100s.
- Activations are the **only memory ledger item you trade against compute**. The static floor and logit spike are addressed by precision choices and kernel swaps; activations require an algorithmic decision.

---

## Questions

1. The activation memory formula has two terms: `s·b·h·(10 + 24/t)` (linear in s) and `5·a·s²·b/(h·t)` (quadratic in s). At what sequence length does the quadratic term equal the linear term for a model with h=4096, a=32, t=1? What does this imply about the sequence length at which checkpointing strategy becomes critical?

2. [[gradient-checkpointing-chen]] states the compute overhead is "one extra forward pass per mini-batch," not one per layer. Walk through the √n scheme: if n=64 layers (√n=8 segments), exactly how many layer-forward operations occur during the backward pass, and why does this add up to n additional operations total (not n√n)?

3. Selective recomputation (Korthikanti 2022) discards the post-softmax attention score matrix and retains MLP activations. Now consider FlashAttention (introduced in [[ch-04]]): FlashAttention's backward pass already recomputes attention scores from stored softmax statistics (logsumexp). If you pair FlashAttention with selective recomputation, is there any double-counting of the recompute overhead? What does this imply for the memory and compute accounting?

4. Sequence parallelism replaces the TP all-reduce with an AllGather + ReduceScatter. The paper claims this is "zero extra communication bandwidth cost." Verify this claim: if the all-reduce volume is V bytes per layer, what is the AllGather + ReduceScatter volume? (For a ring all-reduce over t ranks, the bandwidth cost is 2V·(t-1)/t per-rank; for AllGather + ReduceScatter over t ranks, what is the per-rank cost?)

5. The [[gradient-checkpointing-chen]] excerpt notes an extreme variant achieving O(log n) memory at O(n log n) compute. Describe the recursive structure that achieves this. Why is this variant rarely used in practice despite being theoretically superior to O(√n)?

6. Looking at the activation formula under selective recomputation — `s·b·h·L·(10 + 24/t)` — and under full recomputation — `2·s·b·h·L` — at what tensor-parallel degree t does selective recomputation become cheaper than full recomputation (in bytes)?

---

## References

- Chen, T., Xu, B., Zhang, C., & Guestrin, C. (2016). Training Deep Nets with Sublinear Memory Cost. arXiv:1604.06174. https://arxiv.org/abs/1604.06174 ([[gradient-checkpointing-chen]])
- Korthikanti, V., Casper, J., Lym, S., McAfee, L., Andersch, M., Shoeybi, M., & Catanzaro, B. (2022). Reducing Activation Recomputation in Large Transformer Models. MLSys 2023. https://arxiv.org/abs/2205.05198 ([[selective-recompute-korthikanti]])
- Anthony, Q. et al. (2023). Transformer Math 101. EleutherAI Blog. https://blog.eleuther.ai/transformer-math/ ([[transformer-math-101]])
- Micikevicius, P. et al. (2018). Mixed Precision Training. ICLR 2018. https://arxiv.org/abs/1710.03740 ([[mixed-precision-training]])
- Shoeybi, M. et al. (2019) + Korthikanti et al. (2022). Megatron-LM (TP + SP). https://arxiv.org/abs/1909.08053 + https://arxiv.org/abs/2205.05198 ([[megatron-tp-sp]])
