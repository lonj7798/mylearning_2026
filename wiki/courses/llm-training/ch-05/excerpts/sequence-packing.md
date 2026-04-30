---
chapter: ch-05
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/sequence-packing.md
source_url: https://arxiv.org/abs/2107.02027
created_at: "2026-04-23"
---

# Excerpt: Efficient Sequence Packing (seen through the FSDP lens)

**Paper:** Efficient Sequence Packing without Cross-Contamination
**Authors:** Mario Michael Krell, Matej Kosec, Sergio P. Perez, Andrew Fitzgibbon (Graphcore, 2021)
**arXiv:** 2107.02027

**This excerpt focuses on how packing interacts with FSDP's per-step memory budget.** The general packing mechanics are covered in ch-04; here we look at why an FSDP training job *needs* packing to make sharding arithmetic close.

---

## Why the distributed story depends on packing

The paper's motivating claim is a throughput one:

> "Up to 50% (and in extreme cases 89%) of tokens in BERT/GLUE-style fine-tuning are padding. … [Packing] yields 2× SFT throughput with zero accuracy impact."

For a single-GPU setup this is "nice to have." For an FSDP job at 70B, it is **load-bearing for the memory budget**. The FSDP memory formula ([[excerpts/fsdp-sft]]):

```math
\text{Mem}_{\text{FSDP}} = \frac{16P}{N} + 2P + \underbrace{A(\text{seqlen}, \text{batch}, L)}_{\text{activations}}
```

The first two terms are what the FSDP paper counts. The activation term `A` is independent of FSDP — it scales with `seqlen · batch · L`. At 70B, L ≈ 80, hidden ≈ 8192, seqlen 4096 — activations alone burn 30+ GB of HBM even with checkpointing. If half your tokens are PAD, you are paying full activation cost for zero gradient signal.

**Notice:** padding tokens cost activation memory identically to real tokens. FlashAttention's varlen path is the only way to actually skip them — naive masking still materializes the full `(seqlen × seqlen)` attention matrix.

---

## The packed-block contract — cu_seqlens and varlen attention

Quoted from the Technical Details:

> "Given n short sequences s_1, …, s_n with lengths L_1, …, L_n (Σ L_i ≤ L_max), concatenate along the sequence axis: `packed = [s_1 | s_2 | … | s_n | PAD]` with: `cu_seqlens = [0, L_1, L_1+L_2, …, Σ L_i]` — cumulative start offsets. `position_ids` reset to 0 at each sequence boundary."

The attention mask structure is block-diagonal causal:

```math
\text{mask}[i, j] = \begin{cases} 1 & \text{if } \text{seq}(i) = \text{seq}(j) \text{ and } j \le i \\ 0 & \text{otherwise} \end{cases}
```

And the varlen kernel interface:

> "Modern implementations use `flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen)` which computes block-diagonal causal attention **without materializing the full (L_max × L_max) mask**. Memory: O(sum L_i) not O(L_max²)."

**Notice:** the varlen memory scaling is what closes the FSDP memory budget. Without it, the attention buffer is `O(seqlen²) · num_heads · num_layers` — at seqlen 4096, L 80, heads 64, that's ~170 GB before FSDP even sharded anything.

---

## Interaction with FSDP's ReduceScatter — the token-count bug

This is the subtle part of the distributed story: **packing changes the number of loss-bearing tokens per rank**. In a naive, unpacked batch, each rank has the same micro-batch size B and the same seqlen L, so tokens-per-rank is identical across ranks. Under packing, rank 0 might get a block holding 7 short conversations (say 3100 response tokens), while rank 3 might get a block holding 2 long ones (2900 response tokens).

The FSDP ReduceScatter sums gradients across ranks:

```math
g_{\text{global}} = \frac{1}{N} \sum_{i=1}^{N} g_i
```

If each rank's `g_i` is `∂L_i / ∂θ` computed as a **sum** (not mean) over its local tokens, the ReduceScatter is arithmetically correct: global loss = global sum / global tokens, and the gradient scales linearly. But if each rank divides by its *local* token count first:

```math
g_i = \frac{1}{T_i} \sum_{t \in \text{rank}\ i} \nabla \ell_t
```

then the ReduceScatter averages *per-rank means*, not per-token means. Ranks with few response tokens get disproportionately large influence on the global gradient. This is silent — the loss still trends down, but the effective learning rate is heterogeneous across examples.

**The fix:** compute loss with `reduction="sum"`, then divide by the **global** token count after the ReduceScatter. See [[excerpts/loss-masking-prompt]] for the full normalization contract.

---

## Cross-contamination — why mathematical equivalence matters under sharding

Quoted:

> "Without masking, token t in sequence 2 can attend to sequence 1's tokens → the softmax's partition function leaks across documents → changes gradients even on sequence 1 tokens. The block-diagonal mask restores exactness."

Under FSDP, the equivalence guarantee is what lets you reason about global-batch effective learning rate. If cross-contamination leaks a tiny gradient signal between documents, that signal is still present after ReduceScatter — every rank's gradient is contaminated identically, and the global average preserves the bias. A packed-batch FSDP run that forgets varlen masking trains on a distribution that has no unpacked equivalent; the learner is effectively fine-tuning on a Frankenstein cross-document task.

The paper's equivalence proof is that the block-diagonal mask makes per-token logits bit-identical to the unpacked run. Under FSDP, bit-identical per-rank means the ReduceScatter output is bit-identical too, which means the effective global batch gradient is bit-identical to what a hypothetical `N`-way unpacked DDP run would produce.

---

## SPFHP — why the algorithm's determinism matters for resumes

Quoted:

> "SPFHP (Shortest-Pack-First Histogram-Packing): 1. Compute length histogram of dataset. 2. Greedy: iterate over bins, fill each with the longest remaining sequence first; for each subsequent fit, pick the longest sequence that still fits (shortest gap). 3. Achieves near-optimal packing ratio (>99% fill) in O(N log N)."

The 2025 distributed-training angle: SPFHP is **deterministic** given a fixed input order. This is not a property of the bin-packing problem in general — offline bin-packing has many optimal solutions. But the paper's greedy tie-breaking (always pick the longest fitting sequence) makes the output reproducible.

Under FSDP with sharded checkpointing, resume bit-equivalence requires that each rank sees the exact same packed blocks it would have seen on a first-run. If the packer depends on wall-clock ordering or Python-dict iteration order, different ranks resume from different data distributions — a desync that manifests as an instantaneous loss spike at the resume step. SPFHP's determinism is the contract that rules this out.

**Notice:** TRL's `DataCollatorWithPacking` and Axolotl's `sample_packing` both use SPFHP-family greedy packers for this reason. Never use a custom `random.shuffle`-based packer in an FSDP job unless you've verified that every rank's RNG is seeded identically before the shuffle.

---

## Labels, ignore_index, and the per-rank token count

Quoted:

> "Labels for padding tokens must be set to -100 (ignore_index); loss masks for prompt tokens (see [[loss-masking-prompt]]) are applied per sub-sequence inside the pack."

The `-100` convention is PyTorch's; `F.cross_entropy(ignore_index=-100)` drops those positions from both the numerator and the denominator:

```math
\ell = -\frac{1}{|\{t : y_t \ne -100\}|} \sum_{t : y_t \ne -100} \log \pi_\theta(y_t | y_{<t})
```

Under FSDP, this denominator is computed *locally* on each rank, not globally. Two consequences:

1. If rank 0 has 3100 loss-bearing tokens and rank 3 has 2900, their `ℓ` values are means over different sample sizes. The FSDP gradient ReduceScatter averages the per-rank `∂ℓ / ∂θ`, which is equivalent to weighting each rank's tokens by `1 / T_i` — a heterogeneous weighting.
2. To get unbiased global-mean loss, compute **sum** loss locally and divide by **global** token count after an AllReduce on the token count.

This is the correctness linkage between packing and FSDP that the original paper does not spell out — it matters only under sharding.

---

## Packing's effect on activation-checkpointing arithmetic

Activation checkpointing (the other orthogonal memory knob in ch-05 §6) recomputes forward activations in backward at a 25–35% compute cost. Its memory savings scale with `seqlen · batch · hidden · L`. Under packing, the `seqlen · batch` product is dominated by **real tokens**, not padding — so activation checkpointing's savings are higher-leverage per memory-byte-saved than on an unpacked batch.

Concretely: unpacked batch at 50% padding, seqlen 4096, batch 1 stores:

```math
A_{\text{unpacked}} = 4096 \cdot 1 \cdot 8192 \cdot 80 \cdot 2 \text{ bytes} \approx 5.4\ \text{GB per block}
```

Half of this memory is PAD activations that contribute zero to the gradient — a dead-weight activation footprint. Packing eliminates the PAD fraction, so with activation checkpointing applied on top, the saved memory is real-token memory rather than padding memory.

**Notice:** this is why torchtitan / TRL recipes apply packing *and* activation checkpointing together as a package, never one without the other. The distributed memory formula in [[excerpts/fsdp-sft]] assumes packed, activation-checkpointed batches — using one without the other breaks the arithmetic that justifies FSDP FULL_SHARD at 70B.

---

## Connections

- [[excerpts/fsdp-sft]] — FSDP memory arithmetic; packing closes the activation budget.
- [[excerpts/loss-masking-prompt]] — the per-rank token-count normalization contract lives here.
- [[excerpts/mixed-precision]] — packed varlen attention runs in bf16; padded masked attention in fp32 softmax.
- [[excerpts/gradient-clipping]] — heterogeneous per-rank token counts distort the global-norm calculation too.
- [[excerpts/adam]] — variable per-rank token counts change the per-step gradient statistics the optimizer sees.
- [[ch-05]] — synthesis.
