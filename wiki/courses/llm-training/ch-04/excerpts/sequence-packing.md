---
chapter: ch-04
course: llm-training
phase: read
excerpt_of: Krell, Kosec, Perez, Fitzgibbon — "Efficient Sequence Packing without Cross-Contamination for Language Model Training"
source_url: https://arxiv.org/abs/2107.02027
created_at: "2026-04-23"
---

# Excerpt: Krell et al. 2021 — Sequence Packing without Cross-Contamination

**Source:** `wiki/raw-data/llm-training/papers/sequence-packing.md`
**Paper:** Mario Michael Krell, Matej Kosec, Sergio P. Perez, Andrew Fitzgibbon (Graphcore / Oxford), 2021
**arXiv:** https://arxiv.org/abs/2107.02027

---

## Bibliographic header

> *"Up to 50% (and in extreme cases 89%) of tokens in BERT/GLUE-style fine-tuning are padding. We formalize sequence packing as a bin-packing problem and present efficient algorithms (NNLS, shortest-pack-first) that produce packed batches with minimal padding. Crucially, we design attention masks and position IDs that make the packed model mathematically equivalent to the unpacked one — no accuracy loss — while achieving a 2× speedup on BERT phase-2 pretraining."*

The paper's power is that it is not a new architecture or optimiser — it is a *correctness proof* and a *bin-packing recipe*. Both pieces are load-bearing: the recipe without the proof would be a throughput hack; the proof without the recipe would be a theorem nobody implements. Together they license every modern SFT stack (TRL, Axolotl, Nanotron) to turn packing on by default.

---

## The motivation quote

> *"Instruction datasets have 50–89% padding tokens."*

The raw-data insight line distils the core empirical observation. In SFT, sequences are short (Alpaca median ≈ 150 tokens) and the training `max_seq_length` is long (2048–4096). Padding to `max_seq_length` means most FLOPs in each batch are computed on `[PAD]` tokens whose labels are `-100` and whose attention outputs are discarded. If 50% of tokens are pad, you are paying 50% of your compute for a gradient of zero.

**Notice:** the pad-rate is not a tokeniser artefact, it is a *distributional* fact about instruction data. No amount of tokenizer tuning rescues you; the only structural fix is to pack.

---

## Mechanics of a packed block — the core construction

From the raw-data notes:

> *"Given n short sequences s_1, …, s_n with lengths L_1, …, L_n (Σ L_i ≤ L_max), concatenate along the sequence axis:*
> *`packed = [s_1 | s_2 | … | s_n | PAD]`*
> *with `cu_seqlens = [0, L_1, L_1+L_2, …, Σ L_i]`, `position_ids` reset to 0 at each sequence boundary, and an attention mask where token t in sequence i can attend only to tokens in sequence i with causal structure inside."*

Three invariants are encoded here and each matters independently:

```math
\text{packed}[k] = s_{i(k)}[k - \text{cu\_seqlens}[i(k)]]
```

where `i(k)` is the index of the sub-sequence containing position `k`. This is the definition of the concatenation.

```math
\text{position\_ids}[k] = k - \text{cu\_seqlens}[i(k)]
```

The position IDs must *reset*. If you forgot this step, token 0 of `s_2` would have position ID `L_1` — which under RoPE, ALiBi, or learned absolute embeddings would place it in a region of positional space the model never saw during pretraining for a "start-of-document" token. Reset ensures every sub-sequence begins at position 0 as if it lived in its own unpacked batch.

```math
M[q, k] = \begin{cases} 1 & \text{if } i(q) = i(k) \text{ and } k \le q \\ 0 & \text{otherwise} \end{cases}
```

This is the *block-diagonal causal* mask: block-diagonal because cross-sequence attention is forbidden (`i(q) = i(k)`), causal because within a block tokens only attend to past positions (`k ≤ q`).

**Notice:** the three corrections (`position_ids` reset, block-diagonal mask, `cu_seqlens` offsets) are not independent — they are a *joint* transformation. Skipping any one of them violates the equivalence proof.

---

## Why cross-contamination matters — the correctness argument

From the raw-data notes:

> *"Without masking, token t in sequence 2 can attend to sequence 1's tokens → the softmax's partition function leaks across documents → changes gradients even on sequence 1 tokens. The block-diagonal mask restores exactness."*

Unpack this step by step. The attention softmax for query position `q` is:

```math
\alpha_{q,k} = \frac{\exp(Q_q K_k^\top / \sqrt{d})}{\sum_{k'} \exp(Q_q K_{k'}^\top / \sqrt{d})}
```

Without masking, the denominator `Z_q = Σ_{k'} exp(Q_q K_{k'}^T / √d)` sums over *all* keys in the packed block, including those from other sub-sequences. Consequently:

1. The attention weights `α_{q,k}` within sub-sequence `i(q)` are *smaller* than they would be in the unpacked case, because `Z_q` is inflated by contributions from other sub-sequences.
2. The output vector `O_q = Σ_k α_{q,k} V_k` is contaminated by `V_k` from foreign sub-sequences.
3. By the chain rule, the *gradient* through `O_q` — which flows back through the sub-sequence `i(q)` — picks up signal from tokens in other sub-sequences.

The block-diagonal mask replaces the full `Z_q` with `Z_q^{block} = Σ_{k' : i(k')=i(q), k'≤q} exp(...)` — exactly the denominator that would appear in the unpacked forward pass. This is the paper's equivalence theorem in one line: with block-diagonal causal masking and position-ID resets, the packed forward pass is *bit-identical* (modulo floating-point non-associativity) to running each sub-sequence in its own micro-batch.

**Notice:** this is why "just packing" without mask fixes is not merely a minor approximation — it changes gradients on *every* token, not just on the boundary between sub-sequences. The paper's 2× throughput is conditional on getting the masking right.

---

## FlashAttention varlen — avoiding the O(L²) mask

From the raw-data notes:

> *"Modern implementations use `flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen)` which computes block-diagonal causal attention without materializing the full (L_max × L_max) mask. Memory: O(sum L_i) not O(L_max²)."*

The naive way to implement the block-diagonal mask is to allocate an `L_max × L_max` boolean tensor and set `M[q, k] = (i(q) == i(k)) & (k <= q)`. For `L_max = 8192` this is 64 MiB per head per layer of masking storage alone — prohibitive.

FlashAttention-2's varlen API (https://arxiv.org/abs/2307.08691, building on the original FlashAttention at https://arxiv.org/abs/2205.14135) sidesteps the materialisation by taking `cu_seqlens` directly as a *structural* argument. The kernel signature is:

```python
flash_attn_varlen_func(
    q, k, v,                        # packed tensors of shape (Σ L_i, n_heads, d_head)
    cu_seqlens_q, cu_seqlens_k,     # int32 arrays of length (batch + 1)
    max_seqlen_q, max_seqlen_k,     # ints used for kernel launch sizing
    causal=True,
)
```

Inside the kernel, each CUDA block is assigned to one (query sub-sequence, query-tile) pair; it then iterates over *only the key tiles belonging to the same sub-sequence*, skipping the rest. The block-diagonal structure is never stored as a mask — it is encoded in the kernel's iteration bounds, derived from `cu_seqlens`.

Memory goes from `O(L_max²)` (the full mask) to `O(Σ L_i)` (the packed tokens themselves, which you already have). Throughput improves too, because masked-out tiles are *skipped* rather than computed-and-zeroed.

**Notice:** this is the hidden hinge of the whole 2024+ SFT stack. Without varlen attention, the block-diagonal mask costs more memory than the unpacked batch. With it, packing becomes strictly dominant. See also the FlashAttention v1 paper (https://arxiv.org/abs/2205.14135) for the IO-aware tiling that makes this possible.

---

## SPFHP — Shortest-Pack-First Histogram-Packing

From the raw-data notes:

> *"1. Compute length histogram of dataset. 2. Greedy: iterate over bins, fill each with the longest remaining sequence first; for each subsequent fit, pick the longest sequence that still fits (shortest gap). 3. Achieves near-optimal packing ratio (>99% fill) in O(N log N)."*

Let us walk the algorithm explicitly. Given:
- Dataset `D = {s_1, ..., s_N}` with lengths `L_1 ≥ L_2 ≥ ... ≥ L_N` (after sorting descending).
- Target pack length `L_max`.

```
SPFHP(D, L_max):
  bins = []                                # list of (remaining_capacity, contents)
  for s_i in D (sorted by L desc):
    # Find the pack with smallest remaining capacity that still fits s_i
    candidate = argmin { remaining(b) : b ∈ bins, remaining(b) ≥ L_i }
    if candidate is None:
      bins.append(new_bin(L_max - L_i, [s_i]))
    else:
      candidate.contents.append(s_i)
      candidate.remaining -= L_i
  return bins
```

"Shortest-pack-first" means: among packs that still have room, put the next sequence into the *most-filled* pack (smallest remaining capacity). This greedy heuristic corresponds to the classic **Best-Fit-Decreasing (BFD)** bin-packing strategy and has a provable approximation ratio of `11/9 · OPT + 6/9` for 1D bin-packing.

Why it works well in practice: the length histogram of instruction data is heavy-headed (few long sequences, many short ones). Place the long ones first to seed the packs; fill remaining slots with short sequences. A priority queue keyed on remaining capacity makes each insertion `O(log |bins|)`, giving `O(N log N)` total — cheap preprocessing amortised across epochs.

**Notice:** SPFHP is run *offline* once, not per-step. The output is a permutation of the dataset into packs; the dataloader then streams packs in order. You pay the `O(N log N)` cost once at `prepare_dataset`-time.

---

## NNLSHP — the histogram-LP formulation

From the raw-data notes:

> *"Solves a non-negative least squares problem matching a desired length histogram → provides mathematically optimal packing strategies given histogram constraints."*

NNLSHP formulates packing as: given the length histogram `h ∈ ℕ^{L_max}` (where `h[ℓ]` = count of sequences of length `ℓ`), find a non-negative combination of "pack templates" `T_1, ..., T_K` (each template is a multiset of lengths summing to ≤ `L_max`) such that the sum of templates reproduces `h`:

```math
\min_{x \ge 0} \| \sum_k x_k T_k - h \|_2^2
```

Solved as an NNLS problem. It provides the theoretical optimum and is used in the paper as the benchmark against which SPFHP is measured. In practice SPFHP gets within a fraction of a percent of NNLSHP while being far simpler to implement — which is why every SFT stack ships SPFHP, not NNLSHP.

---

## Loss handling inside packs

From the raw-data notes:

> *"Labels for padding tokens must be set to -100 (ignore_index); loss masks for prompt tokens (see [[loss-masking-prompt]]) are applied per sub-sequence inside the pack."*

Two masking responsibilities live at different levels of the stack:

1. **Packing layer:** sets `labels[pad_position] = -100` for the trailing `[PAD]` tokens in each pack. Position IDs are also set to 0 (or simply ignored, since masked tokens contribute no gradient).
2. **SFT layer:** for each *sub-sequence* within the pack, masks the prompt portion (`labels[prompt_start:prompt_end] = -100`) so only completion tokens contribute. This is per-sub-sequence, not per-pack.

See [[excerpts/loss-masking-prompt]] for the formal loss definition and the IGNORE_INDEX=-100 mechanics in `torch.nn.CrossEntropyLoss`.

**Notice:** incorrect composition is a common silent bug. If you mask the first `prompt_len` tokens of the whole *pack* instead of per-sub-sequence, you accidentally unmask user tokens from every sub-sequence except the first. The SFT loss then trains the model to generate user prompts — the exact pathology prompt-masking exists to prevent.

---

## Empirical result — Table 3 headline

> *"2× throughput on BERT phase 2 with no metric change."*

The gain is not marginal and not scenario-specific. On Wikipedia phase 2 (sequence length 512), packing achieves near-2× throughput because the pad-rate at that length on Wikipedia is ≈ 50%. On SFT datasets with higher pad-rates (Alpaca, ShareGPT at 2048 context), the effective gain is often 3–4×.

Because the mask construction is provably equivalent to unpacked training, the loss trajectory and the downstream benchmark scores are unchanged up to floating-point noise. This is the claim that lets practitioners flip `packing=True` without running an ablation each time.

---

## Connections

- SFT loss definition and multi-turn masking: [[excerpts/loss-masking-prompt]]
- Reference implementation (TRL `SFTTrainer`, packing recipe): [[excerpts/hf-alignment-handbook]]
- Noise-based regulariser that composes with packing: [[excerpts/neftune]]
- Chapter synthesis: [[ch-04]]
- FlashAttention v1 and v2 papers (varlen API backs block-diagonal attention): https://arxiv.org/abs/2205.14135 and https://arxiv.org/abs/2307.08691
