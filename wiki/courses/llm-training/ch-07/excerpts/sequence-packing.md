---
chapter: ch-07
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/sequence-packing.md
source_url: https://arxiv.org/abs/2107.02027
created_at: "2026-04-23"
---

# Excerpt: Krell 2021 — packing's cross-sample attention leak

**Source library:** `wiki/raw-data/llm-training/papers/sequence-packing.md`
**Paper:** Krell, Kosec, Perez, Fitzgibbon 2021, *"Efficient Sequence Packing without Cross-Contamination."*

---

## Why this source anchors ch-07

Ch-07 §4b is entirely Krell 2021. The paper's contribution is simultaneously a 2× throughput optimization *and* the specification of what invariant packing must preserve. When implementations violate that invariant, the symptom is a 0.5–2 point regression on downstream benchmarks with no accompanying loss-curve signal. The failure is perfectly silent, and the class of bug is perfectly specific: attention and positional encodings that don't know a packed block is logically separate sub-sequences.

The source states the invariant in one sentence:

> *"We design attention masks and position IDs that make the packed model mathematically equivalent to the unpacked one — no accuracy loss — while achieving a 2× speedup on BERT phase-2 pretraining."*

Mathematically equivalent means: for any model weights, the output on every token of a correctly-packed block must match, to floating-point rounding, the output on that same token when that sub-sequence is fed alone. Ch-07 §4b's unit test is the operational form of this invariant.

---

## The block anatomy — what "packing" actually means

From the source (Technical Details):

> *"Given n short sequences s_1, …, s_n with lengths L_1, …, L_n (Σ L_i ≤ L_max), concatenate along the sequence axis:*
> *`packed = [s_1 | s_2 | … | s_n | PAD]`*
> *with:*
> *- `cu_seqlens = [0, L_1, L_1+L_2, …, Σ L_i]` — cumulative start offsets.*
> *- `position_ids` reset to 0 at each sequence boundary.*
> *- Attention mask: token t in sequence i can attend to tokens in sequence i only, with causal structure inside."*

Notice: three things must be right for the invariant to hold, not one. This is the trap — many implementations get *two* of the three right and fail silently on the third.

1. **`cu_seqlens`** — the offset array that tells the attention kernel where each sub-sequence starts and ends. If omitted, the kernel defaults to dense causal over the full block and sample 2 attends to sample 1's tokens through the lower triangular.

2. **`position_ids` reset** — RoPE / ALiBi / learned-pos all use token position as input. If position IDs run continuously across the block (0, 1, …, L-1 for the whole pack rather than restarting at each boundary), sample 2's token at local index 0 has global position L_1 — so RoPE rotates it using L_1 as its *own* position, which is wrong. The effect is softer than the attention leak but still violates mathematical equivalence.

3. **Loss / label mask reset** — ch-07 §4a's territory but interacting with this source: each sub-sequence has its own prompt/response split; `labels[:prompt_len] = -100` is single-prompt code that fails on packs (Wrong #2 in [[excerpts/loss-masking-prompt]]).

Ch-07 §4b enumerates three specific kernel-level ways packing breaks:

> *"(1) cu_seqlens is omitted and the kernel defaults to dense causal → sample 2 attends to sample 1's tokens through the lower-triangular.*
> *(2) position_ids are not reset at sub-sequence boundaries → RoPE rotates using absolute offsets across the whole pack, so sample 2's token 0 has position ≈ L₁, distorting relative-position math.*
> *(3) A custom attention override (Liger, TorchTune, HF attn_implementation='eager') silently ignores cu_seqlens because the hand-rolled path did not thread it through."*

(3) is the most insidious. An operator change at the kernel level is invisible from the training loop's perspective; the `model.forward` contract doesn't expose which attention implementation fired. A framework swap that changes the default attention path (like moving from Open Instruct to Olmo-core, mentioned in [[excerpts/olmo-3]]) can silently disable varlen packing without anyone noticing until the eval regresses.

---

## The FlashAttention varlen API — the correct surface

From the source:

> *"Modern implementations use `flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen)` which computes block-diagonal causal attention without materializing the full (L_max × L_max) mask. Memory: O(sum L_i) not O(L_max²)."*

Notice the memory argument: a materialized block-diagonal mask of size `(L_max, L_max)` is `L_max²` elements. For `L_max = 8192`, that is ~67M booleans per head per layer — significant memory that the varlen kernel avoids by encoding the block structure in `cu_seqlens` alone.

The operational test is: if your attention layer takes a `cu_seqlens` argument and threads it through to `flash_attn_varlen_func`, you are correct. If the kernel accepts an `attention_mask` argument but not `cu_seqlens`, you are likely on the `O(L²)` path and possibly the dense-causal fallback path.

The ch-07 §4b unit test in code:

```python
out_packed   = model(input_ids_packed, cu_seqlens=cu)
out_unpacked = torch.stack([model(s) for s in split_by_cu(input_ids_packed, cu)])
assert (out_packed - out_unpacked).abs().max() < 1e-4, "cross-sample leak"
```

This test should be in CI. It takes one forward pass, costs microseconds, and is a regression test for every kernel change the team will ever make.

---

## Why cross-contamination matters — the softmax partition function

The source's explanation of the mechanism:

> *"Without masking, token t in sequence 2 can attend to sequence 1's tokens → the softmax's partition function leaks across documents → changes gradients even on sequence 1 tokens. The block-diagonal mask restores exactness."*

Notice the subtle second-order effect: the leak doesn't just corrupt sequence 2 (which attends to sequence 1's tokens through the unmasked lower triangular). It *also* corrupts sequence 1, because sequence 2's QKᵀ scores appear in the softmax denominator — except they don't, because sequence 1's tokens are attended to only from within sequence 1 by causal masking. Wait — let me re-read.

The leak is asymmetric: under dense causal, token at position `t` attends to tokens at positions `0..t`. If `t` is in sequence 2, `0..t` includes sequence 1's tokens; so sequence 2 leaks. Sequence 1's tokens are at positions `0..L_1-1`, and causal masks out anything past them; so sequence 1 does *not* leak from sequence 2. The source is imprecise on this point — the "changes gradients on sequence 1 tokens" claim is via the *shared weights* backprop, not via direct forward-pass contamination.

Either way, the practical impact is empirical: the paper's Table 3 reports ~2× throughput with no metric change on BERT phase 2; the negative case (packing without the block-diagonal mask) produces 0.5–2 point regressions on GLUE-class benchmarks. For a modern LLM SFT run, the same-sized regression on MMLU, MT-Bench, AlpacaEval is the cost of a wrong kernel — small enough to look like noise, large enough to lose a leaderboard position.

---

## Packing algorithms — SPFHP and NNLSHP

From the source:

> *"SPFHP algorithm (paper's preferred): 1. Compute length histogram of dataset. 2. Greedy: iterate over bins, fill each with the longest remaining sequence first; for each subsequent fit, pick the longest sequence that still fits (shortest gap). 3. Achieves near-optimal packing ratio (>99% fill) in O(N log N)."*

The packing algorithm is independent of the attention-mask correctness. A perfectly-optimized SPFHP pack with 99.3% fill and no `cu_seqlens` threaded through is indistinguishable to the SFT run from a naive 60%-fill pack with correct masking — except the 99.3%-fill version is training silently wrong. The optimization is necessary for throughput; the mask is necessary for correctness; they must both be right.

NNLSHP is the more academic algorithm that solves the histogram match as a non-negative least-squares problem. In practice SPFHP is what ships because it's O(N log N) and the 99.3% vs 99.7% packing difference is dwarfed by the 2× speedup over unpacked.

---

## Loss handling inside a pack — the §4a bridge

From the source:

> *"Labels for padding tokens must be set to -100 (ignore_index); loss masks for prompt tokens (see [[loss-masking-prompt]]) are applied per sub-sequence inside the pack."*

This is the explicit bridge between the two ch-07 §4 failure modes. The correct SFT pack is:

```python
input_ids  = concat([prompt1, resp1, prompt2, resp2, ...])  # up to max length, then pad
labels     = concat([[-100]*len(prompt1), resp1_tokens,
                     [-100]*len(prompt2), resp2_tokens, ...])
cu_seqlens = [0, len(prompt1)+len(resp1),
              len(prompt1)+len(resp1)+len(prompt2)+len(resp2), ...]
position_ids = concat([range(len(prompt1)+len(resp1)),
                       range(len(prompt2)+len(resp2)), ...])
```

Every sub-sequence contributes its own prompt-mask, its own position IDs (reset to 0 at the boundary), and its own `cu_seqlens` entry. Miss any and §4a or §4b fires.

---

## What to take from Krell 2021 for ch-07

1. **Packing's correctness is encoded in three side-channels: `cu_seqlens`, `position_ids`, label mask.** All three must be right.
2. **The FlashAttention varlen kernel is the canonical correct surface.** If your attention layer takes only `attention_mask`, suspect dense-causal fallback.
3. **Custom attention overrides are the highest-risk kernel change.** Liger / TorchTune / eager attention may silently drop `cu_seqlens`.
4. **Unit-test packed == unpacked in CI.** One forward pass, microseconds, catches every variant of the leak.
5. **Packing efficiency (99% fill) is independent of packing correctness.** Optimize throughput only after the correctness test passes.

---

## Connections

- [[excerpts/loss-masking-prompt]] — the per-sub-sequence mask inside the pack; §4a and §4b are siblings in the same collator.
- [[excerpts/fsdp-sft]] — packing + FSDP is the 2025 SFT backbone; the two optimizations compose.
- [[excerpts/karpathy-training-neural-net-recipe]] — "visualize just before the net" is the `decode(input_ids[labels != -100])` audit.
- [[excerpts/olmo-3]] — Open-Instruct-to-Olmo-core migration is exactly the framework swap that risks (3) — the silent `cu_seqlens` drop.
- [[ch-07]] — §4b (the three leak modes), §3 (dead-pipeline via all-padding batch), §7 (packed-vs-unpacked CI assertion).
