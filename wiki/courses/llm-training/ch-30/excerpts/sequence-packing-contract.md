---
chapter: ch-30
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/sequence-packing.md
source_url: https://arxiv.org/abs/2107.02027
created_at: "2026-04-23"
---

# Excerpt: Sequence packing as a correctness contract, not a speed hack

**Source library:** `wiki/raw-data/llm-training/papers/sequence-packing.md`
**Anchor paper:** Krell, Kosec, Perez, Fitzgibbon 2021 — "Efficient Sequence Packing without Cross-Contamination"
**Ablation counterpart:** [[packed-vs-unpacked-ablation]] (attested equivalence when masks are correct)

---

## Why this source anchors ch-30

Packing is ch-30's axis #4. It looks like a throughput optimisation and almost everyone treats it that way in their head. The source argues — and the 2021–2024 ablations confirm — that packing's correctness contract (block-diagonal attention + per-sub-sequence position reset) is *more* consequential than its speedup. When the contract holds, packing is a free 2–3×. When any part of the contract breaks, packing silently contaminates SFT and downstream evals look "mysteriously" worse than the unpacked baseline.

The framing in ch-30 is deliberate: packing is not just about bandwidth, it is the first place in the SFT stack where the tokenizer's abstraction leaks into attention kernels and you must reason about both layers at once.

---

## The attested padding fraction — why we pack at all

From `sequence-packing.md`, abstract:

> Up to 50% (and in extreme cases 89%) of tokens in BERT/GLUE-style fine-tuning are padding.

Modern SFT datasets are worse. Instruction datasets have long-tailed length distributions: median ~300 tokens, 95th percentile ~2000, `L_max` typically 2048 or 4096. Without packing, every short sample gets padded to `L_max` and the attention kernel wastes compute on padding tokens that contribute nothing to the loss (their labels are `-100`). Packing is the mechanical fix.

---

## The three-field packing contract — quoted verbatim

From `sequence-packing.md`, §Mechanics of a packed block:

> Given n short sequences s_1, …, s_n with lengths L_1, …, L_n (Σ L_i ≤ L_max), concatenate along the sequence axis:
> `packed = [s_1 | s_2 | … | s_n | PAD]`
> with:
> - `cu_seqlens = [0, L_1, L_1+L_2, …, Σ L_i]` — cumulative start offsets.
> - `position_ids` reset to 0 at each sequence boundary.
> - Attention mask: token t in sequence i can attend to tokens in sequence i only, with causal structure inside.

These three fields — `cu_seqlens`, `position_ids`, attention mask — are the contract. Ch-30's §4 lifts them verbatim into the `packed_batch.py` code block.

## Notice: the FlashAttention varlen API is the contract made practical

From `sequence-packing.md`, §FlashAttention varlen interface:

> Modern implementations use `flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen)` which computes block-diagonal causal attention **without materializing the full (L_max × L_max) mask**. Memory: O(sum L_i) not O(L_max²).

This is the key. Naive implementations of "block-diagonal mask" allocate an `L_max × L_max` boolean and pass it to dense attention. That costs `L_max²` memory *per head*; at `L_max = 4096` and 32 heads × 32 layers, you are materialising 10+ GB just for the mask. FlashAttention varlen avoids the allocation entirely — it reads `cu_seqlens` and branches inside the kernel.

Practical consequence: you must call `flash_attn_varlen_func`, not `flash_attn_func`. The latter is the dense API and ignores `cu_seqlens`. [[packed-vs-unpacked-ablation]] lists this as failure mode #4.

---

## SPFHP — the bin-packing algorithm that actually gets used

From `sequence-packing.md`, §SPFHP algorithm:

> 1. Compute length histogram of dataset.
> 2. Greedy: iterate over bins, fill each with the longest remaining sequence first; for each subsequent fit, pick the longest sequence that still fits (shortest gap).
> 3. Achieves near-optimal packing ratio (>99% fill) in O(N log N).

Notice what SPFHP does *not* do: it does not shuffle within a pack. The order of sub-sequences in a block is deterministic given the length histogram. This matters for reproducibility — two runs with the same seed and same data will produce the same packed blocks, and so the same loss trajectory. If your SFT run is non-deterministic, the non-determinism is not in the packing; it is in the batching-across-packs or in the optimizer.

NNLSHP (the non-negative least squares variant) solves the bin-packing problem more globally but is rarely used in production SFT stacks — SPFHP's 99% fill ratio is good enough and its linear-time complexity is a big win on large mixes.

---

## The four failure modes — from [[packed-vs-unpacked-ablation]]

The ablation counterpart documents four specific ways packing breaks, and ch-30's §4 lifts them as a numbered list:

> 1. **Missing block-diagonal mask** — tokens in sub-sequence 2 attend to sub-sequence 1 → cross-document leakage → subtle quality drop on multi-turn evals.
> 2. **Un-reset position IDs** — sub-sequence 2 sees positions L_1..L_1+L_2 instead of 0..L_2 → RoPE is effectively position-shifted.
> 3. **Label mask not re-applied per sub-sequence** — prompt tokens of sub-sequence 2 contribute to loss → looks like packed is worse, really a masking bug.
> 4. **Using flash_attn_func (dense) instead of flash_attn_varlen_func** — no mask → silent contamination.

Failure 2 is the most pedagogically interesting. RoPE encodes position as rotation angles; position 0 is the identity rotation, position L_1 is already far along the rotation curve. If sub-sequence 2 starts at position L_1 instead of 0, the model sees its first token as though it were mid-document. At inference the model generates starting from position 0, so there is a train/test mismatch even though no token ever leaked across boundaries.

---

## The diagnostic — how to check the contract holds

From [[packed-vs-unpacked-ablation]], §Diagnostic procedure:

> 1. Train a 100-step unpacked baseline; record train loss curve and first-batch logits.
> 2. Train a 100-step packed run with identical data and seed; compare loss curves.
> 3. Differences > 0.01 nats at matching step indicate a mask/pos-ID bug, not a "packing hurts" phenomenon.

This is the SFT-axis equivalent of Karpathy's "overfit one batch" sanity check. The 0.01 nat threshold is tight because the mathematical equivalence is exact when the contract holds; any gap is a bug.

---

## The throughput-model formula

From `sequence-packing.md` / [[packed-vs-unpacked-ablation]]:

```
speedup ≈ L_max / avg(L_i)
```

Ch-30's HTML companion implements this directly, with a realised-vs-raw discount for FlashAttention overhead and memory-bandwidth ceiling. The typical SFT-mix pair `L_max = 4096, avg(L_i) = 600` → raw 6.8×, realised ~3×. That matches Tülu-3's attested "2.5× throughput, no quality delta."

---

## Connections

- [[excerpts/loss-masking-regimes]] — label mask is the third of the three fields in the contract.
- [[excerpts/tulu-3-sft-recipe]] — the 2.5× throughput claim lives there; this excerpt is the mechanism.
- [[ch-30]] — §4 is entirely built on this source.
- [[ch-36]] (SFT lab) — the diagnostic becomes a hard gate: the lab cannot launch until the 100-step packed vs unpacked loss curves match within 0.01 nats.
