---
chapter: ch-08
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/sequence-packing.md
source_url: https://arxiv.org/abs/2107.02027
created_at: "2026-04-23"
---

# Excerpt: Sequence Packing — the packing-line unit test ch-08 demands

**Source library:** `wiki/raw-data/llm-training/papers/sequence-packing.md`
**Paper:** Krell, Kosec, Perez, Fitzgibbon 2021, "Efficient Sequence Packing without Cross-Contamination"

---

## Why this source anchors ch-08 §2 and §Deliverables

Ch-08 names *packing* as one of the three silent-failure lines in a modern trainer. This paper is where the silent failure was first characterized and the fix (block-diagonal attention indexed by `cu_seqlens`) was proven mathematically equivalent to unpacked training. The lab's packing unit test — run once with and once without the block-diagonal mask, require loss delta > 1e-3 — is a direct consequence of this paper's "cross-contamination" argument.

---

## The `cu_seqlens` contract — what TRL must thread through attention

From the source (lines 32-37):

> Given n short sequences s_1, …, s_n with lengths L_1, …, L_n (Σ L_i ≤ L_max), concatenate along the sequence axis:
> `packed = [s_1 | s_2 | … | s_n | PAD]`
> with:
> - `cu_seqlens = [0, L_1, L_1+L_2, …, Σ L_i]` — cumulative start offsets.
> - `position_ids` reset to 0 at each sequence boundary.
> - Attention mask: token t in sequence i can attend to tokens in sequence i only, with causal structure inside.

Three artifacts must flow from the dataloader to the attention kernel: (1) the packed `input_ids`, (2) `cu_seqlens`, (3) `position_ids` with per-sub-sequence resets. Any one missing and the trainer silently ships cross-contaminated attention.

In TRL's `ConstantLengthDataset` / `DataCollatorWithPacking`, the dataloader assembles (1) and (2); RoPE position-id handling for (3) is the caller's responsibility when the model config does not reset positions automatically. Ch-08's "§2 packing" concept-mapping flags exactly this flow.

---

## The FlashAttention varlen API — the call site ch-08's map highlights

From the source (line 40):

> Modern implementations use `flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen)` which computes block-diagonal causal attention **without materializing the full (L_max × L_max) mask**. Memory: O(sum L_i) not O(L_max²).

Ch-08's companion HTML ([figures/trainer-map.html](../figures/trainer-map.html)) pins this as the `varlen attention` node on the "packing line" (amber). The call is the only place where the block-diagonal structure actually gets enforced; upstream it is just metadata, downstream it is just a scalar loss.

If `cu_seqlens` is derived from a *padded* batch rather than the *packed* batch (a common refactor bug when switching between padded fine-tuning and packed SFT), the varlen kernel runs but treats the whole block as one sequence. Attention leaks across documents. Loss drops faster than it should for the first few hundred steps, then plateaus below the true optimum. The model is mildly degraded, never catastrophically so — classic silent failure.

---

## Why cross-contamination matters — the mathematical identity the unit test exploits

From the source (line 51):

> Without masking, token t in sequence 2 can attend to sequence 1's tokens → the softmax's partition function leaks across documents → changes gradients even on sequence 1 tokens. The block-diagonal mask restores exactness.

Ch-08's packing unit test leverages this directly. Construct a batch with at least two sub-sequences. Run forward twice:

```python
# test 1: correct — block-diagonal mask
out_packed   = model(input_ids, cu_seqlens=cu_seqlens, position_ids=packed_pos)
loss_packed  = ce_loss(out_packed.logits, labels)

# test 2: broken — no block-diagonal; attend across all tokens
out_full     = model(input_ids, cu_seqlens=torch.tensor([0, L_total]),
                     position_ids=torch.arange(L_total))
loss_full    = ce_loss(out_full.logits, labels)

assert abs(loss_packed.item() - loss_full.item()) > 1e-3
```

If the delta is < 1e-3 with a multi-sub-sequence batch, one of three things is true: (a) `cu_seqlens` is ignored in your model wrapper; (b) your batch has only one sub-sequence (rebuild it); (c) position_ids are not being reset and attention is effectively contaminated in both branches. Each case is a real TRL bug I have seen in the wild.

---

## The padding argument — why packing is non-negotiable for SFT

From the source (Abstract, line 15):

> Up to 50% (and in extreme cases 89%) of tokens in BERT/GLUE-style fine-tuning are padding.

Instruction datasets are worse. UltraChat's length histogram is heavily right-skewed: median ~512 tokens, max 8K. Padding to max_seq_length = 4096 wastes ~60–70% of compute on `<pad>` tokens whose labels are `-100`. Packing reclaims this: [[allenai-tulu-sft-recipe]] reports "Packing: 2.5× throughput, no quality delta."

For ch-08's full-budget path this is the difference between a 15-minute 100-step run and a 40-minute one. For the resource-constrained path on a single 8 GB GPU it is the difference between fitting and OOMing.

---

## The labels/-100 interaction with packing — the subtle inner bug

From the source (line 54):

> Labels for padding tokens must be set to -100 (ignore_index); loss masks for prompt tokens (see [[loss-masking-prompt]]) are applied per sub-sequence inside the pack.

This is the line that connects ch-08's packing line (amber) and masking line (red). A packed block of 4096 tokens containing 5 sub-sequences has 5 prompt-response boundaries; the masking collator must find all 5 and zero out 5 prompt spans. A naive implementation that only masks the first prompt span silently trains on four of the five prompts. Loss looks normal; the model overfits on user-style prefixes.

Ch-08's masking unit test catches this by asserting the embed-weight gradient is zero for *every* masked token, not just the first span.

---

## What the paper does not address — where ch-08 adds value

The paper is a throughput paper; it proves the math and reports 2× on BERT phase 2. It does not discuss:

- **FSDP interaction.** Packing + FSDP FULL_SHARD is the 2024–25 standard (per [[fsdp-sft]] §"Typical SFT recipe"), but the paper predates FSDP's public release.
- **Chat-template boundaries.** The paper assumes BERT-style fixed sub-sequence starts; modern SFT has to find response boundaries inside rendered chat templates (ch-04 / [[loss-masking-prompt]]).
- **Per-shard loss logging.** The sub-sequence structure is exactly the right granularity for [[ch-06]] §4's per-shard loss breakdown — this paper does not connect the two.

Ch-08's concept-mapping §2 takes the algorithm as given and shows where it lands inside `SFTTrainer`; this excerpt is the bridge between "the paper is right" and "here is the line of code that can be wrong."

---

## Connections

- [[excerpts/loss-masking-prompt]] — the per-sub-sequence label masking rule, paired with packing always.
- [[excerpts/hf-alignment-handbook]] — `packing=True` in TRL is this paper's algorithm + its unit test, wired to the SFT loop.
- [[excerpts/fsdp-sft]] — packing's throughput gain is what makes the FSDP recipe feasible on 8 × H100 at 7B–70B.
- [[ch-04]] — introduces the packing mechanics at concept level; ch-08 is the lab that unit-tests them.
- [[ch-08]] — §2 (concept map), §Deliverables (packing unit test), figures/trainer-map.html (the `varlen attention` node).
