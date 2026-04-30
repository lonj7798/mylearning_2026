---
chapter: ch-36
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/sequence-packing.md
source_url: https://arxiv.org/abs/2107.02027
created_at: "2026-04-23"
---

# Excerpt: Sequence packing — the invariants §4 of ch-36 unit-tests

**Source library:** `wiki/raw-data/llm-training/papers/sequence-packing.md`
**Artifact:** cu_seqlens offsets, per-sub-sequence position-id reset, block-diagonal attention mask, `flash_attn_varlen_func` API

---

## Why this source anchors ch-36

The SFT track is built around packing as the throughput primitive. Krell 2021 formalizes packing as bin-packing and proves *mathematical equivalence* to unpacked training when three invariants hold:

1. **cu_seqlens** cumulative offsets partition the packed tensor exactly: `cu = [0, L_1, L_1+L_2, ..., Σ L_i]`.
2. **Position IDs reset** to 0 at each sub-sequence boundary — RoPE sees each sub-sequence as starting from position 0.
3. **Attention mask is block-diagonal** — no token in sub-sequence i attends to any token outside `[cu[i], cu[i+1])`.

Ch-36's `§4 Packed-attention unit tests` exists specifically to test these three invariants as *code*. If any one of them is violated, the model trains — but it trains on a subtly different objective, and only the downstream eval reveals the bug.

---

## The varlen API — what ch-36's test_varlen_attention_zeros_cross_sample_leakage checks

From the source (lines 39–41):

> `flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen)` computes block-diagonal causal attention **without materializing the full (L_max × L_max) mask**. Memory: O(sum L_i) not O(L_max²).

Ch-36's test constructs two packed sub-sequences, runs `flash_attn_varlen_func` on the pack, runs it again on sub-sequence 1 alone, and asserts the packed output on positions `[0:L_1]` equals the solo output. If the equivalence does not hold, either `cu_seqlens` is wrong or the kernel was called without varlen — both are the Failure Modes 1 and 4 from [[packed-vs-unpacked-ablation]].

---

## Why cross-contamination is silent

From the source (line 51):

> Without masking, token t in sequence 2 can attend to sequence 1's tokens → the softmax's partition function leaks across documents → changes gradients even on sequence 1 tokens.

The gradients on sequence 1 change because the partition function `Z` in softmax now sums over sequence 2's keys too. Sequence 1's *forward output* changes by a small amount (often < 1e-3 in BF16), but over thousands of training steps this shifts the optimization target. The loss curve stays smooth; MT-Bench drops.

---

## What ch-36 keeps, changes, drops from Krell 2021

| Krell 2021 default | Ch-36 choice | Reason |
|--------------------|--------------|--------|
| BERT phase-2 pretraining | Llama-3.2-3B SFT | decoder-only + instruction data is where 2024+ packing lives |
| SPFHP bin-packing | TRL `DataCollatorWithPacking` | equivalent in practice; SPFHP runs under the hood of modern collators |
| Proved equivalence on GLUE | Verified via unit tests + 100-step loss-curve diff | ch-36 trusts the tests over benchmark replication |
| 2× throughput claim | 2.5–3× realized on 100K mix | attested in [[allenai-tulu-sft-recipe]] |
| Full block-diagonal mask materialized | FlashAttention-2 varlen (O(ΣL) memory) | 2024+ kernel; paper predates it |

---

## The one failure mode ch-36 converts into a unit test

Source lines 50–51 describe cross-contamination. Ch-36 converts this into `test_varlen_attention_zeros_cross_sample_leakage` — if the packed output on sub-sequence 1 differs from the solo output by more than `atol=1e-2`, the test fails, and the commit is blocked. This is the single invariant that matters most: mask correctness.

---

## Connections to the rest of the track

- **ch-33** — the full-read chapter on [[sequence-packing]]; read that before this lab.
- **ch-32** — [[loss-masking-prompt]] is the *label*-side companion: packing resets attention masks; masking resets labels.
- **[[packed-vs-unpacked-ablation]]** — the four failure modes ch-36's unit tests catch.
- **[[fsdp-sft]]** — the memory primitive; packing + FSDP is the 2024+ SFT backbone.
