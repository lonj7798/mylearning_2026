---
chapter: ch-08
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/loss-masking-prompt.md
source_url: https://arxiv.org/abs/2405.14394
created_at: "2026-04-23"
---

# Excerpt: Loss Masking — the masking-line contract ch-08 unit-tests

**Source library:** `wiki/raw-data/llm-training/papers/loss-masking-prompt.md`
**Anchor paper:** Shi et al. 2024, "Instruction Tuning With Loss Over Instructions"; canonical practice per Alpaca (Taori 2023), InstructGPT (Ouyang 2022), HF Alignment Handbook.

---

## Why this source anchors ch-08 §3 and the masking unit test

Ch-08 names masking as silent-failure line #1 (red band in [figures/trainer-map.html](../figures/trainer-map.html)). This source is the formal ablation that says: response-only masking is *strictly better* than full-sequence loss across typical SFT datasets, and full-sequence is only competitive in the tiny-data regime. The lab treats that finding as settled and spends its attention on verifying the mask is actually applied — because "the mask looked right in the config" is exactly the failure mode of [[karpathy-training-neural-net-recipe]]'s "silent abstraction."

---

## The attested masking snippet — the two lines the unit test runs

From the source (lines 46-52), the Python sketch that every SFT trainer in the field reimplements:

```python
labels = input_ids.clone()
labels[:prompt_len] = -100  # mask prompt
loss = F.cross_entropy(logits[..., :-1, :].reshape(-1, V),
                       labels[..., 1:].reshape(-1),
                       ignore_index=-100)
```

Three load-bearing details:

1. **`labels[:prompt_len] = -100`** — the sentinel value PyTorch's `cross_entropy` skips. Any other value (0, the pad-token id, -1) is a silent bug: `cross_entropy` will compute loss and produce nonzero gradient. The default `ignore_index` is `-100`; deviating from it is a documented source of "mask that isn't actually masking."
2. **`logits[..., :-1, :]` paired with `labels[..., 1:]`** — next-token prediction offset. Off-by-one here is the second silent bug: the model is trained to predict token `t` from token `t`, not token `t+1`. Loss looks normal (~0.1 after a few hundred steps, because the model just copies).
3. **`reshape(-1, V)`** — the mask and the logits must be in the same linearized order. Swapping `:-1` and `1:` between them produces a wrong correspondence; gradient is still computed but against the wrong labels.

Ch-08's `masking_unit_test.py` is the defensive check against (1) — it asserts that `embed_tokens.weight.grad` has zero mass on the prompt-token rows, which can only be true if `labels[:prompt_len] == -100` actually made it through to `cross_entropy`'s `ignore_index`.

---

## The multi-turn rule — the bug TRL's collator is supposed to prevent

From the source (lines 38-43):

> For a conversation with turns `[u_1, a_1, u_2, a_2, …, u_k, a_k]`:
> - Mask **all** user turns.
> - Mask **all** prior assistant turns (a_1..a_{k−1}) — they are part of the prompt when generating a_k.
> - Train on a_k tokens only.

This is the silent-failure case ch-08 warns about in §3: *"multi-turn mismatch. Prior assistant turns must also be masked. If only the very first user turn is masked, the model gradient-matches past assistant outputs — turning SFT into a 'copy yourself' task."* The model's own past outputs become self-distillation targets; the distribution shift cascades; eval looks fine on one-turn prompts and silently degrades on multi-turn.

`DataCollatorForCompletionOnlyLM` in TRL implements the "mask everything before the last assistant turn" rule by scanning for the instruction template string (`<|im_start|>assistant` or equivalent) and masking everything before the *last* occurrence. This is why ch-08's §Deliverables requires `chat_template_check.py` to run first — if the template string does not appear in the rendered batch, the collator silently masks nothing.

---

## The per-turn unroll variant — why some trainers look different

From the source (line 44):

> Per-turn-training variant: unroll the conversation k times, each time masking through a_{i−1} and training on a_i → k× more data but identical loss value.

Some trainers (including some `open-instruct` configs per [[allenai-tulu-sft-recipe]]) use this variant: each k-turn conversation becomes k training examples, each masking through the prior assistant turn and training on one assistant span. The *total loss value* is identical to the "train on last turn only" variant — but the *gradient* differs because the data distribution over assistant positions is uniform rather than last-heavy.

For ch-08 this matters as a memo caveat: if the learner's trainer uses per-turn unroll, the masking unit test must still pass (embed gradient zero on all non-target tokens). The mask layout is the same per example; the per-example count differs.

---

## Why not upweight the prompt — the ablation the lab does not repeat

From the source (line 58):

> Upweighting (e.g., loss = α·L_prompt + L_response with α < 1) gives modest gains in some ablations (Shi 2024), but the response-only baseline dominates across most dataset sizes and is simpler.

Ch-08's lab does not explore this knob. The reason is operational: every extra loss weight is a new silent-failure surface (did α make it through the config? is α applied inside or outside gradient accumulation?). The lab is buying diagnostic visibility, not a 0.3% benchmark gain. If a learner wants to explore prompt upweighting in Track 3 (SFT-at-scale) they do it *after* the masking unit test is green.

---

## The packing × masking interaction — the subtle ch-04 bug

From the source (line 54-55):

> In a packed block, every sub-sequence has its own (prompt, response) split → the label mask must be reset per sub-sequence. Incorrect packing + masking is a common bug that silently degrades SFT.

This is the intersection of ch-08's red and amber silent-failure lines. A packed block of 4 sub-sequences requires 4 prompt-masking operations, not 1. `DataCollatorForCompletionOnlyLM` handles this by scanning for the template string at every sub-sequence boundary as identified by `cu_seqlens` — but if `cu_seqlens` is wrong (the packing bug), the masking is also wrong, in a correlated way.

Ch-08's diagnostic ordering — `chat_template_check.py` → `masking_unit_test.py` → `packing_unit_test.py` — is designed so these two failure modes get isolated. Masking test passes on single-sub-sequence batches even if `cu_seqlens` is broken; packing test requires multi-sub-sequence batches and asserts cross-contamination delta. Run in the prescribed order and a failure points to a specific line.

---

## Connections

- [[excerpts/sequence-packing]] — the sub-sequence boundaries the mask resets on.
- [[excerpts/hf-alignment-handbook]] — `train_on_response_only=True` is this paper's algorithm behind one flag.
- [[excerpts/karpathy-training-neural-net-recipe]] — "neural net training fails silently" is exactly why this paper's unit test matters.
- [[ch-04]] — concept-level intro to masking; ch-08 is the lab.
- [[ch-08]] — §3 (concept map), §Deliverables (masking unit test), figures/trainer-map.html ("Masking collator" node).
