---
chapter: ch-30
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/loss-masking-prompt.md
source_url: https://arxiv.org/abs/2405.14394
created_at: "2026-04-23"
---

# Excerpt: Loss masking as the mechanical heart of SFT

**Source library:** `wiki/raw-data/llm-training/papers/loss-masking-prompt.md`
**Anchor paper:** Shi 2024 — "Instruction Tuning With Loss Over Instructions"
**Companion handbook:** [[hf-alignment-handbook]] (the `train_on_response_only=True` default)

---

## Why this source anchors ch-30

Loss masking is axis #2 of the five SFT design axes. It is the cheapest axis to get wrong and the cheapest to get right: one `labels[mask] = -100` line separates "the model learns to generate user prompts" from "the model learns to complete the assistant turn." The source is the clearest published statement of what that line does and when the default flips.

---

## The one-line implementation — quoted verbatim

From `loss-masking-prompt.md`, §Implementation (Python sketch):

```python
labels = input_ids.clone()
labels[:prompt_len] = -100  # mask prompt
loss = F.cross_entropy(logits[..., :-1, :].reshape(-1, V),
                       labels[..., 1:].reshape(-1),
                       ignore_index=-100)
```

Three invariants packed into these four lines:

1. **`-100` is not a magic number.** It is PyTorch `F.cross_entropy`'s default `ignore_index`. Tensorflow users use a mask multiplied into the loss; HuggingFace / TRL follow PyTorch.
2. **`labels = input_ids.clone()` before masking.** The model predicts the *next* token, so the targets are a shift-by-one of the inputs. Masking in-place on `input_ids` would break the attention computation.
3. **The shift-by-one `[..., :-1, :]` / `[..., 1:]`** — done inside the loss, not inside the tokenizer. The tokenizer's `input_ids` are the forward pass; the loss realigns them.

## Notice: the single formula separating response-only from full-sequence

From `loss-masking-prompt.md`, §Standard SFT loss:

> `L_SFT(θ) = −(1 / T_y) Σ_{t=1..T_y} log π_θ(y_t | p, y_<t)`
>
> Prompt tokens' labels are set to `-100` (ignore_index in PyTorch CE) so their gradient contribution is zero.

And §Full-sequence loss (not recommended for instruction SFT):

> `L_full(θ) = −(1 / (T_p + T_y)) Σ_t log π_θ(x_t | x_<t)`
>
> Trains the model to reproduce the user's prompt — never used at inference, wastes capacity.

The two differ only in which positions contribute to the sum. Shi 2024's ablation is a direct test: same model, same data, same optimizer, swap which formula is used. Response-only wins on MT-Bench and AlpacaEval across every dataset size they tested *except* LIMA-sized (≤ 1K) on a strong base, where full-sequence acts as a cheap continued-pretraining regulariser — the one place prompt leakage helps.

---

## The multi-turn rule — where ch-30's table comes from

From `loss-masking-prompt.md`, §Multi-turn chat masking:

> For a conversation with turns `[u_1, a_1, u_2, a_2, …, u_k, a_k]`:
> - Mask **all** user turns.
> - Mask **all** prior assistant turns (a_1..a_{k−1}) — they are part of the prompt when generating a_k.
> - Train on a_k tokens only.

This is the row of ch-30's per-regime loss-mask table for "multi-turn chat." The source also documents the per-turn-training variant:

> Per-turn-training variant: unroll the conversation k times, each time masking through a_{i−1} and training on a_i → k× more data but identical loss value.

Notice: the two variants give the same loss value but *different gradient statistics*. Unrolling makes early turns appear k times in the effective dataset, which is a non-trivial re-weighting. [[allenai-tulu-sft-recipe]] uses the final-turn-only variant; [[hf-alignment-handbook]] offers both via TRL's `SFTTrainer`.

---

## Packed-sequence interaction — the silent-bug class

From `loss-masking-prompt.md`, §Packed-sequence interaction:

> In a packed block, every sub-sequence has its own (prompt, response) split → the label mask must be reset per sub-sequence. Incorrect packing + masking is a common bug that silently degrades SFT.

Chapter 30's §4 (packing) cross-references this: the attention mask, position IDs, *and* label mask must all reset at `cu_seqlens` boundaries. Missing any of the three is a different failure mode:

- Missing attention mask → cross-document leakage (softmax partition).
- Missing position-ID reset → RoPE frequencies shifted.
- Missing **label** reset → prompt tokens of sub-sequence 2 contribute to loss.

The label failure is the hardest to catch because it does not change the loss *value* much — it only changes *what* the model learns. The diagnostic is to decode a packed batch and confirm each sub-sequence's prompt region has `-100` in its corresponding `labels` slot.

---

## Why not upweight the response?

From `loss-masking-prompt.md`, §Why not upweight the response:

> Upweighting (e.g., loss = α·L_prompt + L_response with α < 1) gives modest gains in some ablations (Shi 2024), but the response-only baseline dominates across most dataset sizes and is simpler.

Ch-30's "prompt-weighted" option in the HTML companion comes from here. It is a real option in the literature; it is the attested second-best. Simpler beats it. This is a small but important pedagogic point: the SFT stack has many knobs that *can* move metrics slightly, and the practitioner's discipline is to keep the count of active knobs minimal so that ablations are interpretable.

---

## What ch-30 keeps, changes, drops

| Source default | Ch-30 position | Reason |
|----------------|----------------|--------|
| Response-only loss | Same — framed as the mask axis's default | Attested to dominate; used in both Zephyr and Tülu-3 |
| `ignore_index = -100` | Same — quoted in ch-30 §2 | PyTorch API, TRL convention |
| Mask-all-prior-turns multi-turn | Same — table row for multi-turn chat | Attested; matches [[hf-alignment-handbook]] `train_on_response_only` |
| Full-sequence as a contrarian | Listed as option, warned against at scale | Shi 2024 ablation; tiny-dataset exception noted |
| Prompt-weighted as third way | Listed as option with warn badge | Modest gains, extra knob — discipline axis |

---

## Connections

- [[excerpts/sequence-packing-contract]] — the packing partner of this excerpt; the "reset per sub-sequence" contract.
- [[excerpts/chat-template-matrix]] — the template decides *which* token spans correspond to which role, which in turn decides the mask.
- [[ch-30]] — §2 and §6 both depend on this source.
- [[ch-31]] (iterative SFT↔RL bridges) — rejection-sampling SFT is "response-only on rejection-sampled responses"; same mask.
- [[ch-32]] (reasoning SFT) — extends the table with the `<think>` column, but the loss-mask primitive is the same `-100` technique.
