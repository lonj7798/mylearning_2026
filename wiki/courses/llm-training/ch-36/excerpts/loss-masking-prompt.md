---
chapter: ch-36
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/loss-masking-prompt.md
source_url: https://arxiv.org/abs/2405.14394
created_at: "2026-04-23"
---

# Excerpt: Prompt-masking — the label invariant §3 of ch-36 unit-tests

**Source library:** `wiki/raw-data/llm-training/papers/loss-masking-prompt.md`
**Canonical references:** Shi et al. 2024 ("Instruction Tuning With Loss Over Instructions"), Taori et al. 2023 (Alpaca), Ouyang et al. 2022 (InstructGPT SFT), HF Alignment Handbook
**Venue:** arXiv 2405.14394 (formal ablation) + community practice
**Year:** 2023–2024

---

## Why this source anchors ch-36

If [[sequence-packing]] is the *attention-side* invariant (no key from sequence j attends to sequence i), loss-masking is the *label-side* invariant (no token outside the current assistant turn contributes to cross-entropy). The two are not independent: a packed block with correct attention mask but broken label mask trains fine, hits target throughput, and silently teaches the model to autocomplete user prompts. Ch-36's `§3 Loss-mask unit tests` is the label-side counterpart to `§4 Packed-attention unit tests`. Both must be green before a training step runs.

The canonical formulation — mask prompt tokens, train on completion only — has been community practice since Alpaca (2023), but Shi 2024 is the first systematic ablation proving that response-only strictly dominates full-sequence loss on MT-Bench and AlpacaEval for instruction datasets ≥ a few thousand examples. That ablation is what ch-36 uses to justify *not* running the full-loss axis in its 2×2×2 grid — full-loss is a known loss, not an open question.

---

## The canonical loss equation

Shi 2024 and every downstream handbook define the SFT objective as response-only cross-entropy. For a conversation `(prompt p_{1:T_p}, response y_{1:T_y})`:

```math
\mathcal{L}_{\text{SFT}}(\theta) = -\frac{1}{T_y} \sum_{t=1}^{T_y} \log \pi_\theta(y_t \mid p, y_{<t})
```

Contrast with the *full-sequence* loss Shi 2024 ablates against:

```math
\mathcal{L}_{\text{full}}(\theta) = -\frac{1}{T_p + T_y} \sum_{t=1}^{T_p + T_y} \log \pi_\theta(x_t \mid x_{<t})
```

**Notice:** the difference is two-fold. The *numerator* changes (only completion log-probs are summed). The *denominator* changes (normalization is over `T_y` not `T_p + T_y`). Implementing `ignore_index=-100` in PyTorch's `F.cross_entropy` handles both simultaneously — the `-100` positions are dropped from both the sum and the mean.

If you implement masking by zeroing `loss_per_token` manually but divide by `T_p + T_y`, the loss value is correct-shaped but ~2× smaller than it should be for a prompt-heavy batch, and your learning rate is effectively halved compared to anyone using `ignore_index`. Ch-36's `test_loss_mask_matches_reference` explicitly checks this: compute CE with `ignore_index=-100`, compute it manually with `.masked_select` and `.mean()`, assert `torch.allclose(a, b, atol=1e-6)`.

---

## Multi-turn: the combinatorics that the tests must pin down

Source lines 38–43 specify the multi-turn rule:

> For a conversation with turns `[u_1, a_1, u_2, a_2, …, u_k, a_k]`: mask **all** user turns, mask **all** prior assistant turns (a_1..a_{k−1}), train on a_k tokens only. Per-turn-training variant: unroll the conversation k times, each time masking through a_{i−1} and training on a_i → k× more data but identical loss value.

| Strategy | Label mask on turn k | Data expansion | Effective loss |
|----------|---------------------|----------------|----------------|
| Last-turn-only | `[mask_all_but_a_k]` | 1× | `−log π(a_k \| u_1..u_k, a_1..a_{k-1})` |
| Per-turn unrolled | `[mask_all_but_a_i]` for i=1..k, stacked | k× | `Σ_i −log π(a_i \| u_1..u_i, a_1..a_{i-1})` |
| Full-assistant | `[mask_only_user_turns]` | 1× | `Σ_i −log π(a_i \| ...)` — summed in one pass |

**Notice:** per-turn-unrolled and full-assistant compute the same loss *value* but differ in *batch composition*. Full-assistant puts all of `a_1..a_k` in one packed block with one forward pass; per-turn-unrolled replicates the prompt `k` times across the batch. Per-turn-unrolled is what ch-36's §3 tests target because it's the form the HF Alignment Handbook ships by default (`train_on_response_only=True` + no manual unrolling = full-assistant; unrolling is a dataset preprocessing step).

The ch-36 test `test_multiturn_mask_hides_prior_assistant` constructs a 3-turn conversation, applies the mask, and asserts that the indices of non-`-100` labels lie strictly inside `a_3`'s token span. This is the test that Taori's original Alpaca code did *not* have, and the reason several early forks trained on user-turn echoes for months before anyone noticed.

---

## The regime where full-loss helps — and why ch-36 ignores it

From Shi 2024's abstract:

> Response-only loss is strictly better on helpfulness benchmarks (MT-Bench, AlpacaEval) for typical instruction datasets; full-sequence loss can help in the *tiny-dataset / strong-base-model* regime where it acts as a mild continued-pretraining regularizer.

Ch-36 runs 100K SFT examples (full budget) or 20K (resource-constrained). Both sit comfortably above Shi 2024's "tiny-dataset regime" (< ~2K), so response-only is the answer and full-loss doesn't need to be a live axis. This is a deliberate design call: three axes × two levels = 8 runs, already at the edge of the 8×H100 / 8h compute budget. Spending two of those runs on a variable with a known answer is poor ablation hygiene per [[karpathy-training-neural-net-recipe]]'s "one axis per experiment" rule.

LIMA ([[lima]]) sits at 1K — exactly the regime where full-loss might help. If a future lab revisits LIMA-scale SFT, the full-loss axis should be re-introduced.

---

## Packing × masking: where the silent bug lives

Source line 55 — the one paragraph that motivates the entire `§3 + §4` test pairing:

> In a packed block, every sub-sequence has its own (prompt, response) split → the label mask must be reset per sub-sequence. Incorrect packing + masking is a common bug that silently degrades SFT.

Concretely: if the packer concatenates three conversations `C_1, C_2, C_3` into one block and the label-mask function computes `prompt_len` only from `C_1`, then `C_2`'s prompt and `C_3`'s prompt both fall in the "unmasked" region and contribute to the loss. The model learns to predict user turns — exactly the failure Shi 2024 warns against, reintroduced through a packing bug.

The ch-36 test `test_packed_labels_respect_subsequence_boundaries` constructs a packed block with three conversations, runs the label-masking function, and asserts:

1. For each sub-sequence `i`, labels in `[cu[i], cu[i] + prompt_len_i)` are `-100`.
2. For each sub-sequence `i`, labels in `[cu[i] + prompt_len_i, cu[i+1])` are the corresponding `input_ids[... + 1]` (shifted-by-one CE target).
3. The total count of non-`-100` labels equals `Σ_i response_len_i`.

This test is the intersection of [[sequence-packing]] and [[loss-masking-prompt]]. Neither source alone specifies it; the combination does. Ch-36 exists largely to enforce this intersection.

---

## Implementation sketch from the source (line 45–52)

```python
labels = input_ids.clone()
labels[:prompt_len] = -100  # mask prompt
loss = F.cross_entropy(
    logits[..., :-1, :].reshape(-1, V),
    labels[..., 1:].reshape(-1),
    ignore_index=-100,
)
```

**Notice three things** the snippet bakes in that ch-36's tests must verify on real data:

1. **Shift-by-one alignment.** `logits[..., :-1]` predicts `labels[..., 1:]`. If a packer accidentally strips the final EOS before the label computation, the last real token has no supervision signal and `response_len_i` off-by-one accumulates.
2. **Flatten then reduce.** `.reshape(-1, V)` turns a `[B, L, V]` tensor into `[B*L, V]`; `ignore_index=-100` is what keeps the reduction correct despite variable effective lengths. Manually masking after `F.cross_entropy(reduction='none')` is equivalent only if you also divide by the non-`-100` count.
3. **No explicit division by response length.** The `mean` reduction in `F.cross_entropy` handles it via the `ignore_index` count. This is why the loss value is *per-response-token*, not *per-batch-token*, and comparable across runs with different prompt/response length distributions.

---

## The prompt-upweighting dead-end (and why ch-36 skips it)

Source line 58:

> Upweighting (e.g., `loss = α·L_prompt + L_response` with α < 1) gives modest gains in some ablations (Shi 2024), but the response-only baseline dominates across most dataset sizes and is simpler.

Ch-36 treats `α = 0` (response-only) as the single choice and does not sweep `α`. This is a deliberate simplicity bet: one more sweep axis would triple the ablation grid for a known-small marginal gain. If the track's `sft-run-memo.md` surprise were "MT-Bench dropped unexpectedly on the packed+masked+NEFTune cell", re-introducing `α` would be a reasonable follow-up; it is not a first-order axis.

---

## Connections

- Attention-side companion: [[excerpts/sequence-packing]] — same silent-bug structure, different mask.
- Reference recipe that bakes both in: [[excerpts/hf-alignment-handbook]] — `train_on_response_only=True`.
- Baseline-thesis source for the 1K regime: [[excerpts/lima]] — where full-loss might actually help.
- Ablation-methodology source: [[packed-vs-unpacked-ablation]] — defines Failure Modes 1–4 that include label-mask bugs.
- Full-read chapter on masking: [[ch-32]].
- Full-read chapter on packing: [[ch-33]].
- Lab host: [[ch-36]] — `§3 Loss-mask unit tests`.
