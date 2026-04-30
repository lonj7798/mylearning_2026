---
chapter: ch-07
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/loss-masking-prompt.md
source_url: https://arxiv.org/abs/2405.14394
created_at: "2026-04-23"
---

# Excerpt: Prompt masking — the off-by-one that silently corrupts SFT

**Source library:** `wiki/raw-data/llm-training/papers/loss-masking-prompt.md`
**Paper:** Shi et al. 2024, *"Instruction Tuning With Loss Over Instructions"*; plus canonical practice from Alpaca (Taori 2023), InstructGPT (Ouyang 2022), HF Alignment Handbook.

---

## Why this source anchors ch-07

Every §4a item in ch-07 comes from this source. Prompt masking is the single line of SFT code most often written slightly wrong, and the resulting bug is quantitatively small per step (loss degraded by 0.5–2% absolute) but *systematically* miscalibrated on every step of training. The run completes, the loss curve looks healthy, the MT-Bench score is 1–2 points lower than the baseline — and because you never instrumented "fraction of masked tokens per batch," you never find the bug.

The source's abstract gives the formal ablation:

> *"Response-only loss is strictly better on helpfulness benchmarks (MT-Bench, AlpacaEval) for typical instruction datasets; full-sequence loss can help in the tiny-dataset / strong-base-model regime where it acts as a mild continued-pretraining regularizer."*

So the correct default is hard-constrained: mask the prompt, train only on the response. Any off-by-one is a partial inclusion of prompt tokens in the loss, which pushes toward the full-sequence regime inconsistently — worse than either pure response-only or pure full-sequence.

---

## The canonical masking code — and the three wrong forms

From the source (Technical Details):

```python
labels = input_ids.clone()
labels[:prompt_len] = -100  # mask prompt
loss = F.cross_entropy(logits[..., :-1, :].reshape(-1, V),
                       labels[..., 1:].reshape(-1),
                       ignore_index=-100)
```

Notice: the mask is applied on the *unshifted* labels before the `[..., 1:]` slice. The slicing shifts by one position, so a token at original index `i` becomes the target at shifted index `i-1`. Masking `labels[:prompt_len]` on the unshifted array means that after shifting, positions 0 through `prompt_len - 1` of the target sequence are masked.

The three wrong forms ch-07 §4a enumerates:

```python
# WRONG #1 — mask applied post-shift
labels_shift = labels[..., 1:].clone()
labels_shift[:, :prompt_len] = -100         # off by one:
                                            # position prompt_len-1 not masked

# WRONG #2 — prompt_len computed on packed block
labels[:prompt_len] = -100                  # in a packed block,
                                            # only masks the first pack's prompt

# WRONG #3 — mask applied but not sliced
labels[:prompt_len] = -100
loss = F.cross_entropy(logits.reshape(-1, V), labels.reshape(-1),
                       ignore_index=-100)   # no shift — labels[i] is the target
                                            # for logits[i], which is the wrong alignment
```

Wrong #1 leaks the final prompt token into the loss. That token's label (post-shift) is the response's first token; the gradient flows back through what should be a masked position. Net effect: the model receives a tiny but consistent signal to predict the first response token from the immediate predecessor — benign on single-turn prompts, actively harmful on multi-turn where it collapses the user↔assistant boundary.

Wrong #2 is the packing-specific version ch-07 §3's dead-pipeline branch covers: `prompt_len` inside a packed block is valid only for the *first* sub-sequence; each subsequent sub-sequence has its own prompt-response split whose boundaries are encoded in `cu_seqlens` (see [[excerpts/sequence-packing]]). A packed SFT trainer that computes a single `prompt_len` per batch masks only the first prompt in the pack and trains on every other sub-sequence's prompt as if it were a response.

Wrong #3 is the alignment bug: CE without the shift computes `loss(logits_i, labels_i)`, which asks the model to predict token `i` from its own embedding. This is the "teacher-forcing off" bug; the loss looks plausibly small because embeddings are close to their own identity under any reasonable init, but the model learns nothing about next-token prediction.

---

## Multi-turn masking — the rule that grows with k

From the source:

> *"For a conversation with turns `[u_1, a_1, u_2, a_2, …, u_k, a_k]`:*
> *- Mask **all** user turns.*
> *- Mask **all** prior assistant turns (a_1..a_{k−1}) — they are part of the prompt when generating a_k.*
> *- Train on a_k tokens only."*

Notice: *all* prior assistant turns are masked, not just user turns. A common mis-reading of the rule is "mask user turns, train on every assistant turn." That is wrong — it trains the model to produce assistant turn 1 conditioned on user turn 1, then assistant turn 2 conditioned on user turn 2, *without the intervening assistant turn 1 in the label set*. But the model still sees `a_1` in the input, so the gradient signal is "produce `a_2` given `[u_1, a_1, u_2]`" which is what you want — and the source's "Per-turn-training variant" describes exactly this unrolling:

> *"Per-turn-training variant: unroll the conversation k times, each time masking through a_{i−1} and training on a_i → k× more data but identical loss value."*

The k× data-size multiplier matters for throughput but not for correctness. Both variants (mask-all-prior-assistant-except-last, or unroll-k-times) produce the same gradient in expectation.

---

## The packed-sequence interaction — ch-07 §4b and §4a combined

From the source (Packed-sequence interaction):

> *"In a packed block, every sub-sequence has its own (prompt, response) split → the label mask must be reset per sub-sequence. Incorrect packing + masking is a common bug that silently degrades SFT."*

This is the bridge between ch-07 §4a (prompt-masking off-by-one) and §4b (cross-sample attention leakage). Both bugs exist in the same piece of code — the packed-block collator — and both are invisible unless you explicitly test them. The unit test that catches both together, from ch-07 §4b:

```python
out_packed   = model(input_ids_packed, cu_seqlens=cu)
out_unpacked = torch.stack([model(s) for s in split_by_cu(input_ids_packed, cu)])
assert (out_packed - out_unpacked).abs().max() < 1e-4
```

Plus the mask-specific test from ch-07 §4a:

```python
decoded = tokenizer.decode(input_ids[labels != -100])
assert "<|user|>" not in decoded, "prompt leaked into response mask"
assert "<|system|>" not in decoded, "system leaked into response mask"
```

The second test is the 30-second version of the masking audit. If you read the decoded text and see a template marker, the mask is wrong.

---

## Why upweighting the prompt is a trap

From the source:

> *"Upweighting (e.g., loss = α·L_prompt + L_response with α < 1) gives modest gains in some ablations (Shi 2024), but the response-only baseline dominates across most dataset sizes and is simpler."*

Notice the rhetorical move: *simpler*. The engineering argument against upweighting is not that it's mathematically worse in some regimes — the paper admits it can help in the tiny-data/strong-base regime — but that the additional hyperparameter `α` is a new surface for bugs. Ch-07's organizing rule (Karpathy's "training fails silently") applies: any new knob is a new place for the wrong value to ship. `α = 0` (response-only) is the default that requires no tuning and no bug.

The *"Why not upweight the response?"* section in the source is really a warning: new knobs introduce new silent-failure modes.

---

## The `ignore_index = -100` convention

PyTorch's `F.cross_entropy` treats labels equal to `-100` as "compute no loss, no gradient." This convention is portable across every modern framework (HF Transformers, TRL, Axolotl, TorchTune, OLMo-core). A common SFT bug is to use a different sentinel value — `0`, `IGNORE_TOKEN_ID = vocab_size`, or a custom integer — and then silently train on prompt tokens because the sentinel isn't recognized.

The test: `(labels == -100).sum() > 0` must be true for any SFT batch. If every label is in `[0, vocab_size)`, you are training on the full sequence. Ch-07 §7's `active = (labels != -100).sum()` assertion catches this at step-of-origin.

---

## What to take from Shi 2024 / canonical masking practice for ch-07

1. **Mask prompt tokens to `-100` on the unshifted labels.** Then shift; the alignment is correct.
2. **Per-sub-sequence masking in packed blocks.** A single `prompt_len` per batch is wrong whenever packing is on.
3. **Multi-turn rule: mask all prior assistant turns, train only on the last one** (or unroll k times for k× throughput).
4. **Audit by decoding the unmasked tokens** — should be assistant text only, no template markers.
5. **`active_tokens > 0` assertion** catches the empty-batch / all-masked dead-pipeline variant (ch-07 §3).

---

## Connections

- [[excerpts/sequence-packing]] — the cross-sample attention leak side of §4; the two bugs live in the same collator.
- [[excerpts/karpathy-training-neural-net-recipe]] — "generalize a special case" and "visualize just before the net" map onto the decoded-labels audit.
- [[excerpts/llama-3]] — "loss on response tokens only" is explicit in the Llama-3 SFT recipe; the NLL-stabilizer DPO trick is ch-07 §6-adjacent.
- [[excerpts/olmo-2]] — Tulu-3-style SFT inherits the response-only rule from this source.
- [[ch-07]] — §3 (dead-pipeline all-masked variant), §4a (off-by-one forms), §7 (`active > 0` assertion in the checklist).
