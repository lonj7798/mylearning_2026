---
chapter: ch-04
course: llm-training
phase: read
excerpt_of: Shi et al. 2024 — "Instruction Tuning With Loss Over Instructions" (and canonical SFT practice from Alpaca / InstructGPT / Alignment Handbook)
source_url: https://arxiv.org/abs/2405.14394
created_at: "2026-04-23"
---

# Excerpt: Prompt-Masked vs Full-Sequence Loss in SFT

**Source:** `wiki/raw-data/llm-training/papers/loss-masking-prompt.md`
**Primary paper:** Zhengxiang Shi et al., "Instruction Tuning With Loss Over Instructions", 2024
**arXiv:** https://arxiv.org/abs/2405.14394
**Canonical practice also documented in:** Taori et al. 2023 (Alpaca), Ouyang et al. 2022 (InstructGPT), HF Alignment Handbook

---

## Bibliographic header

Unlike sequence packing (one paper, one theorem), loss masking is a **convention that crystallised across many works**. The canonical recipe — mask user tokens, train on assistant tokens only — appears without ablation in Alpaca and InstructGPT and is systematised in Shi 2024. The core claim:

> *"Response-only loss is strictly better on helpfulness benchmarks (MT-Bench, AlpacaEval) for typical instruction datasets; full-sequence loss can help in the tiny-dataset / strong-base-model regime where it acts as a mild continued-pretraining regularizer."*

This gives the guideline teeth: the default is response-only, and the exceptions are narrow.

---

## The core insight quote

From the raw-data notes:

> *"Computing cross-entropy over the whole sequence (prompt + completion) makes the model learn to generate the prompt, wasting capacity on a distribution it never produces at inference; masking prompt tokens out of the loss (label = -100) keeps the gradient focused on the response distribution."*

The argument is a train/test distribution-match argument. At inference, the model sees a prompt and generates a response — it never generates the prompt itself. If SFT trains the model to produce the prompt (full-sequence loss), that capacity is spent on a distribution that is never sampled from at serving time. Worse, prompts in instruction datasets are stylised (templated user turns) and *not* representative of the base model's pretraining distribution, so full-sequence loss drags the base distribution toward a stilted user-persona distribution that nobody asked for.

**Notice:** this is not merely an efficiency argument ("wasted gradient"). It is a *distributional* argument — full-sequence loss actively mis-shapes the model on a distribution it will never produce.

---

## Response-only loss — the canonical definition

From the raw-data notes:

> *"For a conversation (prompt p_{1:T_p}, response y_{1:T_y}):*
> *`L_SFT(θ) = −(1 / T_y) Σ_{t=1..T_y} log π_θ(y_t | p, y_<t)`*
> *Prompt tokens' labels are set to `-100` (ignore_index in PyTorch CE) so their gradient contribution is zero."*

In math:

```math
\mathcal{L}_{\text{SFT}}(\theta) = -\frac{1}{T_y} \sum_{t=1}^{T_y} \log \pi_\theta(y_t \mid p, y_{<t})
```

The denominator `T_y` — not `T_p + T_y` — is important: the loss is normalised by *response* length, not *total* length. If you average over the whole sequence while masking out the prompt, each batch's effective loss magnitude depends on the prompt-to-response ratio, which varies wildly across examples. Response-normalisation makes the loss scale independent of prompt length.

The full-sequence alternative, shown for contrast:

```math
\mathcal{L}_{\text{full}}(\theta) = -\frac{1}{T_p + T_y} \sum_{t=1}^{T_p + T_y} \log \pi_\theta(x_t \mid x_{<t})
```

trains on prompt tokens too. Shi 2024 Table 2 shows this is *worse* on MT-Bench across Alpaca, ShareGPT, and LIMA, with the penalty growing as dataset size grows.

---

## IGNORE_INDEX = -100 — the PyTorch mechanic

> *"Prompt tokens' labels are set to `-100` (ignore_index in PyTorch CE) so their gradient contribution is zero."*

The `-100` is not arbitrary — it is the default value of `ignore_index` in `torch.nn.CrossEntropyLoss` and `F.cross_entropy`. When a label equals `ignore_index`:

1. That position is **excluded from the numerator** of the loss (log-probability is not added).
2. That position is **excluded from the denominator** (the average is over non-ignored positions only).
3. The gradient with respect to the logits at that position is exactly `0`.

Mechanically, inside the CUDA kernel the implementation looks like:

```python
# conceptual inner loop
loss = 0.0
n = 0
for t in range(T):
    if labels[t] == -100:
        continue
    loss = loss - log_softmax(logits[t])[labels[t]]
    n = n + 1
return loss / n
```

The "gradient is exactly zero" property depends on this early-`continue`: it means `-100` positions contribute no term to `L`, so `∂L/∂logits[t] = 0`. If you tried to implement masking by multiplying the loss by a mask tensor after the fact, you would still pay the CUDA cost of computing `-log(softmax)` over the vocab for masked positions — `ignore_index` skips that work entirely.

**Notice:** `-100` was chosen because it is safely outside the range of any reasonable vocabulary index (vocab sizes are positive integers), so it cannot be confused with a real class. Any negative integer would work, but `-100` is hard-coded as the default in PyTorch.

---

## Multi-turn chat masking — the non-trivial case

From the raw-data notes:

> *"For a conversation with turns [u_1, a_1, u_2, a_2, …, u_k, a_k]: Mask **all** user turns. Mask **all** prior assistant turns (a_1..a_{k−1}) — they are part of the prompt when generating a_k. Train on a_k tokens only."*

Single-turn masking is mechanical — slice at the prompt-response boundary. Multi-turn introduces a subtle question: *which* assistant turns should contribute to the loss?

Option A (last-turn only):

```
mask: [u_1 a_1 u_2 a_2 ... u_k] + train: [a_k]
```

Train only on the final assistant turn. Prior assistant turns are treated as conditioning context — they are "given" and the model should not be trained to regenerate them. This is what Shi 2024 recommends and what the HF Alignment Handbook default implements.

Option B (per-turn-training unroll):

```
example_1: mask [u_1] + train [a_1]
example_2: mask [u_1 a_1 u_2] + train [a_2]
...
example_k: mask [u_1 a_1 ... u_k] + train [a_k]
```

Unroll the k-turn conversation into k separate training examples, each masking through the previous assistant turn. This produces k× more examples with identical per-token loss — it is equivalent to Option A on the final turn and adds supervision for intermediate turns.

From the raw-data notes:

> *"Per-turn-training variant: unroll the conversation k times, each time masking through a_{i−1} and training on a_i → k× more data but identical loss value."*

**Notice:** Option A underutilises earlier assistant turns; Option B costs k× compute. A pragmatic middle path — which TRL's `train_on_response_only` implements by default — is to mask all user turns but train on *every* assistant turn in a single pass (all assistant tokens across all turns, not just the last). This is neither A nor B; it is a cheap compromise that recovers most of B's benefit at A's cost.

---

## Implementation sketch

From the raw-data notes:

```python
labels = input_ids.clone()
labels[:prompt_len] = -100  # mask prompt
loss = F.cross_entropy(logits[..., :-1, :].reshape(-1, V),
                       labels[..., 1:].reshape(-1),
                       ignore_index=-100)
```

Two details in this snippet are easy to miss:

1. **The `[..., :-1, :]` slice on logits and `[..., 1:]` slice on labels** — this is the causal-LM shift. At position `t`, the logits at index `t` predict the token at index `t+1`. Slicing `logits[:-1]` and `labels[1:]` aligns them so `cross_entropy(logit_at_t, label_at_t+1)` computes `-log p(x_{t+1} | x_{<=t})`. Forgetting the shift is the oldest bug in causal-LM training.
2. **`labels[:prompt_len] = -100`** masks positions `0..prompt_len-1` of the label tensor. After the shift, this means the model is not trained to predict tokens `1..prompt_len` — i.e., not trained on the prompt tokens themselves. Exactly the intent.

---

## Interaction with packing — the per-sub-sequence reset

From the raw-data notes:

> *"In a packed block, every sub-sequence has its own (prompt, response) split → the label mask must be reset per sub-sequence. Incorrect packing + masking is a common bug that silently degrades SFT."*

Inside a pack `[s_1 | s_2 | ... | s_n | PAD]`, each `s_i` is its own `(prompt_i, response_i)` pair. The label mask must reflect every individual boundary:

```
labels = input_ids.clone()
for i, (start, prompt_end, end) in enumerate(boundaries):
    labels[start:prompt_end] = -100   # mask prompt of sub-sequence i
    # labels[prompt_end:end] left as input_ids (trained)
labels[pad_start:] = -100             # mask trailing pad
```

If you instead treat the pack as a single example and mask only `labels[:prompt_1_end]`, you inadvertently train on every user turn after the first. This is the "silent bug" the raw-data file warns about: the training loss looks plausible, the model even converges, but it has been contaminated with a gradient signal to regenerate prompts. See [[excerpts/sequence-packing]] for the packing mechanics this masking must compose with.

---

## Why not upweight the prompt instead?

From the raw-data notes:

> *"Upweighting (e.g., loss = α·L_prompt + L_response with α < 1) gives modest gains in some ablations (Shi 2024), but the response-only baseline dominates across most dataset sizes and is simpler."*

The obvious interpolation — weighted loss with `α ∈ (0, 1)` on prompt tokens — has been tried. The empirical finding is: the signal from prompt tokens is not *harmful* in the tiny-data regime (it acts as a weak continued-pretraining regulariser), but in the Alpaca-scale-and-above regime, any `α > 0` underperforms `α = 0`. The Occam answer: set `α = 0` and stop tuning.

---

## When full-sequence loss is actually OK

From the raw-data notes:

> *"For continued pretraining or multi-turn coherent text, full-sequence loss is fine."*

The response-only default is specific to **instruction** SFT. For:
- **Continued pretraining** on raw text (e.g., domain adaptation on medical papers): there is no prompt-response split. The whole sequence is the "response distribution". Full-sequence loss is correct.
- **Multi-turn coherent narrative / dialogue modelling** where the entire conversation is the target distribution (e.g., training a narrative model to produce full dialogues including user turns): full-sequence is correct.

The distinction is whether the prompt distribution at inference matches the prompt distribution at training. For instruction SFT it does not (inference prompts come from users, not from the dataset); for continued pretraining it does (both are raw text).

---

## Connections

- Packing layer that composes with this masking: [[excerpts/sequence-packing]]
- Reference implementation (TRL `train_on_response_only=True`): [[excerpts/hf-alignment-handbook]]
- Regulariser that stacks on top of response-only loss: [[excerpts/neftune]]
- Chapter synthesis: [[ch-04]]
