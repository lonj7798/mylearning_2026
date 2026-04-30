---
chapter: ch-05
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/loss-masking-prompt.md
source_url: https://arxiv.org/abs/2405.14394
created_at: "2026-04-23"
---

# Excerpt: Prompt-Masked SFT Loss under Gradient Accumulation and FSDP ReduceScatter

**Source:** Loss masking for instruction SFT — canonical practice across Taori 2023 (Alpaca), Ouyang 2022 (InstructGPT), HF Alignment Handbook; formal ablation in Shi 2024 "Instruction Tuning With Loss Over Instructions."
**Year:** 2023–2024
**arXiv (Shi 2024):** 2405.14394

**This excerpt focuses on the distributed-correctness surface: how loss masking interacts with gradient accumulation and FSDP's ReduceScatter on heterogeneous token counts.** The single-GPU semantics are covered in ch-01 / ch-04.

---

## The standard SFT loss — and the denominator that matters

Quoted from the Technical Details:

> "For a conversation `(prompt p_{1:T_p}, response y_{1:T_y})`:
> `L_SFT(θ) = −(1 / T_y) Σ_{t=1..T_y} log π_θ(y_t | p, y_<t)`
> Prompt tokens' labels are set to `-100` (ignore_index in PyTorch CE) so their gradient contribution is zero."

```math
L_{\text{SFT}}(\theta) = -\frac{1}{T_y} \sum_{t=1}^{T_y} \log \pi_\theta(y_t \mid p, y_{<t})
```

Single-GPU: the `1 / T_y` denominator is exact — the sum runs over every response token, and `T_y` is the count of unmasked positions. PyTorch's `F.cross_entropy(ignore_index=-100, reduction="mean")` implements this literally: it sums `log π_θ` over non-masked positions, then divides by the non-masked count.

**The distributed problem.** Under FSDP with data-parallel size N, the "effective global batch" contains N micro-batches. If each rank computes `L_SFT` locally with `reduction="mean"`, then ReduceScatter averages N *per-rank means*:

```math
L_{\text{global}} = \frac{1}{N} \sum_{i=1}^{N} \frac{1}{T_y^{(i)}} \sum_{t} \log \pi_\theta(y_t^{(i)})
```

When `T_y^{(i)}` varies across ranks — which is the default under packing — this is **not** equal to the true global mean:

```math
L_{\text{true}} = -\frac{1}{\sum_i T_y^{(i)}} \sum_{i, t} \log \pi_\theta(y_t^{(i)})
```

The divergence is O(variance of `T_y^{(i)}` / mean `T_y^{(i)}`). For packed SFT this can easily be 20–30%. Gradients inherit the same distortion: ranks with few response tokens get disproportionate weight.

---

## The fix — sum-reduce locally, divide globally

The correct distributed pattern (standard in Alignment Handbook / TRL / torchtitan):

```python
# per-rank
loss_sum  = F.cross_entropy(logits, labels, ignore_index=-100, reduction="sum")
n_tokens  = (labels != -100).sum()

# accumulate across grad-accum steps
accum_loss_sum   += loss_sum
accum_n_tokens   += n_tokens

# at optimizer step, reduce across DP group
dist.all_reduce(accum_loss_sum,  op=dist.ReduceOp.SUM)
dist.all_reduce(accum_n_tokens,  op=dist.ReduceOp.SUM)
loss = accum_loss_sum / accum_n_tokens

loss.backward()   # gradients now scale with correct global-mean semantics
```

**Notice:** the FSDP ReduceScatter on parameter gradients happens inside `loss.backward()` and is orthogonal to the loss AllReduce. The loss AllReduce is a *scalar* reduction for logging; the gradient normalization must already be baked into `loss` before backward fires. If you log `loss` post-hoc from a per-rank mean, your loss curve is cosmetic — the actual training signal is whatever the gradient ReduceScatter produced.

---

## Gradient accumulation — the per-step divisor trap

Quoted Python sketch:

```python
labels = input_ids.clone()
labels[:prompt_len] = -100  # mask prompt
loss = F.cross_entropy(logits[..., :-1, :].reshape(-1, V),
                       labels[..., 1:].reshape(-1),
                       ignore_index=-100)
```

The naive gradient-accumulation wrapper divides by the accumulation step count:

```python
for micro_step in range(accum_steps):
    loss = compute_loss(batch[micro_step])
    loss = loss / accum_steps          # <-- naive averaging
    loss.backward()
optimizer.step()
```

With masked loss, this is wrong for the same reason as the per-rank case: each micro-step's `loss` is already a mean over its own `T_y^{(m)}` — micro-batches with more response tokens are under-weighted.

**The correct accum pattern:**

```python
accum_loss_sum = 0.0
accum_n_tokens = 0
for micro_step in range(accum_steps):
    logits = model(batch[micro_step])
    loss_sum = F.cross_entropy(logits, labels, ignore_index=-100, reduction="sum")
    n       = (labels != -100).sum()
    accum_loss_sum += loss_sum
    accum_n_tokens += n
(accum_loss_sum / accum_n_tokens).backward()
optimizer.step()
```

**Notice:** FSDP prefetches the next block's AllGather based on forward-pass progress; the above loop has a single `.backward()`, so FSDP can collapse all micro-step AllGathers into one. Calling `.backward()` inside the loop breaks that collapse and doubles the communication cost. Modern frameworks (torchtitan, TRL) handle this via `no_sync()` context on all but the last micro-step.

---

## Multi-turn chat masking — the distributed edge case

Quoted:

> "For a conversation with turns `[u_1, a_1, u_2, a_2, …, u_k, a_k]`: Mask **all** user turns. Mask **all** prior assistant turns (a_1..a_{k−1}) — they are part of the prompt when generating a_k. Train on a_k tokens only."

The per-turn-training variant:

> "Unroll the conversation k times, each time masking through a_{i−1} and training on a_i → k× more data but identical loss value."

**Distributed subtlety.** Unrolling multiplies the *logical* dataset size but introduces **per-example token-count variance**. Rank 0 might land a 5-turn example where only turn 5 (50 tokens) is trainable; rank 3 lands a 1-turn example where turn 1 (2000 tokens) is trainable. The 40× variance in `T_y^{(i)}` means the sum-reduce contract above is not optional; with per-rank mean, one rank's gradient dominates.

Under HYBRID_SHARD (intra-node FSDP, inter-node DDP), the DDP AllReduce is pre-normalized by the world size. If intra-node loss is sum-reduced but inter-node is mean-reduced, the cross-node normalization double-counts by `N_intra`. Pin the contract end-to-end: **all** reductions are `SUM`, final division by global token count happens once.

---

## The packed-sequence interaction — label reset per sub-sequence

Quoted:

> "In a packed block, every sub-sequence has its own (prompt, response) split → the label mask must be reset per sub-sequence. Incorrect packing + masking is a common bug that silently degrades SFT."

The distributed expression of this: rank-local `T_y^{(i)}` is computed over the *packed* block, not the individual sub-sequences. If the mask is applied only to the first sub-sequence's prompt (a common off-by-one bug when the packer's `cu_seqlens` offsets disagree with the masker's), the rank's denominator inflates by 2–3×, and the effective learning rate on that rank drops.

**Notice:** this bug is invisible on a single GPU — global loss still decreases. It manifests under FSDP as **increasing gradient-norm variance across ranks**, which in turn interacts with the global-norm gradient clip (see [[excerpts/gradient-clipping]]): ranks with inflated denominators get clipped less often, producing asymmetric updates.

---

## Why not upweight the prompt — and what changes under sharding

Quoted:

> "Upweighting (e.g., loss = α·L_prompt + L_response with α < 1) gives modest gains in some ablations (Shi 2024), but the response-only baseline dominates across most dataset sizes and is simpler."

Distributed angle: prompt-weighted loss requires **two** sum-reductions (one for prompt tokens, one for response tokens) and two token-count AllReduces. At FSDP scale this triples the scalar AllReduce traffic — negligible in bytes, but each AllReduce is a synchronization point and adds tail-latency. The 2025 norm is response-only precisely because it maintains the single-reduce contract.

---

## The clip-grad-norm coupling

The token-count heterogeneity described above compounds with the FSDP global-norm clip ([[excerpts/gradient-clipping]]). Each rank's local gradient norm is:

```math
\|g_i\|_2 = \left\| \frac{1}{T_y^{(i)}} \sum_t \nabla \ell_t \right\|_2
```

If `T_y^{(i)}` varies 10× across ranks, local norms vary accordingly. The correct global norm is an AllReduce over *squared* local norms:

```math
\|g_{\text{global}}\|_2 = \sqrt{\sum_{i=1}^{N} \|g_i\|_2^2}
```

FSDP's `model.clip_grad_norm_` implements this AllReduce; a hand-rolled per-rank `torch.nn.utils.clip_grad_norm_` does not. When combined with heterogeneous `T_y^{(i)}`, the hand-rolled version under-counts by `~ √N` and further distorts the relative weighting of high-token vs low-token ranks.

---

## Connections

- [[excerpts/fsdp-sft]] — the ReduceScatter happens on `∂L/∂θ`; the loss normalization must be correct before backward.
- [[excerpts/sequence-packing]] — packing creates the per-rank token-count variance that makes this contract load-bearing.
- [[excerpts/gradient-clipping]] — token-count variance distorts per-rank grad norms.
- [[excerpts/mixed-precision]] — loss is always computed in fp32 regardless of param_dtype.
- [[excerpts/adam]] — the optimizer sees gradients already correctly normalized; it is agnostic to the loss contract.
- [[ch-05]] — synthesis.
