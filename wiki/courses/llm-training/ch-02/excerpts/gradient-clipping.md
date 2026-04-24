---
chapter: ch-02
course: llm-training
phase: read
excerpt_of: "Pascanu, Mikolov, Bengio — On the difficulty of training RNNs (2013)"
source_url: https://arxiv.org/abs/1211.5063
created_at: "2026-04-23"
---

# Excerpt: Gradient Clipping — the precision angle

**Paper:** *On the difficulty of training Recurrent Neural Networks*
**Authors:** Razvan Pascanu, Tomas Mikolov, Yoshua Bengio
**Venue:** ICML 2013
**arXiv:** [1211.5063](https://arxiv.org/abs/1211.5063)

This excerpt treats clipping as a **numerical-stability primitive**, not an optimization trick. The question is: once you decide to clip the global gradient norm, what precision must the norm computation itself use, and how does clipping compose with loss scaling and FSDP gradient reduction? (The optimization theory — why clipping enables higher LR without divergence — is covered in ch-01.)

---

## 1. The clipping operation (§4 of the paper)

From the paper's Algorithm 1:

```math
\hat{g} \gets g
\text{if } \|g\| \ge \text{threshold:}
\quad \hat{g} \gets \frac{\text{threshold}}{\|g\|} \cdot g
```

Two properties make this the right choice:

1. **Direction preserving.** The rescale factor is a scalar; `ĝ` points in the same direction as `g`. Clip-by-value (element-wise clamp) fails this test and is almost never desired.
2. **Global.** `‖g‖` is computed over *all* parameter tensors concatenated; per-tensor norm clipping biases the optimizer toward small-tensor parameters.

Figure 6 of the paper shows the canonical before/after: without clipping, a single "cliff" in the loss surface causes a gradient step that lands the parameters far from the loss manifold and training takes tens of steps to recover. With clipping, the step magnitude is bounded and the direction remains descent-ward.

---

## 2. The global-norm computation and its precision requirements

```math
\|g\|_2 = \sqrt{\sum_{i} \|g_i\|_2^2} = \sqrt{\sum_{i} \sum_{j} g_{i,j}^2}
```

The outer sum is over parameter tensors (embeddings, attention QKV, MLP matrices, norms, head, ...) — for a modern LLM that's hundreds of tensors. The inner sum is over every scalar element — for a 70B model, 70 billion additions.

**Why this must be fp32.** Summing 70B squared bf16 gradients into a bf16 accumulator is a textbook precision disaster:

- bf16 mantissa = 7 bits, so any addition where the running sum is `2^7 ≈ 128×` larger than the increment silently drops the increment.
- A 70B model's per-parameter squared-gradient is typically `~(1e-4)² = 1e-8`. The running sum after 1B additions is `~1e-8 · 1e9 = 10`. Any new increment of `~1e-8` is `10 / 1e-8 = 1e9×` smaller than the running sum — dropped entirely.
- Result: `‖g‖` computed in bf16 is a *deterministic underestimate* of the true norm. Clipping therefore fails to trigger when it should.

The fix: promote each `g_i` to fp32, square, sum, and take the square root in fp32. PyTorch's `torch.nn.utils.clip_grad_norm_` does this correctly by default (it uses `torch.linalg.vector_norm` which promotes to `torch.float32` internally for half-precision inputs). Verify this in any hand-rolled replacement.

---

## 3. FSDP / ZeRO-3: the cross-shard norm problem

Under FSDP / ZeRO-3, each rank holds only a *shard* of each parameter's gradient. Computing the norm locally:

```math
\|g\|_{\text{local}} = \sqrt{\sum_{i \in \text{local shard}} g_i^2}
```

is **wrong** — it omits the contributions from other ranks. The correct global-norm under sharding is:

```math
\|g\|^2_{\text{global}} = \sum_{r=0}^{R-1} \|g\|^2_{\text{local}, r}
\implies \|g\|_{\text{global}} = \sqrt{\text{all\_reduce}_{\text{SUM}}(\|g\|^2_{\text{local}})}
```

One all-reduce over a scalar (the squared local norm) per rank. Then every rank knows the true global norm and rescales its local shard by the same factor `threshold / ‖g‖_global`.

**The precision requirement is strict here.** The all-reduce sum *must* be fp32 for the same reason as §2 — and because all-reduce accumulates across potentially hundreds of ranks, bf16 accumulation compounds the error further. PyTorch's FSDP provides `FullyShardedDataParallel.clip_grad_norm_(max_norm)` which handles the promote-then-reduce-then-scale pattern correctly. DeepSpeed provides an equivalent `engine.clip_fp32_gradients()` / `ZeRO` internal path.

**Hand-rolled naive code that looks right but isn't:**

```python
# BUG: per-rank clip, no cross-rank norm
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
# Each rank rescales using its LOCAL norm, producing a different scale per rank.
# Optimizer step now applies a non-uniform scaling across the full parameter —
# silent divergence.
```

The symptom is specifically *silent divergence that only appears at multi-node scale*; a single-GPU dev run looks fine because the local norm and global norm coincide.

---

## 4. Ordering under mixed precision: `unscale → clip → step`

From [[excerpts/mixed-precision]] §4, the composition rule:

```python
# After scaled backward, grads are fp16/bf16 scaled by S
scaler.unscale_(optimizer)                               # 1. grads become fp32 at real scale
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 2. norm computed in fp32
scaler.step(optimizer)                                   # 3. update
```

**The error case.** If you swap steps 1 and 2 — clipping before unscaling — the threshold is effectively `1.0 / S`. With `S = 2^15 = 32768`, the active threshold is `~3e-5`, which almost every training step exceeds. Every step gets catastrophically rescaled. Loss plateaus at noise level and no one figures out why for days.

**Under bf16, the trap disappears.** bf16 has no loss scaler, so `unscale` is a no-op and the order is just `backward → clip → step`. This is one more reason the 2025 pretraining stack has retired fp16.

---

## 5. Clip-then-accumulate vs. accumulate-then-clip

Gradient accumulation composes `N` micro-batches into one effective batch:

```python
for micro_batch in batch.split(N):
    loss = model(micro_batch).loss / N
    loss.backward()          # grads accumulate in .grad
# After the loop, .grad holds the sum of per-microbatch gradients
clip_grad_norm_(model.parameters(), 1.0)  # CLIP ON ACCUMULATED GRAD
optimizer.step()
```

**The rule.** Clip *after* accumulation, not per micro-batch. Reason: the statistics of a sum of N gradient micro-batches are different from the per-micro-batch gradient. Per-microbatch clipping would bias the accumulated gradient because each micro-batch's scaling factor is independent.

If memory pressure forces you to clip per-microbatch (rare), compensate by adjusting the threshold to match the expected summed norm — but this is brittle and not the 2025 default.

---

## 6. Modern LLM clipping thresholds

From the paper's successor practice across 2020+ LLMs:

| Regime | `max_grad_norm` | Rationale |
|---|---|---|
| Pretrain (GPT / Llama / Qwen / DeepSeek) | **1.0** | Matches ~95th percentile of gradient norms at steady state |
| SFT | 1.0 (or 0.5 for noisy data) | Lower for synthetic-data SFT |
| RL (PPO / GRPO) | 0.5 – 1.0 on policy gradient | Reward spikes produce 10× norm bursts; clip absorbs them |
| Fine-tune small data | 1.0 | Same; the threshold is surprisingly universal |

The universality is not a coincidence. At convergence, the *typical* gradient norm for a reasonably-scaled LLM is `O(1)`, and a clip threshold of `1.0` is targeted at the 5% tail. Tighter thresholds starve the optimizer on hard examples.

**Monitoring: track pre-clip norm.** A sudden 100× spike in pre-clip `‖g‖` predicts an imminent loss spike or NaN with high reliability. This is the single most informative scalar to log. Tools like OLMo-2's "spike detector" hook specifically on this signal.

---

## 7. The "skip-step on inf/NaN" composition

Under fp16, the scaler's `step()` checks all gradients for inf/NaN *after unscale* and skips the optimizer update if any are found:

```python
scaler.unscale_(optimizer)
clip_grad_norm_(...)
scaler.step(optimizer)   # internally: if any inf/NaN in grads, SKIP the .step() call
scaler.update()          # halve S if we just skipped
```

Under bf16 there's no built-in skip. Modern stacks (Llama-3, OLMo-2) bolt on a manual skip:

```python
g_norm = clip_grad_norm_(model.parameters(), 1.0)  # returns the pre-clip norm
if torch.isfinite(g_norm) and g_norm < spike_threshold:
    optimizer.step()
else:
    logger.warning(f"step {step}: grad spike {g_norm}, skipping")
```

`spike_threshold` is typically `10 × max_grad_norm` — i.e. even the clip can't save us, skip. This is one of three Llama-3 loss-spike mitigations (the others: embedding-norm monitoring, warmup-phase LR holds).

---

## 8. The "wall in error surface" intuition (Figure 1)

The paper's Figure 1 shows a sharp cliff in a 2-D loss slice. A normal gradient step near the cliff edge points *along* the cliff (fine), but the magnitude is set by the local slope, which is large. The step lands far off the manifold. Clipping caps the magnitude, so the step direction is preserved but the distance is bounded — the parameters stay near the manifold.

**For an LLM trainer in 2025:** this is the geometric picture behind every pretraining loss-spike mitigation. The cliff is real (catastrophic in data, rare-token perplexity, or a late-stage curriculum transition) and clipping is the last line of defense before the optimizer destroys progress. The *first* lines — initialization, warmup, data quality, embedding-norm monitoring — exist to make the cliff rare; clipping handles the rare case.

---

## Connections

- [[ch-02]] — §5 "stability pitfalls" on `unscale → clip → step` ordering comes from §4 here.
- [[excerpts/mixed-precision]] — the composition rule under fp16 loss scaling.
- [[excerpts/adam]] — AdamW's adaptive scaling does not protect against exploding grads; clipping is still required.
- [[excerpts/batch-vs-layer-norm]] — pre-norm relaxes clipping pressure; post-norm is unworkable without it.
- [[excerpts/deepseek-v3]] — 671B MoE trained with standard `max_grad_norm = 1.0` despite fp8 matmul.
