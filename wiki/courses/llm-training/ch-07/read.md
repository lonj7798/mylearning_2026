<!-- chapter: ch-07
     track: foundations
     title: Common Training Failure Modes
     sources: [[gradient-clipping]], [[mixed-precision]], [[adam]], [[loss-masking-prompt]], [[sequence-packing]], [[fsdp-sft]], [[karpathy-training-neural-net-recipe]], [[olmo-2]], [[olmo-3]], [[llama-3]], [[openrlhf-entropy-debugging]], [[entropy-collapse-ppo]]
     figures: figures/failure-modes-tree.html
-->

# Chapter 7 — Common Training Failure Modes

> **Core insight.** Training a modern LLM never *crashes* first; it *drifts* first. Every expensive incident — a dead NaN at step 84k, a hang in the 712th `all_reduce`, an SFT run that silently trained on `<|pad|>` tokens for a day — was preceded by a logged quantity that changed shape before the loss did. Treat the training loop as a pipeline of invariants (finite loss, non-empty batch, prompt-masked labels, synchronised collectives, bounded entropy) and every failure mode becomes "which invariant broke, and how long ago." The failures in this chapter are the ones that repeatedly break those invariants silently.
>
> **Guideline.** Run Karpathy's two cheap tests on every new pipeline before you even attempt a full run: *initial loss equals* `ln(V)`, *then overfit a single batch to near-zero* ([[karpathy-training-neural-net-recipe]]). Instrument `pre_clip_grad_norm`, per-token entropy, active-token count, and per-rank heartbeat at every step, and assert liveness constraints inline — an `assert torch.isfinite(loss)` that halts a run cheaply is worth more than a clever recovery that hides the bug.

---

## Why this chapter exists

Chapters 1–6 built a mechanically correct training step: an optimizer ([[adam]]) that respects mixed precision ([[mixed-precision]]), a norm-clipped backward pass ([[gradient-clipping]]), a packed and masked SFT batch ([[sequence-packing]], [[loss-masking-prompt]]), an FSDP-sharded forward ([[fsdp-sft]]), and a bit-exact resume ([ch-06]). Everything in those chapters assumes the loop is *healthy*. This chapter is the catalog of how that loop dies.

The shape of the catalog is fixed by the frontier reports. Llama 3's 405B run burned 3.8e25 FLOPs across 15.6 T tokens ([[llama-3]]) — an incident window long enough that *any* low-probability failure mode is guaranteed to fire. OLMo 2 ([[olmo-2]]) devoted an architectural ablation to pre-empting the loss-spike phenotype that killed OLMo 1. OLMo 3 ([[olmo-3]]) ran pretraining on 1 024 H100s, mid-training on 128, and post-training on 256 — three distinct clusters where a single rank dropping its collective hangs the whole world group. Each of these projects earned its stability by cataloging the failures of the prior generation and wiring assertions against them. That catalog is this chapter.

Every section below is organized the same way: **symptom → invariant broken → diagnostic tree → fix → log that would have caught it sooner.** Karpathy's maxim — *"neural net training fails silently, so the only defense is an obsessive, data-first, incremental workflow where every step is verified against an explicit prediction"* ([[karpathy-training-neural-net-recipe]]) — is the organizing rule. Every fix below is an instance of that rule.

---

## 1. NaN / Inf — the three arithmetic sources

NaN is never spontaneous. It comes from one of three operations in the transformer step, and the diagnostic tree is short.

**1a. Softmax / logit overflow.** Attention computes `softmax(QKᵀ / √d_k)`. If `QKᵀ` grows large — which happens when the query/key magnitudes aren't controlled — `exp()` overflows in fp16 (max ≈ 6.5e4) and, more subtly, loses all precision in bf16 (mantissa 7 bits; `exp(89)` already saturates). The output-softmax at the language-model head has the same failure mode on the vocabulary axis. OLMo 1's spike phenotype was exactly this: the paper attributes its cure to a *stack* of logit-scaling interventions — QK-Norm on attention, Z-loss on the output logits — rather than any single fix ([[olmo-2]]).

> *"Architectural stability recipe: RMSNorm + reordered norm + QK-Norm + RoPE + Z-loss. Prevents the training-spike phenotype that plagued OLMo 1."* — [[olmo-2]]

The arithmetic fix inside the kernel is standard: subtract the row-max before `exp`.

```python
# numerically stable softmax — the only one you should ever see
m = x.amax(dim=-1, keepdim=True)       # per-row max
z = (x - m).exp()
p = z / z.sum(dim=-1, keepdim=True)
```

Every production attention kernel (FlashAttention, SDPA, xFormers) does this; if you roll your own and forget the `amax` subtraction, fp16 softmax on a 128-token row will NaN within the first few steps.

**1b. `log(0)` in KL / log-softmax / CE.** The second source is any `log(p)` where `p` can be exactly zero. RL's KL-to-reference penalty (ch ~40) is the most common offender: the k3 estimator `(π_ref/π) − 1 − log(π_ref/π)` ([[openrlhf-entropy-debugging]]) NaNs the instant `π` has a single vocab entry at exactly zero — easy if the policy has collapsed. Cross-entropy with `torch.nn.functional.cross_entropy(logits, targets)` is safe (it fuses log-softmax with CE), but hand-rolled `torch.log(softmax(x)) * y` is not. The defensive guard is either to call `log_softmax` directly or clamp: `logp = torch.log(p.clamp_min(1e-9))`. Under fp16, `1e-9` itself underflows; [[mixed-precision]] warns: *"Keep softmax computation in fp32. Keep cross-entropy loss in fp32."* This is the reason.

**1c. Division by zero in advantage / reward normalization.** RL training normalizes advantages per batch: `A ← (A − mean) / (std + eps)`. If a batch happens to contain K identical rewards (all rollouts got the same binary score), `std = 0`, `eps` is too small, and every advantage in the batch becomes NaN. The same pattern appears in Adam's update `α · m̂ / (√v̂ + ε)` when `v̂` underflows ([[adam]]):

> *"Setting `eps` too small under fp16 → division-by-zero NaNs. Bump to `1e-5` if you see NaN in optimizer step."* — [[adam]]

Defensive pattern: `std.clamp_min(1e-6)`, and on RL you additionally want a log counter for "fraction of batches with `std < 1e-4`." A sudden rise in that counter is the earliest signal of entropy collapse (§6) — the policy has stopped producing diverse completions and every rollout of a prompt earns the same reward.

| NaN symptom | Most likely source | First check |
|---|---|---|
| NaN in `loss.backward()` output only | softmax / attention overflow | log `logits.abs().max()`; add QK-Norm or Z-loss |
| NaN in loss scalar itself | `log(0)` in CE / KL | verify `log_softmax` not `log(softmax(...))`; clamp |
| NaN in `optimizer.step()` | `v̂` underflow or /0 in advantage norm | bump `eps`, use `std.clamp_min(1e-6)`, check fp32 master |
| NaN appears only after resume | loss-scaler state dropped ([[mixed-precision]]) | persist `GradScaler.state_dict()` (see ch-06 §5.2) |

**The liveness assertion.** In any production trainer, the single cheapest bug-catcher is:

```python
loss = model(**batch).loss
assert torch.isfinite(loss), f"non-finite loss at step {step}: {loss.item()}"
```

This catches NaN at the point of origin, not 300 steps later when the gradient history has been poisoned. Combine with `pre_clip_grad_norm` logged per step ([[gradient-clipping]]) — a 100× spike in that scalar predicts the NaN by one to five steps, giving you a buffer for skip-step mitigation.

---

## 2. Loss divergence vs loss spike vs loss plateau — the diagnostic tree

Three loss pathologies share the visual shape "line stops doing the expected thing," but their causes and fixes are disjoint. Wrong diagnosis costs hours. The tree:

**Loss spike.** A single-step or few-step jump (1–20×) followed by either recovery or divergence. Near-universal root cause is an out-of-distribution micro-batch colliding with an already-large weight step. [[gradient-clipping]]: *"a sudden 100× spike usually predicts an imminent loss-spike or NaN."* The OLMo 2 mitigation stack is layered specifically for this shape:

> *"Loss spikes in pretraining: the standard Llama-3 / OLMo-2 mitigation stack is: (1) global-norm clip 1.0, (2) skip-step on loss-spike, (3) embedding-norm monitoring."* — [[gradient-clipping]] (cross-ref)

Skip-step means: if `loss > running_mean + k·running_std` (k≈5), discard the gradient, advance the dataloader, keep optimizer state. This loses one batch of compute; the alternative is a divergence rollback that loses hours.

**Loss divergence.** The curve monotonically climbs over hundreds of steps without a single jump. Spike mitigation is useless here — no single batch is the culprit. Common causes: learning rate too high (try the Karpathy sanity `ln(V)` check — if init loss is already above `ln(V)`, LR is not the only problem), warmup too short (ch-03), softmax / logits overflowing but bf16 is hiding it (no NaN because bf16's range is fp32-class, but precision is gone). Diagnostic: log `||W_embed||` and per-layer weight norms. If weight norm trends upward across the divergence, weight decay is insufficient or disabled; if it trends downward, the gradient is dominated by noise.

**Loss plateau.** The curve flattens at a value above what the data justifies. Three disjoint causes:

1. **Dead learning rate** — schedule decayed to zero (cosine hit zero, WSD decayed past the cooldown), or LR scheduler off-by-one ([ch-06 §5.3]).
2. **Clip threshold too low** — [[gradient-clipping]]: *"Clipping threshold too low (e.g. 0.1) → optimizer never makes a real step on hard examples; loss plateaus."*
3. **Dead data pipeline** — the batch is structurally wrong (all padding, all one label, loss-masked to nothing). §3 is this cause in detail.

The fastest distinguishing signal is `pre_clip_grad_norm`. Dead LR → norm healthy but optimizer step scaled to zero (log `lr` as well). Over-clipping → raw norm large but clipped to threshold every step. Dead pipeline → norm near zero because *there are no real labels in the loss*.

**The diagnostic call-flow (top-to-bottom):**

```
Loss pathology observed
├── single-step jump → SPIKE branch
│   ├── grad_norm pre-clip > 100× running → skip-step + investigate batch
│   └── logits |max| > 50 (bf16) → QK-Norm / Z-loss / lower LR
├── monotone climb → DIVERGENCE branch
│   ├── ||W|| climbing → LR too high or WD off
│   ├── ||W|| falling → gradient noise dominates; check init, lower LR
│   └── no clear trend → try reverting last code change (Karpathy's rule)
└── flat above expected floor → PLATEAU branch
    ├── lr == 0 → scheduler bug (see ch-06 §5.3)
    ├── clipped_fraction == 1.0 → raise clip threshold
    └── active_tokens_per_batch ≈ 0 → §3 dead pipeline
```

Each branch has a persistent metric that distinguishes it. This is the ch-06 instrumentation investment paying back: without `pre_clip_grad_norm`, `clipped_fraction`, `active_tokens_per_batch`, `||W_embed||`, and `lr` logged per step, the tree collapses into guesswork.

---

## 3. Dead data pipeline — training on padding in silence

The most demoralising bug. The loss curve looks plausible (flat, or slowly descending on a fake signal); the model is learning nothing. The mechanism: an upstream filter emits empty or all-padding batches, the collate function pads them into a rectangular tensor, the loss mask zeros out every position, and `reduction="mean"` over zero un-masked tokens silently produces `0.0` or `nan` depending on your framework — which then hits the divide-by-zero guard and produces "training-loss-equals-epsilon" forever.

A concrete SFT version from [[loss-masking-prompt]]: the canonical masking code is

```python
labels = input_ids.clone()
labels[:prompt_len] = -100          # mask prompt
loss = F.cross_entropy(logits[..., :-1, :].reshape(-1, V),
                       labels[..., 1:].reshape(-1),
                       ignore_index=-100)
```

Three ways this silently dies:

- `prompt_len == input_ids.size(-1)` because the generator returned a truncated example where the response was dropped by a length filter; the mask covers everything; `loss` is computed on zero tokens.
- The chat-template renderer gave back only `<|system|> ... <|end|>` with no assistant turn because the assistant field was empty in the raw record.
- A deduplication filter removed all completions longer than K tokens, and the SFT mix happens to have a cluster where every prompt has only long completions; the batch becomes structurally empty.

**The invariant.** An SFT batch must satisfy

```python
active = (labels != -100).sum()
assert active > 0, f"batch {step} has zero active tokens"
logger.log_scalar("tokens/active", active.item(), step=step)
```

Log `active_tokens_per_batch` every step. A sudden drop from `~batch_size · 512` to `~batch_size · 5` is unambiguous. OLMo 3's own report records an operational scare around this class: *"Moving SFT from Open Instruct to Olmo Core reportedly improved throughput by 8×"* ([[olmo-3]]) — throughput jumps like that almost always have a correct and an incorrect explanation. The correct one is the new kernel; the incorrect one, which every team checks for first, is "we started training on mostly padding because the collator changed." Both teams in the OLMo 3 disclosure ran the `tokens/active` assertion before shipping the swap.

**Adjacent failures in the data layer:**

- **Empty-batch from a rank-local filter.** Under FSDP the global batch is `micro_batch · grad_accum · dp_size`; a single rank producing an empty micro-batch under DDP hangs the `all_reduce` because the other ranks are waiting on gradients that never appear. Log per-rank `active_tokens` and alarm on zero.
- **Iterator exhaustion.** A single-epoch iterator without `StopIteration` handling silently restarts at the top when cycled, replaying the first epoch's data under the same labels. Observed as slow loss decrease that looks like a legitimate second epoch but isn't; detectable by hashing the first 100 samples and asserting non-repetition against an older epoch's hash (see ch-06 §5.1).
- **All-one-label batch.** Pure-RL (all rollouts succeed or all fail in a reward-verifier batch) — covered in §1c as the advantage-normalization /0. It is the RL instance of dead pipeline.

---

## 4. Masking bugs — off-by-one, and cross-sample attention leakage

The two masking bugs are nearly invisible because they degrade loss by only ~0.5–2% absolute — small enough to look like "slightly worse hyperparameters" on a short run, large enough to lose a benchmark position on a long one.

**4a. Prompt-masking off-by-one.** The shift-for-next-token pattern in causal LM loss is `logits[..., :-1, :]` vs `labels[..., 1:]`. The prompt mask must be applied *before* the shift, on the original `labels` of length T, so that after the shift a prompt token of index `i` is masked at logit position `i-1`. Common incorrect forms:

```python
# WRONG #1 — mask applied post-shift
labels_shift = labels[..., 1:].clone()
labels_shift[:, :prompt_len] = -100         # off by one: position prompt_len-1 not masked

# WRONG #2 — prompt_len computed on packed block
labels[:prompt_len] = -100                  # in a packed block, this only masks the first pack's prompt

# RIGHT
labels = input_ids.clone()
labels[:prompt_len] = -100                  # full-length mask
loss = F.cross_entropy(logits[..., :-1, :].reshape(-1, V),
                       labels[..., 1:].reshape(-1),
                       ignore_index=-100)
```

Wrong #1 leaks the final prompt token into the loss — that token's label is the response's first token, and the gradient flows back through what should have been a masked position. Net effect: the model receives a tiny but consistent signal to "predict the first response token from the immediate predecessor," which is benign on single-turn prompts but is *actively harmful* on multi-turn where it collapses the user↔assistant boundary. [[loss-masking-prompt]] makes the multi-turn rule explicit:

> *"For a conversation with turns `[u_1, a_1, u_2, a_2, …, u_k, a_k]`: mask **all** user turns; mask **all** prior assistant turns (a_1..a_{k−1}); train on a_k tokens only."*

The debugging move: pick three random samples from the batch, run the tokenizer's `decode` on `input_ids[labels != -100]` — the output must be exactly the assistant text you intended to train on. If it contains a `<|user|>` or template token, the mask is off by at least one.

**4b. Cross-sample attention leakage under packing.** [[sequence-packing]] formalizes the correct kernel: `flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen)` computes block-diagonal causal attention. If any of three things is wrong, samples bleed into each other's attention:

1. **`cu_seqlens` is omitted** and the kernel defaults to dense causal → sample 2 attends to sample 1's tokens through the lower-triangular.
2. **`position_ids` are not reset** at sub-sequence boundaries → RoPE rotates using absolute offsets across the whole pack, so sample 2's token 0 has position ≈ L₁, distorting relative-position math.
3. **A custom attention override** (Liger, TorchTune, HF `attn_implementation="eager"`) silently ignores `cu_seqlens` because the hand-rolled path did not thread it through.

[[sequence-packing]]:

> *"Without masking, token t in sequence 2 can attend to sequence 1's tokens → the softmax's partition function leaks across documents → changes gradients even on sequence 1 tokens."*

The impact is quantitative: ablation reports ([[sequence-packing]]) show packing *with correct masking* is mathematically equivalent to unpacked (no metric change) at 2× throughput; packing *without masking* degrades GLUE-class metrics by 0.5–2 points — small enough to look like noise on a short SFT run, large enough to show on the official eval. The unit test that catches every variant:

```python
# at unit-test time, with deterministic weights
out_packed   = model(input_ids_packed, cu_seqlens=cu)
out_unpacked = torch.stack([model(s) for s in split_by_cu(input_ids_packed, cu)])
assert (out_packed - out_unpacked).abs().max() < 1e-4, "cross-sample leak"
```

Put this in CI. Every framework swap in ch-06-era OLMo 3 had to re-pass it.

---

## 5. Distributed hangs — NCCL, all-reduce rank drop, py-spy / gdb attach

The most terrifying class of failure: the run is not crashed, not progressing, not logging — every rank is sitting in a collective that is never going to complete. Under FSDP FULL_SHARD ([[fsdp-sft]]):

> *"FSDP shards model parameters, gradients, and optimizer states across the data-parallel group and reconstructs full parameters on demand via AllGather; gradients are reduced via ReduceScatter."*

Every transformer block's forward emits an `AllGather`; every backward emits an `AllGather` + a `ReduceScatter`. For a 32-layer model that is ~96 collectives per step. If any rank misses any of them, every other rank is stuck at the NCCL wait with no exception. Causes, ordered by frequency in incident logs:

1. **Rank-local control-flow divergence.** A branch like `if batch["has_images"]:` triggers on some ranks but not others; the ranks that take the branch emit an extra collective (e.g. a vision-tower forward) that the others don't — mismatch, hang.
2. **Empty micro-batch on one rank.** A per-rank data filter emits a zero-token batch, the forward short-circuits, the subsequent `AllGather` never fires. §3's empty-batch bug in its distributed form.
3. **OOM on one rank.** One rank sees a slightly longer sequence after the collator, OOMs, and the CUDA caching allocator aborts that process without tearing down NCCL cleanly; peers wait out the 30-minute NCCL timeout.
4. **Infra: dropped network link, sled reboot, GPU ECC error.** On Llama 3's 405B run ([[llama-3]]) and OLMo 3's 1 024-H100 pretraining ([[olmo-3]]), these are *common* at scale. Llama 3 trained on 15.6 T tokens across thousands of GPUs; any component with MTBF of months fires multiple times per week at that fleet size.

**The NCCL timeout.** PyTorch's default NCCL timeout is 30 minutes. That means a single stuck collective wastes 30 minutes of the *entire cluster* before the first rank raises. For a 1 024-H100 cluster that is 512 H100-hours per hang — at 2024 cloud rates, hundreds of dollars per hang. Set it tight:

```python
import datetime
torch.distributed.init_process_group(
    backend="nccl",
    timeout=datetime.timedelta(minutes=5),   # not the 30-minute default
)
```

5 minutes is long enough to absorb a slow checkpoint save, short enough that a real hang fails fast and dumps a py-spy trace you can act on.

**The py-spy / gdb attach ritual.** When a hang is detected (cadence-less loss curve; tokens/sec → 0 on the dashboard), the first-response protocol:

```bash
# On the node you suspect is the straggler (or all nodes, in parallel via pdsh):
py-spy dump --pid $(pgrep -f "train.py" | head -1)   # native Python stack
# If py-spy shows a C-level wait:
gdb -p $(pgrep -f "train.py" | head -1)
(gdb) thread apply all bt
(gdb) detach
```

What you want to see: every rank stopped at `ncclAllReduce` or `ncclAllGather` at roughly the same layer. If rank 3 is stopped *earlier* in the step than everyone else, rank 3 is the rank that dropped the collective — that is the rank to debug. If rank 3 is *later* (inside an extra op the others skipped), rank 3 is the rank that took a divergent branch. The distinction is worth one `git blame` round of the dataloader code.

**Heartbeat instrumentation.** Under the liveness-invariant framing, a hang is "per-rank heartbeat counter stopped advancing." Implement as a background thread that writes `step, rank, wall_time` to a shared file every second. The monitoring job alarms on "max rank wall_time – min rank wall_time > 60s" — a stale-by-60s rank is the straggler before NCCL's 5-minute timeout fires. OLMo 3 reports moving to *"continuous batching and threading work made RL training about 4× more efficient"* ([[olmo-3]]); the same threading infrastructure is the home of the heartbeat thread.

---

## 6. RL-specific: entropy collapse and reward explosion

Two RL-only failures deserve their own section because they don't appear in pretraining or SFT. Both are loss-curve-healthy — the usual §2 tree fails on them.

**Entropy collapse.** [[entropy-collapse-ppo]]:

> *"Per-token entropy `H(π)` drops from ~2–3 nats to below 0.1 nats within a few hundred PPO updates; reward plateaus; rollouts become repetitive."*

The fingerprint is a *sudden inflection* in entropy rather than a gradual decline. If entropy drops faster than reward rises in the first 200 updates, the policy is collapsing, not converging. [[openrlhf-entropy-debugging]] gives the community-standard triage exactly:

> *"(1) confirm KL-to-reference term is on and finite, (2) bump rollout temperature by 0.1–0.2, (3) raise entropy coefficient an order of magnitude, (4) check advantage normalization is per-batch zero-mean unit-var, (5) only then suspect the reward signal."*

The advantage-norm ON/OFF default is a recurring footgun: OpenRLHF and verl default to ON, TRL defaults to OFF ([[openrlhf-entropy-debugging]]). A framework swap mid-project can silently disable normalization, drive entropy to 0, and leave you wondering why reward plateaued — the sequence of checks above is ordered by cost, not probability.

**Reward explosion / NaN in PPO ratio.** [[openrlhf-entropy-debugging]]: *"`NaN` in PPO ratio → very aggressive update; lower LR and clip range."* The PPO ratio `r = π(a|s)/π_old(a|s)` NaNs when `π_old` has underflowed to zero on a token that `π` still assigns mass to. Mitigation: clamp `π_old` at `exp(-50)` before the division, and log the *pre-clip PPO ratio* every step. A ratio exceeding 10 on any token is out-of-distribution for the clip ε = 0.2 regime and should trigger a skip-step.

**The entropy dashboard.** For any RL run, the minimum survivable log surface is:

| Metric | Cadence | Alarm condition |
|---|---|---|
| per-token entropy | every update | `H < 0.1` or sudden inflection |
| KL(π ‖ π_ref) | every update | climbing monotonically past target |
| PPO ratio mean / max | every update | max > 10 |
| advantage std | every batch | std < 1e-4 |
| response-length histogram | every 50 updates | bimodal or mean diverging |
| clipped-fraction | every update | > 0.5 |

These are exactly the metrics OpenRLHF, verl, and TRL expose by default — the convergence across three independent frameworks is evidence that this surface is the minimum, not a preference ([[openrlhf-entropy-debugging]]).

---

## 7. The silent-failure checklist — one page to paste at the top of every new trainer

Distilled from every section above and from Karpathy's maxim-list ([[karpathy-training-neural-net-recipe]]). Run these as unit tests or inline asserts. Every one of them is the distilled form of a bug that has killed a real run.

```python
# --- one-time, pre-run ---
assert initial_loss == pytest.approx(math.log(vocab_size), rel=0.02)  # Karpathy's init check
overfit_single_batch_to_near_zero(model, batch, steps=200)            # Karpathy's pipeline check

# --- every step ---
assert torch.isfinite(loss), f"non-finite loss at step {step}"
active = (labels != -100).sum().item()
assert active > 0, f"zero-active-token batch at step {step}"
assert grad_norm < cfg.hard_ceiling, f"runaway grad_norm {grad_norm}"     # e.g. 1000× clip threshold

# --- periodic (every N=100) ---
assert packed_vs_unpacked_max_diff(model, batch) < 1e-4                   # §4b leak check
assert all_ranks_heartbeat_within(60)                                     # §5 hang guard
assert abs(embed_norm - embed_norm_prev) / embed_norm_prev < 1e-4         # §1a logit drift

# --- on resume (ch-06) ---
assert bit_exact_resume_loss_delta(ckpt, steps=1) < 1e-6
assert scheduler.last_epoch == saved_step
```

The list is short on purpose. Each assertion costs << 1% of step time, and each one catches a failure mode that has burned ≥ 1 person-week at ≥ 1 lab. The economics are unambiguous.

---

## Connections and what's next

- **[[gradient-clipping]] / ch-01** — `pre_clip_grad_norm` is the earliest of all the warning signals in this chapter.
- **[[mixed-precision]] / ch-02** — bf16 vs fp16 governs which NaN modes you even see; fp32 master weights are the baseline safety.
- **[[adam]] / ch-01** — `eps` placement and `v̂` underflow; the optimizer-step NaN surface.
- **[[loss-masking-prompt]] / ch-04** — the off-by-one surface of §4a.
- **[[sequence-packing]] / ch-04** — the cross-sample attention leak of §4b.
- **[[fsdp-sft]] / ch-05** — the collective topology that makes §5's hangs possible at all.
- **[[karpathy-training-neural-net-recipe]]** — the organizing rule; every section here is an instance of "neural net training fails silently."
- **[[olmo-2]] / [[olmo-3]] / [[llama-3]]** — three frontier-scale incident logs; the engineering response to the failures in this chapter.
- **[[openrlhf-entropy-debugging]] / [[entropy-collapse-ppo]]** — RL-specific extension in §6.
- **ch-06 (checkpointing)** — the resume-time subset of §1, §2 (scaler drop, LR off-by-one, data-iter desync).
- **ch-08 (lab)** — the mandatory first artifact: a failure-mode-checklist.md that enumerates exactly the assertions in §7 for your trainer.

## Further reading

- [[gradient-clipping]] — Pascanu 2013; the 100× pre-spike signal and the FSDP global-norm pitfall.
- [[mixed-precision]] — Micikevicius 2017; why bf16 removes most §1 failures at the cost of 7 mantissa bits.
- [[adam]] — Kingma 2014 / Loshchilov 2017; `eps` placement and `v̂` underflow under fp16.
- [[loss-masking-prompt]] — Shi 2024; response-only loss and the multi-turn mask rule.
- [[sequence-packing]] — Krell 2021; `cu_seqlens` and the block-diagonal invariant.
- [[fsdp-sft]] — Zhao 2023; the AllGather/ReduceScatter topology whose breakage causes §5 hangs.
- [[karpathy-training-neural-net-recipe]] — Karpathy 2019; the `ln(V)` init check and "overfit a single batch."
- [[olmo-2]] — the three-layer spike-mitigation stack; the QK-Norm + Z-loss logit-overflow defense.
- [[olmo-3]] — the 1 024-H100 pretraining incident surface; why staged model flow multiplies §5 hang risk.
- [[llama-3]] — 15.6 T-token 405B run; DPO NLL stabilizer as a chosen-logprob-collapse fix.
- [[openrlhf-entropy-debugging]] — framework-level entropy triage that converged across OpenRLHF / verl / TRL.
- [[entropy-collapse-ppo]] — Andrychowicz 2020 + LLM-RL derivative; the sudden-inflection fingerprint.

## Companion visualization

**[figures/failure-modes-tree.html](figures/failure-modes-tree.html)** — interactive diagnostic tree. Click a symptom (Loss NaN, Loss Spike, Loss Plateau, NCCL Hang, Grad Clip Triggers Every Step, Entropy Collapse) and the page walks you through the decision branch: which logged metrics confirm the diagnosis, which invariant broke, which fix applies, which log would have caught it sooner. The tree is the §2 flowchart made tactile — use it to practice the diagnostic ordering until it is reflex, because in a real incident the cost of a wrong branch is measured in cluster-hours.
