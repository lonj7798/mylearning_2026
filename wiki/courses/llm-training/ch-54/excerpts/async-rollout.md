---
chapter: ch-54
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/async-rollout.md
source_url: https://arxiv.org/abs/2405.11143
created_at: "2026-04-23"
---

# Excerpt: Async actor-learner — the queue is the architecture (ch-54 §5)

**Source library:** `wiki/raw-data/llm-training/papers/async-rollout.md`
**Artifacts:** OpenRLHF async PPO primitives, verl HybridFlow dataflow, IMPALA V-trace ancestry, the `vllm_kl` canary.

---

## Why this source anchors ch-54

§5 of ch-54 asks a single operational question: *how does a 7B RL run get 1.6–2× faster without destabilizing?* The answer is "decouple rollout and training across a bounded queue with an IS-corrected loss" — and that is exactly what this synthesis page documents, drawing on OpenRLHF (Hu 2024), verl HybridFlow, and their IMPALA (Espeholt 2018) ancestor.

---

## The architecture, in three primitives

From the source (line 42), OpenRLHF:

> **OpenRLHF async primitives (from `openrlhf/trainer/ppo_trainer_async.py`):**
>   - `rollout_queue`: `ray.util.queue.Queue`, capacity 1–2.
>   - `rollout_slots`: companion queue carrying `global_step` tokens (backpressure).
>   - `vllm_lock`: `ray` asyncio.Lock to serialize weight-broadcast vs generate.
>   - Partial-rollout flag: `strategy.args.train.partial_rollout_enable`.

And (line 47), verl:

> **verl async primitives (from `verl/workers/rollout/vllm_rollout/vllm_async_server.py`):**
>   - `engine.generate(...)` is async-for-iterated per-request.
>   - `priority` parameter reorders in-flight requests.
>   - Pause state (≈ line 628) blocks new generates and waits for in-flight to drain before a weight broadcast.

ch-54 §5 uses both as a prose diagram: rollout workers → bounded queue → trainer → weight broadcast (under lock / pause) → back to rollout workers. The partial-rollout mode is the newest wrinkle — continue a response after weights have updated, instead of discarding it. That is what verl calls "continuous batching RL."

---

## The staleness bound

From the source (line 48):

> **Staleness bound:** `k = queue_depth + partial_rollout_depth`; typical k=1–2.

k = 1 is already enough to saturate both GPUs (producer busy while consumer steps; consumer busy while producer regenerates). k = 2 helps when rollouts have variable length (long sequences would otherwise block). Beyond k = 2, IS correction starts straining — the per-token ratio `exp(log π_train − log π_rollout)` drifts outside `[low, high]` for too many tokens.

---

## The correctness menu

From the source (line 49):

> - Per-token IS weight `exp(logπ_train − logπ_rollout)`, clipped to `[low, high]`.
> - Sequence-level mask when the IS weight exits `[low, high]` for any token.
> - Drop the rollout entirely if the sequence IS weight exceeds a hard cap.

These correspond exactly to the three `vllm_is_correction_type` modes in [[openrlhf-ppo]] (`tis`, `seq-mask-tis`, `icepop`). ch-54 §5 treats them as a graded defense — `tis` is cheap, `seq-mask-tis` is safer when responses have variable drift, `icepop` zeros out the outlier tokens entirely.

---

## The failure signature

From the source (line 53):

> **Failure signature:** `vllm_kl` divergence > 0.1 plus PPO clipfrac pegged at 1 indicates the async staleness + sampler-mismatch has exceeded what IS correction can handle.

This is the canary. ch-54 §5 makes it the single metric to watch on any async run. Two candidate diagnoses when it fires:

1. Queue backed up (producer faster than consumer, k growing).
2. vLLM/trainer precision mismatch (bf16 rollout vs fp32 train; different cuda kernels).

The fix for (1) is to lower producer throughput or raise consumer throughput; the fix for (2) is to match precisions or force vLLM to return logprobs computed in the trainer's dtype.

---

## Why IMPALA is the ancestor

From the source (line 55):

> Direct descendant of IMPALA V-trace — the IS-weight clip is mathematically the same object.

V-trace's ρ̄ / c̄ bounds and the `[low, high]` clip in PolicyLoss.forward are the same truncated-IS idea, transplanted from discrete-action control to token-sequence control. Knowing this lets you read V-trace proofs as proofs about your LLM loss — the policy parameterization changes, the correctness argument does not.

---

## Connections

- **ch-54 §5** — this is the architecture diagram-in-prose section.
- **[[openrlhf-ppo]]** — the `PolicyLoss.forward` branches that implement each correction mode.
- **[[verl-rollout]]** — the vllm_async_server.py line references.
- **[[trl-ppo]]** — the synchronous counterpart, useful as a baseline corner.
- **[[minibatch-sharing-rl]]** — async keeps the trainer saturated *while the next B × n batch* is being generated.
