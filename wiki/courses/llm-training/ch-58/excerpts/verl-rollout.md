---
chapter: ch-58
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/frameworks/verl-rollout.md
source_url: https://github.com/verl-project/verl
created_at: "2026-04-23"
---

# Excerpt: verl rollout engine — why verl owns the §5 "frontier scale" leaf

**Source library:** `wiki/raw-data/llm-training/frameworks/verl-rollout.md`
**Artifact:** `vllm_async_server.py` L440–525 (async generate), L~628 (pause-state for weight broadcast), `hf_rollout.py` L40–125 (debug path).

---

## Why this source defines ch-58 matrix row 1 + rows 14–15 + §5 tree

The decision tree routes "128 GPU+ or per-request priority" to verl *because* of the async server. Every claim in matrix rows 1, 14, 15 traces back to one of three code sites in this source.

## Row 1 — "rollout backend: async vLLM AsyncLLMEngine"

Source:

> `vllm_async_server.py` ≈ lines 440–530 (`generate` async method, vLLM AsyncLLMEngine path)

> `vllm_rollout.py` ≈ lines 198–214 (`ServerAdapter.generate_sequences` — now raises `NotImplementedError`; SPMD sync mode was retired in PR #4411)

The matrix cell says "SPMD sync retired PR #4411" because the source cites it explicitly. This is the attested fact that TRL's SPMD vLLM integration (`_generate_vllm_server`) is closer to verl's *old* path — a point ch-58 §2 uses to argue that TRL at scale is essentially "pre-2025 verl".

## Row 14 — "async / partial rollout: priority + pause-state"

Source:

> Per-request priority lets the trainer force newly-weighted requests ahead of stragglers — the knob that enables partial-rollout RL (verl's "continuous batching RL" blog post).

> Weight broadcast pauses the engine (`engine to paused state` block around line 628) to avoid on-the-fly weight updates during active decoding — the correctness fix for async RL.

Ch-58 §4 argues that verl's straggler handling operates "at vLLM's scheduling granularity, not Ray's". This is the difference: OpenRLHF's `rollout_queue` batches at the actor level; verl reorders *within* vLLM's in-flight request set. A learner who can't name which level of the stack does the reordering has not understood the row-14 difference.

## Row 15 — "multi-node / 128-GPU+: primary target"

Source:

> verl decouples generation from training via a worker-group abstraction. Each rollout actor wraps a vLLM engine; the trainer holds FSDP shards; weights are broadcast from trainer→rollout between optimizer steps.

The HybridFlow dataflow graph (from the companion `async-rollout.md`) is what makes "sync → async → partial-rollout" a configuration change rather than a code change. Ch-58 §5 Q4 and §6 graduation criteria both depend on this.

## Row 17 — "VLM: multi_modal_data plumbed through async server"

Source:

> ```python
> async def generate(self, prompt_ids, sampling_params, request_id,
>                    image_data=None, video_data=None, priority=0) -> TokenOutput:
> ```

The `image_data` and `video_data` kwargs are the attested VLM surface. This is why §5's VLM branch routes to verl (row 17), and why ch-58 does *not* route VLM runs to TRL despite TRL's mainline `VisionGRPOTrainer` — the rollout engine is the constraint, not the loss.

## The tokens-in-tokens-out discipline

Source:

> Tokens-in-tokens-out — the async server never touches the tokenizer; prompts arrive as `prompt_ids` lists and responses come back as token ids + logprobs. Makes multi-turn tool-use loops simple.

This is the design choice behind verl's agent-RL story. Ch-58 does not dwell on it because the explicit scope is PPO/GRPO/DPO, but a multi-turn tool-use RL run is another silent "route to verl" signal that the §5 tree doesn't enumerate — learners should recognize it from this source.

## What ch-58 inherits verbatim

- "SPMD sync retired PR #4411" (matrix row 1 citation).
- `priority + pause-state (~L628)` (matrix row 14 citation).
- `multi_modal_data` plumbing (matrix row 17 citation).
- The worker-group + FSDP-shards + rollout-actor architecture (§6 graduation criteria).

## Connections

- **[[verl-ppo-loss]]** — the loss that consumes the rollouts this engine produces.
- **[[verl-grpo]]** — the advantage estimator that sits between rollouts and loss.
- **[[async-rollout]]** — the architectural paper (HybridFlow) that formalizes the dataflow.
- **[[openrlhf-ppo]]** — the closest OpenRLHF parallel (Ray `rollout_queue` + `vllm_lock`); see ch-58 §4 for the "queue-level vs engine-level" scheduling difference.
