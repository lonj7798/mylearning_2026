---
chapter: ch-54
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/frameworks/verl-rollout.md
source_url: https://github.com/verl-project/verl
created_at: "2026-04-23"
---

# Excerpt: verl's async vLLM server — what ch-54 §5 cites as the "production" corner

**Source library:** `wiki/raw-data/llm-training/frameworks/verl-rollout.md`
**Artifact:** `verl/workers/rollout/vllm_rollout/vllm_async_server.py` ≈ lines 440–530, plus the PR #4411 retirement of SPMD sync.

---

## Why this source anchors ch-54

§5 of ch-54 uses verl and OpenRLHF as the two reference implementations of async rollout. This page is the verl side — the actual class-level description of how the async vLLM engine is wired, what knobs exist, and why SPMD sync was retired. The architectural choices here (priority, pause state, tokens-in/tokens-out) are the reason verl can do partial-rollout RL that OpenRLHF async cannot (yet) match.

---

## Two backends, very different purposes

From the source (line 14):

> **Core pattern:** Two rollout backends. (1) `HFRollout` is the debug/reference path: FSDP `summon_full_params`, single `.generate()` call with a `GenerationConfig`, no external engine. (2) The production path is `vllm_async_server.py`: an async vLLM engine with per-request tokens-in / tokens-out, priority scheduling, LoRA adapters, MoE routing capture, and weight-transfer pause hooks for on-policy weight broadcasting.

ch-54 §7 ("corners of the cube") puts `HFRollout` at the bottom — a reference for correctness-testing — and `vllm_async_server.py` at the top. The SPMD sync path that sat between them was removed:

> Sync SPMD vLLM mode was removed (PR #4411) — new deployments must use async.

ch-54 §5 flags this as important: if you read old verl tutorials, you may see SPMD sync code that no longer runs on main.

---

## The four knobs §5 uses to describe async

From the source (line 105):

> - **Tokens-in-tokens-out** — the async server never touches the tokenizer; prompts arrive as `prompt_ids` lists and responses come back as token ids + logprobs. Makes multi-turn tool-use loops simple.
> - **Per-request priority** lets the trainer force newly-weighted requests ahead of stragglers — the knob that enables partial-rollout RL (verl's "continuous batching RL" blog post).
> - **`max_tokens` clamp** has three layers: user override, global `response_length`, and context-window residual. Silent truncation is avoided by the final `assert max_tokens <= max_possible_tokens`.
> - **LoRA as adapter** — when on, the trainer only broadcasts adapter weights (much smaller than full state), and the engine loads them as a vLLM LoRA request.

The **priority** knob is what ch-54 §5 points at when introducing partial rollout. Without per-request priority you cannot interrupt an in-flight generate mid-sequence; with it, the trainer can mark a just-updated request as high-priority and have vLLM preempt older in-flight decodes.

---

## The pause state — the correctness fix

From the source (line 110):

> **Weight broadcast pauses the engine** (`engine to paused state` block around line 628) to avoid on-the-fly weight updates during active decoding — the correctness fix for async RL.

ch-54 §5's topology diagram has this as the red rectangle at the learner node. Without it, a weight broadcast mid-decode produces token sequences drawn from a mixture of π_{t−1} and π_t, and no IS correction can fix the mixture because the boundary is not observable at the per-token level. The pause makes `k` deterministic: every token in a given rollout came from a single π_{t−k}.

---

## Comparison §7 uses

From the source (line 114):

> - **vs OpenRLHF async rollout (`openrlhf/trainer/ppo_trainer_async.py`):** OpenRLHF uses a Ray `Queue` between rollout actors and trainer + a `Lock` around vLLM weight broadcast; verl uses async-vLLM's internal scheduler + priority + explicit pause state. Same architectural pattern.
> - **vs TRL `_generate_vllm_server` / `_generate_vllm_colocate`:** TRL's in-process vLLM integration is closer to verl's old SPMD path; no partial-rollout or priority knobs.

So the production-async corner of ch-54's cube is currently *verl + OpenRLHF only*; TRL is one rung down (colocated, sync). This matches framework defaults in [[minibatch-sharing-rl]] and the §7 table.

---

## Why ch-54 does NOT read ppo_loss here

ch-55 (the next chapter) is the deep dive into verl's end-to-end PPO/GRPO loss. ch-54's §5 only cares about *where the policy gradient eats its data* and *how stale that data is*. The actual loss lives next door.

---

## Connections

- **ch-54 §5** — the prose architecture diagram; the pause state; priority and partial rollout.
- **ch-54 §7** — the "corners" table; verl at the async + priority + partial cell.
- **ch-55** — deep dive into verl's ppo_loss + GRPO internals (reads this page's sibling files).
- **[[async-rollout]]** — the synthesis page this implementation is one attestation of.
- **[[openrlhf-ppo]]** — the sibling implementation using `vllm_lock` instead of engine pause.
