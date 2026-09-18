---
chapter: ch-55
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/frameworks/verl-rollout.md
source_url: https://github.com/verl-project/verl/tree/753aed3e1c286ba6825a74342b28669e72c083ea
created_at: "2026-04-23"
updated_at: "2026-09-17"
---

# Excerpt: verl rollout — HFRollout, the async vLLM server, and the agent loop

**Canonical extract:** `wiki/raw-data/llm-training/frameworks/verl-rollout.md` (verified 2026-09-14 against commit `753aed3`). This file was rewritten in the 2026-09 revision to match that card.

---

## HFRollout (`hf_rollout.py` L39–177)

Batched Hugging Face `generate` on the training module. Sampling arguments come from `meta_info` or config; greedy when `do_sample` is false, `val_kwargs` when validating (L64–91). FSDP modules are unsharded with `summon_full_params(writeback=False, recurse=False)`, the comment linking PyTorch issue #100069 for `recurse=False` (L108–110). Output is right-padded to `prompt_length + response_length` (L132–139). The module docstring records that the class hangs under FSDP HybridShard (L16–18).

## Async vLLM server (`vllm_async_server.py` L556–759)

Token ids in, `TokenOutput` out (token ids, log-probs, routed experts, stop reason).

1. PD-disaggregated prefill servers forward to a decode peer (L573–583).
2. `max_possible_tokens = max_model_len − len(prompt_ids)`; below 1 raises `ValueError` (L591–597).
3. `max_tokens` from the request, else `max_new_tokens`, else `min(response_length, prompt_length + response_length − len(prompt_ids))` (L600–611).
4. Clamped to `[1, max_possible_tokens]` (L615–619). The clamp reduces `max_tokens` without a warning; the assert that follows cannot fail after it.
5. `logprobs` is 0 when log-probs are requested and None otherwise; the sampled token's log-prob is read per position (L620, L711–714).
6. A LoRA request is attached when `lora_as_adapter` (L241–245, L650–659).
7. Requests wait while `_submission_paused` is set (L662–665), then pass `priority` to `engine.generate` (L668–674).
8. Aborted requests return empty token ids with `stop_reason="aborted"` (L691–701).

`priority` is a per-sample index `np.arange(len(prompts))` assigned before chunking (`agent_loop.py` L1210–1214), not a staleness signal.

## Weight synchronization (L932–1003)

`abort_all_requests` closes the submission gate, waits up to 60 s for in-flight admissions, then calls `pause_generation(wait_for_inflight_requests=False, clear_cache=...)`. The comment states that weight updates must not proceed unless every in-flight request was aborted and old-weight caches cleared (L989–991). `resume_generation` reopens the gate.

## SPMD retirement

`ServerAdapter.generate_sequences` raises `NotImplementedError`: "The vLLM SPMD mode was retired in PR #4411" (`vllm_rollout.py` L325–341). Config default `mode: async` (`rollout.yaml` L8). `ServerAdapter` still implements `update_weights` (L209–253); it is the server-mode adapter, not the deleted SPMD backend.

## Placement modes (`replica.py` L54–66)

HYBRID: one process, shared GPUs, weight sync. COLOCATED: same placement group, separate process, no weight sync (LLM-as-judge). STANDALONE: separate GPUs, off-policy.

## Agent loop

`AgentLoopBase.run` returns `AgentLoopOutput(prompt_ids, response_ids, response_mask)`; mask 1 = model-generated token, 0 = tool-response token (`agent_loop.rst` L33–72). `align_response_metadata` gives builder-inserted boundary tokens mask 0, assistant tokens mask 1 with rollout log-probs, and tool or user tokens mask 0 with log-prob 0.0 (`continuous_token.py` L390–413). `ToolAgentLoop` is a state machine PENDING → GENERATING → PROCESSING_TOOLS → TERMINATED (`tool_agent_loop.py` L48–52, L166–178), terminating at `response_length` or the turn limits (L282–287). Defaults: `multi_turn.enable: False`, `max_parallel_calls: 1`, `max_tool_response_length: 256`, `tool_response_truncate_side: middle`, `format: hermes`, `tokenization_sanity_check_mode: strict` (`rollout.yaml` L183–230). Every turn appends to one token sequence (L273, L388); nothing resets or summarizes earlier turns.

The docs report that applying the chat template to the final message list "make PPO training not even converged in single-turn", because tool parsers rewrite assistant content and decode-then-encode is not always invertible (`agent_loop.rst` L155–182).

## Timing (fully_async_policy/README.md L333–369)

128 H20 GPUs, Qwen2.5-Math-7B, DAPO, vLLM + FSDP2, 28K max response, `rollout.n` 16: 400 steps took 1d 16h 48m colocated against 17h 22m fully async at 64:64 (2.35×). AIME-2024 acc mean@1 last 0.2958 vs 0.3094; max 0.3573 vs 0.3521. In the colocated baseline `gen` is 177.85 s of a 356.30 s step (L360).

## Corrections to the previous excerpt version

1. "Rollout dominates RL wall-clock (≥70% typical)" → the only number in this artifact is 177.85 s of 356.30 s, about 50%, for one colocated 128-GPU configuration.
2. "`ServerAdapter` … the SPMD backend, now retired" → `ServerAdapter` is the server-mode adapter; the SPMD implementation was deleted and the method now raises.
3. Throughput "~5–8×" for async vLLM against HFRollout — removed; no source.
4. "Per-request priority lets the trainer force newly-weighted requests ahead of stragglers" → priority is a per-sample index.
5. "Paused-state weight broadcast (≈ line 628)" → in-flight requests are aborted, caches cleared, and new submissions parked until `resume_generation` (L932–1003).
6. "The async server never touches the tokenizer" → `generate` calls a VL token dedup with the processor (L633).
7. "HF `.generate` isn't FSDP-aware" — removed as unsupported; the documented limitation is the HybridShard hang.

## Connections

- [[async-rollout]] — the design this code implements.
- [[verl-ppo-loss]], [[verl-grpo]] — consumers of `response_mask`.
- [[gigpo-verl-agent]] — a verl fork that replaces this agent loop with a step-wise paradigm.
