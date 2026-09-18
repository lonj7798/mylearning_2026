<!-- scope: verl rollout code at a pinned commit: HFRollout, the token-in/token-out async vLLM server, abort/resume around weight sync, and the multi-turn tool agent loop with response masks
     deps: [[async-rollout]], [[ppo]]
     see-also: [[verl-ppo-loss]], [[verl-grpo]], [[openrlhf-ppo]], [[trl-grpo]]
-->

# verl (verl-project/verl): rollout workers, async vLLM server, and agent loop
- **Core Insight:** At commit 753aed3, verl's vLLM rollout is token-in/token-out (`vLLMHttpServer.generate` takes prompt token ids and returns token ids and per-token log-probs), and the multi-turn `ToolAgentLoop` gives model-generated tokens response_mask 1 and tool and template tokens mask 0 (`continuous_token.py` L390–413); the policy loss receives this mask (`verl/workers/utils/losses.py` L93–108).
- **Guideline:** When rollouts are multi-turn with tool calls, build the training sequence from the generated token ids plus tokenized tool responses instead of re-applying the chat template to the final messages, because the verl agent-loop documentation reports that re-tokenizing the final chat history kept PPO from converging even in single-turn training (`docs/advance/agent_loop.rst` L149–182; experience report, no numbers given).
- **Authors:** verl community; project initiated by the ByteDance Seed team (`README.md` L3). `fully_async_policy` README author: meituan-search (L3).
- **Year:** 2024 (README news entries start at 2024/08; verl is the open-source version of HybridFlow, arXiv:2409.19256, accepted to EuroSys 2025; `README.md` L26, L90). Code read at commit 753aed3, committed 2026-09-14.
- **URL:** https://github.com/verl-project/verl/tree/753aed3e1c286ba6825a74342b28669e72c083ea
- **Source type:** released config/code
- **Relevant topics:** rollout engines, vLLM AsyncLLM, token-in/token-out generation, weight synchronization, multi-turn agentic RL, response masking, asynchronous and partial rollout

## Summary
verl runs RL rollouts through replaceable rollout backends. `HFRollout` calls Hugging Face `generate` on the training module. The vLLM path launches a `vLLMHttpServer` on each node of a rollout replica (`vllm_async_server.py` L1279–1296); the server wraps vLLM's `AsyncLLM` and exposes a token-in/token-out `generate` method. The synchronous SPMD vLLM mode was retired in PR #4411, so `ServerAdapter.generate_sequences` now raises. Multi-turn and tool-calling rollouts run in `verl/experimental/agent_loop/`, where each prompt runs a user-defined async loop that returns prompt ids, response ids, and a response mask.

## Key Contributions
- `HFRollout`: batched HF `generate` under FSDP `summon_full_params`, bf16 autocast, padding to `response_length` (`hf_rollout.py` L39–177).
- `vLLMHttpServer.generate`: token ids in, `TokenOutput` out (token_ids, log_probs, routed_experts, stop_reason) (`vllm_async_server.py` L556–759; `replica.py` L39–51).
- Weight-sync safety: a submission gate plus `pause_generation` aborts in-flight requests and clears caches before weights change (`vllm_async_server.py` L932–1003).
- Agent loop API: `AgentLoopBase.run` returns `AgentLoopOutput(prompt_ids, response_ids, response_mask)`; mask 1 = LLM-generated token, 0 = tool response token (`agent_loop.rst` L33–72).
- `ToolAgentLoop`: a state machine PENDING → GENERATING → PROCESSING_TOOLS → TERMINATED (`tool_agent_loop.py` L48–52, L166–178).

## Key Figures/Tables to Study
- `verl/workers/rollout/vllm_rollout/vllm_async_server.py` L556–759 (`generate`) and L932–1003 (`abort_all_requests`, `resume_generation`).
- `verl/experimental/agent_loop/tool_agent_loop.py` L128–300 and `verl/utils/tokenizer/continuous_token.py` L363–417 (mask alignment).
- `docs/advance/agent_loop.rst` L146–182 (chat completion vs token-in/token-out).
- `verl/experimental/fully_async_policy/README.md` L333–369 (async vs colocate timing tables).

## Technical Details
**HFRollout.** `generate_sequences` splits the batch into `batch_size // micro_batch_size` chunks (L45–51). Sampling arguments come from `meta_info` or config; greedy when `do_sample` is false, `val_kwargs` when validating, rollout config otherwise, always `num_return_sequences=1` because repetition happens in the trainer (L64–91). FSDP modules are unsharded with `summon_full_params(writeback=False, recurse=False)`; the comment links PyTorch issue #100069 for `recurse=False` (L108–110). Output is right-padded to `prompt_length + response_length` with the pad token (L132–139). The module docstring says the class hangs under FSDP HybridShard (L16–18).

**Async vLLM `generate`.**
1. PD-disaggregated prefill servers forward the request to a decode peer (L573–583).
2. `max_possible_tokens = max_model_len − len(prompt_ids)`; if below 1 a `ValueError` is raised (L591–597).
3. `max_tokens` is taken from `max_tokens`, else `max_new_tokens`, else `min(response_length, prompt_length + response_length − len(prompt_ids))` (L600–611).
4. The value is clamped to `[1, max_possible_tokens]` and then asserted in that range (L615–619). The clamp reduces `max_tokens` without a warning; the assert cannot fail after the clamp.
5. `logprobs` is set to 0 when log-probs are requested and to None otherwise; the sampled token's log-prob is then read at each position (L620, L711–714). `repetition_penalty` and `ignore_eos` default from config (L621–622).
6. A LoRA request is attached when `lora_as_adapter` (rank > 0 and not merged) and the adapter is loaded (L241–245, L650–659).
7. Requests wait while `_submission_paused` is set (L662–665), then pass `priority` to `engine.generate` (L668–674).
8. An aborted request returns empty token ids with `stop_reason="aborted"` (L691–701). With `enable_rollout_routing_replay`, routed experts are returned for MoE router replay (L716–718); this requires vLLM ≥ 0.22.0 (L435–449).

**Priority.** `AgentLoopManager.generate_sequences` assigns each sample `priority = np.arange(len(prompts))` before chunking, "so each sample gets a globally-unique priority that flows to vLLM request scheduling" (`agent_loop.py` L1210–1214). `LLMServerClient` forwards priority only for the vLLM backend and only when non-zero (`llm_server.py` L137–141).

**Weight sync.** `abort_all_requests` closes the submission gate, waits up to 60 s for in-flight admissions (L77, L956–967), then calls `engine.pause_generation(wait_for_inflight_requests=False, clear_cache=...)` (L981–984). The code comment states that weight updates must not proceed unless every in-flight request was aborted and old-weight caches were cleared (L989–991). `resume_generation` reopens the gate (L995–1003). `ServerAdapter.update_weights` sends bucketed weights over IPC or shared memory, then clears the KV cache and records `global_steps` (`vllm_rollout.py` L209–253). Sleep level 1 is used for LoRA adapters ("lora only update adapter weights") and MTP; otherwise level 2 (L1248–1264).

**SPMD retirement.** `ServerAdapter.generate_sequences` raises `NotImplementedError`: "The vLLM SPMD mode was retired in PR #4411" (`vllm_rollout.py` L325–341). Config default `mode: async` (`rollout.yaml` L8).

**Rollout modes.** HYBRID: rollout and training engines in one process, sharing GPUs with weight sync (on-policy). COLOCATED: same placement group, separate process, no weight sync (LLM-as-judge). STANDALONE: separate GPUs (off-policy) (`replica.py` L54–66).

## Findings relevant to agentic training
- **Why token-in/token-out.** The docs report that token ids from applying the chat template to the final messages may differ from the concatenation of per-turn prompt and response ids, because tool parsers rewrite assistant content and decode-encode is not always invertible (`agent_loop.rst` L155–176). They report that "apply_chat_template to the final chat history messages make PPO training not even converged in single-turn" (L180–182).
- **Masking.** `align_response_metadata` gives boundary tokens inserted by the builder mask 0, assistant tokens mask 1 with their rollout log-probs, and context (tool/user) tokens mask 0 with log-prob 0.0 (`continuous_token.py` L390–413).
- **Turn control.** Termination when the response mask reaches `response_length`, or `max_assistant_turns` / `max_user_turns` is reached (`tool_agent_loop.py` L282–287). At most `max_parallel_calls` tool calls run concurrently per turn (L309–313). Tool response text longer than `max_tool_response_length` characters is truncated left, right, or middle (L491–498). Defaults: `multi_turn.enable: False`, `max_parallel_calls: 1`, `max_tool_response_length: 256`, `tool_response_truncate_side: middle`, `format: hermes`, `tokenization_sanity_check_mode: strict` (`rollout.yaml` L183–230).
- **Context.** Every turn appends to one token sequence (`agent_data.prompt_ids = merge_result.token_ids`, L273, L388); `ToolAgentLoop` does not reset or summarize earlier turns.
- **Routing.** `LLMServerClient` uses least-in-flight load balancing and sticky sessions so later turns reach the same server for prefix caching (`llm_server.py` L43–48); each turn gets a fresh vLLM request id unless `full_determinism` (L90–96).
- **Partial rollout (fully_async_policy).** With `partial_rollout=True`, the rollouter interrupts generation before parameter sync and resumes the interrupted samples afterwards (`fully_async_policy/README.md` L55–58, L222–226). On 128 H20 GPUs, Qwen2.5-Math-7B, DAPO: 400 steps took 1d 16h 48m colocated vs 17h 22m fully async 64:64 (2.35x); AIME-2024 (test file, L337) acc/mean@1 last 0.2958 vs 0.3094, max 0.3573 vs 0.3521 (L333–361, L369). In the colocated baseline, `gen` is 177.85 s of a 356.30 s step (L360).

## Recipe ledger
| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| Qwen2.5-Math-7B (fully_async_policy experiment) | 7B | RL | algorithm; engine | DAPO; vLLM + FSDP2 | fully_async_policy/README.md@753aed3 L336, L338 | verified 2026-09-14 | no ablation reported |
| same | 7B | RL | max response length | 28K tokens | L335 | verified 2026-09-14 | no ablation reported |
| same | 7B | RL | samples per prompt (`rollout.n`); `ppo_mini_batch_size` | 16; 32 | L339–340 | verified 2026-09-14 | no ablation reported |
| same (colocate sync baseline) | 7B | RL | steps; `train_batch_size` | 400; 512 | L344–345 | verified 2026-09-14 | no ablation reported |
| same (fully async) | 7B | RL | `require_batches`; `trigger_parameter_sync_step` | 4; 4 | L349–350 | verified 2026-09-14 | require_batches ablation L398 onward |
| same (fully async) | 7B | RL | `staleness_threshold`; `partial_rollout` | 0.5; True | L351–352 (L330 names the stale-samples mode for this run) | verified 2026-09-14 | staleness ablation 0/0.1/0.3/0.5, L380–396 |
| same | 7B | RL | hardware | 128 H20 GPUs (64:64 rollout:train) | L333, L361 | verified 2026-09-14 | 32/64/128-GPU comparison, L353–361 |

Framework defaults (not a model recipe): `temperature: 1.0`, `n: 1`, `response_length: 512` unless `data.max_response_length` is set, `calculate_log_probs: True`, `agent.num_workers: 8` (`rollout.yaml` L17, L127, L39, L237, L246).

## Connections
- [[async-rollout]] — synthesis of asynchronous rollout designs that this code implements for verl.
- [[verl-ppo-loss]], [[verl-grpo]] — the loss and advantage code that consumes `response_mask`.
- [[openrlhf-ppo]], [[trl-grpo]] — other frameworks' rollout integration; compare there, not here.

## Verification
- Checked on 2026-09-14 against: https://github.com/verl-project/verl at commit 753aed3e1c286ba6825a74342b28669e72c083ea (files: `verl/workers/rollout/{hf_rollout.py, llm_server.py, replica.py}`, `verl/workers/rollout/vllm_rollout/{vllm_async_server.py, vllm_rollout.py}`, `verl/experimental/agent_loop/{agent_loop.py, tool_agent_loop.py}`, `verl/utils/tokenizer/continuous_token.py`, `verl/trainer/config/rollout/rollout.yaml`, `docs/advance/agent_loop.rst`, `verl/experimental/fully_async_policy/README.md`, `README.md`).
- Corrections to the previous card version:
  - "`main` branch (fetched 2026-04-21)" with approximate line ranges → pinned commit 753aed3; `generate` is L556–759 (was "≈440–530"); `generate_sequences` raise is `vllm_rollout.py` L325–341 (was "≈198–214"); `HFRollout` is L39–177.
  - "`max_tokens = max(0, min(...))`" → `max(1, min(max_tokens, max_possible_tokens))`, with a `ValueError` when no room for one token (L591–619).
  - "Silent truncation is avoided by the final assert" → the clamp reduces `max_tokens` without a warning; the assert runs after the clamp (L615–619).
  - "Per-request priority lets the trainer force newly-weighted requests ahead of stragglers" → priority is a per-sample index `np.arange(len(prompts))` assigned before chunking (`agent_loop.py` L1210–1214).
  - "Weight broadcast pauses the engine (around line 628) to avoid on-the-fly weight updates during active decoding" → in-flight requests are aborted (not paused mid-decode), caches cleared, and new submissions parked until `resume_generation` (L932–1003, L662–665).
  - "The async server enables overlap of rollout with the next training microbatch" → overlap of generation and training is the `fully_async_policy` design with separate resources (README L42–47); the HYBRID mode shares GPUs (`replica.py` L55–58).
  - HFRollout excerpt omitted the validation branch (`val_kwargs`) and `position_ids` (L70–79, L99, L115).
- Removed as unsupported by the source: "rollout dominates wall time (≥70% for most recipes)" (the README's colocated 128-GPU baseline shows 177.85 s of 356.30 s, L360); "this file is where verl's throughput advantage lives"; the "continuous batching RL" blog post; "much smaller than full state" for LoRA broadcast; "HF .generate isn't FSDP-aware"; "the async server never touches the tokenizer" (`generate` calls a VL token dedup with the processor, L633); OpenRLHF `ppo_trainer_async.py` Queue/Lock and TRL `_generate_vllm_server` comparisons (not in this artifact).
- Not reported by the source: step-independent (per-step context) rollout or memory modules in `ToolAgentLoop`; wall-time share of rollout for the synchronous trainer outside the fully-async README tables.
