<!-- scope: inclusionAI/AReaL at commit bcef492: examples/search_agent (Tongyi-DeepResearch ReAct workflow with search + visit tools and LLM-judge reward), examples/swe (SWE-bench agent RL delegating to AReaL-SWEAgent), examples/sandbox_daytona (DaytonaRunner code grading), and the async staleness bound max_head_offpolicyness (StalenessManager, mask_stale_tokens)
     deps: [[areal-async-rl]], [[async-rollout]]
     see-also: [[asearcher]], [[websailor-tongyi-deepresearch]], [[slime-search-r1-pipeline]], [[verl-search-tool-pipeline]], [[slime-agentic-and-mismatch-examples]], [[grpo]]
-->

# AReaL agent pipelines: search agent, SWE agent, sandbox grading, and the staleness bound
- **Core Insight:** AReaL bounds asynchrony at admission time, not at training time: a new rollout is admitted only while `(max_head_offpolicyness + current_version + 1) × consumer_batch_size − (accepted + running) > 0`, so every sample that reaches a training step was started at most `max_head_offpolicyness` weight versions earlier (`areal/infra/staleness_manager.py` L79–113).
- **Guideline:** For multi-turn agents, let the agent call the policy through AReaL's OpenAI-compatible client and export interactions in `concat` mode; the exporter then builds one sequence per trajectory with `loss_mask` 0 on prompt/tool tokens and 1 on sampled tokens, and refuses to merge a child turn whose prompt does not preserve the parent's sampled token prefix (`areal/experimental/openai/types.py` L190–232).
- **Authors:** inclusionAI / Ant Group AReaL team and contributors.
- **Year:** 2025–2026; code read at commit bcef492a (2026-11-24).
- **URL:** https://github.com/inclusionAI/AReaL/tree/bcef492a63e9c425ac53e04d85d87d4a24e0ebdb
- **Source type:** released config/code
- **Relevant topics:** asynchronous RL, staleness control, decoupled PPO loss, search agent, deep-research agent, LLM-judge reward, SWE-bench RL, sandbox execution, OpenAI-compatible rollout proxy

## Summary
All three examples plug an agent into `PPOTrainer.train(workflow=…)`. The workflow receives an inference engine, runs an episode against it (directly or through a proxy URL), sets a scalar reward on the final completion, and returns the interactions. Rollout and training run on separate GPU groups; the rollout controller applies the staleness bound described above ([[areal-async-rl]]).

## Key Contributions
- `TongyiDeepResearchReactWorkflow`: ReAct loop with `search` (Serper `google.serper.dev`) and `visit` (Jina reader + LLM summary) tools, trained on ASearcher-LRM-35k with Qwen3-1.7B (`tongyi_deepresearch/README.md`; `tool_search.py` L46; `react_agent.py` L80).
- A second SGLang engine served as the LLM judge, with its own `max_head_offpolicyness = int(1e12)` because its weights never change (`train.py` L127–146, L134).
- `SWEAgentWorkflow`: the agent loop, sandbox and grading live in a separate repository (AReaL-SWEAgent, package `aweagent`); AReaL passes a per-rollout proxy `base_url` and `api_key` (`examples/swe/README.md` L5–17; `agent.py` L141–212).
- `DaytonaRunner`: synchronous wrapper that creates one Daytona sandbox, keeps it warm across `run()` calls, and executes code with `sandbox.process.code_run` (`areal/infra/sandbox/runner.py` L102–131, L246–270).
- Optional `mask_stale_tokens`: masks loss targets whose per-token policy-version age exceeds `max_head_offpolicyness` (`cli_args.py` L3312–3319; `rl_trainer.py` L1172–1177).

## Key Figures/Tables to Study
- `areal/infra/staleness_manager.py` L79–113; `examples/search_agent/tongyi_deepresearch/{train.py, react_agent.py, config.yaml}`; `areal/experimental/openai/types.py` L180–275; `examples/swe/agent.py` L102–260; `examples/swe/qwen3_30b_a3b_grpo.yaml`.

## Technical Details
**Pipeline at a glance (search agent)**
1. Data: JSONL (`ASearcher-LRM-35k.jsonl`), `batch_size: 128` (`config.yaml` L127–132; `train.py` L103–110).
2. Admission: `rollout.max_head_offpolicyness: 4`, `max_concurrent_rollouts: 160`, `consumer_batch_size = batch_size` (`config.yaml` L20–28).
3. Episode: `arun_episode` creates `ArealOpenAI(engine, chat_template_type="concat")` and runs the ReAct agent (`train.py` L66–100).
4. Tool parsing: `<tool_call>{json5}</tool_call>`; result appended as a user message `<tool_response>…</tool_response>` (`react_agent.py` L163–177).
5. Limits: 100 LLM calls, 150-minute wall clock, forced answer at 80% of `max_tokens_per_trajectory` (27,648) (`react_agent.py` L78, L140–145, L191–221; `config.yaml` L189–190).
6. Reward: LLM judge compares the first 200 characters of the predicted answer with the gold answer; "correct" → 1.0 else 0.0; set on the last completion (`react_agent.py` L274–317, L319–331; L42–61).
7. Export → loss mask: `client.export_interactions(style="concat")` (`train.py` L98).
8. Train: Megatron actor, `eps_clip 0.4`, `reward_scaling 10.0`, `reward_bias -0.5`, `kl_ctl 0.0`, `use_decoupled_loss: true`, rejection sampling on ratio with `upper: 5.0` (`config.yaml` L41–76).

**Staleness capacity.**
```python
# areal/infra/staleness_manager.py L99-113
with self.lock:
    current_version = self.version_provider.get_version()
    max_concurrent_rollouts = max(1, self.max_concurrent_rollouts)
    concurrency_capacity = max_concurrent_rollouts - self.rollout_stat.running
    ofp = self.max_staleness
    sample_cnt = self.rollout_stat.accepted + self.rollout_stat.running
    consumer_bs = max(1, self.consumer_batch_size)
    staleness_capacity = (ofp + current_version + 1) * consumer_bs - sample_cnt
    capacity = min(concurrency_capacity, staleness_capacity)
    return capacity
```
Default `max_head_offpolicyness` is 0 (synchronous) (`cli_args.py` L3304–3311). `deterministic_sampling` is documented as end-to-end deterministic only with `max_head_offpolicyness=0` (L3491–3495).

**Concat-mode loss mask.** For a child turn whose prompt extends the parent sequence, the exporter inherits the parent's `logprobs`, `loss_mask` and `versions`, pads the new prompt span (tool output, user text) with mask 0 and version −1, and appends the sampled tokens with mask 1 and their per-token policy versions (`types.py` L214–228). For the first turn: `loss_mask = [0] * input_len + [1] * output_len` (L258). Per-token `versions` are what `mask_stale_tokens` compares against.

**SWE agent.** Row fields: `instance_id`, `problem_statement`, `eval_script` (`README.md` L70–77). `run()` wraps the episode in `asyncio.wait_for(…, timeout)` (default 1,800 s; YAML sets 3,600) and re-raises on timeout, discarding the trajectory (`agent.py` L129, L186–205; YAML L205). `_run_episode` calls `aweagent.lifecycle.run_agent_with_reward(data, agent_type, agent_config, override_base_url, override_api_key, …)` and returns its reward (L214–260); `agent_type` is `swe` (built-in tool use) or `cc` (Claude Code) (`README.md` L61). `qwen3_30b_a3b_grpo.yaml`: `n_samples: 4`, rollout `sglang:d4t8p1`, `max_concurrent_rollouts: 32`, `max_head_offpolicyness: 4`, `eps_clip 0.2 / 0.28`, decoupled loss, token-level ratio mask at 5.0 (L26, L38–42, L112–124). The Bailing V3 YAML uses separate actor/rollout nodes and `weight_update_mode: xccl` (NCCL broadcast of HF-layout tensors to SGLang) (`bailing_v3_grpo.yaml` L88–92; README L92–100).

**Sandbox grading example.**
```python
# examples/sandbox_daytona/reward_example.py L11-14
def grade(model_code: str, expected_stdout: str) -> float:
    with DaytonaRunner() as runner:
        result = runner.run(model_code, timeout=5)
        return 1.0 if result.stdout.strip() == expected_stdout.strip() else 0.0
```

## Verification
- Checked on 2026-11-24 against the local clone at bcef492a. All loci refer to that commit.
- `examples/search_agent/local_1.5b_example.yaml` (ASearcher 1.5B, local retrieval) has no workflow or train script in `examples/search_agent/` at this commit; the agent code it expects was not found in-tree.
- The SWE agent loop, container setup and test-based grader are in AReaL-SWEAgent, which is not cloned here; their behaviour is described only from `examples/swe/README.md` and `agent.py`.
- Not found: a retrieval-token masking flag separate from the generic concat exporter; reported throughput or accuracy numbers for these examples (only `reward_curve.png` for the search agent, not read).
