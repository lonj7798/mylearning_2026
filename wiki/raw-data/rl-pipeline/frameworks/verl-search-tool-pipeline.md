<!-- scope: verl-project/verl at commit 8357f1e: the Search-R1-like multi-turn tool pipeline as it exists in-tree — preprocess_search_r1_dataset.py, YAML tool registry (BaseTool / FunctionTool), ToolAgentLoop state machine, Hermes tool-call parser, continuous-token response_mask alignment, search_r1_like_qa_em reward preset; notes which documented files are absent at this commit
     deps: [[verl-rollout]], [[search-r1]]
     see-also: [[slime-search-r1-pipeline]], [[areal-agent-pipelines]], [[zerosearch]], [[grpo]], [[verl-rollout-correction-and-lowprec]]
-->

# verl: Search-R1-like multi-turn tool pipeline (data → tool config → agent loop → mask → reward)
- **Core Insight:** verl separates the search pipeline into four replaceable parts joined by data fields: the dataset row names its reward preset through `data_source` (`searchR1_<name>`) and its per-tool state through `extra_info.tools_kwargs`; a YAML file instantiates tool classes; the generic `ToolAgentLoop` runs generate → parse → call tools → append; and `response_mask` marks assistant tokens 1 and tool/context tokens 0 (`preprocess_search_r1_dataset.py` L71–87; `tool_registry.py` L55–80; `tool_agent_loop.py` L236–413; `continuous_token.py` L392–413).
- **Guideline:** At this commit the concrete `SearchTool` class and the `examples/sglang_multiturn/search_r1_like/` launch scripts are not in the repository, although `docs/sglang_multiturn/search_tool_example.rst` still references them; supply your own `BaseTool` subclass (the test stub `tests/tools/_stub_search_tools.py` shows the minimal contract) and point `actor_rollout_ref.rollout.multi_turn.tool_config_path` at a YAML that names it.
- **Authors:** verl contributors (ByteDance Seed and community); search reward adapted from PeterGriffinJin/Search-R1 (`search_r1_like_qa_em.py` L16).
- **Year:** 2025–2026; code read at commit 8357f1e0 (2026-11-24).
- **URL:** https://github.com/verl-project/verl/tree/8357f1e0a7ce4c06dc970c2c637d49b33815f5cc
- **Source type:** released config/code
- **Relevant topics:** multi-turn agent loop, tool registry, tool-call parsing, response mask for tool outputs, exact-match reward, search-agent RL

## Summary
verl's multi-turn tool use runs inside the agent-loop worker ([[verl-rollout]]). A dataset row that needs tools is routed to the registered `tool_agent` loop; the loop talks to the inference server through `server_manager.generate` with token IDs (not text), so the loop builds the training sequence token by token. Tools are either stateful `BaseTool` subclasses (create → execute → release, with per-row `create_kwargs`) or stateless `@function_tool` functions. The reward is computed after the rollout by the reward manager from the decoded response and `reward_model.ground_truth`.

## Key Contributions
- Row format that carries tool state with the prompt (`extra_info.need_tools_kwargs`, `tools_kwargs.search.create_kwargs` with question, ground truth and data source) (`preprocess_search_r1_dataset.py` L74–87; read in `verl/utils/dataset/rl_dataset.py` L403–407).
- YAML-declared native tools plus Python function tools, merged with a name-collision check (`tool_registry.py` L55–101).
- A four-state loop (PENDING → GENERATING → PROCESSING_TOOLS → TERMINATED) with turn, parallel-call and tool-response-length limits (`tool_agent_loop.py` L48, L113–116, L172–183).
- Registered parsers `hermes`, `gpt-oss`, `qwen3_coder`, `glm`, `seed` (`tool_parser.py` L96–452); default `format: hermes` (`rollout.yaml` L191, L245).

## Key Figures/Tables to Study
- `examples/data_preprocess/preprocess_search_r1_dataset.py` L32–98; `verl/experimental/agent_loop/tool_agent_loop.py` L133–215, L236–413, L445–507; `verl/utils/tokenizer/continuous_token.py` L366–415; `verl/utils/reward_score/__init__.py` L50–82, L145–152; `verl/utils/reward_score/search_r1_like_qa_em.py` L96–128.

## Technical Details
**Pipeline at a glance**
1. Preprocess: download `PeterJinGo/nq_hotpotqa_train`, wrap each question in a system + user prompt, tag `data_source = "searchR1_" + source`, attach `tools_kwargs` (`preprocess_search_r1_dataset.py` L45–98, L163).
2. Tool config: `initialize_tools_from_config` loads `tools: [{class_name, config: {type: native, …}, tool_schema}]` (`tool_registry.py` L55–80); schema shape in `tests/tools/test_mixed_tools_on_cpu.py` L50–75 (`search` with `query_list: array`).
3. Generate: token-level call with the parser's stop tokens injected (`tool_agent_loop.py` L240–254).
4. Parse: Hermes regex `<tool_call>(.*?)</tool_call>` (`tool_parser.py` L103–105); arguments must be JSON (`tool_agent_loop.py` L465–470).
5. Execute: up to `max_parallel_calls` calls with `asyncio.gather`, each `create` → `execute` → `release` (L319–323, L486–498); output truncated to `max_tool_response_length` (default 256, side `middle`) (L501–507; `rollout.yaml` L223, L226).
6. Append tool messages (`role: tool`) through `ct_merge_context_msg`; mask 0 (L388–399; `continuous_token.py` L412–413).
7. Reward: `data_source` → preset `search_r1_like_qa_em` → `compute_score(response_str, ground_truth)` (`reward_score/__init__.py` L72–82, L145–148; `reward_loop/reward_manager/naive.py` L42–79).
8. Loss: `agg_loss(loss_mat, loss_mask=response_mask, …)` (`verl/trainer/ppo/core_algos.py` L1140).

**Prompt (preprocess L33–42).** "…you can call a search engine by `<tool_call>` query `</tool_call>` and it will return the top searched results between `<tool_response>` and `</tool_response>` … provide the answer inside `<answer>` and `</answer>`…".

**Response mask alignment.** Every merge appends to `response_mask` with the same edits applied to the token sequence:
```python
# verl/utils/tokenizer/continuous_token.py L392-401, L412-413
# Boundary tokens are added by CT itself, so they get mask/logprob 0.
inserted_token_count = len(merge_result.inserted_token_ids)
aligned_mask += [0] * inserted_token_count
...
# Model-generated assistant tokens get mask 1 and their logprobs; context tokens get mask/logprob 0.
if merge_result.kind == "assistant":
    aligned_mask += [1] * merge_result.appended_token_count
...
elif merge_result.kind == "context":
    aligned_mask += [0] * merge_result.appended_token_count
```
At the end, `response_ids` is the tail of `prompt_ids` of length `len(response_mask)`, and both are cut to `response_length` (`tool_agent_loop.py` L185–198).

**Termination.** Response budget reached, `max_assistant_turns` or `max_user_turns` reached, or no tool call parsed (L292–310); a tool append that would exceed the budget also terminates (L396–397).

**Tool reward channel.** `execute` returns `(ToolResponse, reward, metrics)`; rewards are collected in `agent_data.tool_rewards` and exported as `extra_fields["tool_rewards"]` (L378, L214). The EM preset does not read them.

**Reward function.**
```python
# verl/utils/reward_score/search_r1_like_qa_em.py L119-128
if answer is None:
    return 0
else:
    if em_check(answer, ground_truth["target"]):
        if open_count > 10 or close_count > 10:  # prevent output a lot of </answer>
            score = score / 4
            return score
        return score
    else:
        return format_score
```
`answer` is the last `<answer>…</answer>` match; `format_score` defaults to 0.0, so the reward is 1 / 0, or 0.25 when more than 10 answer tags appear. A substring-EM variant `compute_score_subem` exists (L131) but no `data_source` maps to it by default.

## Verification
- Checked on 2026-11-24 against the local clone at 8357f1e0. All loci refer to that commit.
- Not present at this commit (searched the tree): `verl/tools/search_tool.py`, `search_r1_like_utils.py`, `examples/sglang_multiturn/` (including `search_r1_like/run_qwen2_5_3b_search_multiturn_fsdp.sh`, `local_dense_retriever/`, `config/tool_config/search_tool_config.yaml`), and `recipe/`. They are referenced by `docs/sglang_multiturn/search_tool_example.rst` L149–243 and by a docstring in `tests/tools/test_mixed_tools_on_cpu.py` L18. The documented YAML used `class_name: verl.tools.search_tool.SearchTool` with `retrieval_service_url`, `num_workers: 120`, `rate_limit: 120`, `timeout: 30` (doc L233–240).
- The preprocess prompt tells the model to put a plain query inside `<tool_call>`, while the Hermes path `json.loads` the tool-call body; whether the chat template's tool schema overrides this was not traced.
- Not found: an IFEval reward preset in `verl/utils/reward_score/`; F1 or LLM-judge search graders in-tree.
