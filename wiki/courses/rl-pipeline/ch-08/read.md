<!-- chapter: ch-08
     track: rl-pipeline
     kind: content
     title: Agentic Rollouts: Tools, Sandboxes, and Graders
     deps: [ch-04]
     sources: [[mai-rl-infrastructure]], [[mai-thinking-1]], [[rstar2-agent]], [[slime]], [[kimi-k2-agentic-data]], [[tito-token-in-token-out]], [[slime-agentic-and-mismatch-examples]], [[areal-agent-pipelines]], [[tau2-bench]], [[verl-rollout]], [[deepseek-train-infer-consistency]], [[agentscaler]]
     figures: none
-->

# Chapter 8 — Agentic Rollouts: Tools, Sandboxes, and Graders

> **Core insight.** An agentic rollout is a long-lived, stateful process, not a single generation call. In Rocket it runs on a CPU rollout worker: an agent loop alternates sampler calls with tool calls that a tool dispatcher routes to environments (sandbox execution environments, search services, MCP tools); when the loop ends, per-rollout graders score the trajectory and, where needed, the environment state; the trajectory payload goes to a rollout store; group graders on the problem worker then score the whole GRPO group before advantages are computed ([[mai-rl-infrastructure]] Fig. 1, p. 3). In one day of a production run, median rollout duration ranged from 2 minutes (instruction following, 1 turn) to 93 minutes (software engineering, 249 turns) (Fig. 2). That spread forces three design decisions that single-turn RL does not need: how a rollout holds tools and sandboxes while it waits (rollouts "may hold VMs and other resources before generation starts", and each rollout in Fig. 4 waited 24 minutes before its first inference request, p. 7); what happens when a tool, sandbox or rollout fails; and how the training sequence is built from many model turns without changing the tokens the sampler produced. Open stacks implement the same pieces with different boundaries: verl's `ToolAgentLoop` state machine, slime's custom generate functions and HTTP adapters for real agent CLIs, AReaL's OpenAI-compatible client with concat export, and rStar2-Agent's separate tool-execution service.
>
> **Guideline.** When rollouts call tools at high rate, run tool execution as a separate CPU service with batching, because rStar2-Agent served up to 45K tool calls per step at 0.3 s mean latency with 32 send workers batching up to 64 calls and 1,024 execution workers per worker node ([[rstar2-agent]] §3.1). When rollouts differ in length, assign them to inference servers by available KV-cache capacity, because rStar2-Agent reports that SGLang evicts half of in-progress rollouts on KV overflow (§3.2). When building the training sequence for a multi-turn rollout, keep the engine's token ids and log-probs for model turns and tokenize only the new tool or user messages as a delta with loss mask 0, because verl reports that re-applying the chat template to the final history kept PPO from converging even in single-turn training ([[verl-rollout]]; experience report, no numbers). Set separate timeouts for the agent, the grader and the whole rollout, and count queue time before the first token in the whole-rollout timeout, because Rocket measured a 24-minute wait before the first request (p. 7) and slime's coding-agent example uses 3,600 / 600 / 7,200 s ([[slime-agentic-and-mismatch-examples]]). Grade agent tasks from environment state where possible, and audit any user simulator, because τ²-bench found simulator errors in 16% (telecom) to 47% (airline) of audited conversations ([[tau2-bench]] Table 2).

---

## Why this chapter matters

Most of Rocket's GPU time and most of its staleness come from agentic rollouts. The Fig. 2 durations are medians from 1.13M rollouts in one production day ([[mai-rl-infrastructure]] Fig. 2):

| Problem type | Median duration | Median turns |
|---|---|---|
| Instruction following | 2 min | 1 |
| Visual Q&A | 2 min | 1 |
| Math & science | 7 min | 1 |
| Tool use | 12 min | 13 |
| Competitive coding | 34 min | 1 |
| Web research | 51 min | 89 |
| Web development | 58 min | 104 |
| Software engineering | 93 min | 249 |

The slowest rollout in Rocket's Fig. 4 GRPO group took 784 turns on weights that started at learner step 104, sampled from six successive weight versions, and made the fastest rollout in the group wait 7.6 hours (p. 6–7). Chapters ch-03 and ch-04 treat the staleness consequences. This chapter treats the rollout worker itself: what it runs, what it holds, how it fails, how it is graded, and what it hands to the learner.

The learner's applied target, a sales voice agent, is a multi-turn agent with tools (catalog lookup, calendar booking) and a simulated customer as its environment. §8 uses it as the one applied example.

---

## §1 The agent loop inside the rollout worker

**Rocket.** Fig. 1 numbers the steps ([[mai-rl-infrastructure]] p. 2–4). The controller (1) selects a problem; the problem worker (2) dispatches one request per rollout in the group; a rollout worker (3) runs the agent loop, which requests model responses from the sampler (4) and routes tool calls through the tool dispatcher to the task environment (5). The agent "can execute actions in sandbox execution environments (SEE), query search services, or invoke tools exposed by MCP servers". Tool results become context for the next model call, and the loop continues "until the agent finishes or reaches a configured limit". Per-rollout graders evaluate the trajectory and, where needed, the environment state; the rollout worker stores the payload and returns scores and references to (2); (2) runs group graders and computes GRPO advantages; the controller packs *training segments* (one example, one or more model turns) into batches of references plus advantages; the learner (6) fetches token ids and sampling log-probs from the rollout store.

**MAI-Thinking-1's description of the same loop** ([[mai-thinking-1]] Sec. 3.3, read from the PDF) adds details the infrastructure post omits:
- Each RL environment consists of a task specification, an SEE session for executing tools, and "a set of verifiable or judged rewards for evaluating task completion while tracking environment state".
- The harness is a ReAct-style loop that parses reasoning and actions, dispatches tool calls to SEE, appends observations, and returns control to the policy. "All the previous steps' tokens are preserved in context and they are strict prefixes for the next steps."
- The loop terminates when the model emits no tool calls, or when the rollout exceeds its step, context-length or time budget. The rollout and its SEE are then sent to graders that "combine format checks, rule-based checks, executable tests, verifiable state comparisons, and AI judges".
- Credit assignment "is applied uniformly across all tokens from all policy steps": one trajectory-level advantage for every generated token.
- Rocket v1's rollout worker retries a failed inference request on another replica, "so individual replica failures degrade pool throughput but do not lose rollouts" (Sec. 3.6.4).

**Why the loop lives on a CPU worker.** The rollout worker spends most of its wall-clock time waiting: for inference capacity, for tool results, for sandboxes. Holding that state in a GPU process would hold GPU memory for idle work. Rocket runs controller, problem workers and rollout workers in CPU pools and Ray heads and CPU actors on CPU-only nodes, which it found "more stable than GPU nodes in our deployments" ([[mai-rl-infrastructure]] p. 11–12).

**verl's equivalent** is an explicit state machine per sample:

```python
# verl@8357f1e verl/experimental/agent_loop/tool_agent_loop.py L48-52, L171-182
class AgentState(Enum):
    PENDING = "pending"
    GENERATING = "generating"
    PROCESSING_TOOLS = "processing_tools"
    TERMINATED = "terminated"
...
        # State machine loop
        state = AgentState.PENDING
        while state != AgentState.TERMINATED:
            if state == AgentState.PENDING:
                state = await self._handle_pending_state(agent_data, sampling_params)
            elif state == AgentState.GENERATING:
                state = await self._handle_generating_state(agent_data, sampling_params)
            elif state == AgentState.PROCESSING_TOOLS:
                state = await self._handle_processing_tools_state(agent_data)
            else:
                logger.error(f"Invalid state: {state}")
                state = AgentState.TERMINATED
```

The generating state terminates when the response mask reaches `response_length`, or when `max_assistant_turns` or `max_user_turns` is reached; otherwise it parses tool calls from the generated token ids and moves to `PROCESSING_TOOLS` if any were found (L291–310). Every turn appends to one token sequence; the loop does not reset or summarize earlier turns ([[verl-rollout]]).

---

## §2 The tool dispatcher and the three tool classes

Rocket names three classes of tool endpoint behind one dispatcher: sandboxes (SEE), search services, and MCP tools ([[mai-rl-infrastructure]] p. 3). They differ in cost, latency and failure behavior.

**Parallel calls within a turn.** A model may emit several tool calls in one turn. verl executes at most `max_parallel_calls` of them concurrently:

```python
# verl@8357f1e verl/experimental/agent_loop/tool_agent_loop.py L318-323
        tasks = []
        for tool_call in agent_data.tool_calls[: self.max_parallel_calls]:
            tasks.append(self._call_tool(tool_call, agent_data.tools_kwargs, agent_data))

        with simple_timer("tool_calls", agent_data.metrics):
            responses = await asyncio.gather(*tasks)
```

Calls beyond `max_parallel_calls` are not executed. The default is 1, multi-turn is disabled by default, and tool responses longer than 256 characters are truncated in the middle (`verl/trainer/config/rollout/rollout.yaml` L197, L220, L223 at 8357f1e). A model trained under the default therefore never sees results of a second parallel call. MAI-Thinking-1 rewards the opposite behavior with a cross-environment grader that "encourage[s] efficient tool usage, including parallel tool calls when possible" ([[mai-thinking-1]] Sec. 3.3.2). The configuration of the dispatcher and the reward must agree.

**Code execution as a service.** rStar2-Agent, Microsoft Research's agentic-math RL system, moved Python execution off the GPU nodes into an environment service on the cluster CPUs: a master node holds a task queue; 32 send workers batch up to 64 calls; each worker node runs 1,024 execution workers. It reports up to 45K tool calls per step at 0.3 s mean latency, and answer verification is offloaded to the same service ([[rstar2-agent]] §3.1). The card notes an internal inconsistency in the paper ("45 calls per step" next to "up to 45K") and uses 45K from Fig. 6 and §1. The environment returns one of four outcomes: stdout, IPython-style output, an error with traceback, or a timeout (§2.1).

**Search tools.** AReaL's Tongyi-DeepResearch workflow exposes `search` (Serper API) and `visit` (Jina reader plus an LLM summary) and caps an episode at 100 LLM calls and 150 minutes of wall clock, forcing an answer at 80% of `max_tokens_per_trajectory` (27,648 tokens) ([[areal-agent-pipelines]] `react_agent.py` L78, L140–145, L191–221). The `visit` tool calls a second model, so a single tool call can itself be an LLM inference request with its own latency.

**MCP and synthetic tools.** Kimi K2 built a tool repository of 3,000+ real MCP tool specifications from GitHub plus over 20,000 synthetic tools, and used a stateful tool simulator with controlled stochasticity to produce trajectories; it kept real execution sandboxes for coding and software engineering because simulated responses were not trusted there ([[kimi-k2-agentic-data]] §3.1.1). That pipeline produced SFT data, not RL rollouts. MAI-Thinking-1's general tool-use RL environments are "backed by mocked backends that simulate realistic API or MCP" behavior, often exceed 50 tools per environment, and number more than 150 synthetic environments and 130,000 tasks, plus tasks whose tools are available but not needed, to reduce over-eager tool calling ([[mai-thinking-1]] Sec. 3.3.2).

**Lightweight sandboxes inside the rollout process.** slime's ReTool example runs Python in a subprocess with 120 s and 4 GB limits and rejects code containing patterns such as `import os`, `open(` and `exec(` before execution; the loop allows up to 16 turns ([[slime-agentic-and-mismatch-examples]] `tool_sandbox.py` L23–31, L112–135). A regex filter is not isolation. It is suitable for a math interpreter, not for repository tasks.

---

## §3 Sandboxes: lifecycle, isolation and capacity

**Lifecycle in MAI's SEE.** SEE "provisions a fresh container for each agentic task and destroys it upon task completion", which "ensures reproducibility, enables safe exploration, and prevents state leakage between tasks". Containers are network-isolated by default, so episodes are "deterministic and free from external side effects such as rate limits or transient failures"; when network access is needed, traffic goes through caching proxies and domain allowlists ([[mai-thinking-1]] Sec. 3.3).

**Lifecycle in slime's coding agent.** The episode boots a sandbox from the task image, prepares the workspace, runs the agent CLI against an adapter URL, extracts a `git diff`, and then grades in a *fresh* sandbox from the same image that applies only the diff and runs the dataset tests ("exit 0 == solved") ([[slime-agentic-and-mismatch-examples]]):

```python
# slime@79aa41a examples/coding_agent_rl/generate.py L281-306
    try:
        async with asyncio.timeout(CONFIG.rollout_guard_sec):
            async with boot_agent_sandbox(md["image"], instance_id, harness=harness, trace=base_sample) as sb:
                with trace_span(base_sample, "workspace_prepare", record_error=False):
                    baseline = await swe.prepare_workspace(sb, md["workdir"], md)
                with trace_span(base_sample, "agent_run", record_error=False) as span:
                    agent_exit_code = await harness.run(
                        sb,
                        workdir=md["workdir"],
                        session_id=session_id,
                        adapter_url=adapter_url,
                        time_budget_sec=CONFIG.agent_time_budget_sec,
                        prompt=swe.SWE_PROMPT,
                    )
                    span.set("exit_code", agent_exit_code)
                agent_metrics = await harness.collect_metrics(sb, workdir=md["workdir"])
                with trace_span(base_sample, "patch_extract", record_error=False):
                    diff_text = await swe.git_diff(sb, md["workdir"], baseline)

            with trace_span(base_sample, "evaluation", record_error=False) as span:
                reward, applied_cleanly = await swe.run_evaluation(
                    md,
                    diff_text=diff_text,
                    timeout_sec=CONFIG.eval_timeout_sec,
                    trace=base_sample,
                )
```

Grading in a fresh sandbox means the agent cannot modify the grader's environment, only submit a diff. MAI's SWE environments use other anti-hacking measures at environment build time: network control, git history scrubbed past the base commit, test files reset, and an LLM monitor ([[mai-thinking-1]] Sec. 3.3.1 table in card).

The README's defaults state the resource budget per trajectory: 8,192 MB memory and 1,024 PIDs per sandbox, a 600 s sandbox-creation deadline, agent / grader / whole-rollout timeouts of 3,600 / 600 / 7,200 s, 8 groups × 8 trajectories per update, and 14 agent slots per inference engine (`examples/coding_agent_rl/README.md` L232–245). The whole-rollout timeout "includes sandbox preparation, agent execution, patch extraction, fresh-sandbox grading and trajectory export" (L229–230). There is no turn-count cap; instead each agent CLI may compact its context twice, and a third compaction attempt stops the agent and grades its existing patch (L226–228).

**Capacity is set by in-flight rollouts, not by active generation.** Rocket states that rollouts "may hold VMs and other resources before generation starts" (p. 7). A sandbox booted for a rollout that is still queued for its first inference request is occupied but idle. Rocket also auto-scales the number of in-flight rollouts to inference capacity, which keeps queues short "under normal conditions", but new rollouts can still wait a long time for their first request (p. 7). The sandbox fleet must therefore be sized for the maximum number of in-flight rollouts plus graders' fresh sandboxes, not for the number currently generating. No source gives a ratio.

**Preemption and replay.** DeepSeek-V4's sandbox platform (DSec) keeps a globally ordered trajectory log per sandbox. After preemption it fast-forwards by replaying cached results of completed, possibly non-idempotent commands instead of re-executing them, and the same log supports provenance and "deterministic replay" of any session ([[deepseek-train-infer-consistency]] Sec. 5.2.5). Re-executing a non-idempotent command (a database write, a `git commit`) after a crash would put the environment in a state the trajectory never saw.

---

## §4 Failure handling: drop, retry, zero reward, or resume

Agentic rollouts fail for reasons unrelated to the policy: sandbox crashes, tool timeouts, inference replica failures, preemption. The four policies used in the sources have different statistical consequences.

| Policy | Where used | What happens to the group | Risk |
|---|---|---|---|
| Drop failed rollout | Rocket for MAI Base 2 ([[mai-rl-infrastructure]] p. 7) | group trains with fewer rollouts; no wait for a retry | baseline noise or bias, since failures "may share features that never make their way to the learner" |
| Retry failed rollout | Rocket v1 problem worker ([[mai-thinking-1]] Sec. 3.6.2) | group waits for the retry | retry finishes late; adds staleness to completed rollouts |
| Keep with reward 0 | slime coding agent on `SandboxTerminatedError` | trajectory trained as a failure | infrastructure failures counted as policy failures |
| Resume from log | DeepSeek-V4 WAL ([[deepseek-train-infer-consistency]] Sec. 5.2.3) | interrupted generation continues | needs token-granular log and saved KV or re-prefill |

slime's choice is explicit in the code:

```python
# slime@79aa41a examples/coding_agent_rl/generate.py L369-391
    except SandboxTerminatedError as error:
        # A terminal sandbox failure is an observed trajectory outcome. Keep
        # its sampled actions and train them with zero reward instead of
        # discarding the trajectory and retrying until execution succeeds.
        metadata = {
            "grading_solved": False,
            "instance_id": instance_id,
            "agent": agent_name,
            "agent_error": str(error),
            "agent_error_type": "sandbox_terminated",
        }
        if evaluation:
            ...
        else:
            with trace_span(base_sample, "trajectory_finalize", record_error=False):
                samples = await adapter.finish_session(
                    session_id, base_sample=base_sample, reward=0.0, extra_metadata=metadata
                )
            if not samples:
                return _abort_result(base_sample, "sandbox_terminated_before_generation", instance_id)
```

A wall-clock timeout or any other exception instead calls `_abort_result`, which sets `loss_mask = [0]`, `reward = 0.0` and `remove_sample = True` (L400–410, L446–455): the sample is removed. AReaL's SWE workflow wraps the episode in `asyncio.wait_for(…, timeout)` and re-raises on timeout, discarding the trajectory (default 1,800 s; the example YAML sets 3,600) ([[areal-agent-pipelines]] `agent.py` L129, L186–205).

**Why the choice matters for GRPO.** The group baseline is the mean reward of the rollouts that reach the problem worker. If failures are dropped and failures correlate with something the policy controls (long trajectories, heavy tool use, a specific tool), the surviving group is a biased sample of the policy's behavior. Rocket names this risk without quantifying it (p. 7). DeepSeek makes the same argument for preemption: regenerating interrupted responses from scratch "is mathematically incorrect" because "shorter responses are more likely to survive interruption" ([[deepseek-train-infer-consistency]] Sec. 5.2.3). Keeping failures with reward 0 avoids the selection effect but adds label noise when the failure was the infrastructure's. No source measures either effect. A reasonable control is to log failure type per rollout and compare the failure rate across groups by trajectory length and tool usage; if the rate is flat, dropping is close to unbiased.

---

## §5 Long and short rollouts in one run

The duration spread of Fig. 2 creates three operational requirements.

**Priority by start time.** Each inference request inherits a priority from its rollout's start time, so earlier rollouts are served first; in early Rocket experiments, multi-turn rollouts "sometimes paused for hours between inference requests while waiting for inference capacity" before this was added ([[mai-rl-infrastructure]] p. 7). verl passes a per-sample priority to vLLM, but it is assigned as `np.arange(len(prompts))` before chunking, i.e. batch order, not rollout age ([[verl-rollout]] `agent_loop.py` L1210–1214 at 753aed3).

**Timeouts that count queue time.** A rollout-level timeout that starts at admission includes the pre-first-token wait (24 minutes per rollout in Fig. 4) and every later wait for capacity. Rocket states that "rollout-level timeouts must account for this wait" (p. 7). slime separates the agent budget (inference waiting plus tool execution), the grader command timeout, and the whole-rollout guard.

**Prefix caching for multi-turn, not single-turn.** Multi-turn rollouts resend a growing prefix each turn. Rocket v1 reached a 97–98% prefix-cache hit rate on multi-turn production runs and disabled prefix caching for single-turn workloads so sliding-window-attention tokens can be evicted during long generations ([[mai-thinking-1]] Sec. 3.6.4). verl's `LLMServerClient` uses least-in-flight load balancing with sticky sessions so later turns of a rollout reach the same server and reuse its prefix cache ([[verl-rollout]] `llm_server.py` L43–48).

**KV-aware scheduling.** rStar2-Agent assigns rollout requests by available KV-cache capacity rather than splitting them equally across engines, because SGLang evicts half of in-progress rollouts on KV overflow ([[rstar2-agent]] §3.2). An evicted rollout must recompute its prefix, which adds latency and therefore staleness.

**Admission bound on concurrency.** AReaL's search-agent example sets `max_concurrent_rollouts: 160` and `max_head_offpolicyness: 4`; its admission rule admits a new rollout only while (max_head_offpolicyness + current_version + 1) × consumer_batch_size − (accepted + running) > 0 ([[areal-agent-pipelines]] `staleness_manager.py` L99–113). This bounds staleness at admission. It also bounds sandbox occupancy, since every running rollout holds its environment.

---

## §6 Graders: per-rollout and group

**Per-rollout graders** see one trajectory and, where needed, the environment state after it. Rocket runs them on the rollout worker ([[mai-rl-infrastructure]] p. 3). Rocket v1's rule: "If individual rollouts can be evaluated in isolation, grading is performed by the rollout worker" ([[mai-thinking-1]] Sec. 3.6.3). Examples from the sources:
- SWE: fail-to-pass and pass-to-pass hidden tests run in the container (MAI); fresh-sandbox test run on the agent's diff (slime).
- General tool use: environment-specific graders on final environment state, tool-usage patterns and final answers; for synthetic environments an LLM judge decomposes the task into sub-tasks and grades each; cross-environment graders for parallel calls, no duplicate calls, and valid argument types ([[mai-thinking-1]] Sec. 3.3.2).
- Customer-service agents: τ-bench's reward is r = r_action × r_output ∈ {0, 1}: the final database must equal a unique gold database, and required strings must appear in the agent's replies ([[tau2-bench]] Summary). AgentScaler accepts a trajectory only if the final database state matches the gold state, or, for read-only tasks, if the calls match exactly ([[agentscaler]] §3.1).
- Search: AReaL's judge compares the first 200 characters of the predicted answer with the gold answer and returns 1.0 for "correct", else 0.0 ([[areal-agent-pipelines]] `react_agent.py` L274–317).

State-based grading accepts different valid call orders, which call-sequence matching would reject ([[agentscaler]] Guideline). It requires the environment to expose its state to the grader after the episode, which is a reason SEE sessions are graded before they are destroyed.

**Group graders** see all rollouts of a problem together. Rocket runs them on the problem worker after the group completes (p. 3). Rocket v1's example is an LLM performing "pairwise comparisons of different model answers"; a problem can have multiple grades, and "a user-defined policy aggregates the individual grades into a single overall reward" ([[mai-thinking-1]] Sec. 3.6.3). MAI's reward model is also a group-level grader in form: it scores k responses in one context and is applied cyclically over k permutations ([[mai-thinking-1]] Sec. 3.4.1). Group graders run once per group but only after the slowest rollout finishes, so their latency adds to the staleness of every rollout in the group. ch-11 covers judge design.

**Graders as served models.** When a grader is itself an LLM, it competes for inference capacity. AReaL serves its judge as a second SGLang engine and sets that engine's `max_head_offpolicyness = int(1e12)` because its weights never change, so the admission bound designed for the policy does not throttle it ([[areal-agent-pipelines]] `train.py` L127–146):

```python
# AReaL@bcef492 examples/search_agent/tongyi_deepresearch/train.py L127-134
        judge_alloc = ModelAllocation.from_str(config.judge_engine.backend)
        assert judge_alloc.backend == "sglang"
        server_args = SGLangConfig.build_args(
            sglang_config=config.sglang,
            tp_size=judge_alloc.parallel.tp_size,
            base_gpu_id=0,
        )
        config.judge_engine.max_head_offpolicyness = int(1e12)
```

verl's COLOCATED rollout mode is described for the same purpose: same placement group, separate process, no weight sync, used for LLM-as-judge ([[verl-rollout]] `replica.py` L54–66).

---

## §7 Token-in/token-out: building the training sequence from many turns

The learner trains on token ids and the sampler's log-probs, not on text. A multi-turn rollout interleaves model turns with tool or user messages. If the trainer rebuilds the sequence by applying the chat template to the final message list, three things can differ from what was sampled ([[tito-token-in-token-out]] Summary): BPE merges across a turn boundary; tokens the template inserts that the engine never sampled (a newline after `<|im_end|>`, an empty `<think></think>`); and string post-processing of the response (tool parsers rewriting assistant content). The sampler log-probs then no longer align with the trained tokens.

**Rocket** stores "tokens/logprobs, model-turn payloads" in the rollout store and trains on segments of one or more model turns; the learner fetches token ids and sampling log-probs, not text ([[mai-rl-infrastructure]] Fig. 1). MAI-Thinking-1 states that previous steps' tokens are strict prefixes for the next step (Sec. 3.3).

**The three open implementations** all append token deltas to the stored generated ids rather than re-tokenizing:

- **verl** — "Continuous Token is the only tokenization path" in `ToolAgentLoop`. Assistant turns append the engine's `output.token_ids` and log-probs; context turns tokenize only the appended messages; builder-inserted boundary tokens "are not model-generated and therefore must not carry loss or model logprobs" ([[tito-token-in-token-out]], `continuous_token.py` L38–48). The default `tokenization_sanity_check_mode` is `strict` (`rollout.yaml` L242 at 8357f1e). A comment in `continuous_token_check.py` warns that nothing at runtime compares the merged stream with the real chat-template render, so a mismatched builder "trains on token sequences the template never produces, without any error" (L15–21).
- **slime** — a `_SampleBuilder` classifies each new turn as CLEAN (prompt extends held tokens; append the tail), REALIGN (short divergence inside the most recent response span; overwrite with loss mask 0), or FORK (divergence too large or too early; close this sample and open a new one), with a default fork threshold of 1,024 tokens. MoE routing-replay snapshots always fork ([[tito-token-in-token-out]], `slime/agent/trajectory.py` L157–209). Forked segments share one outcome reward, and a shared generated prefix contributes loss once (`examples/coding_agent_rl/README.md` L282–286). An end-to-end test asserts `trained["tokens"] == original["tokens"]`.
- **AReaL** — in `concat` export, a child turn inherits the parent's log-probs, loss mask and per-token policy versions, pads the new prompt span (tool output, user text) with mask 0 and version −1, and appends sampled tokens with mask 1 and their versions:

```python
# AReaL@bcef492 areal/experimental/openai/types.py L214-229
            if resp.input_len >= parent_len:
                logprobs = (
                    parent_logprobs
                    + [0.0] * (resp.input_len - parent_len)
                    + resp.output_logprobs
                )
                loss_mask = (
                    parent_loss_mask
                    + [0] * (resp.input_len - parent_len)
                    + [1] * resp.output_len
                )
                versions = (
                    parent_versions
                    + [-1] * (resp.input_len - parent_len)
                    + resp.output_versions
                )
```

The per-token `versions` are what AReaL's optional `mask_stale_tokens` compares against `max_head_offpolicyness` to mask stale tokens ([[areal-agent-pipelines]]). In a multi-turn rollout the turns are sampled under different weight versions, so staleness is per token, which matches Rocket's definition of staleness per token (ch-03).

**Effect size.** No framework reports a measured accuracy effect of re-tokenization drift. The only effect report is verl's documentation that applying the chat template to the final history made PPO "not even converged in single-turn" (experience report, no numbers) ([[verl-rollout]], `agent_loop.rst` L180–182). ch-10 returns to TITO as a rollout-design rule.

---

## §8 The user as part of the environment

For conversational agents the environment includes a user. τ-bench gives a gpt-4-0613 user simulator a hidden task instruction and lets it end the conversation; τ²-bench's telecom domain gives the user tools too (15 write and 15 read tools for the user, 6 write and 7 read for the agent), and dual control cost 18% (gpt-4.1) and 25% (o4-mini) pass^1 relative to the agent acting alone ([[tau2-bench]] Summary, Core Insight). τ²-bench audited its simulator: in airline, 13% of 100 conversations had critical and 34% benign simulator errors; retail 12% / 28% of 50; telecom 6% / 10% of 50 (Table 2). A simulator error is a grader error from the policy's point of view: the reward reflects the simulator's mistake, not the agent's.

τ-bench also shows why a single success rate hides unreliability: pass^k = E_task[C(c, k)/C(n, k)] is the probability that all k trials succeed; the best agent had pass^1 61.2 on retail but pass^8 under 25% ([[tau2-bench]] Core Insight). τ²-bench pass^1 falls toward zero beyond 7 required actions (§4.2, Fig. 5).

**Applied example: a sales voice agent.** The tools are a product catalog and a calendar; the environment state is the calendar and a CRM record; the user is a simulated customer. Following the sources: grade booking correctness from the final calendar/CRM state (state-based, as τ-bench and AgentScaler), check required statements (price, terms) as an output-string term (τ-bench r_output), audit a sample of simulator conversations for simulator errors before trusting the reward (τ²-bench Table 2), and report pass^k over repeated runs rather than one success rate. Conversation quality, which no state check captures, needs a judge (ch-11) and a combination rule (ch-12).

---

## §9 How the open stacks map onto Rocket's pieces

| Piece | Rocket ([[mai-rl-infrastructure]], [[mai-thinking-1]]) | verl | slime | AReaL | rStar2-Agent |
|---|---|---|---|---|---|
| Agent loop | rollout worker (CPU Ray actor) | `ToolAgentLoop` state machine | custom generate function; HTTP adapter for agent CLIs | workflow `arun_episode`; agent calls an OpenAI-compatible client | multi-turn tool loop in VERL v0.2 |
| Tool execution | dispatcher → SEE / search / MCP | in-process `_call_tool`, `max_parallel_calls` | sandbox (Sunabako) or subprocess (ReTool) | Serper/Jina tools; Daytona sandbox runner | separate CPU environment service, 45K calls/step |
| Sandbox lifecycle | fresh container per task, network-isolated | not specified in files read | boot → agent → diff → fresh-sandbox grading | `DaytonaRunner` kept warm across `run()` calls | execution workers on CPU nodes |
| Per-rollout grading | rollout worker | reward function per sample | `swe.run_evaluation` in fresh sandbox | reward set on last completion | answer verification in env service |
| Group grading | problem worker | not in files read | `check_reward_nonzero_std` filter only | LLM judge as separate engine (per rollout) | RoC selection over 2G rollouts |
| Failure policy | drop (Base 2); retry (v1) | not traced | zero reward on sandbox death; remove on timeout | discard on timeout | not stated in card |
| Token handling | rollout store of tokens/log-probs | Continuous Token builders | `_SampleBuilder` CLEAN/REALIGN/FORK | concat export with per-token versions | not stated in card |
| Staleness control | priority, drop, inference-bound, 15 s transfer | fully async with `staleness_threshold` | `staleness/mean`, `staleness/max` logged; no hard bound found | admission bound `max_head_offpolicyness` | not stated in card |

Sources for the open columns: [[verl-rollout]], [[tito-token-in-token-out]], [[slime-agentic-and-mismatch-examples]], [[slime]], [[areal-agent-pipelines]], [[rstar2-agent]].

---

## Recipe

1. **Write each task as an environment spec**: task text, tool schemas, initial environment state, grader(s), and limits (turns, tokens, wall clock).
2. **Put the agent loop on CPU workers**, one async task per rollout, with the sampler, tools and graders behind network calls.
3. **Separate tool execution into a service** with its own concurrency limits and timeouts. Return errors and timeouts to the model as observations, as rStar2-Agent's four outcome types do.
4. **Provision one fresh, network-isolated sandbox per rollout**, and grade in a fresh sandbox or from exported state so the agent cannot edit the grader's environment.
5. **Size the sandbox fleet for in-flight rollouts**, including those waiting for their first token and graders' sandboxes.
6. **Set three timeouts**: agent budget, grader command, whole rollout. The whole-rollout timeout must include queue time.
7. **Choose a failure policy per failure type** and log it per rollout: infrastructure failures before any generation → remove; sandbox death after actions → decide between zero reward and drop, then measure failure rate by trajectory length and tool usage to check for selection bias.
8. **Serve requests by rollout age** and route later turns to the same server for prefix caching; schedule by KV capacity.
9. **Build training sequences token-in/token-out**: engine token ids and log-probs for model turns (mask 1), token deltas for tool/user content (mask 0), per-token policy version recorded. Add a test that the trained tokens equal the sampled tokens.
10. **Grade from environment state where possible**; for conversational tasks, audit the user simulator on a sample and report pass^k.

---

## Common mistakes and how to detect them

| Mistake | Symptom | Detection |
|---|---|---|
| Re-applying the chat template to the final conversation | log-prob/token misalignment; poor convergence | assert trained token ids equal sampled ids; verl strict sanity check |
| Rollout timeout starts at first token | rollouts killed late or never; sandboxes held indefinitely | measure admission-to-first-token time per rollout (Rocket: 24 min) |
| Dropping failed rollouts without checking what fails | baseline biased toward short or tool-light trajectories | failure rate by length and tool count |
| Treating every sandbox crash as reward 0 | reward noise correlated with infrastructure load | failure type logged; reward-0 rate vs cluster health |
| `max_parallel_calls = 1` while the reward asks for parallel calls | parallel calls never executed past the first | count emitted vs executed tool calls |
| Grading in the agent's own sandbox | agent edits tests or environment | grade in a fresh sandbox from the diff or exported state |
| Re-executing non-idempotent commands after preemption | environment state not seen by the trajectory | ordered command log with cached results (DSec) |
| Trusting a user simulator unaudited | reward reflects simulator mistakes | τ²-bench-style audit of sampled conversations |
| One judge engine sharing the policy's admission bound | judge throttled by staleness logic | separate engine and limits (AReaL) |
| Equal static split of long rollouts across engines | KV overflow evictions | KV-capacity-aware scheduling; eviction counters |

---

## Check your understanding

1. Rocket states that waiting before a rollout's first token adds no staleness, yet it requires rollout timeouts to account for that wait. Why are both statements true, and what resource is consumed during the wait?
2. Rocket for MAI Base 2 drops failed rollouts while Rocket v1 retried them and slime trains sandbox failures with reward 0. For each policy, explain which quantity in the GRPO update it can bias and under what condition the bias is zero.
3. Why does slime grade the agent's diff in a fresh sandbox instead of running the tests in the sandbox where the agent worked? What class of reward hacking does that prevent, and what does it not prevent?
4. AReaL records a policy version for every token. Why is staleness a per-token quantity in a multi-turn rollout, and how does that change what an admission bound such as `max_head_offpolicyness` guarantees?
5. State-based grading (τ-bench, AgentScaler) accepts different valid call orders. What does it fail to check that a call-pattern grader would check, and why does MAI-Thinking-1 use both?
6. Group graders run once per group but only after the slowest rollout. How does moving a grader from the rollout worker to the problem worker change staleness and grader cost?
7. rStar2-Agent schedules requests by available KV-cache capacity. Explain the causal chain from equal static assignment to KV overflow to added staleness.

---

## Connections

- ch-02 (Rocket's controller / problem worker / rollout worker / learner split), ch-03 and ch-04 (staleness from long agentic rollouts, priority, drop vs retry), ch-05 (weight transfer during in-flight rollouts), ch-07 (mismatch grows with response length), ch-09 (building agentic tasks), ch-10 (TITO and multi-turn rollout settings), ch-11 (judge design for group graders), ch-15 and ch-16 (search and SWE pipelines end to end), ch-20 (fault tolerance at the job level).
- Algorithm mechanics: [llm-training ch-45b](../../llm-training/ch-45b/read.md) for observation masking, turn-level credit assignment and stability in multi-turn agentic RL; [llm-training ch-40](../../llm-training/ch-40/read.md) for the group baseline that dropped or zero-reward rollouts change; [llm-training ch-44](../../llm-training/ch-44/read.md) for verifiable rewards.
- Sources: [[mai-rl-infrastructure]], [[mai-thinking-1]], [[rstar2-agent]], [[kimi-k2-agentic-data]], [[tito-token-in-token-out]], [[slime-agentic-and-mismatch-examples]], [[slime]], [[areal-agent-pipelines]], [[verl-rollout]], [[tau2-bench]], [[agentscaler]], [[deepseek-train-infer-consistency]].

---

## Sources

- [[mai-rl-infrastructure]] — Fig. 1 loop, tool dispatcher, per-rollout and group graders, Fig. 2 durations, 24-minute pre-first-token wait, VMs held before generation, drop-failed policy, CPU-only control plane.
- [[mai-thinking-1]] — ReAct harness, strict-prefix context, termination rules, SEE lifecycle and isolation, tool-use environments and graders, Rocket v1 grading placement, retry policy, prefix-cache hit rate.
- [[rstar2-agent]] — tool-execution service throughput and latency, KV-capacity-aware scheduling, four tool outcomes.
- [[kimi-k2-agentic-data]] — MCP and synthetic tool repository, stateful tool simulator, real sandboxes for code and SWE (SFT data).
- [[tito-token-in-token-out]] — re-tokenization drift and the slime / verl / AReaL token-delta implementations.
- [[slime-agentic-and-mismatch-examples]] — coding-agent episode, fresh-sandbox grading, timeouts and resource limits, sandbox-failure policy, ReTool sandbox.
- [[slime]] — slime trainer, buffer and fully async rollout function.
- [[areal-agent-pipelines]] — search-agent tools and limits, LLM judge engine, admission bound, concat export, SWE timeout policy, Daytona runner.
- [[verl-rollout]] — agent-loop API, termination, parallel calls, sticky routing, priority assignment, TITO experience report.
- [[tau2-bench]] — state-based reward, pass^k, dual control, simulator error audit.
- [[agentscaler]] — final-state grading and tool-graph task generation.
- [[deepseek-train-infer-consistency]] — WAL-based rollout resumption, short-response survival bias, DSec ordered command log.
