<!-- chapter: ch-46a
     track: rl
     kind: lab
     title: Lab: Small Agentic SFT-then-RL Run with a Generality Gate
     deps: [ch-46]
     sources: [[toucan-mcp]], [[agent-data-protocol]], [[swe-smith]], [[agenttuning]], [[gigpo-verl-agent]],
              [[deepswe]], [[deepswe-recipe]], [[tau2-bench]], [[bfcl]], [[reasoning-trap-tool-hallucination]],
              [[adding-error-bars-evals]], [[mmlu-pro]], [[ifeval]], [[verl-rollout]], [[entropy-logging-patterns]],
              [[grpo]], [[dr-grpo]], [[r2e-gym]], [[swe-gym]], [[rollout-training-mismatch-tis]]
     figures: figures/truncation-advantage.html, figures/generality-gate-power.html
     revised: 2026-09 (generality revision)
-->

# Chapter 46a — Lab: Small Agentic SFT-then-RL Run with a Generality Gate

> **Core insight.** Agentic post-training changes abilities that its own target metric does not cover, in both directions, and the size of the change depends on the share of agent data and on how failed rollouts are treated. At 7B, SFT on agent trajectories alone scored 0.09 on held-out agent tasks and 0.22 on general tasks, while the same trajectories sampled at ratio η = 0.2 against ShareGPT scored 0.67 and 0.63 ([[agenttuning]], Table 5). After think-then-act GRPO on tool data, Qwen2.5-7B-Instruct rose on BFCL Multi-Turn from 13.6 to 23.5 and fell on IFEval from 62.4 to 59.8, while its rate of answering without an available tool rose from 34.8% to 90.2% — a change no tool-calling or instruction-following benchmark in that study reported ([[reasoning-trap-tool-hallucination]], Table 3, §4.4.2). This lab runs agentic SFT then multi-turn RL at a fixed budget, makes the agent share and the failed-trajectory policy explicit axes, and accepts a result only if a held-out environment family, a tool-calling slice, an abstention check, and two non-agentic suites all clear a stated margin with paired intervals.
>
> **Guideline.** When agent trajectories are added to an SFT mixture, choose the share on a held-out agent family rather than on the trained families, because agent-only data at 7B and 13B lowered both held-out agent and general scores relative to a mixture ([[agenttuning]] Table 5), and a 7B model fine-tuned on 119.3K Toucan instances gained 9.74 points of BFCL V3 multi-turn while losing 5.67 points of non-live AST and 5.55 of relevance ([[toucan-mcp]] Table 2). When multi-turn RL rollouts can end by context limit, step limit, or timeout, decide before the run whether those trajectories are masked or scored as failures and record which of the two the group baseline uses, because the two choices give different advantages for the same rollout group (§5) and DeepSWE reports reward collapse without masking ([[deepswe]] §2.3, Fig. 6, curve only). When compute allows only one model, use a 1.5B–4B instruct checkpoint with database-backed tool environments and keep every gate; GiGPO trained Qwen2.5-1.5B-Instruct on ALFWorld and WebShop with 2 H100 GPUs for 150 iterations ([[gigpo-verl-agent]] App. E.1). When reporting any difference, report a paired interval over tasks with the number of trials per task stated, because at 114 tasks and 4 trials per task the minimum detectable effect in the worked example of §7 is 9.6 points.

## Why this chapter matters for a general-purpose model

The pipeline for a general-purpose model runs pre-training → mid-training → SFT → preference optimization → RL → evaluation. This lab is the second of the two RL labs and the last chapter of the RL phase. It follows [[ch-46]], which measured a single-turn preference or RLVR stage against held-out capability, and it precedes the evaluation phase that opens with [[ch-47]].

Agentic training differs from the single-turn case in three ways that change what has to be measured.

1. **The policy sees tokens it did not emit.** Tool results and environment observations enter the context between the policy's own turns. Which of those tokens carry loss is a configuration choice, and a wrong choice produces a trained model without an error message ([[verl-rollout]], `agent_loop.rst` L33-72).
2. **Rollouts end for reasons other than success or failure.** A trajectory can hit the context limit, the step limit, or a wall-clock timeout. Each ending has to be assigned a reward or excluded, and the assignment changes the gradient every other rollout in the group receives (§5).
3. **The target metric covers a narrow slice of behaviour.** DeepSWE reports no evaluation outside SWE-Bench-Verified and its validation curve ([[deepswe]]); GiGPO reports no evaluation outside its trained environments and QA datasets ([[gigpo-verl-agent]] §5). Neither report can say what happened to abilities the environments did not exercise. The one study in this chapter's source set that measured behaviour outside the target metric reports a change of 55.4 points: tool-use RL raised the rate of fabricating or misusing tools on prompts whose required tool is absent, from 34.8% to 90.2% ([[reasoning-trap-tool-hallucination]] Table 3).

The measurable problem of this lab is therefore: **for a fixed compute budget, which agent-data share and which failed-trajectory policy produce a gain on agent tasks that survives a held-out environment family and does not move non-agentic suites down beyond a stated margin.**

The design material comes from [[ch-29c]] (environments and task synthesis), [[ch-30b]] (multi-skill SFT mixtures and agentic shares), [[ch-45b]] (multi-turn RL mechanics), [[ch-45c]] (context management), and [[ch-43a]] (negative gradients). This chapter runs them and reports numbers with intervals.

## §1 Lab question, hypotheses, deliverables, and budget paths

**Question.** Does agentic SFT followed by multi-turn RL at a given agent-data share help or hurt general ability?

**Hypotheses, written before any run.** Each row gives the direction a source measured, the setting of that measurement, and the observation in this lab that would contradict it.

| Axis | Predicted direction | Source and setting | Contradicting observation |
|---|---|---|---|
| Agent share η = 0.2 vs 0.5 in SFT | the higher share gains more on trained families and loses more on the held-out family and on non-agentic suites | [[agenttuning]] Table 5 (Llama-2-chat 7B/13B/70B, six held-in and six held-out agent tasks, four general tasks); endpoints η = 0 and η = 1 only | the η = 0.5 arm's held-out paired interval lies above the η = 0.2 arm's |
| Single-family vs mixed-family agent slice at equal example count | the mixed slice scores higher outside its own family | [[agent-data-protocol]] Table 10 (Qwen-3-8B, OpenHands harness, about 30K samples each: SWE-smith up-sampled 11.0 vs ADP 16.6 on SWE-Bench Verified); [[swe-smith]] Fig. 4 (7B on 700 SymPy trajectories: SymPy subset 21.2 vs 13.6, Verified-without-SymPy 14.0 vs 15.3) | the mixed slice scores lower on the held-out family at equal count |
| Multi-turn RL after agentic SFT | trained-family success rises | [[gigpo-verl-agent]] Table 1 (Qwen2.5-1.5B-Instruct, GRPO: ALFWorld 72.8 ± 3.6, WebShop success 56.8 ± 3.8 over 3 seeds) | trained-family success does not exceed the SFT checkpoint at 3 seeds |
| Failed-trajectory policy: mask vs penalize | masking gives a higher or equal trained-family reward and fewer degenerate trajectories | [[deepswe]] §2.3 (Qwen3-14B, compact filtering on/off; Fig. 6 shows curves, no number) | masking lowers reward at matched steps |
| Abstention after agentic RL | the rate of acting without an available tool rises | [[reasoning-trap-tool-hallucination]] Table 3 (Qwen2.5-7B-Instruct, GRPO on SynTool: 34.8% → 90.2% no-tool-available, 54.7% → 100.0% distractor-tool) | the abstention check does not move outside its paired interval |

**Definitions used throughout.** *Agent share* (η) is the sampling ratio of agent trajectories against general instruction data in the SFT mixture objective, as defined in [[agenttuning]] Eq. 1; it is a sampling weight, not a count of examples, and the two coincide only when every source is sampled once. A *turn* is one policy generation followed by one environment response. A *void turn* is a turn whose action fails to parse or is rejected by the environment, so the environment state is unchanged. A *loop turn* is a turn whose (state, action) pair already occurred in the same trajectory. An *environment family* is a set of environments sharing an action space and an observation format (for example all database-backed tool APIs).

**Deliverables.**
1. `runs.jsonl`: one row per run with base checkpoint, SFT mixture hash, η, RL algorithm and settings, context policy, failed-trajectory policy, seed, code commit, and rollout-engine commit.
2. `rollouts/<run>/*.jsonl`: every rollout with per-turn token counts, response mask, tool-call validity, termination reason, and reward, kept for the post-mortem of §8.
3. `eval/items/<run>/<suite>.jsonl`: per-item and per-trial scores for every suite in §7, for the SFT checkpoint and for every RL checkpoint.
4. `decontamination.md`: n-gram overlap counts of every training slice against every gate suite (§2.4).
5. `agentic-lab-memo.md`: the hypotheses above, the non-inferiority margins (§7.4) written before training, development results, the decision list, and then the gate results.

**Budget paths.**

| Item | Full path | Resource-constrained path |
|---|---|---|
| Base model | Qwen2.5-7B-Instruct | Qwen2.5-1.5B-Instruct or a 3B–4B instruct checkpoint |
| SFT examples per arm | 30,000 (agent + general, mixed at η) | 10,000 |
| RL environment families | three trained, one held out | two trained, one held out; all database-backed |
| Axes | η ∈ {0.2, 0.5}; context policy ∈ {none, keep-recent-2, summary}; failed-trajectory policy ∈ {mask, penalize} | η ∈ {0.2, 0.5}; failed-trajectory policy ∈ {mask, penalize} |
| Seeds per configuration | 3 | 3 |
| Generality gate | required | required |

**Compute anchors (verified values, not estimates).** GiGPO trained Qwen2.5-1.5B-Instruct on 2 H100 GPUs and Qwen2.5-7B-Instruct on 4 H100 GPUs, 150 iterations each, with 8 trajectories per task and 16 tasks per rollout, so 128 environments per iteration ([[gigpo-verl-agent]] App. E.1). DeepSWE's 32B run used 64 H100 GPUs for six days with 512 containers per iteration ([[deepswe]] intro, §2.2), which is outside both paths here. The per-iteration breakdown at 1.5B on ALFWorld was 362.83 s for rollouts, log-probabilities, and the update ([[gigpo-verl-agent]] §5.6); 150 iterations is therefore about 15 GPU-hours on the constrained path under the assumption that the lab's environments have the same rollout cost, which is not verified.

## §2 Stage A: agentic SFT at two agent shares

### 2.1 Choosing the agent slice

Three released slices cover different substrates, and their reported differences are the reason the lab uses more than one family.

| Slice | Size | Substrate | Reported result to compare against |
|---|---|---|---|
| [[toucan-mcp]] | 1,527,259 trajectories, 567,262 multi-turn, from 495 real MCP servers; the paper's SFT subset is 119.3K | real tool servers reached over HTTP | Qwen2.5-7B-Instruct BFCL V3 overall 55.10 → 58.26, multi-turn 12.88 → 22.62, irrelevance 67.93 → 75.18, non-live AST 84.19 → 78.52, relevance 72.22 → 66.67 (Table 2) |
| [[agent-data-protocol]] | 13 datasets unified to 1.3M trajectories, average 10.1 rounds | mixed: API, code, and message actions (53% / 24% / 23%) | Qwen-2.5-7B-Coder-Instruct SWE-Bench Verified 0.4 → 20.2 with SWE-Agent; at about 30K samples, ADP 16.6 vs SWE-smith-only 11.0 on Qwen-3-8B (Tables 3, 10) |
| [[swe-smith]] | 50,137 task instances over 128 repositories; 5,016 rejection-sampled trajectories | one Python execution environment per repository | SWE-agent-LM-32B 40.2 pass@1 on SWE-Bench Verified; 7B trained on 700 SymPy trajectories scores 21.2 on the SymPy subset vs 13.6 for the 100-repository model, and 14.0 vs 15.3 on Verified without SymPy (§3, Fig. 4) |

The SWE-smith specialization result is the most direct measurement of narrowing at the level of an environment family: 700 trajectories from one repository raised that repository's subset by 7.6 points and lowered the rest of the benchmark by 1.3 points at 7B, and at 32B the same substitution moved SymPy from 33.3 to 42.4 and the remainder from 40.2 to 38.3 ([[swe-smith]] §4.1). The repository-pool ablation in the same section gives the opposite direction from diversity: with the trajectory count fixed at 700, drawing from 4, 25, 50, and 100 repositories resolved 10.3%, 11.5%, 12.9%, and 15.1% of SWE-Bench Verified, which the authors describe as approximately logarithmic.

**Lab choice.** Draw the agent slice from at least two families in equal proportion, and hold a third family out of SFT entirely. The held-out family is the one the gate of §7 uses; it never appears in SFT or RL.

### 2.2 Mixing with general instruction data

[[agenttuning]] defines the mixture objective (Eq. 1):

```
J(θ) = η · E_{(x,y) ~ D_agent} [ log π_θ(y | x) ] + (1 − η) · E_{(x,y) ~ D_general} [ log π_θ(y | x) ]
```

where π_θ(y | x) is the fine-tuned model's probability of response y given instruction and history x, D_agent is the agent trajectory set, D_general is the general instruction set, and η is the sampling ratio of agent data. The paper scanned η from 0 to 1 in steps of 0.1 at 7B and selected η = 0.2 on held-out agent tasks; per-η scores are not tabulated, and only the endpoints appear in Table 5 (§2.2.2, App. A).

**Worked example.** With a 30,000-example budget and one pass over the mixture, η = 0.2 draws about 6,000 agent and 24,000 general examples; η = 0.5 draws about 15,000 of each. The agent examples are longer: ADP trajectories average 10.1 rounds and SWE-smith trajectories 26.8 rounds ([[agent-data-protocol]] Table 2), so the agent share of *tokens* is higher than the agent share of *examples* at both settings, by a factor the sources do not allow to be computed in advance. The lab therefore records both the example share and the token share for each arm, and the Recipe table states which one every cited value refers to.

**Evidence for the endpoints.** Held-in / held-out / general normalized scores in [[agenttuning]] Table 5:

| Size | Mixed (η = 0.2) | General only | Agent only |
|---|---|---|---|
| 7B | 1.96 / 0.67 / 0.63 | 0.38 / 0.64 / 0.61 | 1.34 / 0.09 / 0.22 |
| 13B | 2.11 / 0.78 / 0.69 | 0.43 / 0.81 / 0.63 | 1.57 / 0.10 / 0.19 |
| 70B | 2.55 / 1.40 / 0.96 | 0.99 / 0.98 / 1.00 | 2.47 / 0.87 / 0.83 |

**Conditions and limits.** At 7B and 13B the mixture's held-out score is close to general-only training (0.67 vs 0.64; 0.78 vs 0.81); the held-out advantage of mixing appears only at 70B (1.40 vs 0.98). A lab at 1.5B–7B should therefore expect the mixture to protect general ability rather than to raise held-out agent ability, and should say so in the memo before running. The scores are normalized to an average of 1 across the evaluated models per task, so they are not percentages and do not transfer to another model set ([[agenttuning]] §3.1).

### 2.3 Trajectory filtering, and what happens without it

[[agenttuning]] keeps only trajectories with final reward r = 1 (r ≥ 2/3 for Mind2Web), which leaves 1,866 filtered trajectories from 35,341 instructions, a keep ratio of 5.29% (Table 1). Training on the unfiltered set at 7B scored 1.34 held-in and 0.47 held-out against 1.96 and 0.65 for the filtered set (Table 2). [[toucan-mcp]] applies rule filters that remove trajectories with failed tool responses, no tool calls, or local file paths, then keeps instances with question quality and scenario realism 5, completeness and conciseness ≥ 4, and desired tool-use percentage 1.0 (§3.1, §4.1). [[swe-smith]] keeps at most 3 trajectories per instance from a rejection-sampling run in which 17,906 attempts on 8,686 instances resolved 36% (§3).

The lab reproduces one filter decision as a logged number, not as an assumption: the keep rate per family, and the resulting example count per family after filtering.

### 2.4 Decontamination before any gate is run

[[agenttuning]] matches 10-gram token spans with up to 4 mismatched tokens, marks an example dirty above 80% contaminated tokens and clean below 20%, and reports an overall 15.58% rate with 7 dirty examples of 1,021 held-in (App. B, Table 7). None of [[toucan-mcp]], [[agent-data-protocol]], or [[swe-smith]] reports an overlap check against BFCL, τ-bench, τ²-bench, or MCP-Universe; the first two list this under what they do not report.

**Lab rule.** Every training slice is matched against every gate suite in §7 with the same n-gram rule before the gate is opened, and the counts go into `decontamination.md`. A slice with any dirty item against a gate suite is either cleaned or removed from that arm, and the memo records which.

### 2.5 SFT hyperparameters used as the starting point

Verified values from the three slices appear in the Recipe table. [[toucan-mcp]] used LR 2e-5, 2 epochs, effective batch 64, AdamW (0.9, 0.999) with ε 1e-8, and maximum sequence 32,768 (App. C.2, Table 5). [[swe-smith]] used torchtune full fine-tuning at LR 5e-5, at most 3 epochs, maximum context 32,768, on 2–8 H100 GPUs (App. F.1). [[agenttuning]] used LR 5e-5 at 7B and 13B, 1e-5 at 70B, batch 64, sequence 4,096, cosine schedule with warmup ratio 0.02, and loss on model output only (§2.2.3, Table 6). The three disagree on learning rate by a factor of 2.5 at 7B, and the lab records which one each arm used rather than averaging them.

## §3 Stage B: multi-turn RL on a small verifiable environment set

**Environment supply.** The stage needs environment families whose outcome is checked by a program rather than by a judge. Executable software-engineering environment sets are one supply: [[r2e-gym]] and [[swe-gym]] package repository tasks together with the test suites that score them, and DeepSWE trained on 4.5K R2E-Gym problems, each mapped to a Docker image ([[deepswe]] §2.1-§2.2). [[deepswe]] §6 reports limited improvement when the same recipe was run on SWE-Smith and SWE-Gym data instead, so the environment set is a choice that changes the result and is recorded per arm in `runs.jsonl`.

### 3.1 What the rollout must produce

Definition: a multi-turn rollout is a sequence of policy generations interleaved with environment responses, ending at success, failure, the step limit, the context limit, or a timeout. The training sequence is the concatenation of all of these, together with a **response mask** that marks which positions carry loss.

Mechanism, as implemented in verl at commit 753aed3 ([[verl-rollout]]):
1. `AgentLoopBase.run` returns `AgentLoopOutput(prompt_ids, response_ids, response_mask)`, where mask 1 marks an LLM-generated token and mask 0 a tool-response token (`agent_loop.rst` L33-72).
2. `ToolAgentLoop` gives model-generated tokens mask 1 and tool and template tokens mask 0 (`continuous_token.py` L390-413).
3. The policy loss receives this mask (`verl/workers/utils/losses.py` L93-108).
4. `vLLMHttpServer.generate` takes prompt token ids and returns token ids with per-token log-probabilities, so the sequence is never re-tokenized between turns (`vllm_async_server.py` L556-759).

The verl agent-loop documentation states that re-applying the chat template to the final message list, instead of concatenating token ids, kept PPO from converging even in single-turn training (`docs/advance/agent_loop.rst` L149-182; an experience report, no numbers given). **Lab rule:** build the training sequence from token ids, log the fraction of positions with mask 1 per turn, and assert in a test that re-tokenizing the rendered history reproduces the same ids before the first optimizer step.

A second mismatch survives correct masking. The sampler and the learner assign different probabilities to the same tokens under the same weights; on DAPO with Qwen2.5-32B the maximum per-token difference reached 1.0, with some tokens at π_sampler = 1 and π_learner = 0 ([[rollout-training-mismatch-tis]], Fig. 1). The lab logs the maximum and mean sampler-learner probability gap per iteration and treats a rising gap as a reason to stop, not as a metric to optimize.

### 3.2 Algorithm and advantages

The lab uses a group-based critic-free algorithm, as in [[grpo]]: for each task, N trajectories are sampled from the same initial state and the advantage of a trajectory is its return minus the group mean, divided by a normalization factor. [[gigpo-verl-agent]] Eq. 3 writes this as

```
A^E(τ_i) = ( R(τ_i) − mean{R(τ_j)}_{j=1..N} ) / F_norm({R(τ_j)}_{j=1..N})
```

with R(τ_i) the total return of trajectory i and F_norm either the group standard deviation (GRPO's default) or 1, which gives a leave-one-out estimator. [[dr-grpo]] gives the argument against the standard-deviation term: it upweights low-variance groups. [[gigpo-verl-agent]] §5.2 reports that the choice is task-dependent — F_norm = 1 scored higher on the harder subtasks (Look, Pick2, WebShop) and the two were comparable elsewhere — so the lab fixes F_norm = 1 for all arms and records that this is a fixed setting, not a tested axis.

[[deepswe]] uses the leave-one-out form with no standard deviation, no KL loss, and entropy coefficient 0, at LR 1e-6 with `clip_ratio_high` 0.28 and 8 trajectories per task ([[deepswe-recipe]], script L24, L34, L49). The lab starts from those values at 7B and from GiGPO's ALFWorld and WebShop values at 1.5B (LR 1e-6, group size 8, 16 groups per rollout, rollout temperature 1.0, validation temperature 0.4, KL loss coefficient 0.01).

Step-level credit assignment is available at no measured cost and is **not** a tested axis in this lab: [[gigpo-verl-agent]] reports 0.01 s for anchor-state grouping and 0.53 s for the step-advantage computation against 362.83 s for the rest of an iteration, under 0.002% (§5.6). The lab logs the step-group size distribution as an instrument (§6) even when ω = 0, because the distribution measures repeated states, which is the loop-turn signal.

### 3.3 Reward and turn budget

[[gigpo-verl-agent]] App. E.1 gives a rule-based reward of 10 for success and 0 for failure, with an invalid action penalized at −0.1, an episode limit of 50 environment steps on ALFWorld and 15 on WebShop, and maximum response length 512 tokens per turn. [[deepswe]] gives reward 1 when the final patch passes the selected Pass2Pass and Fail2Pass tests within a 5-minute limit and 0 when any test fails or the run times out (§2.2).

Two distinct negative signals appear here and must not be merged: the per-step **invalid-action penalty**, which is reward shaping inside a trajectory, and the **treatment of trajectories that never terminated**, which is §5. The lab keeps the invalid-action penalty fixed at −0.1 across all arms so that the failed-trajectory axis is the only change.

## §4 Ablation axis 1: context policy

Definition: a context policy decides what enters the prompt at turn t. The three arms are **none** (the full history), **keep-recent-k** (the last k observation-action pairs plus the task), and **summary** (a model-written summary of everything older than the last k pairs).

[[gigpo-verl-agent]] App. E.2 uses keep-recent-2 for ALFWorld and WebShop and the full history for search-augmented QA, with maximum prompt 2,048 tokens for ALFWorld and 4,096 for WebShop; the paper reports no ablation across context policies, so the setting is a design choice there, not a measured one. [[deepswe]] reports evaluation at 64k maximum context and 100 maximum environment steps, and a training script with `max_response_length` 32,768 and `agent.max_steps` 50 ([[deepswe-recipe]], script L18-L19, L69) — the blog does not state the training context policy.

**What the lab measures.** For each context arm: trained-family success, the truncation rate (fraction of rollouts ending at the context limit), mean prompt tokens at the last turn, and the held-out-family score. The prediction to write down is that keep-recent-k lowers the truncation rate and that the summary arm lowers it further at the cost of turns spent producing summaries; neither direction is measured by a source in this chapter's set, so both are **open questions** for this setting, and the lab reports its own numbers as a single small-scale result. The theory and the trained-summarization alternatives are in [[ch-45c]].

## §5 Ablation axis 2: failed-trajectory policy

Definition: a trajectory that ends at the context limit, the step limit, or a timeout has no verifier outcome. Two policies are compared.

- **Penalize**: assign reward 0, identical to a verified failure, and train on it.
- **Mask**: assign no loss to that trajectory. This is DeepSWE's compact filtering, which extends DAPO's overlong filtering by also masking trajectories that time out during generation (20 minutes) or reach the maximum step count ([[deepswe]] §2.3).

The authors give two reasons for masking: an agent can pass all tests by accident, for example by answering correctly in the first 10 steps and editing unrelated files afterwards, and rewarding such trajectories leads to collapse; and the average response length per step falls while environment steps rise (Figures 6 and 7). Figure 6 compares Qwen3-14B with and without compact filtering and shows curves only; no number is reported for the size of the effect.

**The choice that is not stated in the source.** Masking the loss and removing the trajectory from the group baseline are two different operations, and the blog states only the first. The lab must pick one and record it. With the leave-one-out advantage of [[deepswe]],

```
A_i = r_i − (1/(n − 1)) · Σ_{j ≠ i} r_j
```

where r_i ∈ {0, 1} is the reward of rollout i and n the rollouts per task, the two variants give different numbers.

**Worked example.** n = 8 rollouts on one task: 2 verified successes, 4 verified failures, 2 truncated.

| Variant | Success advantage | Verified-failure advantage | Truncated trajectory |
|---|---|---|---|
| Penalize (truncated = reward 0, trained) | 1 − 1/7 = **0.857** | 0 − 2/7 = **−0.286** | −0.286, with gradient |
| Mask loss, keep in baseline | 1 − 1/7 = **0.857** | 0 − 2/7 = **−0.286** | −0.286, zero weight |
| Mask loss, drop from baseline (n = 6) | 1 − 1/5 = **0.800** | 0 − 2/5 = **−0.400** | no advantage |

Dropping truncated rollouts from the baseline makes the negative push on verified failures 40% larger (−0.400 against −0.286) and the positive push on successes 7% smaller. The interactive version of this table, including the all-fail case where every advantage is 0, is at [figures/truncation-advantage.html](figures/truncation-advantage.html); it lets the reader set n, the number of successes, and the number of truncated rollouts and read the three variants side by side.

**Zero-variance groups.** When all n rollouts of a task carry the same reward, every advantage is 0 and the task contributes nothing to the update. [[deepswe]] §6 reports that RL on SWE-Smith and SWE-Gym data gave limited improvement with "a high solve-none rate", which is this condition. The lab logs the fraction of zero-variance groups per iteration for every arm; a rising fraction with a flat reward curve means the task set has become too hard or too easy for the current policy, not that the algorithm has stopped working.

## §6 Instrumentation

Each quantity below is logged per iteration and per environment family, so that a change in the aggregate can be attributed.

| Quantity | Definition | Why it is logged | Reference point |
|---|---|---|---|
| Per-turn entropy | categorical entropy of the policy logits over the tokens of turn t, masked to response positions | entropy collapse and entropy explosion have opposite fixes | verl logs `actor/entropy` as categorical entropy aggregated with `loss_agg_mode`; `entropy_coeff` is 0 by default ([[entropy-logging-patterns]]) |
| Entropy split by advantage sign | the same quantity computed separately over positive- and negative-advantage trajectories | the negative branch is where suppression acts (§ Negative samples) | required by the negative-feedback standard; no source in this set reports it for agents |
| Tool-call validity | fraction of turns whose action parses and is accepted by the environment | the direct measure of the failure that abstention checks later penalize | [[gigpo-verl-agent]] penalizes invalid actions at −0.1 per step |
| Void turns | turns where the action fails and the environment state is unchanged | separates "wrong action" from "no action" | — |
| Loop turns | turns whose (state, action) pair already occurred in the trajectory | repeated states are the observable form of a loop | [[gigpo-verl-agent]] §5.5: anchor-state groups of size ≥ 10 exceed 20% at iteration 10, fall to 12.1% (10 ≤ size < 50) and 3.1% (size ≥ 50) by iteration 75, and concentrate at 6–8 by iteration 140 |
| Trajectory length | turns per rollout and tokens per rollout | interacts with the context policy of §4 | [[deepswe]] Fig. 7: response length per step falls while environment steps rise |
| Truncation rate | fraction of rollouts ending at the context, step, or time limit | the denominator of §5 | [[deepswe]] §2.3 |
| Zero-variance group fraction | fraction of tasks where all n rewards are equal | see §5 | [[deepswe]] §6 |
| Reward by environment family | mean reward per family, not pooled | a pooled curve hides one family collapsing | families defined in §1 |
| Sampler-learner probability gap | max and mean per-token \|log π_sampler − log π_learner\| | off-policy drift from the inference engine | [[rollout-training-mismatch-tis]] Fig. 1 |

The step-group size distribution of [[gigpo-verl-agent]] §5.5 is the cheapest loop instrument available: it is computed by hashing environment states across the N trajectories of a group, costs 0.01 s per iteration at 1.5B, and needs no extra rollouts.

## §7 The generality gate

A gate is a set of measurements taken after the decisions are written down, with margins fixed in advance. The gate has four parts.

### 7.1 Held-out environment family

One family that appears in no SFT slice and no RL environment. Reported as success rate with a paired interval against the SFT checkpoint. This is the only part of the gate that tests agentic transfer rather than retention.

### 7.2 Tool-calling and dual-control slices

- A **BFCL** slice, reported per category rather than as a single number. [[bfcl]] shows why: gpt-4o-2024-11-20 in prompting mode scores 95.5 and 94.0 on single-turn AST multiple and parallel calls but 59.0 on multi-turn base and 6.0 on memory (Table 1, §5.6), and a model can move in opposite directions on relevance and irrelevance — Qwen2.5-72B-Instruct in prompting mode scores 100.0 relevance and 72.8 irrelevance. The lab reports Simple, Multiple, Parallel, Irrelevance, Relevance, and Multi-Turn separately, in one fixed mode (function-calling or prompting), because [[bfcl]] analyses mode effects in §5.1: GPT-4-turbo-2024-04-09 scores 83.8 irrelevance in function-calling mode and 35.6 in prompting mode (Table 1).
- A **τ²-bench** slice for dual-control behaviour. [[tau2-bench]] telecom has 114 tasks with 15 write and 15 read user tools; each task is run four times at temperature 0, and the metric is pass^k, "the fraction of k independent runs that succeed". pass^k falls with k, unlike pass@k. Reference points: gpt-4.1 pass^1 is 74% on retail, 56% on airline, and 34% on telecom; in the telecom mode ablation gpt-4.1 moves from 0.34 default to 0.67 with no user and 0.88 with an oracle plan, and o4-mini from 0.42 to 0.73 to 0.96 (Fig. 4 left). A model whose pass^1 rises while pass^4 does not has become less consistent, not more capable.

### 7.3 Abstention and non-agentic regression

- **Abstention.** The construction in [[reasoning-trap-tool-hallucination]] gives two settings: No-Tool-Available, where the required tool is absent from the system prompt, and Distractor-Tool, where one irrelevant tool is supplied. The benchmark is 296 × 2 tool-query pairs built from 349 Agent-SafetyBench tools, with a DeepSeek-R1 judge whose agreement rate with humans is not reported (App. A.1, A.4). The reference result is the reason this check exists: after think-then-act GRPO, Qwen2.5-7B-Instruct moved from 34.8% to 90.2% on No-Tool-Available and from 54.7% to 100.0% on Distractor-Tool, while BFCL Multi-Turn rose from 13.6 to 23.5 (Table 3). GRPO on GSM8K alone, with no tools in the training data, also raised both rates (Fig. 3).
- **MMLU-Pro**, 12,032 questions over 14 disciplines with up to ten options each (83% of questions have ten; the average is 9.47), scored with the paper's 5-shot chain-of-thought protocol. Two harness properties matter for this lab: a failed answer extraction falls back to a random option, so a model that stops emitting the expected format is scored at chance rather than flagged; and the prompt-variation range across 24 prompts is about 2% with a maximum of 3.74% ([[mmlu-pro]] §4, §6.3). The lab freezes one prompt and one extraction regex for every checkpoint and reports the frozen setting.
- **IFEval**, 541 prompts over 25 verifiable instruction types, reported as prompt-level strict accuracy ([[ifeval]] §1, §2.2). The reference change after tool RL is −2.6 points ([[reasoning-trap-tool-hallucination]] Table 3), which is smaller than the half-width a single 541-item paired comparison resolves (§7.4).

### 7.4 Intervals, resampling, and what this size of lab can detect

[[adding-error-bars-evals]] gives the estimators. For one score with per-item scores s_i and mean s̄ over n items, SE = sqrt((1/(n−1)) Σ_i (s_i − s̄)² / n) (Eq. 1). For a comparison of two checkpoints on the same items, the paired standard error is SE_paired = sqrt(SE²_A + SE²_B − 2 SE_A SE_B Corr(s_A, s_B)) (§4.2). The author recommends the paired form "wherever practicable" and regards bootstrapping as unnecessary "unless a complicated sampling scheme or estimator is being used" (§2.1). pass^k over four trials per task is such an estimator, so this lab uses a paired bootstrap over tasks for pass^k and the analytic paired form for item-level suites, and says which was used for each number (Interpretation: the source does not discuss pass^k).

**Worked example 1 — IFEval.** At n = 541 and a score near 0.60, SE_Bernoulli = sqrt(0.6 × 0.4 / 541) = 0.0211, so 2.11 points. An unpaired difference of two checkpoints has SE = sqrt(2) × 2.11 = 2.98 points and a 95% half-width of 5.84 points. With a per-item correlation of 0.7 between the two checkpoints, the paired SE is 2.11 × sqrt(2 − 2 × 0.7) = 1.63 points and the half-width is 3.20 points. The −2.6-point IFEval change reported after tool RL is inside both intervals, so a single 541-item run cannot resolve it; the lab therefore states IFEval results as a non-inferiority check against a declared margin, not as a measured drop.

**Worked example 2 — held-out agent family.** [[adding-error-bars-evals]] Eq. 10 gives the minimum detectable effect δ = (z_{α/2} + z_β) · sqrt((ω² + σ²_A/K_A + σ²_B/K_B)/n), with ω² the variance of the paired per-task conditional means, σ² the mean conditional variance, K the trials per task, and n the tasks. Take the paper's illustrative binary-score values σ²_A = σ²_B = 1/6 and per-task difficulty variance 1/12 for each checkpoint (§3.1, §4.2), together with a correlation of 0.7 between the two checkpoints chosen for this example rather than taken from the paper, which uses 0.5. Then ω² = 2 × (1/12) × (1 − 0.7) = 0.05 (derived). At α = 0.05 and β = 0.20, z_{0.025} + z_{0.20} = 1.96 + 0.8416 = 2.8016. With n = 114 tasks and K = 1: δ = 2.8016 × sqrt((0.05 + 0.3333)/114) = 0.162, or 16.2 points. With K = 4: δ = 2.8016 × sqrt((0.05 + 0.0833)/114) = 0.096, or 9.6 points. To reach δ = 5 points at K = 4, Eq. 9 gives n = 2.8016² × 0.1333 / 0.05² = 419 tasks.

This is the number that sets the lab's task budget, and it is why the memo states non-inferiority margins instead of claiming small gains. [figures/generality-gate-power.html](figures/generality-gate-power.html) computes δ and n for the reader's own n, K, ω², and σ², so a gate can be sized before the run instead of explained after it.

**Two rules that follow.** Resampling each task K times reduces only the conditional-variance term: Var(s_i) = σ²_i / K. In the paper's worked example K = 4 reduces the total variance by 1/2, and the reduction reachable by raising K without limit is 2/3 (§3.1). Lowering the sampling temperature is not a substitute: it can move variance into the conditional means instead of removing it, and in the paper's single-token example Var(x) rose from 1/12 at T = 1 to 1/4 at T = 0 (§3.3). [[tau2-bench]] runs at temperature 0 with four trials per task; the lab keeps the benchmark's own setting for benchmark numbers and uses its training temperature only for training diagnostics.

### 7.5 Gate decision rule

The run passes when, at 3 seeds: (a) trained-family reward improves with a paired interval excluding zero; (b) the held-out family does not fall below its declared margin; (c) no BFCL category, and no τ²-bench pass^1 or pass^4 value, falls below its declared margin; (d) the two abstention rates do not rise beyond their declared margins; (e) MMLU-Pro and IFEval each clear their non-inferiority margins. A failure on (b), (c), (d), or (e) with a pass on (a) is the outcome this lab exists to detect, and it is reported as such rather than repaired by re-tuning.

## §8 Post-mortem on one agentic failure

Pick the single worst-scoring gate cell and trace it to rollouts. The four candidate causes and the evidence that distinguishes them:

| Candidate | Distinguishing evidence in `rollouts/` | Reference |
|---|---|---|
| Reward hack | high reward with the verifier's own check passing for a reason unrelated to the task; for example a patch that passes selected tests while editing unrelated files | [[deepswe]] §2.3 gives this exact example as the reason for masking |
| Loop | loop-turn rate rising with trajectory length; anchor-state groups of size ≥ 10 growing | [[gigpo-verl-agent]] §5.5 |
| Context loss | failures concentrated in rollouts whose prompt at the last turn is near the context limit; truncation rate rising | §4, §6 |
| Tool hallucination | tool-call validity falling, or the abstention rates of §7.3 rising while trained-family reward rises | [[reasoning-trap-tool-hallucination]] Table 3 |

The memo records one causal chain with the rollout ids that support it, one change that would test it, and whether that change was run.

## Negative samples and negative feedback

**Which kind of negative.** Following the four-way distinction: (1) *negative marginal value* — trajectories discarded by a filter; (2) *negative as content* — a failure written into an ordinary cross-entropy target; (3) *negative as conditioning* — a failure trained under a control token; (4) *negative as gradient* — an explicit decrease of a sample's likelihood. This lab uses (1) in Stage A, (2) in one Stage A slice, and (4) in Stage B. Type (3) does not appear in any source in this chapter's set, which is an open question for agentic training rather than a settled choice.

**Where the negatives come from.** In Stage A they come from the trajectory generator and its verifier: [[agenttuning]] keeps r = 1 only, leaving 1,866 trajectories from 35,341 instructions (Table 1); [[toucan-mcp]] discards trajectories containing failed tool responses (§3.1); [[swe-smith]] keeps at most 3 of the 36% of rejection-sampling attempts that resolved (§3). In Stage B they come from the environment verifier: a test suite, an assertion, or an answer check returns 0. False-negative rates are not reported by any of these sources.

**What practice does with them.**
- *Discard* (type 1) is the Stage A default, and the one measurement of the alternative is [[agenttuning]] Table 2: unfiltered trajectories at 7B scored 1.34 held-in and 0.47 held-out against 1.96 and 0.65 for filtered.
- *Use as content* (type 2) appears in [[toucan-mcp]] Ext.1: server metadata is shuffled so tasks are unsolvable with the given tools, and only trajectories with zero tool calls are kept, giving 40K irrelevance instances in the 119.3K SFT subset. The measured trade in Table 2 at 7B is irrelevance 67.93 → 75.18 against relevance 72.22 → 66.67. Abstention training and willingness to act move together, and both must be on the gate.
- *Use as gradient* (type 4) is Stage B: a failed rollout receives a negative advantage through the group baseline, as computed in §5.

**Mechanism.** For a softmax over logits z with target token y, the gradient of the log-probability is

```
∂ log p_y / ∂ z_j = 1[j = y] − p_j
```

where p_j is the model's probability of token j. A negative advantage reverses the sign, so the update lowers p_y and raises the other tokens in proportion to their current probability. The mass removed from a rejected continuation therefore lands on whatever the model already considered most likely next. In an agentic setting the most likely alternative to a rejected tool call is frequently a different tool call or a plain answer with no call, which is the direction [[reasoning-trap-tool-hallucination]] observes: the No-Tool-Available rate, where answering without the tool counts as a hallucination, rose from 34.8% to 90.2% after tool RL (Table 3). That the mechanism explains the observation is an **Interpretation**; the paper reports the representational analysis (CKA below 0.75 in early and middle layers on tool inputs, §5.1) but no token-level account.

**Evidence for benefit and for failure.**
- Benefit: GRPO with binary outcome rewards, which is negative-as-gradient for every failed rollout, raised ALFWorld overall success at 1.5B from 4.1 (prompting) to 72.8 ± 3.6 and WebShop success from 5.2 to 56.8 ± 3.8 ([[gigpo-verl-agent]] Table 1). DeepSWE raised SWE-Bench-Verified pass@1 from 23% to 42% in about 200 RL steps with the same kind of signal ([[deepswe]] Fig. 2).
- Failure: the abstention results above; and the DPO mitigation in [[reasoning-trap-tool-hallucination]] §6, where pairs with a fabricated tool call as the rejected response lowered No-Tool-Available from 90.2% to 55.8% and Distractor-Tool from 100.0% to 71.4% while the SynTool task reward fell from 0.45 to 0.34. Negative gradients fixed the failure they targeted and cost target performance.

**Controls this lab applies.**
1. *Mask instead of penalize* for trajectories with no verifier outcome ([[deepswe]] §2.3), with the baseline treatment recorded (§5).
2. *Localize* the negative signal to the step rather than the trajectory where a step-level advantage is available ([[gigpo-verl-agent]] Eq. 7-8); the weighting coefficient ω is fixed at 1 in the source, and its WebShop sweep peaks at ω = 0.8 (Table 5).
3. *Bound* the update with the clipped objective; [[deepswe-recipe]] records `clip_ratio_high` 0.28 with no lower clip set in the released script.
4. *Keep on-policy*: log and bound the sampler-learner gap ([[rollout-training-mismatch-tis]]).
5. *Anchor abstention with positive targets*: the irrelevance instances of [[toucan-mcp]] are positive cross-entropy targets for not calling a tool, and they enter the SFT mixture rather than the RL reward.

**Diagnostics.** Log chosen and rejected trajectory returns separately; entropy split by advantage sign; the zero-variance group fraction; tool-call validity; and the two abstention rates. Report trained-family success at k = 1 and at k = 4 trials, since a rise at k = 1 with no rise at k = 4 is the pattern [[tau2-bench]] describes as less consistent performance (§4.2).

**Honesty about the size of the effect.** No source in this chapter's set decomposes an agentic gain into a positive-sample and a negative-sample share. The comparable decomposition exists only for single-turn reasoning and is covered in [[ch-43a]]. This lab therefore does not claim that negatives drive the agentic gain; it measures what changes when their treatment changes, which is the §5 axis.

## Recipe

All rows were read at the stated locus on the stated date. "Samples per task" means full multi-turn trajectories; "examples" means SFT instances.

| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value | Lab value | Reason for difference |
|---|---|---|---|---|---|---|---|---|---|
| AgentLM-7B / 13B | 7B; 13B | SFT | agent sampling ratio η | 0.2 | arXiv:2310.12823v2 §2.2.2, App. A ([[agenttuning]]) | verified 2026-09-14 | η scanned 0–1 in 0.1 steps at 7B, best held-out; per-η scores not tabulated | η ∈ {0.2, 0.5} | the share is the tested axis |
| AgentLM-7B / 13B | 7B; 13B | SFT | peak LR; batch (unit not stated); sequence; warmup ratio; schedule | 5e-5; 64; 4,096; 0.02; cosine | §2.2.3, App. A Table 6 | verified 2026-09-14 | no ablation reported | LR from the slice used (see next two rows); batch 64; sequence 32,768 | agent trajectories exceed 4,096 tokens |
| AgentLM-7B / 13B / 70B | all | SFT | loss masking | loss on model output only | §2.2.3 | verified 2026-09-14 | no ablation reported | same, plus a unit test on the response mask | §3.1 |
| Toucan SFT models | 7B / 14B / 32B | SFT | LR; epochs; effective batch; optimizer; max sequence | 2e-5; 2; 64; AdamW (0.9, 0.999) ε 1e-8; 32,768 | arXiv:2510.01179v1 App. C.2 Table 5 ([[toucan-mcp]]) | verified 2026-09-15 | no ablation reported | LR 2e-5, 2 epochs at 7B | matches the slice the mixture draws from |
| SWE-agent-LM-7B / 32B | 7B; 32B | SFT | LR; epochs; max context; hardware | 5e-5; at most 3; 32,768; 2–8 H100 | arXiv:2504.21798v2 App. F.1 ([[swe-smith]]) | verified 2026-09-15 | Fig. 4, Fig. 5, Tables 4-5 ablations on data, not on these values | not used | the 2.5× higher LR is not reconciled with the other two slices |
| ADP-trained models | 7B / 8B / 14B / 32B | SFT | corpus; sampling multipliers | 13 datasets, 1.3M trajectories; w_d from 0.001 (orca agentinstruct) to 3 (swe-gym openhands) | arXiv:2510.24702v2 §2.1, App. C Table 9 ([[agent-data-protocol]]) | verified 2026-09-15 | Table 6 and Table 10: ADP 16.6 vs SWE-smith-only 11.0 at about 30K samples | multipliers not reused; the lab mixes families in equal proportion | the lab's family count is 3, not 13 |
| Qwen2.5-1.5B/7B-Instruct + GiGPO | 1.5B; 7B | RL | LR; group size N; groups per rollout; rollout / validation temperature; mini-batch; KL loss coefficient; γ; ω | 1e-6; 8; 16; 1.0 / 0.4; 256 (ALFWorld), 64 (WebShop); 0.01; 0.95; 1 | arXiv:2505.10978v3 App. E.1 ([[gigpo-verl-agent]]) | verified 2026-09-15 | Table 5 ω sweep on WebShop peaks at ω = 0.8; F_norm comparison in §5.2 | same, with F_norm = 1 fixed | F_norm is task-dependent (§5.2); fixing it keeps §5 the only negative-signal axis |
| Qwen2.5-1.5B/7B-Instruct + GiGPO | 1.5B; 7B | RL | reward; invalid-action penalty; episode step limit; max prompt / response tokens | 10 / 0; −0.1; 50 (ALFWorld), 15 (WebShop); 2,048 / 512 and 4,096 / 512 | App. E.1 | verified 2026-09-15 | no ablation reported | same reward and penalty; step limit per family | the invalid-action penalty is held fixed so §5 is isolated |
| Qwen2.5-1.5B/7B-Instruct + GiGPO | 1.5B; 7B | RL | history in prompt | last 2 observation-action pairs (ALFWorld, WebShop); full history (search QA) | App. E.2 | verified 2026-09-15 | no ablation reported | keep-recent-2 is one of three arms | the context policy is a tested axis (§4) |
| Qwen2.5-1.5B / 7B + GiGPO | 1.5B; 7B | RL | compute; iterations | 2×H100 (1.5B), 4×H100 (7B); 150 iterations | App. E.1 | verified 2026-09-15 | n/a | same | the constrained path targets this scale |
| DeepSWE-Preview | 32B | RL | LR; clip_ratio_high; trajectories per task; rollout temperature; entropy coefficient; KL | 1e-6; 0.28; 8; 1.0; 0.0; `use_kl_loss=False` | rllm@709dec43b740 `scripts/agent/swe/deepswe_32b.sh` L24, L34, L49, L47, L53, L33 ([[deepswe-recipe]]) | verified 2026-09-14 (script only; the blog prints none of these) | no ablation reported | LR 1e-6, 8 trajectories per task at 7B; KL loss coefficient 0.01 from the GiGPO row | a KL term is kept because the lab starts from an SFT checkpoint it wants to stay near |
| DeepSWE-Preview | 32B | RL | masked trajectories | blog: max context, max environment steps, 20-minute generation timeout; script: `agent.overlong_filter=True`, `algorithm.mask_truncated_samples=False` | Blog §2.3; script L55, L70 | verified 2026-09-14 | Blog Fig. 6 (Qwen3-14B, curves only) | mask and penalize are both run | the treatment is the tested axis (§5) |
| DeepSWE-Preview | 32B | RL | max response length; max environment steps (training) | 32,768; 50 | script L18-L19, L69 | verified 2026-09-14 (script only) | no ablation reported | set per family from the step limit above | environments differ |
| τ²-bench | n/a | eval-gate | trials per task; temperature; telecom task count | 4; 0; 114 evaluated of 2,285 generated | arXiv:2506.07982v1 §4.1, Table 1 ([[tau2-bench]]) | verified 2026-09-15 | n/a | same | benchmark settings are not changed |
| MMLU-Pro | n/a | eval-gate | prompting; items; answer extraction | 5-shot CoT; 12,032 questions; two regexes then a random option | arXiv:2406.01574v6 §4 ([[mmlu-pro]]) | verified 2026-09-15 | §6.3: 24-prompt range about 2%, maximum 3.74% | same, with one frozen prompt and extraction regex | the fallback scores format failures at chance |
| IFEval | n/a | eval-gate | items; metric | 541 prompts, 25 instruction types; prompt-level strict accuracy | arXiv:2311.07911v1 §1, §2.2 ([[ifeval]]) | verified 2026-09-15 | n/a | same | — |

**Starting point for a small general-purpose run.** For a 1.5B instruct checkpoint: SFT at LR 2e-5 for 2 epochs with effective batch 64 and maximum sequence 32,768 on a mixture at η = 0.2, with loss on model output only; then RL at LR 1e-6 with 8 trajectories per task, 16 tasks per rollout, rollout temperature 1.0, validation temperature 0.4, KL loss coefficient 0.01, discount 0.95, F_norm = 1, reward 10 for success and 0 for failure with −0.1 per invalid action, and a step limit set per family. Every one of those numbers comes from a `verified` row above: the SFT values from Toucan's 7B–32B runs on 119.3K instances, the η from AgentTuning's 7B scan on Llama-2-chat, and the RL values from GiGPO's Qwen2.5-1.5B runs on ALFWorld and WebShop with 2 H100 GPUs for 150 iterations. None of them was tuned for the combination used here.

## Generalization lens

**(a) What increases breadth.**
- Mixing agent data with general instruction data rather than training on agent data alone: at 7B, held-out 0.67 and general 0.63 for η = 0.2 against 0.09 and 0.22 for agent-only ([[agenttuning]] Table 5).
- Drawing the agent slice from many environments rather than one: with the trajectory count fixed at 700, drawing from 4, 25, 50, and 100 repositories resolved 10.3%, 11.5%, 12.9%, and 15.1% of SWE-Bench Verified ([[swe-smith]] §4.1, Fig. 5).
- Mixing agent-data *sources* at equal scale: Qwen-3-8B with the OpenHands harness scored 16.6 on SWE-Bench Verified with about 30K ADP samples against 11.0 with up-sampled SWE-smith alone ([[agent-data-protocol]] Table 10).
- Explicit abstention data as positive targets: Toucan's irrelevance extension raised BFCL irrelevance from 67.93 to 75.18 at 7B ([[toucan-mcp]] Table 2).
- Multi-turn RL itself, within the trained families: ALFWorld overall 4.1 → 72.8 ± 3.6 at 1.5B under GRPO ([[gigpo-verl-agent]] Table 1).

**(b) What causes narrowing or forgetting.**
- Concentrating the agent slice in one environment: 700 SymPy trajectories at 7B raised the SymPy subset from 13.6 to 21.2 and lowered Verified-without-SymPy from 15.3 to 14.0; at 32B, SymPy 33.3 → 42.4 and the remainder 40.2 → 38.3 ([[swe-smith]] §4.1).
- Agent-only SFT: general score 0.22 at 7B and 0.19 at 13B against 0.63 and 0.69 for the mixture ([[agenttuning]] Table 5). Even the mixture moved the 70B general benchmarks in both directions: MMLU 62.1 → 59.5, HumanEval 30.8 → 28.7, GSM8K 54.7 → 59.7, MT-Bench 6.85 → 7.26 (Table 4).
- Tool-use SFT shifting the willingness to act: Toucan raised multi-turn from 12.88 to 22.62 and lowered non-live AST from 84.19 to 78.52 and relevance from 72.22 to 66.67 at 7B ([[toucan-mcp]] Table 2).
- Reasoning RL, tool-related or not, raising tool hallucination: 34.8% → 90.2% and 54.7% → 100.0% after think-then-act GRPO on SynTool, and both rates also rising under GRPO on GSM8K alone ([[reasoning-trap-tool-hallucination]] Table 3, Fig. 3). Removing the `<think>` block lowered the effect and the task reward together (41.4 / 63.6 at reward 0.28 against 90.2 / 100.0 at reward 0.45, Table 2).
- Training in one language or one runtime: SWE-agent-LM-32B scored 8.4% on SWE-bench Multilingual against 43% for Claude 3.7 Sonnet, with edits that "reflected syntax closer to Python" ([[swe-smith]] App. F.4).

**(c) How generality is measured for this stage.** A held-out environment family, reported with a paired interval (§7.1); per-category tool-calling scores in one fixed mode, not a single BFCL number (§7.2); pass^k at more than one k, since pass^1 and pass^4 can move in different directions ([[tau2-bench]] §4.2); an abstention check on prompts whose required tool is missing, because the study that ran one found the target benchmarks did not reveal the change ([[reasoning-trap-tool-hallucination]] §4.4.2); and non-agentic suites with declared non-inferiority margins sized by the power calculation of §7.4. Known measurement errors: pass^k with four trials per task on 114 tasks resolves about 9.6 points (§7.4); MMLU-Pro moves about 2% across 24 reasonable prompts ([[mmlu-pro]] §6.3); the τ²-bench telecom user simulator itself has a 16% error rate with 6% critical errors ([[tau2-bench]] §1); and the abstention judge's agreement with humans is not reported ([[reasoning-trap-tool-hallucination]] App. A.4).

## Common mistakes and how to detect them

| Mistake | Observable symptom | Check |
|---|---|---|
| Tool-response tokens carry loss | training loss falls faster than in a single-turn run at the same data size; the model starts emitting text that looks like tool output | assert the response mask is 0 on every tool and template token before step 1 ([[verl-rollout]] `continuous_token.py` L390-413) |
| The history is re-tokenized between turns instead of concatenated | reward flat or falling with no other anomaly | compare token ids from the rollout against ids produced by re-applying the chat template; the verl documentation reports non-convergence from this even in single-turn training |
| Truncated rollouts scored as failures without saying so | the negative advantage on verified failures differs from the value implied by the group composition | recompute advantages by hand for one group with the §5 table; log the termination reason for every rollout |
| Reward curve pooled across environment families | aggregate reward rises while one family falls to zero | plot reward per family; the gate of §7.1 uses the held-out family only |
| Gains reported without trials per task | pass^1 rises, pass^4 does not | report both, with the trial count, following [[tau2-bench]] §4.2 |
| Temperature lowered to make scores stable | the variance of the score drops but the mean also moves | keep the benchmark's own temperature; [[adding-error-bars-evals]] §3.3 shows Var(x) rising from 1/12 to 1/4 when T goes from 1 to 0 in a single-token example |
| Unpaired intervals on paired runs | intervals wide enough that every gate passes | use the paired form; at n = 541 and correlation 0.7 the half-width falls from 5.84 to 3.20 points (§7.4) |
| Gate suites overlap the training slices | held-out family improves as much as the trained families | run the n-gram decontamination of §2.4 before the gate; none of the three data sources reports such a check against agent benchmarks |
| Format breakage read as a capability drop | MMLU-Pro falls several points with no other change | count failed answer extractions; the harness scores them at chance by selecting a random option ([[mmlu-pro]] §4) |
| Abstention treated as safety, not capability | abstention rate rises while relevance falls | report irrelevance and relevance together; Toucan moved them in opposite directions ([[toucan-mcp]] Table 2) |

## Check your understanding

1. At 7B and 13B, mixing agent data with general data gives a held-out agent score close to general-only training, while at 70B the mixture is far ahead (0.67 vs 0.64; 0.78 vs 0.81; 1.40 vs 0.98). What does this pattern imply about which claim a 1.5B lab is entitled to make about agent-data share, and which claim it is not?
2. Masking a truncated rollout's loss and removing it from the group baseline are different operations. Using the leave-one-out advantage with n = 8, 2 successes, 4 failures, and 2 truncated rollouts, compute both and explain which verified rollouts are affected and in which direction.
3. A run shows rising trained-family reward, a rising zero-variance group fraction, and a falling number of gradient-carrying trajectories per iteration. Give two different causes consistent with these three observations and one measurement that separates them.
4. The Reasoning Trap study found that GRPO on GSM8K alone raised tool-hallucination rates, even though the training data contained no tools. Using the softmax logit gradient, give a mechanism-level account of why a negative gradient on one distribution of outputs can change behaviour on a different distribution of inputs, and state which part of your account the paper measures and which part it does not.
5. Toucan SFT raised BFCL irrelevance from 67.93 to 75.18 and lowered relevance from 72.22 to 66.67 on the same checkpoint. Why does reporting the BFCL overall score alone hide the thing a general-purpose model needs to know here, and what would you have to add to the training mixture to test whether the trade is necessary?
6. Your held-out environment family has 114 tasks. You want to detect a 5-point change with 80% power at α = 0.05. Using the §7.4 values, how many tasks or how many trials per task would be needed, and why does increasing trials per task run into a limit that increasing tasks does not?
7. GiGPO reports that anchor-state groups of size ≥ 10 exceed 20% at iteration 10 and fall by iteration 75. Explain what the policy is doing in each regime, and why this distribution is a better loop instrument than trajectory length alone.
8. DeepSWE reports no evaluation outside SWE-Bench-Verified. Suppose it had run this chapter's gate and the held-out family had fallen. Which of DeepSWE's stated design choices would be the first two candidates, and what would you change in a re-run to separate them?

## Connections

- **Previous chapter.** [[ch-46]] — *Lab: DPO or RLVR Experiment with Negative-Signal Ablation and Held-Out Capability Retention*. That lab measures a single-turn preference or RLVR stage; this one adds turns, an environment, and a rollout that can end without a verdict.
- **Next chapter.** [[ch-47]] — *Evaluation Harness and Suite Design for General Capability*. The suites assembled ad hoc in §7 are designed systematically there.
- **Dependency.** [[ch-46]] supplies the negative-signal ablation design, the paired-interval protocol, and the held-out retention suite that §7 extends to agents.
- [[ch-29c]] — *Agentic Environment and Task Synthesis at Scale*: where the Toucan, ADP, and SWE-smith slices come from, and how new environments are built.
- [[ch-30b]] — *Multi-Skill SFT Mixtures: Interference, Transfer, and Agentic and Long-Context Shares*: the mixture theory behind the η axis of §2.
- [[ch-45b]] — *Multi-Turn Agentic RL: Observation Masking, Credit Assignment, and Stability*: the mechanics of §3 and §5 in full, including step-level credit assignment and the stability failures this lab only instruments.
- [[ch-45c]] — *Context Management for Long-Horizon Agents*: the context policies compared in §4, including trained summarization.
- [[ch-45d]] — *Open Agentic Recipes Side by Side: Stage Placement, Data Mixture, and Agentic RL*: which labs put agentic capability at which stage, and which of them report general ability afterwards.
- [[ch-43a]] — *Negative Samples and Negative Gradients: Likelihood Displacement, Squeezing, and Negative Advantages*: the derivations behind the negative-feedback section.
- [[ch-51a]] — *Evaluating Agent Generality and Reliability*: the evaluation counterpart of §7, including pass^k and harness-level agent measurement.
- [[ch-30a]] — *Forgetting and Alignment Tax in Fine-Tuning: Measurement and Control*: the forgetting-report format the memo reuses.
- [[ch-36]] — *Lab: SFT Run with Masking Tests, a Forgetting Report, and a Held-Out Evaluation Split*: the masking unit tests of §3.1 are the multi-turn version of that lab's gate.

## Sources

- [[agenttuning]] — the agent-share axis: the mixture objective, the η = 0.2 scan, and Table 5's agent-only / general-only / mixed comparison at three sizes.
- [[toucan-mcp]] — an MCP-server trajectory slice, its SFT hyperparameters, its irrelevance extension, and the BFCL category trade it produces.
- [[agent-data-protocol]] — a unified multi-source agent corpus and the equal-scale comparison of mixed against single-source agent SFT.
- [[swe-smith]] — environment-family specialization and repository-pool diversity, and the cross-language transfer limit.
- [[gigpo-verl-agent]] — the multi-turn RL configuration used by the constrained path, the episode and step advantage formulas, the anchor-state group distribution used as a loop instrument, and the per-iteration cost breakdown.
- [[deepswe]] and [[deepswe-recipe]] — compact filtering of trajectories with no verdict, the leave-one-out advantage, the released script values, and the absence of any evaluation outside the target benchmark.
- [[tau2-bench]] — the dual-control slice of the gate, the pass^k definition and trial protocol, and the user-simulator error rate.
- [[bfcl]] — per-category tool-calling evaluation, function-calling versus prompting mode effects, and the relevance / irrelevance split.
- [[reasoning-trap-tool-hallucination]] — the abstention check, and the result that tool-calling and instruction-following benchmarks did not reveal the change it measures.
- [[adding-error-bars-evals]] — the paired standard error, the resampling relation, the temperature warning, and the power formula used to size the gate.
- [[mmlu-pro]] — item count, prompting protocol, the random-option fallback on extraction failure, and the prompt-variation range.
- [[ifeval]] — item count, instruction types, and the metric reported in the regression check.
- [[verl-rollout]] — the response-mask contract, token-in-token-out generation, and the re-tokenization warning.
- [[entropy-logging-patterns]] — what `actor/entropy` and the KL metrics mean in verl, and the framework defaults the lab must record.
- [[rollout-training-mismatch-tis]] — the sampler-learner probability gap logged as a stop condition.
- [[grpo]] — the group-baseline advantage the lab's algorithm is built on.
- [[dr-grpo]] — the argument against the standard-deviation normalization, which is why F_norm = 1 is fixed.
- [[r2e-gym]] and [[swe-gym]] — executable SWE environment sets referenced as alternative task supplies for the RL stage.
