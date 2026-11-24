<!-- chapter: ch-13
     track: rl-pipeline
     kind: content
     title: Diversity Axes of the RL Task Distribution
     deps: [ch-12]
     sources: [[diversity-axes]], [[system-prompt-diversity]], [[kimi-k2-agentic-data]], [[demystifying-agentic-rl]], [[deepseek-v3-2]], [[glm-4-5-recipe]], [[apigen-mt]], [[toolace]], [[tau-bench]], [[bfcl]], [[persona-hub]], [[consistentchat]], [[user-oriented-dialogue-generation]], [[rest-multi-problem-stress]], [[tau2-bench]], [[agentscaler]], [[simulator-collapse]], [[agent-world]], [[agentrl]], [[agentgym-rl]], [[general-reasoner-verifier]], [[ifbench]], [[scaling-reasoning-losing-control-mathif]], [[qwen-3]], [[mai-rl-infrastructure]], [[mai-thinking-1]], [[olmo-3-rl-pipeline]], [[open-instruct-rlvr-pipeline]], [[nemotron-ultra-recipe]], [[magistral]], [[minimax-m1]], [[deepseek-v3.1]], [[big-math]], [[rl-design-guide]], [[polaris]], [[scalerl]], [[dapo]], [[verifier-pitfalls]], [[deepseek-r1-recipe]], [[asearcher]]
     figures: figures/diversity-axes.html
-->

# Chapter 13 — Diversity Axes of the RL Task Distribution

> **Core insight.** An RL prompt set is a distribution over many independent properties: domain, difficulty, tool-call structure per turn, horizon, number of tasks per episode, toolset and environment count, user behaviour, system prompt, constraint count, answer format, output length and language. The optimizer only sees what the distribution contains, so each property the set leaves constant becomes an untrained input condition. For some axes a training ablation measured the cost of leaving the axis narrow (domain, difficulty, parallel tool calls, environment count, user-simulator diversity, constraint count, output length). For others only evaluation evidence exists. The clearest case of the second kind is the number of questions per prompt. Reasoning models lose 29 AIME24 points when several problems share one prompt ([[rest-multi-problem-stress]] Table 1), and several open pipelines remove multi-part prompts on purpose. No study was found that trains with and without multi-question prompts, so the causal claim is still untested.
>
> **Guideline.** When an axis has a training ablation, cover it in the RL set at least to the range the deployment needs, because the ablations show losses when it is missing: removing parallel-call data cut BFCL overall from 58.19 to 50.60 at a fixed 25,000 SFT samples ([[toolace]] App. E.2 Table 7), and RL against one frozen user simulator ended near the untrained 40.4% on held-out τ²-Retail, while population co-training reached 62.2% ([[simulator-collapse]] Table 1, §4.2). When an axis has only evaluation evidence, add a stress evaluation for it (REST-style stress levels for multi-question prompts) and treat the training fix as a hypothesis to test, because REST trains no model and its RL-versus-base comparison is one model lineage with unknown data ([[rest-multi-problem-stress]] Verification). When several domains share one policy, normalize advantages per task, because removing per-task normalization lowered a five-environment average from 65.0 to 59.4 ([[agentrl]] Table 6). Check what your framework does by default before assuming an axis is covered: verl at 8357f1e executes only the first tool call per assistant turn unless `max_parallel_calls` is raised (`rollout.yaml` L219–220, `tool_agent_loop.py` L319).

---

## Why this chapter matters

ch-09 to ch-12 treated one prompt at a time: how a task is written, rolled out, graded and selected. This chapter treats the prompt set as a distribution. The question is which properties of that distribution decide what the trained policy can still do on inputs the set did not contain.

The question matters for two reasons.

1. **RL updates every parameter from a narrow input slice.** A math-only RL set, a single-environment agent set, or a set in which every prompt contains exactly one question is narrow along at least one axis. The policy is never penalized for failing on inputs that differ along that axis, and there is no term that keeps the old behaviour (ch-40 in llm-training covers why negative advantages move mass to already-likely outputs).
2. **Pipelines narrow axes on purpose.** Graders are easiest to write for one question with one canonical answer. Qwen3's long-CoT cold start removes queries that "contain multiple sub-questions" ([[qwen-3]] §4.1). MiniMax-M1 drops proofs, multi-part and binary questions from its math set ([[minimax-m1]] §4.1), and MAI Thinking 1 rewrites MCQ and proof items into open-ended form ([[mai-thinking-1]] Sec. 3.2). Each of these choices improves grader precision and narrows the task distribution along one axis.

The applied target for this course, a sales voice agent, depends on axes that math RL sets keep constant: several requests in one user turn, users who reveal information gradually, persona-specific system prompts, and tool calls inside a dialogue. §6 and §8 use it as a short example.

---

## §1 What a diversity axis is, and how evidence is labelled

**Definition.** A diversity axis is a property of each RL item that (a) can be measured per item, (b) varies at deployment, and (c) the pipeline can control at construction or sampling time. Pass rate is an axis: it is measured from N rollouts, varies across user queries, and is controlled by filters. Topic is an axis when it is tagged.

**Evidence labels.** [[diversity-axes]] assigns each axis one of three labels, and this chapter keeps them:

| Label | Meaning | What it can support |
|---|---|---|
| causal | a training ablation changes only this axis and measures the effect | "leaving the axis narrow costs X on Y, in this setup" |
| observational | models are compared or evaluated; no training varies the axis | "models trained some way perform worse on inputs with this property" |
| practice | a lab does it; no ablation | "this is how it is done", nothing about effect size |

Almost every causal row comes from a single study on one model family ([[diversity-axes]] "Reading the table"). An effect size from one study is a statement about that setup only.

**The figure.** [figures/diversity-axes.html](figures/diversity-axes.html) lists all thirteen axes and can be filtered by evidence label. Each row shows the measurement, the measured failure, and the control. A second panel shows the REST numbers used in §6. Open it next to §2–§9.

**Where the axes are controlled in an RL system.** Rocket's controller selects each problem "from the curriculum" (Fig. 1, step 1), and its runtime settings include "mixture weights", typed and validated, and applied at step or sampling boundaries ([[mai-rl-infrastructure]] p. 2–4, p. 12–13). In open-instruct the mixture is fixed when the dataset is built: `--dataset_mixer_list` takes alternating dataset names and amounts, where a float is a proportion of the dataset and an integer is an absolute sample count.

```python
# open-instruct@f020e9a open_instruct/dataset_transformation.py L2935-L2944
dataset_mixer_list: Alternating list of [dataset_name, amount, dataset_name, amount, ...].
    The 'amount' value determines how many samples to use:

    - Float values (must contain a decimal point): Interpreted as a PROPORTION of the dataset.
      Examples: "1.0" = 100% of dataset, "0.5" = 50%, "3.0" = 300% (3x upsampling)

    - Integer values (no decimal point): Interpreted as an absolute SAMPLE COUNT.
      Examples: "100" = exactly 100 samples, "1000" = exactly 1000 samples

    IMPORTANT: "1" means 1 sample, NOT 100% of the dataset. Use "1.0" for 100%.
```

The Olmo 3 7B Think script passes one pre-mixed dataset, `allenai/Dolci-Think-RL-7B 1.0` (`scripts/train/olmo3/7b_think_rl.sh` L28), so the domain mixture is decided when that dataset is assembled and cannot change during the run. The report describes that mixture only as "roughly equal amounts of data … for each domain (with slightly more focus on math and instruction following)" ([[olmo-3-rl-pipeline]] §4.4.2). The two designs differ in one property: Rocket can change an axis's weight mid-run without a restart, and open-instruct cannot.

---

## §2 Domain

**Measurement.** The share of prompts per domain tag (math, code, science, IF, agentic, chat).

**Causal evidence.**
- Zero-RL on mixed-domain data against math-only data at matched settings: SuperGPQA 34.2 vs 29.8 at 7B and 39.5 vs 35.6 at 14B. Math-related averages were 48.5 vs 49.1 at 7B and 53.9 vs 48.6 at 14B ([[general-reasoner-verifier]] Table 4). At 7B, adding other domains cost 0.6 points on math. At 14B, math scored higher with the mixed data.
- Single-environment agent policies fail on other environments. The DB-trained policy scores 0.2 on ALFWorld, while one policy trained on all five environments averages 67.7, compared with 67.8 for the best single-task policy on each task ([[agentrl]] Table 4).
- Cross-domain transfer is not symmetric. From one 24B checkpoint at AIME'24 32.2 and LiveCodeBench v5 22.7, math-only RL reached 62.5 / 38.3 and code-only RL reached 49.7 / 42.7 ([[magistral]] §6.1, Table 5). Each single-domain run improved both benchmarks, by different amounts.

**Practice evidence.** MAI runs its agentic climb jointly with STEM tasks, stating that this "helps to stabilize the RL climb" and that agentic tasks transfer "neither positively nor negatively" to STEM ([[mai-thinking-1]] p. 40). No numbers are printed.

**Control.** Domain tags and balanced coverage; a per-task advantage normalization when domains differ in length and difficulty (removing it: 65.0 → 59.4, [[agentrl]] Table 6); and the stage-order decision of ch-17.

---

## §3 Difficulty and pass rate

ch-12 covers this axis in full. In diversity terms the point is that pass rate must be measured per policy and re-measured as the policy improves, so the axis is controlled online as well as offline.

- MAI applies two filters per group. It samples G_early = 16 rollouts and continues to G = 128 only if the early pass rate lies in [0.05, 0.8], then applies a second filter of [0.1, 0.8] on the full group ([[mai-thinking-1]] Sec. 3.1.3, hyperparameter table).
- open-instruct drops groups with `np.std(reward_scores) == 0` and, with `active_sampling`, pulls a replacement prompt ([[open-instruct-rlvr-pipeline]], `data_loader.py` L817, L1131–1135).
- Nemotron 3 Nano fixes the domain ratio per batch and draws each domain's tasks from a Gaussian target pass-rate distribution whose mean decreases linearly over steps. When progress plateaus, tasks are re-profiled with the best RL checkpoint ([[nemotron-ultra-recipe]] NR §3.2.2). Only curves are shown (Figure 7).

The causal rows are in [[diversity-axes]]: keeping only pass ≤ 4/8 hurt as much as not filtering ([[polaris]]), zero-variance filtering raised the fitted asymptote ([[scalerl]] Fig. 6a), and dynamic sampling added 8 AIME24 points ([[dapo]] Table 1).

---

## §4 Tool calls per turn: single, parallel, dependent, none

**Measurement.** Each sample's call-structure category. BFCL defines Simple (one candidate function, one call), Multiple (several candidates, one call), Parallel (one function, several calls), Parallel Multiple, Irrelevance (no call expected) and Relevance ([[bfcl]] §3.1).

**Causal evidence (SFT, not RL).** ToolACE varied data types at a fixed 25,000 samples. The full data gave 58.19 BFCL overall. Removing parallel-call data gave 50.60. Removing non-tool ("multi-type") data gave 42.71 overall and 6.99 on irrelevance detection. Removing dependent or nested data cost under 1.3 points ([[toolace]] App. E.2, Table 7). No RL ablation of this axis was found.

**Observational evidence.** Real user queries contain more multiple-function and fewer parallel cases than the curated single-turn set ([[bfcl]] §5.2). Claude-3.5-Sonnet-20241022 in FC mode scores 3.5 on AST parallel, and o1-2024-12-17 in FC mode scores 0.0 ([[bfcl]] Table 1). The authors attribute part of this to the calling mode: Claude cannot emit parallel calls in FC mode but can in prompting mode (§5.1).

**Practice.** MAI's general tool-use climb adds cross-environment graders "for parallel calls, no duplicates, valid arguments" ([[mai-thinking-1]] p. 43–44).

**A framework default that removes the axis.** In verl the tool agent loop slices the parsed tool calls before executing them:

```python
# verl@8357f1e verl/experimental/agent_loop/tool_agent_loop.py L318-L320
tasks = []
for tool_call in agent_data.tool_calls[: self.max_parallel_calls]:
    tasks.append(self._call_tool(tool_call, agent_data.tools_kwargs, agent_data))
```

```yaml
# verl@8357f1e verl/trainer/config/rollout/rollout.yaml L219-L220
# max parallel call for tools in single turn
max_parallel_calls: 1
```

With the default, a turn that emits three calls executes only the first. The other two get no tool response, and the model's tokens for them still carry loss mask 1. If parallel calls matter at deployment, raise the limit and log the per-turn call count, so that the training distribution of this axis is measured rather than assumed.

---

## §5 Horizon: turns and tool calls per episode

**Measurement.** Interaction turns or tool calls per trajectory, or the number of actions a task requires.

**Causal evidence.** In one Deep Search environment, a fixed 10-turn cap gave higher early reward and then collapsed, a fixed 5-turn cap plateaued, and a staged cap reached the highest reward ([[agentgym-rl]] §4.2, Fig. 7; one environment, no seeds reported). ASearcher argues that small turn limits (4–10 in Search-R1-style training) cap the learned tool-use horizon and trains with 32 or 128 turns ([[asearcher]] §1, §4.1). That is a design argument, not a matched ablation.

**Observational evidence.** τ²-bench pass^1 falls toward zero beyond 7 required actions ([[tau2-bench]] §4.2, Fig. 5). AgentScaler reports a "clear negative correlation" between accuracy and tool-call count (Fig. 5, values figure-only, [[agentscaler]]).

**Production distribution.** Rocket reports median duration and turns per domain from 1.13M rollouts in one day of a production run ([[mai-rl-infrastructure]] Fig. 2):

| Domain | Median duration | Median turns |
|---|---|---|
| instruction following | 2 min | 1 |
| visual Q&A | 2 min | 1 |
| math & science | 7 min | 1 |
| tool use | 12 min | 13 |
| competitive coding | 34 min | 1 |
| web research | 51 min | 89 |
| web development | 58 min | 104 |
| software engineering | 93 min | 249 |

The horizon axis is therefore also an infrastructure axis. Median turns range from 1 to 249 inside one job. When a group's slowest rollout takes hours, its fastest members wait for it, which is the staleness problem of ch-03 and ch-04 (Fig. 4 of the same post: the fastest rollout of one group waited 7.6 h).

**Control.** Staged turn caps ([[agentgym-rl]]). Tasks composed from atomic subtasks so the action count is set by construction: τ²-bench composes at most one subtask per group out of 15 groups, giving 2,285 tasks ([[tau2-bench]] §3.2, Table 3). Tool-graph walks with a maximum step count ([[agentscaler]] §2.2).

---

## §6 Tasks per episode: the learner's hypothesis as a worked case

**Hypothesis (learner).** "RL mostly on single-question prompts makes the model lose multi-question ability." This section asks what evidence exists, what it does and does not show, and how the hypothesis could be tested.

### §6.1 Is the axis actually narrow in RL sets?

Several pipelines remove multi-part prompts on purpose:

| Pipeline | What it removes or rewrites | Location |
|---|---|---|
| Qwen3 long-CoT cold start (SFT before reasoning RL) | queries that "contain multiple sub-questions" | [[qwen-3]] §4.1 |
| MiniMax-M1 math RL set | proofs, multi-part and binary questions; MCQ converted to open-ended | [[minimax-m1]] §4.1 |
| Big-Math (recommended filter) | MC, yes/no, multi-part and proof items | [[rl-design-guide]] §1 rule 1, [[big-math]] |

The graders add a structural reason. Single-answer extractors read one answer per response. open-instruct's `MathVerifier` reads one `\boxed{}` answer, the last one (`last_boxed_only_string` uses `string.rfind("\\boxed")`, open-instruct@f020e9a `open_instruct/math_utils.py` L13–14), plus one Minerva-format answer. verl's Search-R1 EM reward takes the last `<answer>` match ("If there are 2 or more matches, return the last one", verl@8357f1e `verl/utils/reward_score/search_r1_like_qa_em.py` L84–85). A prompt with two questions therefore needs a different grader: one that splits the response and checks each answer. So the extractors that make RLVR precise also make multi-question prompts expensive to include.

The table above shows that the axis is narrowed in specific public pipelines. It does not measure the share of multi-question prompts in any lab's RL set. That share is not reported in any card read.

### §6.2 What the evidence shows

**REST** concatenates s problems into one prompt, rotates indices so every question appears once in each of the s positions, and averages accuracy over stress levels above 1 ([[rest-multi-problem-stress]] §3.1, §4.1). Results (Table 1):

- DeepSeek-R1: AIME24 81.66 → 52.49; AIME25 68.75 → 37.17.
- Models with similar single-question scores split under stress. On MATH500, R1-7B goes 93.00 → 66.75 and R1-32B goes 94.60 → 88.97.
- Simple tasks are not affected. On GSM8K, R1-7B goes 89.49 → 89.06.
- The RL-versus-base pair: at stress 9 on MATH500, AReaL-boba-RL-7B scores 38.07 and Light-R1-7B-DS 39.11, against 49.42 for their base R1-7B (§4).
- Error types for DeepSeek-R1 on AIME24: question omission plus reasoning error account for 42.00% of the degradation (Fig. 5d). Earlier questions are answered more accurately than later ones (Fig. 7).

**τ-bench** finds that 19% of 36 analysed gpt-4o retail failures are partial resolution of compound requests ([[tau2-bench]] §5.2).

### §6.3 Why this does not establish the hypothesis

The evidence is **correlational**, for four separate reasons:

1. **REST trains no model.** All comparisons are between released checkpoints ([[rest-multi-problem-stress]] Verification).
2. **The RL-versus-base comparison is one lineage with unknown data.** AReaL-boba-RL-7B is RL on R1-7B. Its RL prompt composition, length settings and step count are not part of REST, so the drop cannot be attributed to the single-question share of its RL data.
3. **Length is a confound.** Length-reduced models do better under stress: L1-Qwen-1.5B-Max 67.11 vs R1-1.5B 22.40 at stress 9 on MATH500, and Efficient-R1-7B 66.02 vs R1-7B 49.42, despite lower single-question accuracy (88.20 vs 93.00) ([[rest-multi-problem-stress]] §4). Long-CoT RL lengthens responses, so an RL-induced drop could come from length and truncation rather than from the absence of multi-question prompts. REST reports that endless repetition plus truncation make up 48.90% of Light-R1-7B's degradation (Fig. 5b), and also that truncation is "not the sole cause".
4. **No training study was found.** The REST card records a web search for training studies on multi-question prompts, with none found ([[rest-multi-problem-stress]] Verification). [[diversity-axes]] states the same: "a controlled run (same base, with vs without multi-question prompts in RL) has not been found."

The closest causal result is on a **different axis**. In controlled runs on four Qwen2.5 bases, long-CoT SFT and GRPO lowered constraint compliance in 15 of 16 trained variants while raising math accuracy in 15 of 16 ([[scaling-reasoning-losing-control-mathif]] Table 4). Raising the RL maximum rollout length from 1k to 8k raised math from 28.73 to 39.82 and lowered hard IF accuracy from 19.05 to 14.29 (Table 5). This shows that reasoning RL can remove a skill the RL set does not reward. It is an analogue for the hypothesis, not a test of it.

**Status.** The hypothesis is consistent with the evidence: multi-question accuracy drops, one RL model dropped more than its base, and pipelines remove multi-part items. Its causal claim is unverified.

### §6.4 A controlled experiment that would test it (open problem)

The design must separate three causes: absence of multi-question prompts, response length, and RL itself.

1. **Base.** One instruct or base checkpoint, with REST measured before training at the stress levels of [[rest-multi-problem-stress]] §4.1.
2. **Arms.**
   - A0: RL on single-question prompts only.
   - A1: same pool, with a fraction f of prompts replaced by REST-style concatenations of 2–5 pool questions, using the same position rotation.
   - Dose arms at f ∈ {0.1, 0.3} give a dose-response curve instead of one contrast.
   - A no-RL control (the base checkpoint) and, if compute allows, an SFT arm on the same multi-question prompts to separate RL from exposure.
3. **Matched quantities.** Total prompts, total gradient steps, total generated tokens (multi-question prompts produce longer responses, so match tokens, not only prompts), and the maximum response length. Per question, the multi-question arm must not see more unique questions than A0, or it gets more data instead of only a different format.
4. **Grader.** Split the response into per-question answers. Score each answer with the same verifier as A0. Use either the fraction correct (SAcc-style) or all-correct (HAcc-style) reward, as an explicit sub-arm, because MathIF shows the two metrics diverge as constraint count grows ([[scaling-reasoning-losing-control-mathif]] §4.2).
5. **Length control.** Report mean response length per arm. Add one arm with a length penalty, or compare at matched mean length, because REST's long2short result makes length the leading alternative explanation.
6. **Metrics.** REST score per stress level, the per-position accuracy curve (Fig. 7 of REST), the omission rate (questions with no answer), single-question accuracy, and a held-out compound-request agent task if the model is an agent (τ-bench partial-resolution rate).
7. **Replication.** At least 3 seeds per arm and a second model family, because single-family RLVR results do not transfer reliably (llm-training ch-40, spurious-rewards discussion).

**Predictions.** If the hypothesis is right, A0 loses REST score relative to the base checkpoint, A1 loses less or gains, the gap grows with f, and it remains after matching length. If the drop is a length effect, the gap between A0 and A1 closes once length is matched.

**Practice that already exists.** APIGen-MT builds compound tasks by "reverse task recombination": it concatenates validated tasks that share a persona, reruns the policy check, writes a combined instruction and revalidates ([[apigen-mt]] §4.1.3). τ²-bench composes tasks from subtask groups ([[tau2-bench]] §3.2). Both are data-construction methods for agent SFT or evaluation. Neither paper reports an RL ablation on them.

**Applied example.** In a sales call a customer says, in one turn, "What does the premium plan cost, and can I keep my current number?" A policy trained only on one-intent turns has never been rewarded for answering both. Per §6.1, the cheapest test is to add compound turns to the evaluation set before adding them to training.

---

## §7 Toolset size and environment count

**Causal evidence.**
- At about 30,000 SFT instances, sampling APIs from 6, 14 or 30 clusters raised overall BFCL accuracy as the number of clusters grew, with the largest gain in relevance detection ([[toolace]] §3.3.3, Fig. 5; values figure-only).
- RL on executable environments: the four-subdomain average rose from 18.4% with 0 training environments to 38.5% with 1,978, with larger jumps at 10 → 100 and 100 → 500 and smaller gains after 500 ([[agent-world]] §4.3.3, Fig. 8).
- DeepSeek-V3.2 ran RL from V3.2-SFT on synthetic general-agent tasks only and reports improvement on Tau2Bench, MCP-Mark and MCP-Universe. V3.2-Exp, trained with RL only in search and code environments, did not improve on those benchmarks ([[deepseek-v3-2]] §4.3, Fig. 5; values only in the figure). The synthetic set has 1,827 environments and 4,417 tasks, kept only at non-zero pass@100 ([[deepseek-v3.1]] §3.2.3).

**Practice: scale reported by labs.**

| Lab / source | Tools / environments |
|---|---|
| ToolACE | 26,507 APIs over 390 domains ([[toolace]] Table 1) |
| Kimi K2 | 3,000+ real MCP tools + 20,000+ synthetic tools; synthesized system prompts per agent ([[kimi-k2-agentic-data]] §3.1.1) |
| AgentScaler | 30,000+ APIs clustered into 1,000+ database-backed domains ([[agentscaler]] §2.1) |
| MAI Thinking 1 | >150 synthetic environments, 130,000 tasks, often >50 tools per environment ([[mai-thinking-1]] p. 43–44) |

**Control.** Generate environments as a database plus tools and grade by final-state match ([[agentscaler]] §2, §3.1). Kimi K2 keeps simulated tools for coverage and uses real sandboxes for coding and SWE, where test-suite pass rates provide ground truth ([[kimi-k2-agentic-data]] §3.1.1).

---

## §8 User behaviour, persona and system prompt

### §8.1 User simulators

**Causal evidence.** On τ²-Retail with Qwen3-4B-Instruct, RL against one frozen GPT-5-mini simulator peaks at 46.1% held-out success (base 40.4%) and returns toward the base by the end of training. Persona-guided simulation reached 49.2%, an ensemble of 3 simulators 57.1%, Verbalized Sampling 55.5%, Co-Training 60.5%, and Population Co-Training 62.2% ([[simulator-collapse]] Table 1, §4.2). Under single-simulator RL, zero-variance batches rose from 60% to over 85% and entropy fell from 1.9 to 0.4 nats (App. F.3). The comparison is at matched steps (250), not matched compute, and Co-Training doubles per-step compute (App. C.4, Table 8).

**Simulator error rates.** τ²-bench audited its gpt-4.1 user simulator and found critical plus benign errors in 47% of airline, 40% of retail and 16% of telecom conversations ([[tau2-bench]] Table 2: 13/34, 12/28, 6/10). The authors attribute the lower telecom rate to user tools and observable state constraining the simulator.

**Practice.** APIGen-MT uses a persona-guided human LM that reveals the task over several turns ([[apigen-mt]] §4.2). [[user-oriented-dialogue-generation]] separates user simulation from task generation so that users disclose information incrementally (no numerical outcome is recorded in the card). [[consistentchat]] plans query sequences from intent and information-flow skeletons. Its generator sees future queries during response generation, which differs from deployment (card Limits).

### §8.2 Persona and system prompt

**Partly measured.** Removing the domain policy document from the agent's system prompt lowered τ-bench airline from 33.2 to 10.8 and retail from 61.2 to 56.8 ([[tau2-bench]] Table 3, evaluation only). Persona-conditioned synthesis produces problems that are less similar to each other than their personas are, and adding a topic constraint raises similarity ([[persona-hub]] §4.1.2, Fig. 10). [[system-prompt-diversity]] claims system-prompt pools give more behavioural variety than topic pools. The card is a practice summary for SFT and prints no RL ablation. No RL ablation of system-prompt diversity was found ([[diversity-axes]]).

**Practice.** MAI's IF data includes conflicting system, developer and user instructions across more than 40 domains ([[mai-thinking-1]] Sec. 3.4.2). Kimi K2 synthesizes a system prompt per agent ([[kimi-k2-agentic-data]] §3.1.1).

**Applied example.** For a sales voice agent trained against a simulated customer, §8.1 implies that one simulator configuration (one model, one persona, one temperature) is the measured failure case. The minimum design is several simulator families or personas, with held-out simulators kept for evaluation, plus an audit of simulator errors on a sample of conversations, as in τ²-bench Table 2.

---

## §9 Constraint count, answer format, output length, language

| Axis | Evidence (label) | Control |
|---|---|---|
| Constraint count per prompt | **causal**: n = 1…6 constraints per instance gave IFEval 71.2 at n = 1 and 85.8 at n = 6. Training on up to 5–6 generalized better than up to 3, although test prompts have ≤ 3 ([[ifbench]] Table 1, Fig. 2). **observational**: GPT-4.1 and Claude 3.7 Sonnet score below 50% on 58 unseen constraints (§1) | many constraint types, several per prompt, variable ranges that include and extend the test range ([[ifbench]] §4.3); detail in ch-14 |
| Answer format | **measured on graders, not policies**: rule-verifier recall 0.78–0.95 on math responses, 47% on general science ([[verifier-pitfalls]] Table 5, §4.3); 117,018 MC items in a 643K pool are guessable ([[big-math]] Table 2) | remove or rewrite MC, yes/no and proof items; convert to integers ([[dapo]] §3.5); model verifier for non-canonical formats ([[general-reasoner-verifier]]) |
| Output length range | **causal**: RL max length 1k → 8k raised math 28.73 → 39.82 and lowered hard IF 19.05 → 14.29 ([[scaling-reasoning-losing-control-mathif]] Table 5); 14K → 32K raised the asymptote and lowered efficiency ([[scalerl]] Fig. 9) | staged context; soft overlong punishment; per-task advantage normalization for mixed lengths ([[agentrl]]) |
| Language | **partly measured**: R1 without a language-consistency reward mixes languages; with it, consistency is stable and coding is slightly lower ([[deepseek-r1-recipe]] B.6, Fig. 7). Magistral Medium scores 73.6 AIME'24 in English and 63.7–69.3 on six translations ([[magistral]] Table 4) | language-consistency reward. MAI: R_lang = max(1 − α·n_non-English, 0), w_lang = 0.5, α = 0.005, added because mixed-language CoTs "correlate with spikes in log-probability divergence" between training and inference policies ([[mai-thinking-1]] Sec. 3.1.2) |

**Dropped axis.** [[diversity-axes]] found no RL ablation over prompt languages as a diversity lever, so multilingual prompt mixture is not given an evidence label here.

---

## §10 Balancing axes with mixture weights

Axes interact. Adding SWE tasks extends the horizon axis and changes the token share. Adding IF prompts adds the constraint axis and short outputs. A mixture weight set by prompt count is therefore not a weight by gradient contribution.

**The sample-versus-token gap.** MAI's consolidation SFT mixture is 56% STEM + coding, 11% agentic and 33% helpfulness + safety by sample, and 89%, 9% and 2% by token ([[mai-thinking-1]] Sec. 3.5, Table 10). With a token-level loss over the global batch (MAI's RL objective normalizes over all tokens in the batch, Sec. 3.1.1), a domain's share of the gradient follows its token share, not its prompt share. The 33% helpfulness + safety share by sample contributes 2% of tokens.

**What labs report about weights.**

| Lab | Mixture control | Reported ablation |
|---|---|---|
| Rocket / MAI | mixture weights as typed runtime config, applied at step or sampling boundaries ([[mai-rl-infrastructure]] p. 12–13); controller selects from a curriculum (Fig. 1) | not reported |
| Olmo 3 | "roughly equal amounts" per domain, slightly more math and IF, fixed in the dataset ([[olmo-3-rl-pipeline]] §4.4.2) | not reported; card: "Not reported: per-domain reward weights in the final mix" |
| Nemotron 3 Nano | fixed domain ratio per batch (ratios not printed) plus a per-domain pass-rate curriculum ([[nemotron-ultra-recipe]] NR §3.2.2) | curriculum vs random sampling, curves only |
| MiniMax-M1 | rule-verified reasoning first, general tasks mixed in "gradually" with dynamic weighting ([[minimax-m1]] §4.3) | not reported |
| AgentRL | per-task advantage normalization | 65.0 vs 59.4 without ([[agentrl]] Table 6) |

The only measured balancing lever in this table is per-task advantage normalization. Mixture weights themselves are set without a printed ablation in every recipe carded here.

---

## Recipe

1. **List the deployment's axes.** For each axis in §2–§9, write the range the deployed model will see, for example "1–3 requests per user turn, 1–4 parallel tool calls, 5–40 turns, 3 personas".
2. **Measure the RL set on the same axes.** Count per-item values: domain tag, pass rate under the current policy, call structure, turns, tasks per prompt, distinct tools and environments, constraint count, answer type, output-length percentiles, language. An axis that is never measured is effectively held constant.
3. **Find axes your pipeline narrows on purpose.** Check filters that drop multi-part, MC or proof items, and graders that read only the last answer (§6.1). Decide for each whether the narrowing is acceptable at deployment.
4. **Check framework defaults that cap an axis.** In verl these are `max_parallel_calls` (default 1), `max_tool_response_length` (default 256 characters) and the turn limits (`rollout.yaml` L200–223). In your own loop, check the turn cap and timeout.
5. **For axes with causal evidence** (domain, difficulty, parallel calls, horizon, environment count, simulator diversity, constraint count, output length), cover the deployment range in training and keep a held-out slice per axis.
6. **For axes with observational evidence only** (tasks per episode, system-prompt diversity), first add a stress evaluation (REST-style levels, τ-bench compound requests). Run the §6.4 ablation before spending large compute on a training fix.
7. **Set mixture weights in tokens, not prompts.** Compute each domain's token share under your loss normalization (§10). Use per-task advantage normalization when domains differ in length or difficulty.
8. **Make weights changeable mid-run** if the framework allows it (Rocket's runtime config). Otherwise plan restarts at checkpoint boundaries for mixture changes.
9. **Diversify simulators** for any user-facing agent: several families or personas in training, held-out simulators in evaluation, and a sample audit of simulator errors.
10. **Log per-axis metrics during training**: zero-variance group share per domain, per-turn call count, turns per episode, per-domain reward and entropy. [[simulator-collapse]] App. F.3 shows that these diagnostics move before held-out reward does.

---

## Common mistakes and how to detect them

| Mistake | Observable symptom | Check |
|---|---|---|
| Treating a REST-style drop as proof that single-question RL caused it | a training fix is chosen without an ablation; the fix may be a length effect | run the §6.4 design with length matched; compare against a length-penalty arm |
| Leaving the framework's per-turn call limit at 1 | the model emits several calls, only the first gets a response; parallel-call accuracy flat or falling | log `len(tool_calls)` per turn versus executed calls; check `max_parallel_calls` |
| Mixing by prompt count | a domain with long outputs dominates the gradient; short-output domains regress | compute token share per domain per batch (MAI 56% sample vs 89% token for STEM) |
| One frozen user simulator | training reward rises; held-out simulator reward peaks early and falls; zero-variance batches > 85% | evaluate on held-out simulator families every N steps; log entropy ([[simulator-collapse]] Fig. 3) |
| Graders that read only the last answer, on prompts with several questions | multi-question items score as if only the last question mattered | split responses per question; unit-test the grader on 2- and 3-question inputs |
| Single-environment agent RL | in-domain success rises; other environments fall (DB policy 0.2 on ALFWorld) | keep at least one held-out environment; train multi-environment with per-task normalization |
| Constraint ranges that stop at the test range | IFEval rises, unseen constraints stay low | evaluate on held-out constraint types (IFBench's 58); widen variable ranges ([[ifbench]] §4.3) |
| Removing an axis's items because the grader is weak, without recording it | the deployment input type is silently absent | keep a list of every filter that drops an item type, and its share of the raw pool |
| Reading a single-study effect size as general | an expected gain does not appear on another model family | replicate on a second family before adopting the number |

---

## Check your understanding

1. Rocket can change mixture weights mid-run; open-instruct fixes the mixture when the dataset is built. Explain what kind of experiment each design makes cheap, and why mid-run changes complicate the attribution of a later score change.
2. MAI's helpfulness + safety data is 33% of consolidation samples and 2% of tokens. Under a loss normalized over all tokens in the global batch, explain why this domain's influence on the gradient is closer to 2% than to 33%, and what you would change if the domain regressed.
3. The REST results show R1-7B and R1-32B with near-equal single-question MATH500 scores but very different stressed scores. Why does this make single-question accuracy a poor selection metric for a model that will receive compound requests, even before any causal claim about training?
4. Give two mechanisms other than "no multi-question prompts in RL" that could explain why AReaL-boba-RL-7B scores below its base under stress. For each, name the measurement in the §6.4 design that would separate it from the hypothesis.
5. Why does a last-answer extractor make multi-question RL prompts expensive, and how does the choice between a fraction-correct and an all-correct reward change the gradient a partially correct multi-question response receives?
6. In [[simulator-collapse]], held-out reward rises and then returns toward baseline while training reward keeps rising. Using the paper's mode-collapse argument, explain why a single frozen simulator produces this pattern, and why zero-variance batches rise at the same time.
7. verl's default `max_parallel_calls: 1` executes only the first call of a multi-call turn, but the extra calls are still trainable model tokens. What does the policy learn from those tokens under an outcome reward, and why can this push the model away from emitting parallel calls?

---

## Connections

- **ch-09, ch-10, ch-11, ch-12** — task, rollout, grader and selection design for one item. This chapter is the distribution-level view of the same decisions. Difficulty filtering is treated in full in ch-12.
- **ch-03, ch-04** — staleness. The horizon axis (§5) is the main source of rollout-duration spread, and so of staleness.
- **ch-14** — constraint count and constraint types for instruction-following RL.
- **ch-15, ch-16** — horizon, toolset and environment counts for search, code and SWE pipelines.
- **ch-17** — how labs combine domains (staged or joint) and set mixture weights.
- **ch-19** — subjective rewards for persona and style axes.
- **ch-22** — capstone: applying the axis inventory to the learner's own setting.
- **llm-training ch-16** (RL prompt distribution: difficulty filtering, domain breadth, prompt reuse) — the prompt-selection mechanics this chapter assumes.
- **llm-training ch-44b** (multi-domain RL for general capability) — domain mixing and non-verifiable rewards.
- **llm-training ch-45b** (multi-turn agentic RL) — horizon and credit assignment for long episodes.
- **llm-training ch-44a** (length in RL) — the output-length axis.
- **llm-training ch-26** (tool and function-calling data) — SFT-side construction of call-structure diversity.

---

## Sources

- [[diversity-axes]] — library guide: the axis table, evidence labels, and the statement that no causal study of multi-question RL was found.
- [[rest-multi-problem-stress]] — REST stress test: Table 1 drops, the AReaL-boba RL-vs-base pair, long2short comparisons, error taxonomy. Observational only.
- [[tau2-bench]] — τ-bench / τ²-bench: compound-request failure share, action-count trend, compositional task generator, simulator error audit, policy-document ablation.
- [[toolace]] — SFT data-type and API-cluster ablations for call structure and toolset size.
- [[bfcl]] — call-structure categories, curated-vs-real distribution shift, FC-mode parallel-call scores.
- [[apigen-mt]] — reverse task recombination; persona-guided human LM.
- [[agentscaler]] — database-as-environment construction; accuracy vs tool-call count.
- [[agent-world]] — environment-count scaling for agent RL.
- [[agentrl]] — single- vs multi-environment policies; per-task advantage normalization.
- [[agentgym-rl]] — staged turn caps.
- [[simulator-collapse]] — single-simulator RL collapse and the diversified-simulator results.
- [[persona-hub]] — persona-conditioned synthesis and similarity measurements.
- [[system-prompt-diversity]] — practice claim on system-prompt pools (SFT, no RL ablation).
- [[consistentchat]], [[user-oriented-dialogue-generation]] — skeleton-guided and user-simulator multi-turn synthesis (no numerical RL claims).
- [[kimi-k2-agentic-data]] — tool repository scale; synthesized system prompts; simulated vs real sandboxes.
- [[demystifying-agentic-rl]] — real end-to-end tool trajectories vs stitched ones for agentic cold start (context for §7).
- [[deepseek-v3-2]], [[deepseek-v3.1]] — synthetic general-agent environments and their transfer.
- [[glm-4-5-recipe]] — GLM-4.5 RL ledger (holistic RL categories 7/33/139; IF constraint taxonomy) used for practice rows.
- [[general-reasoner-verifier]] — math-only vs mixed-domain zero-RL.
- [[magistral]] — cross-domain transfer table; multilingual AIME.
- [[ifbench]] — constraint-count ablation; unseen-constraint evaluation.
- [[scaling-reasoning-losing-control-mathif]] — controlled evidence that reasoning RL lowers constraint compliance.
- [[qwen-3]], [[minimax-m1]], [[big-math]] — filters that remove multi-part items.
- [[mai-thinking-1]] — pass-rate filters, tool-use environment scale, IF conflict cases, language reward, consolidation mixture.
- [[mai-rl-infrastructure]] — Rocket curriculum controller, runtime mixture weights, per-domain rollout durations and turns.
- [[olmo-3-rl-pipeline]], [[open-instruct-rlvr-pipeline]] — static mixture in open-instruct and Olmo 3.
- [[nemotron-ultra-recipe]] — fixed domain ratio with per-domain pass-rate curriculum.
