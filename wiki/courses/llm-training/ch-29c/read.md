<!-- chapter: ch-29c
     track: synthetic
     kind: content
     title: Agentic Environment and Task Synthesis at Scale
     deps: [ch-29b, ch-27, ch-32d]
     sources: [[swe-gym]], [[excerpts/swe-gym-v2-extract]], [[r2e-gym]], [[swe-smith]], [[swe-rebench]], [[glm-5]], [[kimi-k2]], [[agentscaler]], [[toucan-mcp]], [[deepseek-v3.1]], [[webshaper]], [[quest-deep-research]], [[tongyi-deepresearch]], [[agent-data-protocol]], [[agent-world]], [[reasoning-gym]], [[swe-bench-illusion]], [[swe-bench-pro]], [[agentic-benchmark-checklist]], [[browsecomp-plus]]
     figures: figures/task-funnels-and-filters.html
     revised: 2026-09 (generality revision)
-->

# Chapter 29c — Agentic Environment and Task Synthesis at Scale

> **Core insight.** For agent training, the scarce resource is a verifiable task inside an environment that runs, not a trajectory. The cost of building one fell as pipelines moved from human environment setup (SWE-Gym: about 200 annotation hours for 2,438 tasks, [[excerpts/swe-gym-v2-extract]] §3.1) to automated setup and synthesized tasks (SWE-smith: about 20 hours of human labor and $1,360 for 50,137 tasks, [[swe-smith]] §2.1-2.2). Environment breadth is a measured lever for generality: in Agent-World the four-domain tool-use average rose from 18.4% with 0 training environments to 38.5% with 1,978 ([[agent-world]] Fig. 8), SWE-smith resolve rate rose from 10.3% to 15.1% as training repositories grew from 4 to 100 at a fixed 700 trajectories ([[swe-smith]] Fig. 5), and RL on DeepSeek-V3.2-SFT using only synthesized general-agent tasks improved held-out Tau2Bench, MCP-Mark, and MCP-Universe while RL restricted to search and code environments did not ([[deepseek-v3.1]] §4.3, Fig. 5). Coverage is limited by the substrate: the Python-only SWE-smith model scored 8.4% on SWE-bench Multilingual against 6.5% for its base model ([[swe-smith]] App. F.4).
>
> **Guideline.** When building tasks for agentic SFT or RL, write the verifier first (fail-before/pass-after tests, database end state, verification function, or rubric tree), and keep a task only after a reference solution passes it and a trivial or shortcut solution fails it, because unvalidated tasks give false rewards: an empty response passes 38% of τ-bench Airline tasks ([[agentic-benchmark-checklist]] §5.2) and flawed tests penalize correct agents ([[swe-rebench]] §2.3). When the goal is breadth at a fixed trajectory budget, add distinct environments, repositories, and task families before adding trajectories per task, because the SWE-smith repository comparison was run at a fixed 700 trajectories ([[swe-smith]] Fig. 5) and the ADP comparison at an equal sample count ([[agent-data-protocol]] Table 10). Agent-World does not state whether its RL sample count was held fixed across environment counts. When the target includes real APIs, include real executable or real-server substrates, because a model trained with simulated environments (Simulator-8B) scored 31.8 on τ²-Bench but 2.4 on MCP-Mark ([[agent-world]] Table 1; single study). Otherwise, when the purpose is algorithm iteration rather than a released checkpoint, use simulated or offline substrates, which cost less per rollout, as Tongyi DeepResearch did with an offline Wikipedia environment ([[tongyi-deepresearch]] §3.4.3). When a benchmark score is used as evidence of general agent ability, exclude that benchmark's repositories, servers, and task templates from the environment pool and report a second score on tasks created after the training cutoff, because models score lower on equally popular repositories outside SWE-Bench (under 53% file-path accuracy against 60-76%, [[swe-bench-illusion]] §4.1) and SWE-rebench flags results whose issues predate a model's release ([[swe-rebench]] §3.2).

## Why this chapter matters for a general-purpose model

The pipeline for a general model has six stages: pre-training, mid-training, supervised fine-tuning (SFT), preference optimization, reinforcement learning (RL), and evaluation. Agentic behavior enters mid-training as trajectory data ([[ch-32d]]), SFT as filtered teacher trajectories ([[ch-27]]), and RL as rollouts scored by an environment ([[ch-45b]]). All three need the same input: a large set of tasks, each paired with an environment that executes actions and a check that scores the outcome. This chapter covers how that input is produced.

The measurable problem has three parts. (1) **Yield:** the fraction of candidate tasks that become runnable and verifiable, and the human and compute cost per surviving task. (2) **Verifier error:** the rate at which a task's check accepts a wrong solution (false positive) or rejects a correct one (false negative). (3) **Coverage:** the number of distinct environments, languages, tool ecosystems, and task families, and whether training on them changes scores on environments that were not used. The previous chapter, [[ch-29b]], synthesized long conversations in which context accumulates; this chapter adds an environment whose state the agent changes. The next chapter, [[ch-29d]], covers simulated users, trajectory verification, and what to do with failed trajectories once tasks exist.

## §1 Terms and the substrate taxonomy

**Definitions.**
- An **environment** is a program that accepts an agent action (a tool call, a shell command, a file edit) and returns an observation, while holding state that later actions can read. [[agent-world]] §2 formalizes an environment as e = (D, F): a database D and a toolset F, with state observed only through tool outputs.
- A **task** is an instruction plus an initial environment state.
- A **verifier** is a function from the final environment state, the agent's answer, or both, to a score. Examples: Fail-to-Pass (F2P) tests that fail before a gold patch and pass after it; a database end-state comparison; a Python verification function; a rubric tree.
- A **substrate** is the kind of system an environment runs on.

**Problem addressed.** Environments differ in fidelity (how closely observations match the deployed system) and in cost per task (setup time, storage, latency, API fees). A training run that uses one substrate inherits its failure modes.

**Taxonomy.** Tongyi DeepResearch describes three forms: a prior-world environment that provides tasks, tools, and state definitions with no real responses ("perfect stability, zero interaction cost ... but lacks real-world feedback signals"), a simulated environment with "a notable sim-to-real gap", and a real-world environment with "expensive interactions, significant non-stationarity, and exploration risks" ([[tongyi-deepresearch]] §2). The table refines this into six substrates used by the sources in this chapter.

| Substrate | Example | Observation source | Verifier | Cost per task (as reported) | Main fidelity limit |
|---|---|---|---|---|---|
| Real executable repository | SWE-Gym, R2E-Gym, SWE-smith, SWE-rebench, GLM-5 SWE and terminal | Docker container running code and tests | F2P and pass-to-pass tests | SWE-Gym 6 TB images for 2,438 tasks; SWE-smith 295 GB for 50k | test quality, flaky tests, language coverage |
| Real MCP / API servers | TOUCAN (495 servers) | live remote servers | LLM judge plus rule checks | servers needing API keys excluded | non-stationary outputs, availability |
| Stateful simulated tools (LLM) | Kimi K2 tool simulator | LLM that tracks state | LLM judge against task rubric | not reported | hallucinated or inconsistent responses |
| Programmatic database environment | AgentScaler, DeepSeek-V3.2 general agent, Agent-World | Python tools reading and writing a database | end-state match, tool-sequence match, verification function | not reported | schema realism, reach of synthesized tools |
| Offline search corpus | Tongyi 2024 Wikipedia environment; BrowseComp-Plus corpus | local retrieval tools | answer match | not reported | corpus coverage and retriever choice |
| Procedural generator | Reasoning Gym (over 100 generators) | none (single-step) or a game state | algorithmic check | generated on demand | task families limited to what can be coded |

Sources: [[excerpts/swe-gym-v2-extract]] §3.1; [[swe-smith]] Table 2; [[toucan-mcp]] §3.1; [[kimi-k2]] §3.1.1; [[agentscaler]] §2; [[deepseek-v3.1]] §3.2.3; [[agent-world]] §3.1; [[tongyi-deepresearch]] §3.4.3; [[browsecomp-plus]]; [[reasoning-gym]] §2.

**Evidence on fidelity.**
- Kimi K2 generates SFT trajectories with a simulator that "maintains and updates state after each tool execution" and "introduces controlled stochasticity", and adds real sandboxes for coding and software engineering "for scenarios where authenticity is crucial" ([[kimi-k2]] §3.1.1). No ablation separates the two.
- In Agent-World Table 1, Simulator-8B (a baseline trained with simulated environments) scores 31.8 on τ²-Bench, 23.9 on BFCL V4, and 2.4 on MCP-Mark, while the untrained Qwen3-8B listed in the same table scores 26.2, 40.4, and 2.4 ([[agent-world]] Table 1). The paper does not state which base model Simulator-8B was trained from. The authors interpret the result as simulated environments not capturing real state transitions (Interpretation; the baselines differ in data and method, so this is not a controlled comparison).
- Tongyi DeepResearch reports that the RL reward curve in its offline Wikipedia environment "closely matches" the curve in the real environment (§4.4, Fig. 10b vs Fig. 8) and describes the offline environment as a platform for rapid algorithm iteration (§4.4). Whether the released model's final RL used this environment is not stated.

**Implication for a general model.** A single substrate covers a subset of deployment conditions. Real servers and executable repositories supply realistic failures; programmatic and offline environments supply scale and reproducibility. The sources that report the strongest held-out agent results combine substrates (§4, §6).

## §2 The SWE environment-factory lineage

**Definition.** An SWE environment factory is a pipeline that turns repositories into tasks of the form (repository snapshot, issue text, hidden tests) with a runnable container.

**Problem.** Human environment setup does not scale. SWE-Gym extracted 64,689 raw task instances from 358 repositories, but only 11 repositories were converted into executable environments, at "approximately 200 human annotation hours and 10,000 CPU core hours" for 2,438 validated instances ([[excerpts/swe-gym-v2-extract]] §3.1).

**Mechanism, generation by generation.**
1. **SWE-Gym (arXiv 2024-12).** Real issues and PRs; dependencies configured manually per task instance; validation requires the gold patch to pass more tests than the original code; repositories disjoint from SWE-Bench "to avoid contamination" (§3). Fine-tuning Qwen-2.5-Coder-32B on 491 trajectories raised SWE-Bench Verified from 7.0% to 20.6% (Table 3).
2. **R2E-Gym (arXiv 2025-04).** SWE-Gen builds environments from commits, not issues: collect existing F2P tests or generate them, and back-translate the code change into a problem statement ([[r2e-gym]] §2). This gives 8,135 problems and a 4,578-problem subset with no repository overlap with SWE-Bench (Table 1). With 400 trajectories each, synthetic problem statements gave 27.8% Pass@1 and real issues 28.0% (Fig. 3).
3. **SWE-smith (arXiv 2025-04).** The order is inverted: "define an execution environment first, and then synthesize task instances within the environment" ([[swe-smith]] §2). One Docker image per repository (installed by SWE-agent in at most 100 steps and checked by a person); bugs are injected by LM rewrites, AST transformations, PR reversal, and bug combination; a candidate is kept only if it breaks at least one passing test.
4. **SWE-rebench (arXiv 2025-05).** Installation is automated: tasks are grouped by version, Qwen2.5-72B-Instruct writes up to 3 candidate JSON installation recipes per task and revises a recipe from error logs; a task is valid only if a test-patch test fails before the solution patch, all such failing tests pass after it, and initially passing tests stay passing; tests are rerun to exclude flaky tasks ([[swe-rebench]] §2.2-2.3).
5. **DeepSeek-V3.2 and GLM-5 (2025-12, 2026-02).** Environment-setup agents extend the method to more languages. DeepSeek-V3.2 counts an environment as built only if the gold patch gives F2P > 0 and pass-to-fail = 0, across Python, Java, JavaScript, TypeScript, C, C++, Go, and PHP, and uses 24,667 code-agent RL tasks ([[deepseek-v3.1]] §3.2.3, Table 1). GLM-5 uses a RepoLaunch-based pipeline with LLM-written log parsers to build "over 10k verifiable environments across thousands of repositories spanning 9 programming languages" ([[glm-5]] §4.2.1).

**Worked example: cost per surviving task.** The values below are derived from the printed totals.

| Pipeline | Human time | Surviving tasks | Human time per task | Storage per task |
|---|---|---|---|---|
| SWE-Gym | about 200 h | 2,438 | 200 × 60 / 2,438 = 4.9 min | 6 TB / 2,438 = 2.5 GB |
| SWE-smith | about 20 h | 50,137 | 20 × 3,600 / 50,137 = 1.4 s | 295 GB / 50,137 = 5.9 MB |

SWE-smith's storage saving comes from sharing one image per repository instead of one per task; the authors estimate 50-150 TB for the same number of SWE-bench-style images ([[swe-smith]] §2.2). The saving has a cost: tasks from one repository share one dependency snapshot, so versions of the code before that snapshot are not covered.

**Yield.** SWE-rebench keeps about 153,400 of about 450,000 issue-linked PRs after task filters (34.1%) and 21,336 of those after installation and validation (13.9%), so 4.7% of candidates survive (§2.1, §2.4; percentages derived). SWE-smith's overall yield of candidates that break tests is 50.1%, ranging from 33.8% (PR Mirror) to 96.9% (Combine) (Table 1). The companion figure [task-funnels-and-filters.html](figures/task-funnels-and-filters.html) lets the reader step through the SWE-Gym, SWE-rebench, TOUCAN, and QUEST funnels and see the stage-to-stage and cumulative retention.

**Evidence on what the synthesized tasks are worth.**
- Strategy (1,000 instances each, 507 trajectories per student, Qwen 2.5 7B): PR Mirror 9.2%, LM Rewrite 8.8%, Procedural 8.6%, LM Modify 5.7% on SWE-bench Verified ([[swe-smith]] Table 4). Procedural bugs cost 0.00¢ per candidate against 5.53¢ for PR Mirror (Table 1).
- Issue text (259 trajectories each): original issue 7.8%, LM-generated 7.7%, F2P test as issue 7.3%, fixed template 6.4% (Table 5). With test-based issues the student attempted to reproduce the bug in 127 of 500 runs against 379 with LM-generated issues, which the authors attribute to leaking the evaluation criterion into the prompt (Interpretation).
- Difficulty did not predict training value: SFT sets at difficulty 2/4/6/8 scored 12.4 / 10.8 / 13.6 / 12.2% (§4.1).

**Conditions and limits.** All SWE-smith and SWE-Gym results are rejection-sampling SFT with a Claude or GPT teacher; neither paper trains RL on the environments. SWE-rebench trains no agent, so the effect of its validation filters on training is not measured ([[swe-rebench]] Guideline).

**Implication for a general model.** Automated installation and bug synthesis make the number of repositories and languages a controllable variable. §6 shows that this variable, not the trajectory count alone, is where the measured transfer comes from.

## §3 Tool and task diversification beyond code

**Definition.** Tool diversification increases the number and variety of callable tools and the domains they belong to; task diversification increases the variety of goals composed from those tools.

**Problem.** Tool-use models fail on unfamiliar tools and on tasks that need several tools in sequence. Real APIs are limited by credentials, availability, and cost; the TOUCAN crawl of about 2,800 MCP servers kept 871 (30.6%) that were remote and needed no API keys, and 495 after testing each tool ([[toucan-mcp]] §3.1).

**Mechanisms.**
1. **Real plus synthetic tools with hierarchical domain evolution (Kimi K2).** "we directly fetch 3000+ real MCP (Model Context Protocol) tools from GitHub repositories" and "systematically evolve synthetic tools through a hierarchical domain generation process: we begin with key categories (e.g., financial trading, software applications, robot control), then evolve multiple specific application domains within each category", producing over 20,000 synthetic tools ([[kimi-k2]] §3.1.1). Thousands of agents are created from system prompts and tool combinations; each task carries a rubric with success criteria, expected tool-use patterns, and checkpoints.
2. **Database-backed environment scaling (AgentScaler).** Every function is treated as a read or write operation on a domain database, API(func, α) ≡ op(func)(α; D) ([[agentscaler]] §2). Over 30,000 APIs are connected into a graph, split into more than 1,000 domains by Louvain community detection, and each tool is rewritten as Python code over the domain database (§2.1).
3. **Real-server tasks (TOUCAN).** Five open models write tasks for one server, several servers, or 25 featured servers; Kimi-K2 rates task difficulty, uniqueness, quality, realism, verifiability, and stability; three teacher models generate trajectories against the live servers (§3.1). An irrelevance extension shuffles server metadata so that the task cannot be solved with the tools given and keeps only trajectories with zero tool calls (§3.2).
4. **Environment synthesis agents (DeepSeek-V3.2, Agent-World).** DeepSeek-V3.2's agent receives a task category and a sandbox with bash and search, stores retrieved data in a database, writes task-specific tools, then proposes a task "along with its solution and verification functions implemented in Python", where "The solution function is restricted to invoking tool functions or performing logical computations, and cannot call other functions or directly access the database" ([[deepseek-v3.1]] §3.2.3). Agent-World mines databases with a deep-research agent, keeps a tool only if it compiles and passes more than 50% of its unit tests, and composes tasks by random walks on a tool dependency graph ([[agent-world]] §3.1).

**Formula: AgentScaler tool-graph edges** ([[agentscaler]] §2.1, Eq. 1).

```
E = { (i, j) | sim( φ(P_func_i), φ(P_func_j) ) > τ ,  i ≠ j }
```

- `P_func_i` is the parameter list of tool i; `φ` maps it to a vector; `sim` is cosine similarity.
- `τ` is a similarity threshold (value not printed).
- An edge means the two tools can be composed; an LLM then re-checks dependencies inside each domain.

**Worked example.** Suppose `φ` gives tool A the vector (1, 0, 1) and tool B the vector (1, 1, 0). Their dot product is 1 and each norm is √2, so cos = 1 / 2 = 0.5. With an illustrative threshold τ = 0.6, no edge is added; with τ = 0.4, A and B join the same candidate domain. The threshold therefore sets how many domains Louvain finds and how long a composable tool chain can be.

**Worked example: Agent-World weighted walk.** Agent-World assigns edge weight 3 to a strong dependency, 2 to a weak one, and 1 to an independent pair, and samples the next tool "with a probability distribution biased by the edge weights" ([[agent-world]] §3.1.1). If the distribution is proportional to the weights (the paper does not print the normalization), a tool with one strong, one weak, and one independent successor moves to them with probabilities 3/6 = 0.50, 2/6 = 0.33, and 1/6 = 0.17. Agent-World raises difficulty by increasing the sampling probability of weak and independent edges, which lengthens chains of tools whose outputs are less directly connected.

**Evidence.**
- AgentScaler-30B-A3B (Qwen3-Thinking-30B-A3B backbone) raised τ²-Bench Telecom from 26.3 to 55.3 and ACEBench-en Overall from 67.2 to 75.7 ([[agentscaler]] Table 1). On ACEBench-zh, which the authors call out-of-distribution, Overall rose from 74.2 to 81.5 (Table 2).
- TOUCAN SFT (119.3K instances) raised BFCL V3 Overall for Qwen2.5-32B-Instruct from 61.73 to 70.45 and multi-turn from 26.38 to 46.50 ([[toucan-mcp]] Table 2).
- DeepSeek-V3.2 kept synthesized general-agent tasks only if pass@100 under V3.2 was non-zero, which left 1,827 environments and 4,417 tasks; on 50 sampled tasks, pass@1 was 12% for DeepSeek-V3.2-Exp, 34% for Sonnet-4.5, 51% for Gemini-3.0 Pro, and 62% for GPT-5-Thinking ([[deepseek-v3.1]] §3.2.3, Table 5).

**Conditions and limits.** AgentScaler reports no learning rate, epochs, or data size, and trains only SFT up to 30B ([[agentscaler]] Limitation). Kimi K2 reports no ablation of synthetic against real tools. None of AgentScaler, TOUCAN, or Kimi K2 reports an overlap check between its tool pool and the benchmarks it evaluates on (§7).

**Implication for a general model.** Tool diversity is produced by two different operations: collecting real tools (bounded by what exists and is accessible) and evolving synthetic tools (bounded by the generator's knowledge of domains). The evidence that either transfers to unseen tool ecosystems comes from held-out suites such as MCP-Mark, MCP-Universe, and ACEBench-zh, not from the benchmarks whose environments inspired the synthesis.

## §4 Verifier-first task design

**Definition.** Verifier-first design writes and validates the check before the task enters the pool, and discards tasks whose check cannot be validated.

**Problem.** A verifier error becomes a reward error. On the benchmark side, an empty response passes 38% of τ-bench Airline tasks and an agent that overwrites SWE-Lancer's test files scores 100% ([[agentic-benchmark-checklist]] §5.2). On the training side, SWE-Bench Pro reports that removing the required-behavior and interface fields lowered GPT-5 (high) from 25.9% to 8.40%, which its authors attribute to verifier false negatives ([[swe-bench-pro]] §6.2, Table 3).

**Four verifier families and their validation rules.**

| Family | Validation before a task is kept | Source |
|---|---|---|
| Hidden tests | gold patch: F2P > 0 and pass-to-fail = 0; rerun to remove flaky tests; injected bug must break ≥ 1 passing test | [[deepseek-v3.1]] §3.2.3; [[swe-rebench]] §2.3; [[swe-smith]] §2.1 |
| Database state or verification function | reference solution must pass the verification function, repaired until it does; tool-only solution path; end-state match or exact tool-sequence match | [[deepseek-v3.1]] §3.2.3; [[agentscaler]] §3.1; [[agent-world]] §3.1.1 |
| Answer check for search | ground truth verified correct and all candidate answers from other agents verifiably incorrect; non-unique answers rejected | [[deepseek-v3.1]] §3.2.3; [[glm-5]] §4.2.3 |
| Rubric tree for open-ended tasks | tree refined and verified; executable Python evaluation script generated; tasks whose tree cannot be made consistent are discarded | [[quest-deep-research]] §2.2 |

**Mechanism: difficulty filters.** Most pipelines also filter by how often a model solves the task. Let `p` be a model's per-attempt solve probability on a task and assume independent attempts (a simplification; attempts are correlated in practice).

```
keep_pass@k>0      = 1 − (1 − p)^k                    (DeepSeek-V3.2, k = 100)
keep_≥2_of_5       = 1 − (1 − p)^5 − 5p(1 − p)^4       (Agent-World)
keep_mixed_G       = 1 − p^G − (1 − p)^G               (Tongyi DeepResearch; G not printed)
keep_unsolved_8    = (1 − q)^8                         (GLM-5 stage 1, q = tool-free solve rate)
```

- `k`, `5`, `G`, `8` are the numbers of attempts in each rule; `q` is the solve probability of a model without tools.

**Worked example.** A task with p = 0.01 survives the DeepSeek-V3.2 rule with probability 1 − 0.99^100 = 0.634; a task with p = 0.001 survives with probability 0.095. Under Agent-World's rule a task with p = 0.10 survives with probability 1 − 0.9^5 − 5(0.1)(0.9)^4 = 0.081, and one with p = 0.30 with probability 0.472. Under the Tongyi rule with G = 8, p = 0.10 gives 0.570 and p = 0.50 gives 0.992. Under GLM-5's first stage, a question a tool-free model answers with q = 0.10 is kept with probability 0.9^8 = 0.430. The rules therefore select different difficulty bands from the same pool: pass@100 keeps rare-success tasks, ≥2-of-5 removes them, and the GLM-5 rule removes tasks answerable from parametric knowledge. Panel B of the companion figure plots the four curves and lets the reader vary p, k, and G.

**Formula: QUEST reward** ([[quest-deep-research]] §4.4, Eq. 1).

```
R = 0.75 · s_rubric + 0.25 · min(s_fact, s_rubric)
```

- `s_rubric ∈ [0, 1]` is the rubric-tree score (open-ended tasks: pairwise score J(r_cand)/(J(r_cand)+J(r_ref)) mapped to 1.0 above 0.5, 0.75 on [0.475, 0.5), 0.5 on [0.45, 0.475), 0.25 on [0.425, 0.45), else 0).
- `s_fact` is the fraction of determinate citations labeled supported.

**Worked example.** A report with J(r_cand) = 0.60 against a reference with J(r_ref) = 0.62 has pairwise score 0.60 / 1.22 = 0.492, so s_rubric = 0.75. With s_fact = 0.9, R = 0.5625 + 0.25 × 0.75 = 0.75. With s_fact = 0.2, R = 0.5625 + 0.05 = 0.6125. A report with s_rubric = 0 and s_fact = 1 gets R = 0, so well-cited but incorrect reports receive no reward.

**Evidence on verifier quality.**
- QUEST: of 50 sampled objective tasks, 2 evaluation scripts were non-executable and 6 contained rubric-related errors (App. A.2). Objective tasks fell from 17,000 generated to 5,934 kept (34.9%) through rubric refinement, tree verification, and script checks (App. A.1 Table 4).
- QUEST pointwise 0/0.5/1 rubric scoring reached about 1 in about 50% of cases, which the authors attribute to judge high-score bias; their pairwise design was introduced to fix it (§7.4).
- R2E-Gym: for the majority of problems, under 20% of generated tests distinguish correct from incorrect patches, and up to 10% of tests are toxic for some problems ([[r2e-gym]] §4.2).
- GLM-5 terminal tasks: a refine agent checks that "the environments are robust against potential exploits or shortcuts", with Docker construction accuracy above 90% ([[glm-5]] §4.2.2). The rate of exploitable tasks remaining is not reported.

**Conditions and limits.** No source in this chapter reports a measured false-positive or false-negative rate for its training verifiers. The difficulty filters use the model being trained or a sibling model, so the surviving pool shifts as that model changes; Tongyi DeepResearch refreshes the pool during RL for this reason ([[tongyi-deepresearch]] §3.4.3).

**Implication for a general model.** A verifier that accepts shortcuts rewards the shortcut in every environment built with the same template. Verifier validation (reference passes, trivial agent fails) is the check that prevents a training pool from teaching behavior that no held-out benchmark will reward.

## §5 Synthetic search and research tasks

**Definition.** A search task asks for an answer that must be assembled from several web or corpus sources; a research task asks for a report whose quality is judged on several criteria.

**Problem.** Question-first generation from retrieved pages produces questions whose reasoning structure does not match the information retrieved, or whose answer is wrong ([[webshaper]] Abstract). Single-answer tasks give a binary reward and cover fact seeking, not report synthesis ([[quest-deep-research]] §2).

**Mechanisms.**
1. **Knowledge-graph subgraphs (GLM-5, Tongyi).** GLM-5 builds a Web Knowledge Graph from over two million pages, samples low- to mid-frequency seed entities, expands multi-hop subgraphs, and converts each into a question ([[glm-5]] §4.2.3). Tongyi DeepResearch raises difficulty by "strategically increasing the uncertainty within the question", for example merging entities with similar attributes ([[tongyi-deepresearch]] §3.4.1).
2. **Formalization-driven expansion (WebShaper).** A Knowledge Projection is R(V) = {u | ∃v ∈ V, (u, v) ∈ R or (v, u) ∈ R}, with E the entity set, R ⊆ E × E a relation, and V ⊆ E ([[webshaper]] Eq. 2). A task is ?T for T = R_1(T_1) ∩ ... ∩ R_k(T_k) (Eq. 6-7). The layer-wise Expander replaces a leaf constant C with a sub-question whose answer is C, so q_{n+1}(T) = Expander(C, q_n(T)) keeps the same answer (Eq. 11). A validation call rejects the sub-question if QwQ answers it directly (§3.2.3).
3. **Multi-agent verified QA (DeepSeek-V3.2).** A question-construction agent explores long-tail entities; several answer agents with different checkpoints and prompts answer; a verification agent keeps the sample only if the ground truth is correct and all candidates are verifiably incorrect ([[deepseek-v3.1]] §3.2.3). This yields 50,275 search-agent RL tasks (Table 1).
4. **Rubric trees (QUEST).** Trending keywords seed web exploration; objective tasks get a verifiable rubric tree and a GPT-5-written evaluation script; open-ended tasks get four fixed criteria with task-specific children and a reference report (§2.2).

**Worked example: WebShaper formalization.** The question "Which player of a team in the 2004-05 season, who was born in 90s? This team is founded in 1966 and is an East German football team." is written as [[V@T, playIn, V@X], [V@T, playAt, C@2004_05], [V@T, bornIn, C@90s], [V@X, foundIn, C@1966], [V@X, isA, C@East German football team]] ([[webshaper]] Eq. 10). The leaf constant C@1966 can be expanded into a sub-question whose answer is 1966, for example about an event in that year. Because the target is unchanged, the answer check written for the seed question remains valid after each expansion.

**Evidence.**
- At 5,000 SFT samples per dataset, GAIA averages for QwQ-32B were WebShaper 53.3, WebWalkerQA 45.6, E2HQA 45.6, MHQA 41.7 ([[webshaper]] Table 2).
- GLM-5 removes questions a tool-free reasoning model answers in at least 1 of 8 attempts and questions an early agent solves within a few steps ([[glm-5]] §4.2.3); no ablation of the filters is printed.
- At the 30B scale, the recipe that trained on single-answer synthetic tasks (Tongyi-DR) was best on BrowseComp 43.4, HLE 32.9, and GAIA 70.9; OpenResearcher was best on BrowseComp-Plus 54.8, "a fully offline benchmark that closely matches its data synthesis recipe"; QUEST-30B was best on 4 of 8 benchmarks ([[quest-deep-research]] Table 3, §6.2). The authors conclude that "the capabilities exhibited by a deep research agent are shaped by its data synthesis recipe" (Interpretation).

**Conditions and limits.** WebShaper, Tongyi DeepResearch, and QUEST report no overlap check between synthesized questions and GAIA, BrowseComp, or HLE, except QUEST's exclusion of benchmark runs from its context-summarization data (App. A.3). On BrowseComp-Plus, accuracy depends on the retriever (gpt-5 55.90% with BM25, 70.12% with Qwen3-Embedding-8B), so offline-corpus tasks need a fixed retriever to be comparable ([[browsecomp-plus]] Table 1).

**Implication for a general model.** The task format selects the capability: single-answer tasks train fact seeking, offline corpora train retrieval against that corpus, and rubric trees train report synthesis. A general research agent needs all three formats in the task pool.

## §6 Environment diversity as the generality factor

**Definition.** Environment diversity is the number of distinct environments (repositories, servers, databases, domains, generators) represented in training. Where a study holds the number of trajectories or RL samples fixed, the comparison isolates diversity from data volume; where it does not, the two vary together.

**Problem.** An agent trained on few environments can raise in-distribution scores while failing on new environments, languages, or tool ecosystems.

**Evidence.**
1. **Environment count (Agent-World).** Training with 0, 10, 100, 500, 1,000, and 1,978 environments raised the four-subdomain average from 18.4% to 38.5% (+20.1 points), with the largest increases at 10 → 100 and 100 → 500 ([[agent-world]] §4.3.3, Fig. 8). The model size of this run and the number of training samples at each environment count are not stated, so data volume is not held fixed by the text. **Result (single study).**
2. **Repository count (SWE-smith).** At 700 Procedural Modification trajectories, 4, 25, 50, and 100 repositories gave 10.3, 11.5, 12.9, and 15.1% on SWE-bench Verified ([[swe-smith]] §4.1, Fig. 5; the §4.1 text says 4 repositories for the smallest pool while the Fig. 5 axis prints 5). The student is Qwen-2.5-Coder-Instruct 7B.
3. **Synthetic general-agent RL (DeepSeek-V3.2).** RL on DeepSeek-V3.2-SFT with only synthetic general-agent tasks improved Tau2Bench, MCP-Mark, and MCP-Universe; V3.2-Exp, trained with RL only in search and code environments, did not improve on them ([[deepseek-v3.1]] §4.3, Fig. 5; values only in the figure). The environments of these benchmarks were "not encountered during RL training" (§4.1).
4. **Mixed agent data (ADP).** With the same OpenHands harness and Qwen-3-8B, training on about 30K samples of the non-web ADP mixture gave 16.6% on SWE-Bench Verified against 11.0% for SWE-smith up-sampled to the same size ([[agent-data-protocol]] Table 10). On GAIA, AgentInstruct-only training gave 0.6% and ADP 9.1% (Table 6).
5. **Procedural cross-domain transfer (Reasoning Gym).** GRPO on algorithmic generators raised held-out algebra from 23.83 to 52.89 and geometry from 0.83 to 23.17 for Qwen2.5-3B-Instruct ([[reasoning-gym]] Table 2).

Items 1-5 are separate studies in different settings that agree in direction: more distinct environments or task families gave higher held-out scores. Items 2 and 4 hold the data budget fixed (700 trajectories; about 30K samples); items 1, 3, and 5 do not report a fixed budget. **Replicated** in direction; the size of the effect is not comparable across studies.

**Worked example: gain per doubling.** From SWE-smith's endpoints, 4 → 100 repositories is log2(25) = 4.64 doublings for a 15.1 − 10.3 = 4.8-point gain, about 1.0 point per doubling (derived from two endpoints; the paper describes the curve as approximately logarithmic). From Agent-World's 10 → 1,978 environments, the authors describe smaller gains after 500 environments; the per-step values are shown only in Fig. 8.

**Counter-evidence and narrowing.**
- Language: the Python-only SWE-smith model scored 8.4% on SWE-bench Multilingual against 6.5% for Qwen 2.5 Coder Instruct and 43% for Claude 3.7 Sonnet; the authors observed edits "reflected syntax closer to Python" in non-Python repositories ([[swe-smith]] App. F.4).
- Specialization: two 7B models each trained on 700 trajectories, one from 100 repositories and one from SymPy only, scored 13.6% and 21.2% on the 22 SymPy instances of SWE-bench Verified created after 2022-01-01, and 15.3% and 14.0% on the rest of Verified ([[swe-smith]] §4.1, Fig. 4).
- Tool-call format: AgentScaler-4B's ACEBench-en Special score fell from 84.7 to 76.7 after training (Table 1), and QUEST-35B after SFT became "more likely to call disallowed tools due to overfitting to the training-time tool-use pattern" on BrowseComp-Plus ([[quest-deep-research]] §6.3).
- Procedural transfer is not uniform: ARC fell from 6.49 to 4.18 after algebra training and to 4.26 after games training ([[reasoning-gym]] Table 2).

**Implication for a general model.** A training report that gives only a trajectory count leaves the diversity variable unmeasured, so the count of distinct environments belongs next to it. When generality is the claim, whole environments, languages, and tool ecosystems have to be held out, because the narrowing results above appear only when the evaluation environment differs from the training environments.

## §7 Decontamination of environment tasks against agent benchmarks

**Definition.** Environment decontamination removes from the training pool any repository, server, database schema, task template, or question that also appears in an evaluation benchmark, and prefers evaluation tasks created after the training data cutoff.

**Problem.** Agent benchmarks reuse public repositories, public MCP servers, and public tool APIs, which are the same sources environment factories mine. Given only the repository name and issue text, ten OpenAI and Anthropic models named a file changed by the gold patch in 60-76% of SWE-Bench Verified instances but in under 53% of 245 comparable tasks from repositories outside SWE-Bench ([[swe-bench-illusion]] §4.1).

**Mechanism.**
1. Exclude benchmark repositories at the repository level: SWE-smith removes the 12 SWE-bench test repositories ([[swe-smith]] §2.1); SWE-Gym and R2E-Gym-Subset use repositories disjoint from SWE-Bench ([[excerpts/swe-gym-v2-extract]] §3; [[r2e-gym]] §2).
2. Exclude benchmark environments at the environment level: list the servers, databases, and tool schemas of each agent benchmark and remove matches from the tool pool. AgentScaler reports that its generated τ-bench-domain databases and tool code "exhibit a high degree of consistency with the official implementations provided by τ-bench" ([[agentscaler]] §2.1) and then evaluates on τ-bench; no removal is reported.
3. Evaluate on fresh tasks: SWE-rebench marks a model's results as potentially contaminated when the benchmark includes issues created before the model's release ([[swe-rebench]] §3.2). GLM-5 evaluates on SWE-rebench because SWE-bench Verified "is a static, public, human-validated test set and released for more than 2 years" ([[glm-5]] §6.2.4).
4. Evaluate on environments not used in RL: DeepSeek-V3.2 states that MCP-Universe, MCP-Mark, and Tool-Decathlon environments were not seen in RL ([[deepseek-v3.1]] §4.1).

**Worked example: reading two benchmarks side by side.** On SWE-bench Verified, GLM-5 scores 77.8 and Claude Opus 4.5 80.9 ([[glm-5]] Table 7). On January 2026 SWE-rebench tasks, they resolve 42.1% and 43.8% ([[glm-5]] Table 9). The drop from 77.8 to 42.1 is not a contamination measurement, because the two benchmarks differ in scaffold, task filters, and difficulty. The informative quantity is the change in ordering or gap between models evaluated under one protocol. SWE-rebench makes this comparison for two DeepSeek-V3 versions: 39.7 and 35.2 on SWE-bench Verified, 21.3 and 21.9 on its March-April 2025 slice, which its authors read as possible contamination of the older benchmark ([[swe-rebench]] Table 2, §3.3; Interpretation).

**Conditions and limits.** None of TOUCAN, AgentScaler, Agent-World, WebShaper, Tongyi DeepResearch, or ADP reports an overlap check against its evaluation benchmarks. Agent-World's self-evolution rounds choose weak environments from held-out arena tasks and report gains on MCP-Mark (Postgres) ([[agent-world]] Table 2); when an evaluation benchmark guides which tasks are synthesized, its later scores are no longer an unbiased estimate of performance on untargeted environments (Interpretation). Contamination methods in general are in [[ch-48]].

## §8 Agent data unification (Agent Data Protocol)

**Definition.** The Agent Data Protocol (ADP) is a schema that represents any agent trajectory as alternating actions (`APIAction`, `CodeAction`, `MessageAction`) and observations (`TextObservation`, `WebObservation`) ([[agent-data-protocol]] §3.2).

**Problem.** Agent datasets use different action syntaxes and observation formats, so mixing D datasets for A agent harnesses needs D × A converters.

**Mechanism.**
1. Convert each raw dataset to ADP once.
2. Write one ADP → SFT converter per harness, which sets the system prompt, action syntax, and context handling.
3. Run automated checks: tool-call format, most tool calls paired with a thought (threshold 80%), proper conversation ending.
4. Resample each dataset with a multiplier w_d (sample ⌈w_d · n_d⌉ examples per epoch; w_d < 1 without replacement, w_d > 1 with replacement) (App. C).

**Worked example: converter cost.** The 13 Raw → ADP converters took 4,892 lines of code, and an ADP → SFT converter averages 77 lines (Tables 7-8). For A = 100 harnesses, per-pair conversion costs about 100 × 4,892 = 489,200 lines, and ADP costs 4,892 + 77 × 100 = 12,592 lines (§6.3). **Worked example: resampling.** Orca AgentInstruct has 1,046.1K trajectories and w_d = 0.001, so about 1,046 are drawn per epoch; SWE-Gym has 0.5K with w_d = 3 (Table 1, Table 9).

**Evidence.** Qwen-2.5-32B-Coder-Instruct in SWE-Agent reached 40.3% on SWE-Bench Verified, against 40.2% for SWE-smith data alone; in OpenHands it reached 36.8% against 20.6% for SWE-Gym data (Table 5). AgentLab 7B rose from 4.5% to 21.0% on WebArena (Table 3).

**Conditions and limits.** Coding harnesses were trained on the non-web portion (about 30K samples) and AgentLab on the web portion (about 20K samples) (App. C.1), so "cross-task transfer" in Table 6 is transfer within those portions. ADP reports no deduplication across sources, no decontamination against its four evaluation benchmarks, and no general (non-agent) benchmarks.

**Implication for a general model.** A shared schema makes the environment mixture a variable that can be ablated. It does not remove harness dependence: SWE-smith-only training scored 1.0% in OpenHands for Qwen-2.5-7B-Instruct (Table 6), while SWE-smith trajectories used in their own SWE-agent harness gave 15.2% with Qwen-2.5-7B-Coder-Instruct (Table 3). The base models differ, so the harness is not the only difference between the two numbers.

## Agentic training pipeline: where environment tasks enter

1. **Mid-training.** Tongyi DeepResearch generates function-calling data from AgentScaler-style database environments and includes it in agentic continued pre-training at 32K and 128K, with "a small proportion of general pre-training data" interleaved ([[tongyi-deepresearch]] §3.3). Stage placement across reports is compared in [[ch-32d]].
2. **SFT.** Tasks with a teacher success become rejection-sampled trajectories: SWE-smith 5,016 from 6,457 resolved runs, capped at 3 per task ([[swe-smith]] §4); TOUCAN 119.3K selected instances ([[toucan-mcp]] §4.1). Details of trajectory filtering are in [[ch-27]] and [[ch-29d]].
3. **RL.** Tasks the teacher never solves are kept for RL: QUEST reserves 864 objective tasks that Tongyi DeepResearch failed in all rollouts ([[quest-deep-research]] App. A.5). DeepSeek-V3.2 and GLM-5 run RL on their code, search, terminal, and synthesized environments ([[deepseek-v3.1]] Table 1; [[glm-5]] §3.3). Multi-turn RL mechanics are in [[ch-45b]]; rollout infrastructure in [[ch-54]].
4. **Evaluation.** Held-out environments (MCP-Mark, MCP-Universe, SWE-rebench fresh tasks) are the measurement of transfer ([[ch-51a]]).

## Negative samples and negative feedback

In this stage "negative" appears in all four senses of the course standard: (1) negative marginal value, (2) negative as content, (3) negative as conditioning (not used by any source here), and (4) negative as gradient.

1. **Where negatives come from.** (a) Candidate tasks that fail validation: 49.9% of SWE-smith candidates do not break a test (Table 1); 95.3% of SWE-rebench candidates do not survive (derived from §2.1, §2.4). (b) Tasks outside the difficulty band: DeepSeek-V3.2 removes tasks with zero pass@100; GLM-5 removes questions a tool-free model answers; Tongyi removes always-solved and always-failed problems (§4). (c) Trajectories that fail the verifier. (d) Rollout failures caused by the environment, not the policy: GLM-5 notes that "coding-agent sandboxes can be inherently unstable and may fail for reasons unrelated to the model" ([[glm-5]] §4.1.2). (e) Verifier false negatives from flawed tests ([[swe-rebench]] §2.3; [[swe-bench-pro]] §6.2). No source reports the false-negative rate of its training verifier.
2. **What current practice does with them.**
   - Discard (sense 1): failed candidates, out-of-band tasks, failed SFT trajectories ([[swe-smith]] §3; [[toucan-mcp]] §3.1).
   - Re-route: QUEST sends teacher-failed tasks to the RL pool instead of discarding them (App. A.5); Agent-World sends failed arena traces to a diagnosis agent that writes new targeted tasks (§3.2.2).
   - Keep as content (sense 2): AgentScaler keeps trajectories with tool-call errors when the final state is correct (§3.1); GLM-5 SFT keeps "Erroneous segments within trajectories" but masks them from the loss ([[glm-5]] §3.1); TOUCAN irrelevance tasks train a zero-tool-call response to unsolvable requests (§3.2).
   - Use as gradient (sense 4) with exclusions: in RL, failed rollouts get negative advantages, but GLM-5 excludes environment-collapse failures and Tongyi excludes some negative rollouts such as length-limit truncations ([[tongyi-deepresearch]] §3.4.3).
3. **Mechanism.** For a token with logits z and softmax p, ∂ log p_y / ∂ z_j = 1[j = y] − p_j, where y is the sampled token and j ranges over the vocabulary. A negative advantage multiplies this gradient by a negative number, so z_y decreases and the removed probability mass goes to the other tokens in proportion to p_j, most of it to the highest-probability alternative. GLM-5's agentic RL objective uses the mean reward of the group as the baseline, A_i = r_i − (1/K) Σ_k r_k, where K is the number of traces sampled per problem and r_i is the reward of trace i ([[glm-5]] §4.1; the reasoning-RL section §3.2 divides the same quantity by the group standard deviation). Worked example under the §4.1 form: K = 8 rollouts, 3 correct (r = 1), 4 wrong (r = 0), and 1 correct rollout whose sandbox crashed and was scored 0. The mean is 3/8 = 0.375, so the crashed but correct rollout gets A = −0.375 and its actions are pushed down. GLM-5's rule removes the crashed sample; 7 valid samples exceed half of 8, so the group is padded by repeating valid samples. With only 4 valid samples (not more than half) the whole group is dropped ([[glm-5]] §4.1.2).
4. **Evidence.** Tongyi DeepResearch reports that "directly optimizing on an unfiltered set of negative rollouts significantly degrade training stability and can lead to policy collapse after extended training" (§3.4.3; no table). GLM-5 states that removing environment-failure samples "reduces spurious reward noise and improves training stability" (§4.1.2; no table). DeepSeek-V3.2 masks negative-advantage sequences whose policy divergence exceeds a threshold δ and reports improved stability in some otherwise unstable runs ([[deepseek-v3.1]] §3.1). The share of improvement attributable to negatives is not measured in any of these sources. SWE-Gym's self-improvement run, which trained on 868 on-policy successes plus 491 off-policy successes, dropped SWE-Bench Lite from 15.3% to 8.7% ([[excerpts/swe-gym-v2-extract]] §4.2); this involves positives only and shows that on-policy filtering alone does not guarantee gains.
5. **Controls.** Record a failure reason per rollout and exclude infrastructure failures from the loss; validate verifiers with a reference solution and a do-nothing agent before training; mask or drop rollouts truncated by length limits instead of assigning reward 0; bound negative updates for off-policy samples (DeepSeek-V3.2 sequence masking); keep teacher-failed tasks for RL rather than treating them as bad tasks.
6. **Diagnostics.** Per-environment failure-reason histogram (policy vs infrastructure vs timeout); fraction of groups dropped or padded; log-probabilities of positive- and negative-advantage samples tracked separately; pass@1 and pass@k per environment family; rate at which a do-nothing agent passes each task template.
7. **Effect on generality.** False negatives concentrated in one environment family (for example flaky tests in one language) push down correct behavior in that family and shift the policy toward families with cleaner verifiers. Irrelevance targets (sense 2) teach abstention when no tool fits; TOUCAN's 7B model rose on BFCL irrelevance from 67.93 to 75.18 while relevance fell from 72.22 to 66.67 ([[toucan-mcp]] Table 2), which shows the abstention trade-off.

## Recipe

| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| SWE-agent-LM-32B (Qwen2.5-Coder-32B-Instruct) | 32B | distill-SFT | trajectories; teacher; limits | 5,016 (≤ 3 per task, from 6,457 resolved runs); claude-3-7-sonnet-20250219 in SWE-agent; ≤ 75 steps, $2.00 | arXiv:2504.21798v2 §3, §4 | verified 2026-09-15 | §4: repeated easy trajectories degrade performance (no table); Fig. 1 trajectory scaling |
| SWE-agent-LM-32B | 32B | distill-SFT | LR; epochs; context; hardware | 5e-5; max 3; 32,768; 2-8 H100 (torchtune, full fine-tuning) | App. F.1 | verified 2026-09-15 | no ablation reported |
| SWE-Gym-32B (Qwen2.5-Coder-32B-Instruct) | 7B, 14B, 32B | distill-SFT | trajectories; LR; epochs; batch; context | 491; 1e-4; max 5; global 8; 32,768 | arXiv:2412.21139v2 §4.1, App. B.2 | verified 2026-09-15 | Fig. 1: resolve rate vs 100-491 trajectories |
| R2E-Gym-32B | 7B, 14B, 32B | distill-SFT | trajectories; teacher temperature; max context | 3,321 successful Sonnet-3.5-v2 trajectories; 0.2; 20K | [[r2e-gym]] card (arXiv:2504.07164v1 §3, App. B) | verified 2026-09-14 | Fig. 2: 100-3,200 trajectories |
| SWE-rebench dataset | — | data | installation recipe candidates; validator model | up to 3 per task; Qwen2.5-72B-Instruct | [[swe-rebench]] card (arXiv:2505.20411v2 §2.2, App. C) | verified 2026-09-14 | App. C Table 3: 1/3/10 candidates configure 6/8/9 of 18 instances |
| DeepSeek-V3.2 | not reported in arXiv:2512.02556 (V3.1-Terminus start point: 671B total, 37B activated, V3.1 model card) | RL | agent task counts (code / search / general / code interpreter) | 24,667 / 50,275 / 4,417 / 5,908 | arXiv:2512.02556v1 §3.2.3 Table 1 | verified 2026-09-15 | §4.3 Fig. 5: synthetic general-agent RL vs search+code RL |
| DeepSeek-V3.2 | see row above | RL | general-agent task filter | non-zero pass@100 under DeepSeek-V3.2; 1,827 environments | §3.2.3 | verified 2026-09-15 | Table 5: pass@1 12%-62% on 50 sampled tasks |
| GLM-5 | 744B total, 40B active (§2.1, Table 10) | RL | environment counts; stale-sample rule; failed-group rule | > 10k SWE environments, 9 languages; drop if w′ − w_0 > τ (τ not reported); pad group if valid > K/2, else drop | arXiv:2602.15763v2 §4.1, §4.1.2, §4.2.1 | verified 2026-09-15 | no ablation reported |
| GLM-5 | 744B total | RL | search-task difficulty filter | remove if a tool-free model answers in ≥ 1 of 8 attempts | §4.2.3 | verified 2026-09-15 | no ablation reported |
| Kimi K2 | 1.04T total | SFT | tool pool | 3000+ real MCP tools; over 20,000 synthetic tools | arXiv:2507.20534v2 §3.1.1 | verified 2026-09-15 | Fig. 9: t-SNE coverage (no ablation) |
| AgentScaler-30B-A3B | 30B-A3B | SFT | API pool; domains; stages; LR, epochs, data size | > 30,000 APIs; > 1,000 domains; 2 stages; not reported (checked §2-4, Limitation) | arXiv:2509.13311v1 §2.1, §3.2 | verified 2026-09-15 / not reported | Fig. 3: stage 1 vs stage 2 on ACEBench-en |
| TOUCAN SFT (Qwen2.5-Instruct) | 7B, 14B, 32B | SFT | instances; LR; epochs; batch; max length | 119.3K (28.3K core, 40K irrelevance, 15.8K diversify, 35.2K multi-turn); 2e-5; 2; 64; 32,768 | arXiv:2510.01179v1 §4.1, App. C.2 Table 5 | verified 2026-09-15 | App. C.3 Table 6: extension ablation on 14B |
| Agent-World-8B / 14B | 8B, 14B | RL | RL samples; tasks per step × rollouts; clip ε_low / ε_high; max trajectory | 5K; 32 × 8; 0.2 / 0.28; 80K tokens | [[agent-world]] ledger (arXiv:2604.18292v1 §4.1) | verified 2026-09-14 | no ablation reported |
| Agent-World | — | data | task keep rule; tool keep rule | ≥ 2 of 5 ReAct runs succeed; compiles and > 50% unit tests pass | §3.1, §3.1.1 | verified 2026-09-14 | no ablation reported |
| QUEST-35B (Qwen3.5-35B-A3B) | 35B-A3B | distill-SFT | tasks / trajectories / sessions; keep threshold | objective 5,070 / 19,435 / 39,861; open-ended 1,958 / 4,485 / 11,903; ε = 1 objective, 0.475 open-ended | arXiv:2605.24218v1 Table 1, §4.1 | verified 2026-09-15 | no ablation reported |
| QUEST-35B | 35B-A3B | distill-SFT | rollouts; retries; max tool calls; context threshold | 5 per task; 3 per failed trajectory; 100; 16K | App. A.4 | verified 2026-09-15 | no ablation reported |
| QUEST-35B | 35B-A3B | RL | tasks; reward; KL | 864 objective + 269 open-ended; R = 0.75 s_rubric + 0.25 min(s_fact, s_rubric); no KL penalty | Table 1, §4.4 Eq. 1 | verified 2026-09-15 | §6.3 Fig. 5: MT, SFT, RL stage ablation |
| Tongyi DeepResearch 30B-A3B | 30.5B total, 3.3B active | mid-train | context stages; general-data share | 32K then 128K (64K-128K agentic data); "small proportion" (value not reported) | arXiv:2510.24701v3 §3.3.1 | verified 2026-09-15 / not reported | no ablation reported |
| Tongyi DeepResearch 30B-A3B | 30.5B total | RL | reward; policy regime; baseline; clip values, group size, LR | 0/1 answer correctness; strict on-policy; leave-one-out; not reported | §3.4.3 | verified 2026-09-15 / not reported | Fig. 9: 32K/48K/64K context comparison |
| WebShaper (Qwen-2.5-32B/72B, QwQ-32B) | 32B, 72B | distill-SFT + RL | seeds; rollouts; trajectories | 18k seeds (keep if ≥ 1 of 5 correct); 5 rollouts per question; 5,000 trajectories | arXiv:2507.15061v1 §3.1, §3.3 | verified 2026-09-15 | Table 2: 5,000-sample dataset comparison |
| ADP-trained Qwen-2.5-Coder-Instruct | 7B, 14B, 32B | SFT | sampling multipliers; samples | orca agentinstruct 0.001, synatra 0.01, code feedback 0.1, nebius SWE 0.2, agenttuning 2, SWE-Gym 3, others 1; about 30K (coding) / 20K (web) | arXiv:2510.24702v2 App. C Table 9, C.1 | verified 2026-09-15 | Table 10: equal-scale ADP vs SWE-smith |
| Reasoning Gym RG-Math (Qwen2.5-3B-Instruct) | 3B | RL | steps; rewards; curriculum rule | 800 GRPO steps; accuracy 1.0 + format 0.2; raise level when performance > 70% over 20 steps | arXiv:2505.24760v2 §4.3, Fig. 5, §5 | verified 2026-09-15 | Table 5: curriculum vs uniform levels |

**Starting point for a small general-purpose run.** For a 7B-32B Qwen2.5-Coder-Instruct base and SWE environments, rejection-sampling SFT on a few thousand teacher trajectories with at most 3 trajectories per task, LR 5e-5, at most 3 epochs, and 32,768-token context reproduces the SWE-smith 32B setting (5,016 trajectories, 2-8 H100). For tool-use breadth on a 7B-32B Qwen2.5-Instruct base, TOUCAN used LR 2e-5, 2 epochs, effective batch 64, and 32,768 tokens on 119.3K instances that included 40K irrelevance examples. For RL on synthesized tool environments at 8B-14B, Agent-World used 32 tasks × 8 rollouts per step, ε_low 0.2 and ε_high 0.28, and an 80K-token trajectory cap on 5K RL samples. These rows come from different papers, models, and substrates; none was tuned for the others' settings.

## Generalization lens

**(a) What increases breadth.**
- More distinct environments: 10.3% → 15.1% over 4 → 100 repositories at a fixed 700 trajectories ([[swe-smith]] Fig. 5); 18.4% → 38.5% over 0 → 1,978 environments, with no fixed-budget statement ([[agent-world]] Fig. 8). Replicated in direction.
- More environment types in RL: synthetic general-agent RL improved unseen Tau2Bench, MCP-Mark, and MCP-Universe where search+code RL did not ([[deepseek-v3.1]] §4.3).
- Mixed agent datasets over a single source at equal size: 16.6% vs 11.0% on SWE-Bench Verified ([[agent-data-protocol]] Table 10).
- Multiple task formats: single-answer, offline-corpus, and rubric-tree tasks each lead on different benchmarks at 30B ([[quest-deep-research]] Table 3).
- Multiple languages in the environment pool: DeepSeek-V3.2 (8 languages) and GLM-5 (9 languages) build them ([[deepseek-v3.1]] §3.2.3; [[glm-5]] §4.2.1); no ablation isolates the language count.

**(b) What causes narrowing or forgetting.**
- Single-language environments: 8.4% on SWE-bench Multilingual after Python-only training ([[swe-smith]] App. F.4).
- Repository specialization: the SymPy-only 7B model is 7.6 points above the 100-repository 7B model on the SymPy subset and 1.3 points below it on the rest of Verified ([[swe-smith]] Fig. 4).
- Overfitting to training tool patterns: disallowed tool calls on BrowseComp-Plus after SFT ([[quest-deep-research]] §6.3); ACEBench-en Special 84.7 → 76.7 ([[agentscaler]] Table 1); BFCL relevance 72.22 → 66.67 and non-live AST 84.19 → 78.52 at 7B ([[toucan-mcp]] Table 2).
- Specialized RL objectives: QUEST RL "slightly sacrifices performance on HLE and GAIA" (§6.3).
- Simulated-only substrates: Simulator-8B BFCL V4 23.9 below the untrained Qwen3-8B's 40.4 ([[agent-world]] Table 1; not controlled).
- Recipe-matched benchmarks: OpenResearcher leads only on the offline benchmark that matches its synthesis ([[quest-deep-research]] §6.2).

**(c) How to measure it at this stage.**
- Hold out whole environments, servers, repositories, and languages; report scores on environments never used in training (MCP-Mark, MCP-Universe, SWE-bench Multilingual) next to in-family scores.
- Use date-split tasks: SWE-rebench flags results as potentially contaminated by issue creation date ([[swe-rebench]] §3.2).
- Report 5 runs with SEM and pass@5 under a fixed scaffold, as SWE-rebench does (§3.2); Llama-4-Maverick had pass@5 27.6 at 12.2% mean resolved (§3.3), so a single run can mislead.
- Run null agents against every grader: a do-nothing agent passes 38% of τ-bench Airline ([[agentic-benchmark-checklist]] §5.2).
- Known measurement errors: SWE-Bench Verified memorization signals ([[swe-bench-illusion]] §4.1); retriever dependence on offline search benchmarks ([[browsecomp-plus]] Table 1); benchmark-guided task synthesis ([[agent-world]] Table 2); general-benchmark comparisons that give the agent tools and the base model none ([[tongyi-deepresearch]] Fig. 11), which do not measure forgetting. General-ability forgetting across agentic training is covered in [[ch-30a]] and [[ch-51a]].

## Common mistakes and how to detect them

| Mistake | Observable symptom | Check |
|---|---|---|
| Keeping tasks whose verifier was never run on a reference solution | RL reward stays near 0 on a task family while teacher trajectories look correct | Run the gold patch or solution function; require F2P > 0 and pass-to-fail = 0, or verifier pass |
| Verifier accepts trivial outputs | Reward rises while held-out success is flat; short or empty answers get reward | Run a do-nothing agent and an answer-enumerating agent on every task template |
| Scoring environment crashes as policy failures | Reward variance spikes on one substrate; failures cluster by container host or time | Log a failure reason per rollout; exclude infrastructure failures; report dropped-group rate |
| Counting trajectories instead of environments | More trajectories do not change held-out scores | Report distinct repositories, servers, domains, and languages; ablate environment count at fixed trajectories |
| Mining benchmark repositories or servers into the training pool | Large gap between the public benchmark and a date-split or private-repository benchmark | Repository and server exclusion lists; SWE-rebench-style fresh tasks; SWE-Bench Pro commercial subset |
| Synthesizing tool environments that replicate a benchmark's domain code | In-domain benchmark rises more than other tool suites | Compare generated schemas and tool code against benchmark implementations before training |
| Leaking the evaluation criterion into the task text | Agent stops writing reproduction scripts or skips exploration | Count reproduction attempts per run (SWE-smith: 379 vs 127 of 500) |
| Single-language SWE pool | Non-Python edits contain Python syntax | Evaluate on SWE-bench Multilingual or a per-language held-out set |
| Difficulty filter tied to a stale model | RL batches become all-solved or all-failed; zero-advantage groups rise | Track the fraction of groups with all-equal rewards; refresh the pool with the current checkpoint |
| Offline search tasks evaluated with a different retriever | Score changes when only the retriever changes | Fix and report the retriever and corpus |
| Mixing agent datasets without harness-specific conversion | Data that trains well in one harness gives near-zero scores in another (SWE-smith data: 1.0% in OpenHands with Qwen-2.5-7B-Instruct vs 15.2% in SWE-agent with the 7B Coder model; base models also differ) | Convert through a shared schema and one converter per harness; evaluate in each target harness |

## Check your understanding

1. SWE-smith injects bugs into the latest commit, while SWE-Gym uses real historical issues. Explain how this choice lowers cost per task and what kind of task it cannot represent.
2. A pool is filtered with non-zero pass@100 and a second pool with at least 2 of 5 successes, both using the same model. Explain which tasks each pool keeps and how this changes the RL learning signal.
3. Explain why a verifier false negative in GRPO pushes down a correct trajectory, and trace where the removed probability mass goes.
4. AgentScaler's τ-bench domain databases closely match the official τ-bench implementations. Explain why its τ-bench gains are weaker evidence of general tool use than its ACEBench-zh gains.
5. At 30B, a single-answer recipe leads on BrowseComp and a rubric-tree recipe leads on DeepResearch Bench. Explain what this implies about how task format determines capability, and what a general research agent's task pool must contain.
6. SWE-smith's 32B model reached 40.2% on SWE-bench Verified and 8.4% on SWE-bench Multilingual. Explain the causal chain from the environment pool to both numbers.
7. GLM-5 scores 77.8 on SWE-bench Verified and 42.1% on SWE-rebench. Explain why this gap alone does not measure contamination and what comparison would.
8. Agent-World chooses which tasks to synthesize from failures in held-out arena environments and reports gains on MCP-Mark. Explain the conditions under which MCP-Mark remains a valid measure of generality.

## Connections

- Previous: [[ch-29b]] — Long-Conversation and Accumulating-Context Synthesis.
- Next: [[ch-29d]] — User Simulators, Trajectory Verification, and Failed Trajectories.
- Dependencies: [[ch-27]] — Agentic Trajectory Data; [[ch-32d]] — Agentic Mid-Training: Repository, Execution-Trace, and Trajectory Data Before Post-Training.
- Earlier chapters used here: [[ch-18]] — The Synthetic-Data Design Pattern: Generate, Filter, Deduplicate, Verify, Select, Mix; [[ch-23]] — Model Collapse and Verification of Synthetic Data; [[ch-26]] — Tool and Function-Calling Data.
- Later chapters that use these environments: [[ch-30b]] — Multi-Skill SFT Mixtures: Interference, Transfer, and Agentic and Long-Context Shares; [[ch-44]] — Process Supervision and Verifiable Rewards; [[ch-45b]] — Multi-Turn Agentic RL: Observation Masking, Credit Assignment, and Stability; [[ch-45d]] — Open Agentic Recipes Side by Side: Stage Placement, Data Mixture, and Agentic RL; [[ch-46a]] — Lab: Small Agentic SFT-then-RL Run with a Generality Gate; [[ch-54]] — Rollout Infrastructure: Off-Policy Data, Asynchronous RL, and Agentic Environments.
- Evaluation phase: [[ch-48]] — Contamination Detection and Its Effect on Reported Scores; [[ch-51a]] — Evaluating Agent Generality and Reliability.

## Sources

- [[swe-gym]] — library card for SWE-Gym; this chapter takes its numbers from the primary paper through [[excerpts/swe-gym-v2-extract]], because several card values (K = 10 per task, ~10K H100-hours, 7B 3.0% → 15.3% on Verified) are not in arXiv:2412.21139v2.
- [[excerpts/swe-gym-v2-extract]] — 64,689 raw instances, 200 annotation hours, 2,438 tasks, 6 TB images, Table 3 results, App. B.2 recipe, self-improvement drop.
- [[r2e-gym]] — SWE-Gen commit-based environments, synthetic vs real issue statements, generated-test distinguishability, recipe row.
- [[swe-smith]] — environment-first design, bug strategies and yields, storage and cost, repository scaling, specialization, Multilingual result, recipe.
- [[swe-rebench]] — automated installation, fail/pass validation, funnel, date-based contamination flags, DeepSeek-V3 comparison, recipe row.
- [[glm-5]] — SWE, terminal, and search environment pipelines, search difficulty filter, environment-failure and stale-sample handling, SWE-rebench evaluation.
- [[kimi-k2]] — real MCP and synthetic tools with hierarchical domain evolution, simulator and real sandboxes, rubric-filtered trajectories.
- [[agentscaler]] — read-write database abstraction, tool-graph construction, state-alignment filters, τ-bench consistency statement, results and regressions.
- [[toucan-mcp]] — real MCP server funnel, task and trajectory filters, irrelevance extension, BFCL results, SFT recipe.
- [[deepseek-v3.1]] — DeepSeek-V3.2 agent task counts, code-environment build rule, general-agent synthesis with verification functions, pass@100 filter, synthetic RL transfer, off-policy sequence masking.
- [[webshaper]] — Knowledge Projection formalization, layer-wise expansion, validation, dataset comparison.
- [[quest-deep-research]] — rubric trees, evaluation scripts and their error rate, retention funnel, reward formula, recipe-dependent capabilities, stage ablation.
- [[tongyi-deepresearch]] — three environment forms, offline Wikipedia environment, RL data curation, negative-rollout exclusion, mid-training placement.
- [[agent-data-protocol]] — schema, converter cost, sampling multipliers, mixed vs single-source results, harness dependence.
- [[agent-world]] — environment-count scaling, task keep rules, weighted tool-graph walks, Simulator-8B comparison, self-evolution rounds, recipe rows.
- [[reasoning-gym]] — procedural generators, cross-domain transfer and regressions, curriculum rule.
- [[swe-bench-illusion]] — SWE-Bench Verified memorization probes used in §7.
- [[swe-bench-pro]] — verifier false negatives when task specifications omit interface details.
- [[agentic-benchmark-checklist]] — do-nothing and shortcut agents that pass flawed graders.
- [[browsecomp-plus]] — retriever dependence of offline search evaluation.
