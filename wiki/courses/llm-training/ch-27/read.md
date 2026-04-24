<!-- chapter: ch-27
     track: synthetic
     title: Modality — Agentic Trajectories
     sources: [[agentinstruct]], [[agenttuning]], [[lumos]], [[fireact]], [[autoact]], [[agent-flan]], [[webarena-data]], [[swe-gym]], [[swe-rl]], [[openhands-data]], [[kimi-k2-agentic-data]], [[kimi-k2]], [[terminal-bench-trajectories]], [[explorer]]
     figures: figures/action-space.html
-->

# Chapter 27 — Modality: Agentic Trajectories

> **Core insight.** An "agent trajectory" is not a long chat. It is a sequence of `(observation, thought, action)` tuples where each action has *environmental side-effects* the next observation must reflect. This changes what data means: the dataset is not `(prompt, response)` pairs — it is *conditional on an executable world*. Every design choice in agentic post-training flows from that one structural fact: how you observe the world, what actions are legal, and how the world grades the final state.
>
> **Guideline.** Pick the environment first, the teacher second, the model third. The action space defines the trajectory format, the trajectory format defines the data pipeline, and only then do teacher-model and student-model choices matter. Mixing modalities at the data stage (AgentInstruct's multi-skill pipelines, Kimi-K2's pretraining-mix injection) works; mixing modalities at the model stage (one monolithic agent LoRA on everything) does not.

---

## Why this chapter exists

By [[ch-26]] you know how to synthesize reasoning and tool-call data inside a single autoregressive context. An agent is the next conceptual step: the model is no longer producing a self-contained response, it is producing *one turn* in a loop where an external world — a shell, a browser, a Python kernel, a Git repo — reads its action, mutates state, and replies with an observation. The loop can run for dozens of turns; trajectories routinely hit 15K–100K tokens ([[openhands-data]], [[swe-gym]]); and the reward that matters is *whether the final world-state satisfies a predicate*, not whether any single token was fluent.

This chapter is the design-space map for the data side of that loop. Six lineages cover the SFT corner (AgentInstruct, AgentTuning, Lumos, FireAct, AutoAct, Agent-FLAN); two benchmarks cover the environment corner (WebArena, SWE-Gym); one frontier RL recipe shows how rule-based reward scales (SWE-RL); one frontier model report shows what a 1T-class lab actually does end-to-end (Kimi-K2). The thread tying them together is a four-axis taxonomy — environment × action space × observation format × success signal — that you will use to read any future agent paper.

---

## 1. The six SFT recipes — a design-space tour

All six were published between Oct-2023 and Apr-2024. Each picked a different axis to vary.

| Paper | Year | Trajectories | Teacher | Core design move |
|---|---|---|---|---|
| [[agenttuning]] | 2023.10 | 1,866 | GPT-4 | **Mixing ratio**: 1:10 agent:ShareGPT preserves chat quality |
| [[fireact]] | 2023.10 | ~2,000 | GPT-4 | **Method diversity**: CoT + ReAct + Reflexion in one corpus |
| [[lumos]] | 2023.11 | 40K tasks → 200K triples | GPT-4 | **Module decomposition**: Plan / Ground / Execute separable heads |
| [[autoact]] | 2024.01 | ~10K | *none* (self) | **Self-differentiation**: one base model plays Plan/Tool/Reflect |
| [[agent-flan]] | 2024.03 | ~85K | GPT-4 | **Negative examples**: four hallucination modes explicitly corrected |
| [[agentinstruct]] | 2024.07 | 25M | GPT-4 | **Agentic pipeline**: 43-generator flows per skill |

Read them as a conversation. AgentTuning establishes that a small curated corpus works if you mix it correctly. FireAct adds that *how* you collect trajectories matters — three prompting methods beat one. Lumos adds that trajectories have *internal structure* (plan / ground / execute) and that structure should show up in the training data, not be hidden inside the monolithic ReAct blob. AutoAct tests whether you even need a GPT-4 teacher — the answer is "within a narrow QA domain, no." Agent-FLAN answers a complaint none of the first four addressed: *what if the trained model hallucinates tool calls on prompts that don't need tools?* It introduces four explicit negative-example classes (format / action / parameter / relevance hallucination). Finally, AgentInstruct scales the idea to 25M pairs by making the *generation pipeline itself* a multi-agent system.

The corpus-size column hides a critical fact. AgentTuning and FireAct sit at ~2K trajectories each; Agent-FLAN at 85K; AgentInstruct at 25M. Three orders of magnitude in four papers. The quality claim of the small-corpus papers is that **diversity of structure** (method, environment, or decomposition) compensates for volume; the scaling claim of AgentInstruct is that once the pipeline is in place, you may as well run it to exhaustion. Both claims are empirically true on their respective benchmark suites; the open question is whether small-diverse corpora hit a lower ceiling than large-pipeline corpora on held-out tasks. Agent-FLAN's ablation table hints at the answer — removing any single capability type (instruction-follow, agent-reason, generalization) costs 0.3–0.5 AgentBench points; removing negatives triples the hallucination rate. Structure matters at every scale, but at 85K+ scale the *ablation deltas* shrink, which is why you see AgentInstruct skip the decomposition debate and just scale.

### 1.1 AgentInstruct's six-flow taxonomy — the pipeline-of-specialists template

[[agentinstruct]] (Mitra et al. 2024, Microsoft) is the most ambitious single paper in this list. Its central abstraction is an **agentic flow** — a pipeline of specialized LLM agents where each stage has its own prompt, its own tool access, and its own output schema. The four generic stages for every skill:

1. **Content Transformation** — one agent rewrites raw input (a web document, a codebase, an API spec) into a canonical intermediate structure (passage + candidate-questions list, function + test stub, schema + example call).
2. **Seed Instruction Generation** — 10–40 *parallel* "generator" agents, each prompted to produce a distinct sub-skill (literal question / inferential question / multi-hop / numerical reasoning / …). The reading-comprehension skill alone uses **43 generator agents**, one per question category.
3. **Instruction Refinement** — a "suggester" agent proposes improvements; an "editor" agent applies them. Loop up to 3 iterations per instruction. This is how you get diversity of *phrasing* without losing structural coverage.
4. **Answer Generation + Validation** — GPT-4 produces the gold answer; an LLM-judge filter drops low-quality pairs.

Aggregated across 17 skills (reading-comprehension, math, code, tool-use, RAG, creative-content, web-agent, long-context, …) this produced the proprietary **AgentInstruct-25M** corpus. Orca-3 (Mistral-7B base + AgentInstruct SFT) outperformed Mistral-7B-Instruct by 40% on AGIEval, 54% on GSM8K, 3× on MATH.

Two takeaways for your own pipelines. First: **the generator count is a diversity knob**, not a budget waste — 43 narrow-prompt agents cover a wider sub-skill distribution than one broad-prompt agent sampled 43×. Second: **the refinement loop is non-negotiable** — single-shot GPT-4 generation plateaus fast; iterative suggester/editor adds ~3 points on hard sub-skills at ~1.5× cost.

Skill-specific variants worth noting. The **tool-use flow** seeds from real API docs (not synthetic schemas), lets generator agents synthesize queries at varying tool-count complexity (1 tool → 2 tools → composed chains), and routes refinement through schema-correctness checks. The **RAG flow** uses content agents to build passage clusters, then query agents generate questions that require evidence fusion across passages. The **long-context flow** stitches documents up to 8K+ tokens before generator prompts fire. In all three, the pattern is: one *upstream* agent prepares the substrate, then *many* downstream agents sample from it at varied difficulty. This is the opposite of the self-instruct lineage's "one prompt, many samples" approach; the AgentInstruct bet is that **substrate diversity + prompt specialization** beats **one diverse prompt**.

### 1.2 Lumos's Plan/Ground/Execute format spec

[[lumos]] is the one you copy when designing a new agent-trajectory format. Every trajectory is decomposed into three aligned supervised targets:

```
Plan:    (task, gold_answer) → list[subtask]
Ground:  (subtask, env_state) → action in unified grammar
Execute: action → observation (from real env or tool)
```

The unified action grammar is explicit: `Search[query]`, `Retrieve[doc_id]`, `Calculate[expr]`, `Click[element]`, `Type[element, text]`, `Back`, `Finish[answer]`. This is the grammar Lumos trains against. Conversion from an existing dataset (HotpotQA, ALFWorld, WebShop, Mind2Web, Musique, GSM8K, MATH, StrategyQA, ScienceQA) goes through GPT-4 as an *annotator* — a prompt that takes a raw trajectory and emits the three-layer decomposition.

Two training modes emerge: **Lumos-I (iterative)** replans after every observation; **Lumos-O (onetime)** plans the whole task upfront, then executes sequentially. Each module can be its own LoRA or its own head. On generalization to held-out environments, the modular decomposition costs only ~8 points versus ~20 for monolithic ReAct fine-tunes — the reason is that the *action grammar* is shared across environments even when the concrete tools differ.

Use Lumos's format when your downstream plan is "swap a retriever / browser / code executor without retraining." Use AgentInstruct's format when your downstream plan is "one giant SFT blob." They are not competitors; they are different API-stability choices.

The three-module decomposition also produces a cleaner supervision signal per module than a monolithic ReAct trace. The Planning module sees `(task → list[subtask])` pairs with clear structural targets; Grounding sees `(subtask + env_state → action)` where the action is constrained by the unified grammar; Execute is pure environment interaction. A 7B model can specialize each module effectively because each has a narrower output distribution than a unified ReAct agent. This is the same insight AutoAct reuses ([[autoact]] splits Plan/Tool/Reflect); it recurs in Kimi-K2's sub-agent orchestration (planner / executor / critic). **Role specialization is a persistent design pattern** across the 2023→2025 literature — keep it in your toolbox even when your final-deployment model is monolithic, because the *data* can still be role-partitioned during synthesis.

### 1.3 Agent-FLAN's four hallucination classes — the negative-example ontology

[[agent-flan]] is the paper to re-read if your SFT-trained agent over-calls tools in chat. It classifies hallucinated tool calls into four distinct failure modes:

| Mode | Trigger | Gold response |
|---|---|---|
| **Format** hallucination | Model emits malformed tool-call JSON | Corrected call *or* a text refusal |
| **Action** hallucination | User query doesn't need a tool | Text-only answer, no call |
| **Parameter** hallucination | Right tool, wrong args | Tool call with correct args |
| **Relevance** hallucination | Tool list doesn't contain a needed tool | "I cannot help with this tool set" refusal |

Each class gets its own synthetic-negative-example pool, generated by prompting GPT-4 with common failure patterns drawn from the base model's errors. Agent-FLAN-7B cuts hallucinated tool calls on AgentBench held-out by 5× vs AgentTuning baseline. The lesson: **agent SFT without negatives is an open-loop controller**. Adding the four negative classes is the closed-loop correction.

Agent-FLAN's second contribution — easy to miss — is **format alignment**. The paper rewrites agent trajectories to avoid special tokens and delimiters that don't appear in Llama-2 pretraining (e.g., custom `<tool>`/`</tool>` pairs become plain markdown-code-fenced JSON blocks). Keeping the training distribution close to the pretraining distribution reduces catastrophic forgetting and improves preserved-chat-quality metrics (MT-Bench within 0.5 points of base Llama-2-Chat). This is a detail the 2024 papers increasingly converged on — don't invent new tokens for agent formatting unless you're also planning to pretrain on them ([[kimi-k2-agentic-data]] does; most SFT-only papers shouldn't).

### 1.4 FireAct — method diversity beats method depth

[[fireact]] (Chen et al. 2023, Princeton + Cambridge) sits at the opposite end from AgentInstruct in scale but makes an orthogonal claim. For the same ~2K HotpotQA + Bamboogle question pool, they collect trajectories via **three** prompting methods in parallel: Chain-of-Thought (GPT-4 reasoning only), ReAct (GPT-4 with Wikipedia search in `Thought/Action/Observation` loop), Reflexion (GPT-4 attempts, reflects on failure, retries up to N=3). Each question gets one trajectory per method; each trajectory is labeled with its method name so the student can learn method-specific formatting.

The ablation is the paper's core result: CoT-only SFT hits 38.9 HotpotQA EM, ReAct-only 37.3, Reflexion-only 35.2 — but the three-method mix hits **40.0**. Strict improvement from diversity, at the same data volume. At inference, a method-specific system prompt lets the same model switch styles. The implication: **what prompting method you collect under is itself a hyperparameter**, and the training-time answer is "all of them."

### 1.5 AutoAct — the zero-teacher lower bound

[[autoact]] is the recipe to consult when your API budget is zero and you're OK with narrow-domain QA. A single base model (Llama-2-7B/13B) plays three roles — Plan, Tool, Reflect — via separate LoRAs, and the loop produces its own training data:

1. Meta-agent prompts the base model to classify its role for each turn.
2. Base model rolls out trajectories on raw HotpotQA questions under each role.
3. Self-consistency filter: keep trajectories whose final answer matches gold *or* matches self-consistency majority.
4. Fine-tune the three LoRAs on their role-specific subsets.
5. Iterate — retrained sub-agents generate new rollouts for the next round.

At 13B the AutoAct model hits ~36 EM on HotpotQA — within ~4 points of GPT-4-teacher-distilled baselines, with zero API spend. Saturates by iteration 4. The open weakness: **self-consistency anchors on the base model's biases**. If the 7B base systematically misreads a question type, no amount of iteration will correct it. This is why AutoAct's successors (and the self-improvement-at-frontier-scale papers) pair self-play with an external verifier — purely self-referential loops drift.

---

## 2. Environment-grounded corpora — when the world does the grading

The 2024→2025 shift is away from teacher-distilled trajectories (which inherit the teacher's ceiling) toward **environment-grounded** trajectories where the world itself labels success.

### 2.1 WebArena — a self-hosted browser with deterministic state

[[webarena-data]] packages five real open-source apps — GitLab, Reddit-clone (Postmill), Shopping (Magento), OpenStreetMap, Calendar — into a Docker-compose bundle with a deterministic initial DB state and per-task reset scripts. 812 tasks span retrieval, browsing, form-filling, and multi-step transactions.

The action vocab: `click [element_id]`, `type [element_id] [text]`, `hover`, `press [key]`, `scroll`, `tab`, `new_tab`, `goto [url]`, `go_back`, `stop [answer]`. The observation is either the accessibility tree (text representation of the DOM — preferred for text-only agents) or screenshot+tree (multimodal, used in VisualWebArena).

Success is a **predicate** over final URL / page content / DOM state — not a similarity to a reference trajectory. Three predicate categories: info-lookup (gold-string match), content-producing (predicate over created content), state-modifying (predicate over DB state). Trajectory-collection practice: run GPT-4 with a SeeAct-style scaffold, run the success predicate, keep only trajectories that pass. Community dataset scale: tens of thousands of successful trajectories at per-trajectory cost $5–$20 (GPT-4V).

The **environment-drift** hazard is real and under-appreciated. Docker images must be pinned; app versions (GitLab, Magento) upgrading silently break tasks whose success predicates depended on old DOM structure. A dataset built on WebArena v1.0 may not be re-executable against v1.2 without re-running the success-predicate pass. For long-lived agent corpora, either pin the bundle images forever (storage cost, but reproducible) or plan periodic re-validation. A second hazard: **shortcut learning** — some tasks are solvable by URL-hacking a known-shortcut URL rather than navigating through the UI. Strict success predicates are the mitigation, but some leakage is always present, which is why the frontier numbers (GPT-4 ~35%, VisualWebArena ~20%) have a scaffold-sensitivity variance of 5–10 points.

### 2.2 SWE-Gym — 2,438 executable GitHub issues

[[swe-gym]] (Pan et al. 2024, Berkeley + CMU + Apple) is the SWE-side analogue. 2,438 real GitHub issues from 11 Python repos (astropy, sympy, django, matplotlib, …), each packaged as a Docker image containing the repo at pre-PR commit + the PR's test files applied (so tests exist but code doesn't satisfy them). Hidden test command included.

The action space is the **OpenHands scaffold** ([[openhands-data]]): `str_replace_editor` (view/create/str_replace/insert/undo_edit), `execute_bash` (shell + pytest), `browse` (filesystem), `finish` (submit patch). Trajectories routinely median ~15K tokens, tail past 100K for long debugging sessions.

Recipe: (1) run OpenHands with a teacher (Qwen-2.5-Coder-32B or Claude-3.5) on each SWE-Gym task, up to K=10 rollouts; (2) run hidden tests, label each trajectory pass/fail; (3) filter to all-pass only; (4) rejection-sampling SFT the student. Numbers: Qwen-2.5-Coder-7B goes 3.0% → **15.3%** on SWE-Bench Verified after RS-SFT, → **20.3%** with a trained verifier doing best-of-N at inference. 32B hits **32.0%** — open SOTA at release (Dec 2024). Verifier adds +5 points over SFT alone.

The verifier is a separate model trained on (trajectory, success) pairs from SWE-Gym — it learns to rank trajectories by predicted success from execution-labeled data. At inference you sample K trajectories with the SFT policy, score each with the verifier, and pick the highest-scoring one. Scaling behavior: trajectory count and verifier-N both show log-linear returns on SWE-Bench Verified; no plateau visible at 32B + K=10. This is the cleanest empirical case for **execution-labeled rejection-sampling SFT** as an agent-training recipe, and the reason the OpenHands scaffold became the 2025 de facto agent data pipeline.

Two SWE-Gym practicalities to remember. First, Docker-image maintenance cost is non-trivial — 2,438 images each with a full Python environment + hidden test suite + issue metadata consume storage on the order of hundreds of GB to a few TB depending on layer dedup. The "491 tasks immediately compatible with SWE-Bench Lite" subset is what you use for fast iteration during pipeline development; the full 2,438 is for production training runs. Second, **language-narrow is a real ceiling** — SWE-Gym is Python-only, and transfer to Go/Rust/TypeScript is completely untested in the published numbers. Multi-language SWE trajectories are the 2026 frontier (expect papers adding Rust cargo-integrated tasks, TypeScript jest-integrated tasks) but as of now the recipe is Python-bounded.

### 2.3 Terminal-Bench trajectories and Explorer — the terminal and web trajectories

[[terminal-bench-trajectories]] (2026) releases the full agent traces (messages, tool calls, observations) for tens of thousands of trials over Terminal-Bench 2.0 CLI tasks — turning a benchmark into a reusable trajectory corpus. [[explorer]] (Pahuja et al. 2025, MS) goes the other direction on the web side: rather than use a fixed 812-task benchmark, it *explores* the web first (broad intent generation) then refines successful trajectories into training data. Released 94K successful multimodal web trajectories across 49K unique URLs. The design pattern — **decouple intent discovery from trajectory refinement** — is now the default for web-agent data at scale.

---

## 3. SWE-RL — rule-based reward at open-source scale

[[swe-rl]] (Wei et al. 2025, Meta FAIR) is the paper that proved you don't need executable environments to do RL on SWE tasks at scale. The trick is a rule-based reward that's dense, cheap, and surprisingly hard to game:

$$
r = \texttt{difflib.SequenceMatcher(None, predicted\_patch, ground\_truth\_patch).ratio()}
$$

That's it. `difflib.SequenceMatcher.ratio()` returns a float in `[0, 1]` based on matching-block coverage. For an (issue, code_context, ground_truth_patch) triple scraped from GitHub, you have the agent emit a unified-diff patch and score it against the human PR diff. No unit tests run during training — only at eval time (SWE-Bench Verified).

**The data:** 11M (issue, context, patch) triples mined from GitHub Archive BigQuery. Filters: PR merged, linked issue, ≤10 files, ≤500 lines, Python-primary, MinHash dedup. **The algorithm:** GRPO with group size G=8, KL coefficient β=0.02, LR 1e-6. **The base:** Llama-3.1-70B-Instruct. **The cost:** ~1M H100-hours.

Headline result: **Llama3-SWE-RL-70B hits 41.0% on SWE-Bench Verified** — open SOTA at release, beating DeepSeek-Coder-V2-Instruct (18.0%) and matching SWE-Gym-32B. The paper's most provocative finding is out-of-domain transfer: training only on SWE pushes HumanEval+ by +6, MATH by +4, BBH by +3 over the baseline. Hypothesis: RL on software-engineering tasks teaches "long-horizon grounded planning" transferable across domains. (This connects to [[front-loading-reasoning]] from ch-26.)

**Why similarity beats execution as a training signal.** Execution rewards are sparse — many tests fail for unrelated reasons (dependency version, unrelated test flakiness, setup error). Similarity reward is dense: *every* sample gets a gradient signal. The cost is gameability — a patch that copies context verbatim gets partial credit without fixing anything. Mitigation: format filters (must be a diff, must modify code, not just comments). The authors also experiment with binary thresholding vs continuous reward; continuous wins.

One caveat: SWE-RL is **single-turn** — issue → patch, no file navigation, no test running mid-trajectory. Multi-turn RL on executable envs is still SWE-Gym's territory. The two recipes are complementary, not competing; SWE-RL does the cheap-dense-signal stage and SWE-Gym does the environment-grounded multi-turn stage.

Decontamination deserves a paragraph. SWE-Bench Verified comes from the same GitHub universe that SWE-RL scrapes, so date-based filtering (training data predates the benchmark issues) and commit-hash blocklists are mandatory. The paper reports both. A second, subtler risk: **the similarity reward is biased toward patches that look like human diffs**, which may mask the model's ability to generate *better* patches than humans wrote. At training time this looks like ceiling behavior; at eval time on SWE-Bench Verified (which tests functional correctness via unit tests, not string similarity) the gap between training reward and eval reward shows up as a training-curve-plateau while the eval curve still climbs. Reading this mismatch correctly requires you to hold the two metrics separately in your head — something new ML practitioners on agent pipelines routinely collapse.

---

## 4. Kimi-K2 — what a frontier lab actually does

[[kimi-k2]] and [[kimi-k2-agentic-data]] (Moonshot AI, 2025) together describe the most complete public frontier-lab agentic recipe. K2 is a 1T-parameter MoE with 32B active, pretrained on 15.5T tokens with zero loss spikes using the **MuonClip** optimizer (Muon + QK-Clip — rescales Q/K projection matrices post-update to cap attention logit magnitude, which plain Muon otherwise lets drift past 1000).

The agentic recipe has four stages:

1. **Agentic PRETRAINING data (~1T tokens).** A synthetic environment simulator generates tens of thousands of tool schemas (web search, file ops, code exec, DB, calendar, enterprise APIs) and plausible tool-response shapes. Multi-agent simulation produces trajectories: a "user" agent issues queries, a "planner" decomposes, an "executor" emits tool calls, a "critic" reviews. Up to 5 sub-agents per trajectory. Critic-LLM rates success/coherence/tool-call validity; top-scoring trajectories enter the pretraining mix at **~3–5%** of the total token budget.
2. **SFT.** Real-world agentic tasks (SWE-Bench-style issues, τ-bench tasks, tool-calling datasets) over the 20,000+ tool library from the K2 technical report.
3. **Joint RL stage.** Combines two reward streams into a single scalar: **RLVR** (verifiable rewards — math, code, tool-call correctness) + **self-critique rubric reward** (the model produces a rubric appropriate for the task and scores its own completion against it). The self-critique stream is the alignment-for-open-ended-tasks component; the RLVR stream is the capability-for-verifiable-tasks component. Combined, one RL stage trains both skill slices.
4. **Evaluation.** K2-Instruct hits **~65% on SWE-Bench Verified**, leads open models on τ-bench, competitive with Claude-3.5-Sonnet on tool-use.

The contentious claim is stage 1: Moonshot argues agentic behavior is best **installed during pretraining**, not bolted on as a post-training afterthought. Mixing agent-format tokens into the pretrain distribution gives the base model a native tool-calling "vocabulary" so post-training isn't fighting the base distribution. This is the sharpest break from the SFT-only lineage in §1. Whether it generalizes is still an open empirical question — nobody outside Moonshot has reproduced the 1T-token agentic-pretrain mix at scale.

Three details from the K2 report worth holding in mind. First, the self-critique rubric reward extends alignment to **open-ended** tasks where no automated verifier exists — the model generates (a) a task-appropriate rubric, then (b) its own completion, then scores completions against the rubric. Joint RL combines this with RLVR into a single scalar, so one stage trains both verifiable and open-ended skills. This is the descendant of [[constitutional-ai]]'s self-rating idea, operationalized at 1T-MoE scale. Second, **MuonClip** is not optional for trillion-scale stability — plain Muon without QK-Clip observed attention-logit max blowing past 1000 and diverging; QK-Clip rescales Q/K projection matrices after each update so the max stays below a threshold, yielding zero loss spikes across the full 15.5T-token run. Third, the **20K+ tool library** is the surface area from which both pretraining and post-training trajectories are sampled — a single tool-schema registry feeds all stages, which is why agent behavior stays coherent across the flow. The unification is as much an engineering contribution as a scientific one.

A fourth detail the K2 report treats cautiously but is worth extracting: **synthetic-env mismatch**. The simulator-generated tool responses are plausible but not identical to real-world API failure modes — real APIs return `429 rate limited`, `500 upstream timeout`, partial results, stale caches; simulators tend to emit cleaner responses. This is the analogue of ALFWorld / WebShop being "stylized" versus real web. K2 mitigates by mixing in real-env trajectories during SFT (τ-bench is real; SWE-Bench-style tasks execute real code). The methodology-vs-dataset opacity is the other caveat — Moonshot publishes the *how* (four-stage flow, critic-filter, 3–5% pretrain weight) but not the *what* (the 1T-token agentic corpus itself). You can reproduce the recipe; you cannot reproduce the corpus without spending comparable dollars.

---

## 5. Action-space design — the one table you need

The action space is the single design choice that constrains everything else: observation format, trajectory length, success signal, teacher cost, and eval harness all follow. See [[figures/action-space.html]] for the interactive side-by-side.

| Environment | Observation type | Action vocab | Reward signal | Typical traj length |
|---|---|---|---|---|
| **Web browser** ([[webarena-data]], [[explorer]]) | Accessibility tree (text) *or* screenshot+tree (multimodal) | `click[id] / type[id,text] / hover / press / scroll / tab / new_tab / goto[url] / go_back / stop[answer]` | Predicate over URL + DOM + DB state (three categories: info-lookup, content-producing, state-modifying) | 5–30 steps, 10K–50K tokens |
| **Terminal** ([[terminal-bench-trajectories]], [[openhands-data]]) | Shell stdout/stderr + exit code | `execute_bash[cmd]`, possibly `execute_ipython_cell` | Test-suite pass, file-state check, or CLI-predicate | 5–50 steps, 5K–40K tokens |
| **Repo / SWE** ([[swe-gym]], [[swe-rl]], [[openhands-data]]) | File contents + dir listing + test output | `str_replace_editor.view/create/str_replace/insert/undo_edit`, `execute_bash`, `browse`, `finish` | Hidden pytest pass (SWE-Gym) *or* difflib-ratio vs gold PR (SWE-RL) | 10–100+ steps, 15K–100K tokens |
| **Sandbox** ([[kimi-k2-agentic-data]], [[agentinstruct]] tool-use subset) | Simulated tool-response JSON | OpenAI-style `tool_calls` JSON over 20K+ schemas | Critic-LLM score + JSON-schema validity + no-repeat-loop | 1–20+ tool calls, variable |

Three structural lessons from the table:

- **Observation format dictates teacher cost.** Accessibility trees are cheap — text-only, GPT-4 can process them at normal rates. Screenshots are expensive — GPT-4V at $5–$20 per web trajectory. Terminal stdout is free. This is why WebArena text-mode datasets are 10× larger than VisualWebArena datasets at equal budget.
- **Reward signal determines RL tractability.** Dense rule-based signals (SWE-RL's difflib, predicate-with-partial-credit) support RL at 11M-sample scale. Binary test-suite rewards (SWE-Gym) support rejection-sampling SFT cheaply but are sparse under RL — most samples get 0 gradient. Critic-LLM rewards (Kimi-K2, Agent-FLAN filter) are dense but drift-prone; they require self-critique rubric anchoring or fixed-scale calibration.
- **Action vocab size trades generalization for safety.** OpenAI-style `tool_calls` JSON over 20K+ schemas is maximally general but trivially hallucinatable (Agent-FLAN's four negative classes are almost all this failure mode). A tight 10-action WebArena vocab is safer — the model can't invent a `hack_database` action because the grammar doesn't permit it.
- **Trajectory length determines the training infrastructure.** Median 15K tokens on SWE is barely different from a normal SFT example; tail to 100K+ needs FlashAttention-3-class long-context support plus gradient checkpointing over the full trajectory. Web agents at 50K tokens sit in between. Kimi-K2's simulated-sandbox trajectories at "several thousand tokens median" are cheap to train on; SWE-Gym trajectories at 15K median are not. If you're building the data pipeline before the training stack, check the length distribution first — the training team will thank you.

A final point the table can't show: **teacher choice leaks into the action distribution**. GPT-4 has its own priors about when to navigate, when to search, when to stop. Collecting trajectories with GPT-4 as the teacher imprints those priors onto the student. Changing teacher mid-dataset (mixing Claude-3.5 and GPT-4 rollouts, as OpenHands community releases often do) diversifies the action distribution but also makes the corpus slightly inconsistent — Claude tends to write longer `think` blocks, GPT-4 tends to move faster. Either is fine; mixing needs to be labeled so downstream filtering can stratify.

---

### 5.2 Rough cost reckoning — what each recipe actually costs

The numbers below are order-of-magnitude estimates from the papers, not precise disclosures. Hold them loosely; use them to sanity-check claims.

| Recipe | Data scale | Compute (rough) | Notes |
|---|---|---|---|
| AgentTuning SFT | 1,866 trajectories + GPT-4 distill | ~$20K API + 100s of GPU-hours for SFT | Cheapest real agent recipe |
| FireAct SFT | ~2,000 traj × 3 methods + GPT-4 | ~$3K API + SFT compute | Method-mix dominates spend |
| AgentInstruct 25M | 25M pairs + multi-agent flows | >$500K GPT-4 API est. | Pipeline cost is the bottleneck |
| SWE-Gym RS-SFT (32B) | 2,438 tasks × K=10 rollouts | ~10K H100-hours | Docker + teacher cost dominates |
| SWE-RL 70B | 11M GH triples + GRPO | ~1M H100-hours | RL dominates; teacher-free |
| Kimi-K2 agentic pretrain | ~1T agentic tokens | Part of 15.5T-token pretrain | Integrated into base training |

## 6. What you should take into ch-28

1. **The dataset is the environment.** Every agent-SFT paper you read, mentally reduce to `(environment, action vocab, observation format, success predicate)`. Everything else is teacher-model plumbing.
2. **Negatives are load-bearing.** [[agent-flan]]'s four hallucination classes are not optional add-ons. Any agent corpus missing them will produce a model that over-calls tools in chat.
3. **Mixing ratio matters more than raw agent-data volume.** [[agenttuning]]'s 1:10 rule generalizes: if agent data dominates the SFT mix, general chat quality drops; if general data dominates, agent skill never installs.
4. **Pretrain injection beats post-training retrofit at frontier scale.** [[kimi-k2]] argues this explicitly; [[swe-rl]]'s out-of-domain transfer hints at why — agentic tokens in the base distribution change what "next token" even means.
5. **Rule-based dense rewards scale; execution-based sparse rewards do not (for RL, at training time).** SWE-RL's 41.0% at 1M H100-hours is the existence proof. Save execution for eval.
6. **Role specialization recurs — use it.** Lumos (Plan/Ground/Execute), AutoAct (Plan/Tool/Reflect), AgentInstruct (suggester/editor/generator), Kimi-K2 (user/planner/executor/critic). Even when your deployed model is monolithic, role-partitioning the *data* gives cleaner supervision signals.
7. **Environment-drift is a long-lived-corpus problem.** WebArena DOM upgrades, SWE-Gym dependency drift, terminal-env library changes — pin your images, or plan periodic re-validation. Benchmarks that shipped in 2024 are not re-executable in 2026 without work.

Next chapter extends the structural shift from "conditional on executable world" to "conditional on very long context" — long-context synthesis, where the world is replaced by a 128K–1M-token document and the action is "read carefully." The action space collapses to `read(span)` + `attend(position)`, but the trajectory length (in tokens, not steps) explodes from 100K to 1M. Different modality, same structural lesson: the environment shapes the data.
