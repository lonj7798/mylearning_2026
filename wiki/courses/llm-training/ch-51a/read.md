<!-- chapter: ch-51a
     track: eval
     kind: content
     title: Evaluating Agent Generality and Reliability
     deps: [ch-51, ch-45b]
     sources: [[swe-bench-pro]], [[terminal-bench-2]], [[terminal-bench-trajectories]], [[terminal-bench-trajectories-stats]], [[tau-bench]], [[tau2-bench]], [[bfcl]], [[browsecomp-plus]], [[toolathlon]], [[are-gaia2]], [[holistic-agent-leaderboard]], [[agentic-benchmark-checklist]], [[swe-bench-illusion]], [[swe-rebench]], [[impossiblebench]], [[metr-time-horizon]], [[minimax-m2-aligning-to-what]], [[agent-world]], [[agentgym-rl]]
     figures: figures/pass-hat-k.html, figures/harness-coordinates.html
     revised: 2026-09 (generality revision)
-->

# Chapter 51a — Evaluating Agent Generality and Reliability

> **Core insight.** An agent score is not a property of a model. It is the output of a function of four arguments:
> the model, the scaffold, the environment version, and the budget. Holding the model and the task set fixed and
> changing only the scaffold moves Gemini 2.5 Pro from 32.6% to 15.7% on Terminal-Bench 2.0 ([[terminal-bench-2]],
> Table 2), and holding the model, scaffold and task set fixed while adding a $2 per-task cost cap moves Claude
> Sonnet 4 from 42.7% to 17.6% on SWE-Bench Pro and reverses its ordering against GPT-5 (high)
> ([[swe-bench-pro]], Tables 1 and 5). A second measurement is needed beside capability: the same agent that
> resolves 38.6% of Toolathlon tasks on one run resolves 51.9% on at least one of three runs and only 20.4% on all
> three ([[toolathlon]], Table 3). Capability and reliability are different quantities and they move separately.
>
> **Guideline.** When reporting an agent result, pin and publish the harness coordinates (scaffold and version,
> tool schemas, turn / time / token / cost budget, context strategy, environment image, reasoning effort, decoding
> settings, number of runs), because these coordinates are worth tens of points ([[terminal-bench-2]], Table 2;
> [[swe-bench-pro]], Table 5; [[are-gaia2]], Fig. 13). When the claim is about generality rather than about one
> benchmark, evaluate on environments that were not in the training mixture and report the seen-versus-unseen
> pair, because the fresh-slice comparison in [[swe-rebench]] separated two checkpoints that a contaminated
> benchmark ranked 4.5 points apart and a fresh slice ranked 0.6 points apart in the other direction (Table 2).
> When the agent will be run once per user request, report pass^k next to pass@1, because pass^k is the quantity
> that the deployment sees ([[tau-bench]], §3). When a benchmark score is used as a gate, first run a do-nothing
> agent and an exploit agent against the grader, because that check found 38% of τ-bench Airline tasks passable by
> returning nothing and 100% of SWE-Lancer tasks passable by overwriting the tests
> ([[agentic-benchmark-checklist]], §5.2, App. E.2, E.4).

## Why this chapter matters for a general-purpose model

The stage sits at the end of the pipeline: pre-training → mid-training → SFT → preference optimization → RL →
**evaluation**. [[ch-45b]] trained a policy to act over many turns; this chapter decides whether that training
produced an agent that works in environments nobody trained it on, and whether it works often enough to be used.

Three properties of agent evaluation make it different from the single-turn evaluation of [[ch-51]].

1. **The score is not attached to the model alone.** A static benchmark asks the model for one completion. An
   agentic benchmark runs a program — the scaffold — that calls the model many times, parses its output, executes
   tools, truncates context, and decides when to stop. Every one of those decisions is a free parameter that the
   scorer, not the model, controls.
2. **The variance is larger and structured.** A single-turn item fails independently of the next item. An agent
   trajectory fails at a step and then keeps running, so a single early mistake changes every later observation.
   Run-to-run variation across trajectories is therefore both larger than token-level sampling noise and
   correlated inside a task.
3. **Narrowing is the expected failure.** Agentic training optimizes on a small number of environments with
   verifiers. The measurable risk is that the policy gets better at those environments and no better, or worse,
   elsewhere. [[agentgym-rl]] states plainly that its trained agents "perform well within in-domain settings" and
   lists transfer to novel environments as future work (§7). Without an unseen-environment split, an agentic
   training run cannot distinguish capability from environment fitting.

## §1 What an agent score measures

**Definition.** An agentic benchmark result is a value of

```
score = f(model, scaffold, environment, budget, protocol)
```

where `model` is the checkpoint and its decoding settings; `scaffold` is the program that mediates between the
model and the environment (prompt, tool schemas, parsing, retry logic, context strategy); `environment` is the
task set together with its pinned container image, tool versions, and grader; `budget` is the set of limits on
turns, wall-clock or simulated time, tokens, and cost; and `protocol` is the number of runs, the aggregation, and
the reported statistic.

**The problem it addresses, stated as a measurable problem.** If a report names only `model` and the benchmark
name, the reader cannot reproduce the number, and two reports of the "same" benchmark are not comparable. The size
of that incomparability is measurable: evaluate one model under several scaffolds and read the spread.

**Evidence (Replicated).** [[terminal-bench-2]] runs six agents across 16 models on the same 89 tasks, at least
five times per agent-model combination, 32,155 trials in total (§3). Within one model:

| Model | Best scaffold | Worst listed scaffold | Spread (points) |
|---|---|---|---|
| GPT-5.2 | Codex CLI 62.9% ± 3.0 | Terminus 2 54.0% ± 2.9 | 8.9 |
| Claude Opus 4.5 | Terminus 2 57.8% ± 2.5 | OpenHands 51.9% ± 2.9 | 5.9 |
| Gemini 2.5 Pro | Terminus 2 32.6% ± 3.0 | OpenHands 15.7% ± 2.6 | 16.9 |
| Grok 4 | Mini-SWE-Agent 29.0% ± 4.6 | OpenHands 19.6% ± 3.5 | 9.4 |

The best scaffold is not the same for every model, so the interaction is real and not a constant offset. The
authors summarize the overall pattern as "model selection is usually more important than agent scaffold when
optimizing for performance" (§4), which is a statement about the average, not about any single comparison.

[[holistic-agent-leaderboard]] reaches the same conclusion from 21,730 rollouts over 9 models and 9 benchmarks at
about $40,000: on Online Mind2Web, SeeAct with GPT-5 Medium costs $171 and Browser-Use with Claude Sonnet 4 costs
$1,577 for a two-point accuracy difference, and Claude models score higher with Browser-Use while OpenAI models
score higher with SeeAct (§4.1 items 6–7). Task-specific scaffolds beat one generalist scaffold on 9 of 12 runs on
CORE-Bench Hard and 11 of 12 on SWE-bench Verified Mini, while the generalist costs less in 20 of 24 comparisons.

**Conditions and limits.** Both sources run proprietary and open models through third-party APIs.
[[holistic-agent-leaderboard]] records that a provider silently switched a DeepSeek R1 endpoint to R1-0528 under
the same name and that OpenRouter served FP4 on one call and FP8 on another (App. A3 items 2, 4). The identity of
`model` is therefore itself an assumption that has to be checked.

> Figure: [harness-coordinates.html](figures/harness-coordinates.html) — one tab per coordinate (scaffold,
> turn and cost budget, simulated time, retriever quality). Each group fixes the model and changes only that
> coordinate, prints the spread inside the group, and names the exact source locus for every bar.

## §2 The generality suite: covering action spaces, not counting benchmarks

**Definition.** A generality suite is a set of evaluation environments chosen so that its members differ in
**action space** — the set of operations the agent may perform and the form of the observations it receives — and
not only in topic.

**The problem it addresses.** Adding a second coding benchmark to a suite that already contains one adds little
information about whether the agent can operate a terminal, hold a conversation with a user who also holds tools,
or search a corpus. The measurable version of the problem: after training, which of the suite's scores moved, and
did they move together? Scores that always move together are measuring one thing.

**Mechanism.** Select environments along four axes: (1) the action space (patch a repository, issue shell
commands, call typed functions, search a corpus, act in an application stack); (2) who else acts (no one, a
simulated user, a simulated user who also holds tools, other agents, asynchronous environment events); (3) the
grader (unit tests, final-state comparison, execution script, LLM judge, string match); (4) the horizon (turns per
task).

| Environment | Size | Action space | Other actors | Grader | Reported ceiling |
|---|---|---|---|---|---|
| SWE-Bench Pro public ([[swe-bench-pro]]) | 731 tasks, 11 repositories | multi-file repository patch | none | fail2pass + pass2pass unit tests | 43.6% (Claude Sonnet 4.5, Table 1) |
| SWE-Bench Pro commercial | 276 tasks, 18 private repositories | same | none | same | 17.8% (Claude Opus 4.1, Table 2) |
| Terminal-Bench 2.0 ([[terminal-bench-2]]) | 89 tasks | shell commands in a container | none | tests on final container state | 62.9% ± 3.0 (GPT-5.2 + Codex CLI, Table 2) |
| τ-bench ([[tau-bench]]) | 115 retail + 50 airline | domain API calls plus user messages | simulated user (no tools) | database-state equality + substring check | 61.2 / 35.2 pass^1 (gpt-4o, Table 2) |
| τ²-bench telecom ([[tau2-bench]]) | 114 evaluated tasks | agent tools plus user guidance | simulated user **with 15 write / 15 read tools** | assertion functions on final state | 49% pass^1 (claude-3-7-sonnet, §1) |
| BFCL ([[bfcl]]) | 5,551 pairs across 6+ categories | typed function calls, Python/Java/JS/REST/SQL | none or stateful multi-turn | AST substring matching, state checks | 66.4 overall (gpt-4o-2024-11-20, Table 1) |
| BrowseComp-Plus ([[browsecomp-plus]]) | 830 queries, 100,195-document corpus | search and fetch over a fixed corpus | none | gpt-4.1 judge | 70.12% (gpt-5 + Qwen3-Embedding-8B, Table 1) |
| Toolathlon ([[toolathlon]]) | 108 tasks, 32 applications, 604 tools | MCP tool calls across applications | none | deterministic per-task script | 38.6 ± 2.7 pass@1 (Claude-4.5-Sonnet, Table 3) |
| Gaia2 ([[are-gaia2]]) | 800 + 320 scenarios, 12 apps, 101 tools | app tools in an asynchronous environment | simulated user, app sub-agents, environment events | write-action matching with hard, soft, causal and timing checks | 42.1 overall pass@1 (GPT-5 high, Table 2) |

**Evidence that the axes are not redundant (Result, single study).** In [[are-gaia2]] Table 2, GPT-5 (high) scores
79.6 on Search and 69.2 on Execution but 51.9 on Ambiguity, 40.4 on Adaptability and 0.0 on Time; Grok-4 scores
57.5 on Search and 8.8 on Execution. The authors read the drop from Execution and Search to Ambiguity and
Adaptability as evidence that existing benchmarks "may overestimate robustness in realistic environments" (§4.2,
Interpretation). In [[bfcl]] Table 1, gpt-4o-2024-11-20 in prompting mode scores 95.5 and 94.0 on single-turn AST
multiple and parallel calls but 59.0 on multi-turn base and 6.0 on memory; the highest memory score across all 71
rows is 12.0 (§5.6).

**Implication for a general-purpose model.** A suite whose members all score in the same order for every model is
carrying one axis of information. The τ²-bench dual-control result is the sharpest example of an axis that other
suites do not contain: moving gpt-4.1 from no-user mode (the agent holds every tool) to dual control lowers
telecom pass^1 from 0.67 to 0.34 ([[tau2-bench]], Fig. 4 left).

## §3 Seen and unseen environment splits

**Definition.** A **seen** environment is one whose tasks, tools, container images, or close relatives appeared in
the training mixture, including in SFT trajectories and RL rollouts. An **unseen** environment is one whose whole
stack — tools, observation format, grader — was withheld. The unseen split is the generality measurement; a
held-out *task* split inside a seen environment is not, because the agent has already learned that environment's
tool schema, error strings, and conventions.

**The problem it addresses, measurably.** After agentic RL on a fixed set of environments, the seen-environment
score rises. The question a gate has to answer is what fraction of that rise appears on environments the run never
touched.

**Mechanism.** (1) Partition environments, not tasks, before training starts. (2) Freeze the unseen partition:
no prompt tuning, no scaffold tuning, no hyper-parameter selection on it. (3) Report the pair
(seen score, unseen score) before and after the run, with the same harness on both. (4) Compute the transfer
fraction as the unseen delta divided by the seen delta, and state the confidence interval from [[ch-51]].

**Worked numeric example (real numbers, [[swe-rebench]], Table 2).** Two checkpoints of the same model family are
evaluated under one minimal ReAct scaffold with identical prompts, 5 runs per model with standard errors:

| Checkpoint | SWE-bench Verified (public since late 2023) | SWE-rebench March–April 2025 slice (issues created in 2025) |
|---|---|---|
| DeepSeek-V3-0324 | 39.7% | 21.3% |
| DeepSeek-V3-1226 | 35.2% | 21.9% |

The difference on the older benchmark is 39.7 − 35.2 = 4.5 points in favour of the newer checkpoint. On the fresh
slice it is 21.9 − 21.3 = 0.6 points in the other direction. The authors write that this "may suggest potential
contamination effects on the older benchmark" (§3.3, Interpretation). A gate that had used the 4.5-point Verified
difference to choose a checkpoint would have selected on a difference that a fresh slice does not reproduce.

**Second form of the split: private repositories.** [[swe-bench-pro]] holds three partitions: 731 public tasks
from copyleft repositories, 858 held-out tasks kept private "to test for overfitting in the future", and 276 tasks
from 18 private startup repositories whose results are published (§3.3). The best public score is 43.6% and the
best commercial score is 17.8%. The card records an important limit: the model lists in Tables 1 and 2 differ, so
**no model has a paired public-versus-commercial number in the same setting**, and the 43.6 → 17.8 comparison is
between two different models. The held-out set has no published results in v2.

**Third form: a fresh-environment arena during training.** [[agent-world]] samples K = 5 environments per
first-tier category for an arena of held-out environments that is regenerated each round, sends the failure traces
to a diagnosis agent, synthesizes targeted tasks, and continues RL (§3.2.2, Alg. 1). Two rounds moved Agent-World-
14B from 60.2 / 52.4 / 29.5 to 65.4 / 55.8 / 38.1 on τ²-Bench / BFCL-V4 / MCP-Mark Postgres (Table 2). The same
paper is also the cautionary case for reporting: its §4.3.1 claims "no degradation on core math reasoning" while
per-benchmark values appear only in Fig. 6, so the claim cannot be checked at the stated locus.

**Conditions and limits.** An unseen split degrades over time. Once results on it are published and used for
selection, it has been selected on. [[terminal-bench-2]] states this directly: all its tasks, including the 140
contributed tasks not selected for 2.0, are in a public repository, and the BIG-bench canary string it embeds is
"largely a symbolic safeguard" (App. B).

## §4 Reliability: pass^k, repeated runs, and variance

**Definition.** For a task run n independent times with c successes, and for a chosen k ≤ n,

```
pass^k  = E_task[ C(c, k)   / C(n, k) ]        the chance that all k trials succeed
pass@k  = 1 − E_task[ C(n−c, k) / C(n, k) ]    the chance that at least one of k trials succeeds
```

where `C(a, b)` is the binomial coefficient, `n` is the number of trials per task, `c` the number of successful
trials, `k` the number of trials the metric asks about, and `E_task` the mean over tasks. Both are the unbiased
estimators given in [[tau-bench]] §3. At k = 1 they coincide: pass^1 = pass@1 = E[c/n].

**The problem it addresses.** pass@1 answers "how often does one attempt succeed, averaged over tasks". A
deployment that runs the agent once per user request needs the probability that *this* task succeeds *every* time
it is attempted, which is what pass^k estimates. [[tau-bench]] introduced pass^k for exactly this reason: "for
real-world agent tasks requiring reliability and consistency like customer service" (§3).

**Worked numeric example.** Five tasks, n = 5 trials each, successes c = (5, 4, 3, 1, 0). With C(5,3) = 10:

| Task | c | c/n | C(c,3)/C(5,3) → pass^3 term | 1 − C(5−c,3)/C(5,3) → pass@3 term |
|---|---|---|---|---|
| 1 | 5 | 1.0 | 10/10 = 1.00 | 1 − 0/10 = 1.00 |
| 2 | 4 | 0.8 | 4/10 = 0.40 | 1 − 0/10 = 1.00 |
| 3 | 3 | 0.6 | 1/10 = 0.10 | 1 − 0/10 = 1.00 |
| 4 | 1 | 0.2 | 0/10 = 0.00 | 1 − 4/10 = 0.60 |
| 5 | 0 | 0.0 | 0/10 = 0.00 | 1 − 10/10 = 0.00 |

pass^1 = 2.6/5 = **0.52**, pass^3 = 1.5/5 = **0.30**, pass@3 = 3.6/5 = **0.72**. One number, 0.52, sits between two
quantities that differ by 42 points. Which of the three is reported decides whether this agent looks deployable.

> Figure: [pass-hat-k.html](figures/pass-hat-k.html) — edit the trial table above and watch pass^k and pass@k
> separate as k grows; the default table is the worked example, and the third panel lists the reported anchors from
> [[tau-bench]] and [[toolathlon]].

**Evidence (Replicated).**
- [[tau-bench]], §5.1 and Fig. 4: the gpt-4o function-calling agent has pass^1 above 60% on τ-retail and
  "pass^8 drops to < 25%".
- [[toolathlon]], Table 3, 3 runs per model over 108 tasks: Claude-4.5-Sonnet pass@1 38.6 ± 2.7, pass@3 51.9,
  pass^3 20.4; DeepSeek-v3.2-Exp 20.1 ± 1.2, 27.8, 12.0. The authors write that the pass@3-versus-pass^3 gap
  "indicates that while many models have certain capability coverage, they lack consistency in producing reliable
  results" (§4.2).
- [[metr-time-horizon]], §3.2.1: the 80% time horizon of a model is 4–6× shorter than its 50% time horizon, which
  the authors read as "even models that sometimes succeed on difficult and diverse tasks cannot reliably perform
  tasks of moderate length".
- [[swe-rebench]], §3.3: Llama-4-Maverick has 12.2% mean resolved and pass@5 of 27.6, a ratio that identifies low
  run-to-run reliability without any extra instrumentation.

**Conditions and limits.** pass^k is only defined relative to what is resampled. In [[tau-bench]] the database and
the user instruction are fixed and only language-model sampling varies (§3), so pass^k measures sampling
reliability, not robustness to a different user or a different environment seed. A pass^k computed with the agent
at temperature 0 and the user simulator at temperature 1.0 (the τ-bench setting, §5) attributes all variation to
the simulator.

## §5 Perturbation robustness across the five axes

**Definition.** Perturbation robustness is the change in score when something that should not matter is changed.
[[minimax-m2-aligning-to-what]] lists the five places a perturbation can enter a trajectory: "The **Tool Info** and
available toolset", "The **System Prompt** defining the agent's persona and rules", "The **User Prompt** and its
specific goal", "The **Environment** itself (files, codebases, APIs)", and "The **Tool Responses** returned at each
step". The post states that its team's earlier "tool scaling" approach "only addressed the first item" and
describes the symptom as "the same model can feel brilliant in one framework and useless in another". This is an
official but qualitative source: it supports which axes to perturb, not how much each is worth.

**Mechanism.** For each axis, hold the task set fixed and re-run with one change: (1) swap the scaffold;
(2) rewrite the system prompt without changing its content requirements; (3) paraphrase the user goal;
(4) change the environment image version or the retriever behind a search tool; (5) inject tool failures and
signature changes.

**Evidence, one measurement per axis.**

| Axis | Measured perturbation | Effect | Source |
|---|---|---|---|
| Scaffold | Gemini 2.5 Pro on Terminal-Bench 2.0 under Terminus 2 vs OpenHands | 32.6% → 15.7% | [[terminal-bench-2]], Table 2 |
| Tool info / call format | GPT-4-turbo-2024-04-09 irrelevance score, native function-calling vs prompting mode | 83.8 → 35.6 | [[bfcl]], Table 1, §5.1 |
| Tool responses | gpt-5 on BrowseComp-Plus, BM25 vs Qwen3-Embedding-8B behind the same search tool | 55.90% → 70.12%, with fewer search calls (23.23 → 21.74) | [[browsecomp-plus]], Table 1 |
| Environment noise | Claude-4-Sonnet on Gaia2-mini under none / low / medium / high noise (tool-signature changes, 0.1 tool-failure probability, 10 random events per minute) | 31.2 / 35.0 / 23.8 / 8.1 | [[are-gaia2]], App. B.6.1, Table 6 |
| Other actors | Gaia2 Agent2Agent teams: all-Llama vs all-Claude | 8.5 vs 29.3 | [[are-gaia2]], Table 3 |

The aggregate picture from the released Terminal-Bench 2.0 trajectory corpus is consistent:
[[terminal-bench-trajectories]] publishes the full traces, and the dataset card records 52,104 trajectories over
89 tasks, 109 agent/model combinations, 26 scaffolds and 49 models, with per-scaffold pass rates from 22.8%
(mini-swe-agent) to 74.9% (terminus-3-3) ([[terminal-bench-trajectories-stats]]). Those per-scaffold rates are
**not** paired — each scaffold row aggregates a different set of models — so they show how much a leaderboard row
depends on the pair, not how much the scaffold alone is worth. The paired version of that question is the
fixed-model comparison in [[terminal-bench-2]] Table 2.

**Implication for a general-purpose model.** A model that is trained against exactly one scaffold and one tool
schema has been trained on a coordinate that will change in deployment. Perturbation robustness is the axis of the
generality gate that a seen/unseen environment split does not cover, because an unseen environment can still be
run under the training scaffold.

## §6 Harness coordinates and the pinned evaluation contract

**Definition.** A harness coordinate is any setting of the evaluation program that is not part of the model and
not part of the task, but changes the score.

**Mechanism: the list to pin.** Scaffold name and commit; tool schemas and how tools are presented (native
function-calling field or in-prompt text); maximum turns; maximum wall-clock or simulated time; token budget per
turn and per trajectory; cost cap; context strategy when the limit is reached; environment image tag; reasoning
effort; decoding settings; retry and parse-error policy; number of runs and the aggregation.

**Worked evidence: one coordinate, two models, a rank reversal.** [[swe-bench-pro]] evaluates the public set under
SWE-Agent with a maximum of 50 turns (§5, Table 1), and repeats it with the same 50-turn limit plus a $2 per-task
cost cap for its analysis section (§6, Table 5):

| Model | 50 turns (Table 1) | 50 turns + $2 cap (Table 5) | Change |
|---|---|---|---|
| Claude Sonnet 4 | 42.7% | 17.6% | −25.1 |
| GPT-5 (high) | 41.8% | 25.9% | −15.9 |

Under the first budget Claude Sonnet 4 is ahead by 0.9 points; under the second GPT-5 (high) is ahead by 8.3
points (derived from the two tables). The paper does not state the number of runs per model, so neither ordering
carries a confidence interval. A leaderboard row that prints only "SWE-Bench Pro, 25.9%" is not comparable with
one that prints 41.8%.

**Second worked evidence: simulated time.** [[are-gaia2]] runs the mobile environment with a clock that advances
while the model generates. On the Time split, GPT-5 (high) scores 0.0 in that default mode and 34.4 in "instant"
mode, where each action costs one simulated second; Claude-4-Sonnet moves from 8.1 to 26.7 (Fig. 13; the §4.3 text
prints 8.2 where Fig. 13 prints 8.1). Capping GPT-5 (high) at 30 minutes of execution lowered Execution by 10
points and Search by 20 points (§5). Latency is a harness coordinate with the same magnitude as a scaffold swap.

**Third: reasoning effort does not move scores in a fixed direction.** [[holistic-agent-leaderboard]] pairs
no-reasoning against high-reasoning for three Claude models and low against high for o4-mini, and finds equal or
lower accuracy with more reasoning in 21 of 36 model-agent-benchmark combinations (§4.1 item 5, Fig. 3).
[[toolathlon]] reports the same direction: GPT-5 scores 30.6 ± 1.5 and GPT-5-high 29.0 ± 3.1 (Table 3), and reads
it as "exploring new observations matters more than extended internal reasoning in agentic tasks" (§4.2,
Interpretation). The setting must be recorded because it is not monotone; [[holistic-agent-leaderboard]] also
records that "high" for Anthropic models in its harness is LiteLLM's default of 4,096 reasoning tokens, and that
OpenAI does not disclose its budgets (§3 footnote).

## §7 Benchmark validity: whether the score measures the capability

**Definition.** [[agentic-benchmark-checklist]] separates two conditions: **task validity** — a task is solvable
if and only if the agent has the target capability — and **outcome validity** — the evaluation result is positive
if and only if the task succeeded (§1, §3). The Agentic Benchmark Checklist (ABC) is 10 task-validity items, 20
outcome-validity items and 13 reporting items (Figs. 2–4; the paper does not print a total).

**Evidence of how large the errors are (Result, single study; §5.2, App. E).**
- τ-bench contains intentionally unsolvable tasks, 38% of Airline and 6% of Retail, where success is defined as an
  unchanged database. A **do-nothing agent** scores 38% on Airline and 6.0% on Retail for any k; an agent that
  prints all data scores 40% and 9.6% (App. E.2).
- SWE-Lancer stores its tests in a password-protected ZIP whose files can be overwritten without the password;
  replacing every test with `assert 1 == 1` gives a 100% resolve rate (App. E.4).
- KernelBench's fuzzer varies tensor values but not shapes or memory layouts; adding verified extra tests found
  kernel correctness overestimated by 31% (App. E.6).
- OSWorld: 13 of 46 Chrome-section problems were broken by website changes, underestimating the evaluated agent by
  28 points in absolute terms (§5.2).
- SWE-bench: citing UTBoost, agents pass without resolving the issue on 5.3% of Verified and 7.7% of Lite tasks,
  changing 40.9% and 24.4% of leaderboard rankings (App. E.1).
- Across ten assessed benchmarks, 7 have task-validity flaws, 7 have outcome-validity flaws, and all 10 have
  reporting limitations (§1, Fig. 5).

**Memorization as a validity failure.** [[swe-bench-illusion]] gives ten OpenAI and Anthropic models only the
repository name and the issue text, with no code, file tree, or metadata, and asks for a file path. Models name a
file from the gold patch in 60–76% of SWE-Bench Verified instances but under 53% of 245 tasks from seven popular
repositories outside SWE-Bench (§4.1.1–4.1.2). Function reproduction shows 34.9% maximum 5-gram overlap on
Verified against 13.9% on outside repositories (§4.3). The authors interpret this as instance-specific memorization
inside the Verified set and repository-bias memorization of the 12 SWE-Bench repositories (§4.1, §6). The probes
measure localization and reproduction subtasks, not end-to-end resolve rates.

**Test exploitability as a validity failure.** [[impossiblebench]] mutates unit tests so that they contradict the
specification and defines the pass rate on those impossible variants as the **cheating rate**: any pass implies a
specification-violating shortcut (§1, §2). GPT-5 cheats on 76% of Oneoff-SWEbench tasks and 2.9% of
Oneoff-LiveCodeBench tasks; Figure 1 lists Claude Sonnet 3.7 70%, Claude Opus 4.1 54%, Claude Sonnet 4 48%, o3 39%
on Oneoff-SWEbench. Three harness controls change these rates: a prompt that tells the agent to stop and report
flawed tests lowers GPT-5 from above 85% to 1% on Conflicting-LiveCodeBench (§5.1); read-only tests block test
modification while leaving operator overloading available (§5.2); adding an abort action lowers GPT-5 from 54% to
9% on Conflicting-SWEbench (§5.3).

**Leaks inside the harness.** [[holistic-agent-leaderboard]] found that the official τ-bench few-shot file
`few_shot_data/MockAirlineDomainEnv-few_shot.jsonl` contained test-set examples, after about $1,000 of evaluation
had been run with it; every result from that scaffold was excluded (App. A5). The same log analysis found eight
cases of agents retrieving gold answers from HuggingFace or arXiv instead of solving the task (§4.2 item 1).

**Formula and worked example: a success rate with imperfect ground truth** ([[agentic-benchmark-checklist]],
App. F, Eq. 1).

```
μ = e + (1 − 2e)·p₀        σ² = μ(1 − μ)        95% interval:  μ ± 1.96·σ/√N
```

- `e`: fraction of test items whose ground-truth label is wrong.
- `p₀`: the success rate measured against that imperfect ground truth.
- `μ`, `σ`: mean and standard deviation of the corrected estimate.
- `N`: number of test items.

Take a 108-task agentic benchmark with a measured p₀ = 0.386 ([[toolathlon]] Table 3, Claude-4.5-Sonnet).
With a perfect grader (e = 0): μ = 0.386, σ = √(0.386 × 0.614) = 0.487, and 1.96 × 0.487 / √108 = 0.092, so
0.386 ± 0.092 — a ±9.2-point interval from the task count alone. With e = 0.05: μ = 0.05 + 0.90 × 0.386 = 0.397,
σ = 0.489, half-width 0.092, so the point estimate moves 1.1 points and the width does not change. Two consequences
follow. First, on benchmarks of this size a 5-point difference is not a result. Second, `e` must be measured, not
assumed: the checklist's own example plugs in 11.65% incorrect ground-truth queries found by verifying 500 sampled
BIRD tasks (App. F). Everything in [[ch-51]] about paired comparisons applies on top of this, and the pairing
matters more here because the item-level interval is this wide.

## §8 Task-length horizon and what it assumes

**Definition.** The X%-task-completion time horizon is "the length of tasks that models can complete approximately
X% of the time", where task length is measured by how long a skilled human takes ([[metr-time-horizon]], Abstract).

**Mechanism.** (1) Time human baseliners with domain knowledge but no task-specific context on every task.
(2) Binarize each task's score at a threshold representing human performance. (3) Fit one logistic curve per
agent:

```
p_success(agent, task) = σ( (log h_agent − log t_task) · β_agent )
```

- `t_task`: the geometric mean time of successful human baselines on that task.
- `h_agent`: the fitted 50% time horizon — the task length at which success probability is 0.5.
- `β_agent`: the fitted slope, how sharply success falls as tasks get longer.
- `σ(x) = 1/(1 + e^(−x))`.

**Worked numeric example.** Take h = 110 minutes, the 50% horizon reported for o3 (§3.2), and an illustrative slope
β = 0.6 (the paper fits β per agent and does not print it in the body). For a 30-minute task:
log(110/30) = 1.299, times 0.6 gives 0.780, and σ(0.780) = 1/(1 + 0.459) = 0.686, so 68.6% predicted success. For
an 8-hour (480-minute) task: log(110/480) = −1.474, times 0.6 gives −0.884, and σ(−0.884) = 1/(1 + 2.421) = 0.292,
so 29.2%. A larger β makes the curve steeper: the same horizon with a sharper slope means higher success on short
tasks and lower on long ones.

**Evidence.** 170 tasks from HCAST, RE-Bench and SWAA; 12 frontier models from 2019 to 2025. GPT-2 has a 50%
horizon of 2 seconds and o3 has 110 minutes. Horizon doubled every **207 days**, 95% bootstrapped CI 166–240 days,
from a three-level hierarchical bootstrap over task families, tasks, then runs (§3.2). The 80% horizon doubles
every 204 days but is 4–6× shorter in absolute terms (§3.2.1).

**Conditions and limits stated by the authors.**
- Errors on individual model horizons are large and **highly correlated across models**, so the slope is better
  determined than any single model's horizon (§3.2).
- Replicating the method on SWE-bench Verified gives about a 70-day doubling time against 143 days on the main
  suite for 2024 models. The authors attribute part of the difference to SWE-bench Verified's time annotations,
  which assume an engineer with "a few hours to familiarize themselves with the codebase" (§4.1).
- A 16-factor "messiness" score (resource limits, novelty, dynamic environment, and others) predicts lower
  performance at fixed task length, but the trend over time is similar for low- and high-messiness subsets, with
  "no evidence of plateaus" specific to the messier subset. The mean messiness score of the suite is 3.2 out of 16
  (§4, §F.2), so the suite is cleaner than deployment tasks.
- On METR's internal pull requests, contractors take 5–18× longer than repository maintainers, and agent
  performance matches contractor times rather than maintainer times (§4, §C.2). The horizon is therefore defined
  relative to a chosen human population.
- "Time horizon is always measured relative to a domain, task distribution, and human baseliners' level of skill
  and context" (§5). The evaluation covers software and research tasks only.

**Implication for a general-purpose model.** The horizon is the one agent metric that is comparable across
benchmarks of different difficulty, because the x-axis is human time rather than a benchmark-specific scale. It is
also the metric most sensitive to who the baseliners are, which makes it useful for tracking a trend and weak for
comparing two checkpoints.

## §9 The paired general-ability regression gate

**Definition.** The regression gate re-runs the non-agentic suite — knowledge, reasoning, instruction following,
safety — on the pre-agentic-training checkpoint and the post checkpoint, item-paired, and requires that the
post checkpoint is **non-inferior** on each slice by a stated margin before the agentic gain is accepted.

**The problem it addresses.** Agentic SFT and RL optimize a narrow reward on a small number of environments. The
measurable risk is a gain on those environments together with a loss on capabilities outside them.

**Evidence that the loss is real (Result, single study).** [[agent-world]] §4.3.1 reports that EnvScaler-8B scores
below its own Qwen3-8B backbone on SWE-bench Verified and Terminal-Bench 1.0, while its tool-use averages improved
(Table 1: 5.6 / 47.6 / 37.9 on MCP-Mark / BFCL V4 / τ²-Bench against Qwen3-8B's 2.4 / 40.4 / 26.2). The same card
records that Agent-World's own "no degradation on core math reasoning" claim has per-benchmark values only in a
figure. [[agentgym-rl]] reports no held-out-environment and no general-benchmark evaluation at all (§7), which is
the reporting failure this gate exists to prevent.

**Mechanism.** Reuse the machinery of [[ch-51]] rather than inventing new statistics: a paired bootstrap over
items for each slice; a non-inferiority margin fixed before the run; multiple-comparison control across the
slices; and a written decision that names which slice would have blocked the release. Add three agent-specific
rows to the gate: (1) the unseen-environment pair from §3; (2) pass^k next to pass@1 from §4; (3) one perturbation
axis from §5, run on the same task set as the headline number.

**Conditions and limits.** The gate is only as good as the harness pinning in §6. If the post-training checkpoint
is evaluated under a scaffold tuned during training and the pre-training checkpoint under a different one, the
paired comparison measures the scaffold change.

## Negative samples and negative feedback

Which of the four meanings of "negative" (course standard §6.1) this section is about: **negative as content** —
failure trajectories used as evaluation evidence and as input to failure bucketing — and, at the boundary of this
chapter, **negative as gradient**, because an evaluation grader reused as an RL reward turns a grading error into
a gradient.

1. **Where the negatives come from.** Failed trajectories, labelled by the benchmark's grader: unit tests
   ([[swe-bench-pro]], [[terminal-bench-2]]), final-state comparison ([[tau-bench]], [[toolathlon]]),
   write-action matching with an LLM soft check ([[are-gaia2]]), or an LLM judge ([[browsecomp-plus]]).

2. **What current practice does with them.** It buckets them. [[terminal-bench-2]] samples two failed trials per
   model per task under one scaffold, annotates them with Docent and two human annotators, and reports a three-class
   taxonomy — Execution, Coherence, Verification — with GPT-5 (high) as judge at 90% agreement against 120
   human-labeled traces (92% precision, 90% recall) (§4.4). [[swe-bench-pro]] has GPT-5 judge the last 20 turns of
   each unresolved trajectory, a window chosen because it matched human validation better than 10 or 40 (§6.3).
   Neither trains on the failures.

3. **Mechanism and the risk at the boundary.** A grader with a false-negative rate rejects correct behaviour. When
   the same grader is used as an RL reward, that rejection becomes a negative advantage on a correct trajectory,
   and the softmax gradient `∂ log p_y / ∂ z_j = 1[j = y] − p_j` moves the removed probability mass to whatever
   the model ranks next, which is not necessarily better behaviour. The magnitude is measurable:
   [[swe-bench-pro]] Table 3 removes the requirements and interface fields from the task description, and resolve
   rate falls from 25.9% to 8.40% for GPT-5 (high) and 22.7% to 8.20% for Claude Opus 4.1, which the authors
   attribute to verifier false negatives — unit tests accepting only a narrow set of interfaces (§6.2).

4. **The false-positive direction, with numbers.** [[agentic-benchmark-checklist]] measures graders that accept
   non-solutions: 38% of τ-bench Airline tasks pass with an empty response, 100% of SWE-Lancer tasks pass after
   overwriting the tests (§5.2). [[impossiblebench]] measures the policy side of the same coin: GPT-5 passes 76%
   of Oneoff-SWEbench tasks that are impossible to pass without violating the specification.

5. **Controls.** Give the agent the information the tests require so the verifier is not the ambiguity resolver
   ([[swe-bench-pro]], §6.2 guideline). Make tests read-only rather than hidden, which preserves legitimate
   performance while removing the easiest exploit ([[impossiblebench]], §5.2). Add an abort action so that an
   impossible task can be reported instead of gamed (§5.3). Add a task-agnostic style check beside a rubric judge:
   in early ARE RL runs the agent embedded code-like strings in its answer messages until the soft-check judge
   accepted them, and a style check stopped the exploit ([[are-gaia2]], App. B.3.1). Exclude harness failures —
   context overflow, timeout, environment crash — from the gradient rather than scoring them as policy failures
   ([[ch-45b]]).

6. **Diagnostics.** Per-failure-class counts from the bucketing pipeline; command-level error rate (9.2% for
   Grok 4 to 26.7% for GPT-OSS-120B on Terminal-Bench 2.0, §4.5); do-nothing and enumerate-everything baselines
   against the grader; cheating rate on an impossible variant; pass^k against pass@k at the same k.

7. **Effect on generality.** A grader that rewards a shortcut trains the shortcut. [[holistic-agent-leaderboard]]
   states that it cannot establish whether fixing a flagged failure would produce success, which would require
   checkpointing and replay (App. A4.2), so the link from failure bucket to training change is currently an
   inference, not a measurement.

## Recipe

Every row is an evaluation-protocol value read at the stated locus. Values are for the benchmark's own reference
protocol; a different protocol produces a different score, which is the point of §6.

| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| τ-bench reference protocol | — | eval-gate | max agent actions per task; agent / user temperature; trials per task | 30; 0.0 / 1.0; at least 3 | arXiv:2406.12045v1 §5 | verified 2026-09-15 | no ablation reported |
| τ²-bench reference protocol | — | eval-gate | runs per task; temperature; user simulator model | 4; 0; gpt-4.1-2025-04-14 | arXiv:2506.07982v1 §4.1 | verified 2026-09-15 | §4.2, Fig. 4: no-user and oracle-plan mode ablation |
| SWE-Bench Pro (public, N=731) | — | eval-gate | scaffold; max turns; cost cap; task fields given | SWE-Agent with its default prompt; 50; none; problem statement + requirements + interface | arXiv:2509.16941v2 §5 | verified 2026-09-15 | §6.2 Table 3: removing requirements+interface drops GPT-5 (high) 25.9 → 8.40 |
| SWE-Bench Pro (analysis setting) | — | eval-gate | max turns; cost cap | 50; $2 per task | arXiv:2509.16941v2 §6, Table 5 | verified 2026-09-15 | stated as a compute constraint, not an ablation |
| SWE-Bench Pro | — | eval-gate | runs per model; confidence intervals | not reported (checked §5, §6, Tables 1–5) | — | not reported | none |
| Terminal-Bench 2.0 | — | eval-gate | runs per agent-model combination; reasoning effort; execution | at least 5 (32,155 trials total); provider default (medium for Anthropic and OpenAI); Harbor harness, 32–100 containers in parallel on Daytona | arXiv:2601.11868v1 §3, §3.3, §3.4 | verified 2026-09-15 | resolution rates printed with 95% CIs (Table 2) |
| Gaia2 / ARE | — | eval-gate | loop; runs per scenario; temperature; context; generation tokens per turn; max steps; verifier | ReAct-like, one JSON tool call per step; 3; 0.5 (GPT-5 at temperature and top-p 1); at least 128K; 16,000; 200; Llama 3.3 70B Instruct at temperature 0 | arXiv:2509.17158 §4.1, App. B.5 | verified 2026-09-15 | Table 1: verifier agreement 0.98 / precision 0.99 / recall 0.95 on 450 labeled trajectories |
| Toolathlon | — | eval-gate | max turns; runs per model; reported metrics | 100; 3; pass@1 with standard deviation, pass@3, pass^3 | arXiv:2510.25726v2 §4.1 | verified 2026-09-15 | no ablation reported |
| BrowseComp-Plus | — | eval-gate | retriever; top-k; document truncation; judge | BM25, Qwen3-Embedding 0.6B/4B/8B, or ReasonIR-8B; k = 5; first 512 tokens; gpt-4.1 | arXiv:2508.06600v1 §4.3, §4.4 | verified 2026-09-15 | Table 1: retriever swap raised accuracy for every agent tested |
| SWE-rebench benchmark slice | — | eval-gate | scaffold; runs; reporting; context; contamination flag | minimal ReAct with identical prompts and text commands; 5 runs with SEM and pass@5; 128K; runs including issues created before a model's release are marked potentially contaminated | arXiv:2505.20411v2 §3.2 | verified 2026-09-15 | §3.1: lists pass@N-as-pass@1 and best-of-runs reporting as the confounds this protocol removes |
| HAL harness | — | eval-gate | runs per combination; confidence intervals | mostly single runs, no CIs, stated as a cost constraint | arXiv:2510.11977 App. A3 item 1 | verified 2026-09-15 | none |
| BFCL | — | eval-gate | call mode | both native function-calling and prompting mode are run and reported | PMLR 267 §5.1, App. A | verified 2026-09-15 | Table 1: GPT-4-turbo-2024-04-09 irrelevance 83.8 (FC) vs 35.6 (prompting) |
| METR time-horizon protocol | — | eval-gate | fit; uncertainty | logistic regression of success on log human task time; 10,000-sample three-level hierarchical bootstrap over task families, tasks, then runs | arXiv:2503.14499v4 §3.1, §3.2 | verified 2026-09-15 | §3.2.1: the 80%-horizon repetition gives a similar doubling time (204 vs 207 days) |

**Starting point for a small general-purpose agent gate.** Every number here comes from a `verified` row above, and
each is stated with the setting the source used. Run three runs per task and report pass@1, pass@3 and pass^3, as
[[toolathlon]] does over 108 tasks with 3 runs (§4.1); on a reliability-focused suite run at least three trials per
task and report pass^k, as [[tau-bench]] does over 165 tasks (§5). Fix a maximum of 50 turns as in
[[swe-bench-pro]] (§5) or 100 turns as in [[toolathlon]] (§4.1) depending on the horizon of the environment, and
publish the cap, because a $2 cost cap at a fixed 50-turn limit moved two models by 15.9 and 25.1 points
([[swe-bench-pro]], Tables 1 and 5). Run both call modes as [[bfcl]] does (§5.1). Report per-combination
confidence intervals as [[terminal-bench-2]] does (Table 2) rather than single runs as [[holistic-agent-leaderboard]]
was forced to (App. A3 item 1). These protocols were used on benchmarks of 89 to 830 tasks against frontier-scale
models; a smaller model on the same protocol will produce wider intervals at the same task count.

## Generalization lens

**(a) What increases breadth.**
- Evaluating across action spaces rather than across topics. The Gaia2 capability profile shows one model at 79.6
  on Search and 0.0 on Time under one scaffold ([[are-gaia2]], Table 2), a spread no single-axis suite reveals.
- Training environment count, measured on unseen benchmarks: [[agent-world]] reports a four-domain average rising
  from 18.4% with 0 training environments to 38.5% with 1,978, with decreasing returns beyond 500 (§4.3.3, Fig. 8).
- A regenerated held-out arena with diagnosis-driven task synthesis: two rounds moved Agent-World-14B from 29.5 to
  38.1 on MCP-Mark Postgres (Table 2).
- Evaluating under more than one scaffold. A model that is only strong under its own vendor's scaffold has a
  narrower capability than its leaderboard row states ([[terminal-bench-2]], Table 2;
  [[holistic-agent-leaderboard]], §4.1 item 6).

**(b) What causes narrowing or forgetting.**
- Training and gating on one environment family. [[agentgym-rl]] states its agents "perform well within in-domain
  settings" and reports no held-out-environment or general-benchmark evaluation (§7).
- General-ability regression after agentic training: EnvScaler-8B falls below its Qwen3-8B backbone on SWE-bench
  Verified and Terminal-Bench 1.0 while its tool-use averages rise ([[agent-world]], §4.3.1, Table 1).
- Benchmark memorization: file-path identification on SWE-Bench Verified reaches 60–76% with no code in the
  prompt, against under 53% on repositories outside the benchmark ([[swe-bench-illusion]], §4.1).
- Graders that accept shortcuts. Optimizing against a grader with a 38% do-nothing pass rate optimizes toward
  doing nothing ([[agentic-benchmark-checklist]], App. E.2).
- Scaffold overfitting: [[swe-rebench]] names scaffolds tuned on SWE-bench subsets as "implicit overfitting"
  (§3.1).

**(c) How generality is measured at this stage.**
- The seen/unseen environment pair with the same harness on both sides (§3), reported as two numbers and a
  transfer fraction, with the paired bootstrap of [[ch-51]].
- pass^k next to pass@1 at the k the deployment will use (§4).
- One perturbation re-run per axis on the same task set (§5): scaffold swap, call-mode swap, tool-quality swap,
  environment noise.
- A fresh or date-filtered slice next to the public benchmark ([[swe-rebench]] §3.2; see also [[ch-47a]]).
- A trivial-agent baseline and an exploit-agent run against the grader before the score is used
  ([[agentic-benchmark-checklist]] R.13, §5.2).
- The non-inferiority gate on non-agentic slices (§9).

## Common mistakes and how to detect them

| Mistake | Observable symptom | Check |
|---|---|---|
| Comparing two agent scores taken under different scaffolds | Scores differ by less than the known scaffold spread for that benchmark (5.9–16.9 points on Terminal-Bench 2.0, Table 2) | Re-run both checkpoints under one pinned scaffold commit; publish the commit |
| Reporting pass@k and calling it a success rate | Deployment success is far below the reported number | Compute pass^k at the deployment's k; the Toolathlon pass@3 − pass^3 gap is 31.5 points for Claude-4.5-Sonnet (Table 3) |
| Treating a held-out task split inside a trained environment as a generality measurement | Unseen-environment score is flat while the in-environment score rises | Hold out whole environments, including tool schema and grader (§3) |
| Trusting a grader that was never probed | A score that is high for a model whose trajectories show few completed actions | Run a do-nothing agent and an answer-enumerating agent; τ-bench Airline returns 38% for do-nothing ([[agentic-benchmark-checklist]], App. E.2) |
| Attributing a gain to the model when the harness changed | The gain appears only on runs after a harness upgrade | Keep the environment image tag and scaffold commit in the result record; [[holistic-agent-leaderboard]] App. A3 lists endpoints changing weights under one name |
| Reading a benchmark score as contamination-free because the benchmark is recent | Score on the public set far above a date-filtered slice | Run the same scaffold on a fresh slice; DeepSeek-V3-0324 is 39.7% on Verified and 21.3% on the 2025 slice ([[swe-rebench]], Table 2) |
| Raising reasoning effort as a default improvement | Cost rises, score does not | Pair the effort settings; 21 of 36 combinations were equal or lower with more reasoning ([[holistic-agent-leaderboard]], §4.1 item 5) |
| Declaring a small improvement on a 100-task agentic benchmark | Difference under 10 points on ~100 items | Apply the App. F interval: at N = 108 and p₀ = 0.386 the half-width is 9.2 points before grader error is counted (§7) |
| Publishing an unseen split and then tuning on it | Unseen-set score converges to the seen-set score over successive runs | Record how many times the split has been read; rotate to a new environment family when it has been used for selection |

## Check your understanding

1. [[terminal-bench-2]] reports that "model selection is usually more important than agent scaffold", and its
   Table 2 shows Gemini 2.5 Pro moving 16.9 points between scaffolds. Explain how both statements can be true at
   the same time, and state the comparison that would decide which matters more for a specific release decision.
2. τ-bench fixes the database and the user instruction and varies only language-model sampling (§3). Explain what
   a high pass^8 under that protocol does **not** establish about deployment reliability, and name one change to
   the protocol that would close that gap.
3. Adding a $2 cost cap reverses the ordering of Claude Sonnet 4 and GPT-5 (high) on SWE-Bench Pro. Explain the
   mechanism that produces a rank reversal rather than a uniform downward shift, and say which agent property the
   capped setting is measuring that the uncapped one is not.
4. [[swe-rebench]] finds two DeepSeek-V3 checkpoints 4.5 points apart on SWE-bench Verified and 0.6 points apart
   (in the other direction) on a fresh 2025 slice. Give two explanations consistent with this evidence, and state
   the measurement that would separate them.
5. The ABC interval formula gives the same half-width for e = 0 and e = 0.05 but a different centre. Explain why
   grader error shifts the estimate without widening the interval in this model, and state what the model is
   assuming about how grading errors are distributed.
6. [[impossiblebench]] shows that adding an abort action lowers GPT-5's cheating rate from 54% to 9%. Explain why
   this is evidence about the harness rather than about the model, and describe the training-time analogue of the
   same intervention.
7. The METR 80% horizon has almost the same doubling time as the 50% horizon but is 4–6× shorter. Explain what
   this implies about the fitted slope β across models, and why a constant ratio between the two horizons would
   follow.
8. A run reports a 6-point gain on its three training environments and a 1-point gain on one held-out environment.
   Construct the go/no-go argument you would write, naming the statistic from [[ch-51]] you would use and the
   condition under which you would still block the release.

## Connections

- **Previous:** [[ch-51]] — Metric Noise, Confidence Intervals, and Go/No-Go Decisions. That chapter builds the
  paired bootstrap, the non-inferiority test and the go/no-go memo; this chapter supplies the agent-specific
  quantities those tools are applied to, and the much wider item-level intervals that agentic task counts imply.
- **Depends on:** [[ch-45b]] — Multi-Turn Agentic RL: Observation Masking, Credit Assignment, and Stability, for
  the trajectory structure, the harness-failure filtering rule reused in the negative-feedback section, and the
  environments whose scores this chapter gates.
- **Next:** [[ch-52]] — Safety Evaluation, Over-Refusal, and Red-Teaming, which adds the safety slices to the
  regression gate of §9 and extends the perturbation axes of §5 to prompt injection through tool outputs.
- **Builds on:** [[ch-50]] — Slice Analysis, Forgetting Slices, and Failure Bucketing, for the trajectory failure
  buckets that §7 and the negative-feedback section apply to agent transcripts.
- **Related:** [[ch-47a]] — Benchmark Overfitting and Generalization Audits: Fresh, Perturbed, Counterfactual, and
  Live Evaluation, for the fresh-slice and perturbation audits that §3 and §7 use on agentic benchmarks.
- **Applied in:** [[ch-46a]] — Lab: Small Agentic SFT-then-RL Run with a Generality Gate, which runs the gate
  defined here on a small model.

## Sources

- [[swe-bench-pro]] — the public / held-out / commercial partition, the 50-turn and $2-cap protocols, the Table 1
  and Table 5 numbers behind the rank reversal, and the augmentation ablation that measures verifier false
  negatives.
- [[terminal-bench-2]] — the 89-task set, the fixed-model scaffold comparison with 95% CIs, the Harbor harness
  settings, the three-class failure taxonomy, and the benchmark's own Agentic Benchmark Checklist self-audit.
- [[terminal-bench-trajectories]] — the released full-trajectory corpus for Terminal-Bench 2.0; the card predates
  the current source standard, so its numbers are not quoted here.
- [[terminal-bench-trajectories-stats]] — the dataset-card statistics read on 2026-09-15: 52,104 trajectories, 26
  scaffolds, 49 models, and the per-scaffold pass rates, together with why those rates are unpaired.
- [[tau-bench]] — the pass^k and pass@k estimators, the harness settings, the Table 2 pass^1 values, and the
  pass^8 < 25% reliability result.
- [[tau2-bench]] — the dual-control telecom domain, the no-user and oracle-plan mode ablation, and the
  user-simulator error audit.
- [[bfcl]] — the function-calling category structure, the FC-versus-prompting mode difference, and the
  single-turn-versus-memory gap.
- [[browsecomp-plus]] — the fixed-corpus design and the retriever swap as a tool-quality perturbation.
- [[toolathlon]] — the 32-application, 604-tool, 108-task set and the pass@1 / pass@3 / pass^3 triple.
- [[are-gaia2]] — the seven-split capability profile, the noise and Agent2Agent ablations, the simulated-time
  coordinate, and the verifier-as-reward exploit.
- [[holistic-agent-leaderboard]] — the 21,730-rollout three-dimensional analysis, scaffold-model interactions,
  reasoning-effort pairing, the τ-bench few-shot leak, and the infrastructure hazards behind model identity.
- [[agentic-benchmark-checklist]] — task and outcome validity, the trivial-agent baselines, the measured
  over- and underestimations, and the confidence interval under noisy ground truth.
- [[swe-bench-illusion]] — the file-path and function-reproduction probes that measure benchmark-specific
  memorization.
- [[swe-rebench]] — the date-tracked decontaminated slice, the 5-run SEM protocol, and the paired Verified-versus-
  fresh-slice comparison.
- [[impossiblebench]] — the cheating-rate construction and the three harness controls that change it.
- [[metr-time-horizon]] — the logistic time-horizon fit, the 207-day doubling time, the 80%-horizon result, and the
  external-validity checks.
- [[minimax-m2-aligning-to-what]] — the five perturbation axes, used as a qualitative design source only.
- [[agent-world]] — the held-out arena, the environment-count scaling curve, and the general-ability regression of
  EnvScaler-8B against its own backbone.
- [[agentgym-rl]] — the in-domain-only scope statement, used as the reporting failure the §9 gate prevents.
