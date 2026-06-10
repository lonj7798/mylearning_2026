<!-- scope: cross-source insight index for the automation-bench raw library
     deps: [[LIBRARY]], [[COLLECTION-PLAN]]
     see-also: [[automationbench-overview]], [[automationbench-harness]],
               [[automationbench-tasks-grading]], [[taubench]], [[benchmark-comparison]]
-->

# AutomationBench — Insights Index

Cross-source map for the library. The course's spine is one idea: **a good agent benchmark
is a deterministic world + an honest grader**, and every design choice in AutomationBench
falls out of taking that seriously.

## Benchmark design

- **Outcomes, not output.** Grade the final world state, never the agent's prose. Both
  AutomationBench (assertion rubric) and τ-bench (DB-state hash) commit to this; it is the
  single most important design decision and what makes "no LLM-judge" possible.
  ([[automationbench-overview]], [[taubench]])
- **Determinism is a design output, not luck.** AutomationBench simulates the whole world
  as one in-process Pydantic object and seeds all noise by `example_id` → <1% run variance.
  τ-bench's LLM user simulator makes variance *intrinsic* → 10+ repeats needed. If you want
  cheap repeats (and thus pass^k), you must engineer determinism in. ([[automationbench-harness]])
- **Make the agent discover its tools.** Handing the agent the right tools (limited_zapier,
  or τ-bench's fixed set) measures execution; forcing BM25 search over ~400 tools measures
  discovery too. The same harness can ablate the two by switching toolset mode.
  ([[automationbench-harness]])
- **Hardness lives in the data and the negative assertions, not the prompt.** Decoys,
  near-match entities, compliance-hold traps, scope-creep traps, recency conflicts, and
  must-not-occur guards are what separate "measures reasoning" from "measures luck." A
  benchmark with only positive assertions is reward-hackable by shotgun behavior.
  ([[automationbench-tasks-grading]])
- **Two scores from one rubric.** `partial_credit = passed/total` doubles as an RL reward;
  `task_completed_correctly` (all-or-nothing) is the honest headline. Reporting both — and
  keeping the headline binary — resists partial-credit gaming. ([[automationbench-tasks-grading]])
- **A sanity domain is a harness-validity control.** `simple` (small models ~97%) proves a
  low main score is real difficulty, not a broken harness. Every benchmark should ship one.

## Metrics

- **pass-rate + cost vs pass^k are different questions.** Capability-at-a-price (independent
  runs) vs reliability-across-k-trials (recurring task). AutomationBench reports the former
  and *could* add the latter for free (it's deterministic); τ-bench *must* use pass^k
  because variance is built in. ([[benchmark-comparison]])

## Landscape

- **The three-way gap is the thesis.** Cross-app coordination + autonomous API discovery +
  policy adherence — prior benchmarks each had one or two; AutomationBench combines all
  three, which is why frontier models sit <20%. ([[automationbench-overview]])
- **They measure different agents.** AutomationBench ≈ back-office automator (no user, many
  apps, find tools, buried policy); τ-bench ≈ customer-service conversationalist (user
  simulator, one app, given tools, posted policy). Score on the wrong-shaped benchmark
  predicts little. ([[benchmark-comparison]])

## Open gaps (see COLLECTION-PLAN for the full log)

- Private task set + task-generation recipe are not in the repo; teach authoring from the
  paper + handcrafted examples.
- Both benchmarks share the **sim2real gap**; neither replaces a measured online trial —
  the through-line to the learner's own end-model / sim-to-real eval problem (ch-09, ch-10).
