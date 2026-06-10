<!-- scope: AutomationBench vs the τ-bench family — structural head-to-head
     deps: automationbench-harness, automationbench-tasks-grading, taubench
     see-also: automationbench-overview
-->

# AutomationBench vs τ-bench — Structural Comparison

- **Core Insight:** They measure *different agents*. AutomationBench measures a **back-office
  automator** (one trigger, no human, many apps, find-your-own-tools, follow buried policy);
  τ-bench measures a **customer-service conversationalist** (multi-turn user you must
  interrogate, one app, given tools, follow a posted policy).
- **Guideline:** Pick the benchmark whose *structure* matches your agent's deployment shape;
  a high score on the wrong-shaped benchmark predicts little.
- **Source:** synthesis of [[automationbench-harness]], [[automationbench-tasks-grading]],
  [[taubench]].
- **Relevant chapters:** ch-09 (centerpiece), seeded in ch-02/03/05/06/08.

## Side by side

| Axis | AutomationBench | τ-bench (τ²/τ³) |
|------|-----------------|-----------------|
| Interaction | **No user.** Single natural-language trigger; agent runs to completion | **Multi-turn user simulator** (LLM); agent must *elicit* the hidden goal |
| Apps per task | **Many** (cross-application is the point) | **One** app/domain per task (τ³ banking adds doc discovery) |
| Tools given? | **No** — generic Search+Execute over ~400 tools; discovery is tested | **Yes** — fixed per-domain tool set handed to the agent |
| Where policy lives | **Buried in the seeded world** (an inbox msg, a sheet row) | A **posted policy wiki** the agent may read |
| Grading | **End-state assertion rubric** (must-pass + must-not-occur), pure Python | **Final DB-state == goal-state**, exact, pure Python |
| Partial signal | `partial_credit` exists (RL reward); official score binary | Binary `r∈{0,1}`, no partial |
| Anti-reward-hacking | Negative assertions + count-locks + free-assertion exclusion | Irreversible writes; confirm-before-write policy |
| Headline metric | **pass-rate + cost/task** | **pass^k** (reliability over k trials) |
| Noise / determinism | Seeded in-process world; **<1% variance** | LLM-user stochasticity; **high variance**, 10+ repeats |
| Hardest sub-skill | tool discovery + cross-app coordination + policy-under-noise | information elicitation + user coordination + consistency |

## What each tests that the other cannot

- **Only AutomationBench**: autonomous API discovery (which of 400 tools?), multi-app
  coordination, policy you must *find*, and explicit *cost* as a reported axis.
- **Only τ-bench**: conversational information-gathering, user-coordination (τ² dual-control,
  where the *user* also acts), and **reliability** as a first-class metric (pass^k) — the
  thing AutomationBench's single-shot pass-rate cannot see.

## Two metric philosophies (the deepest contrast)

- **pass-rate + cost** answers *"capability at what price, right now?"* — apt when each run
  is independent and you care about throughput/economics.
- **pass^k** answers *"will it succeed every time a customer asks?"* — apt when the same task
  recurs and a 30% failure rate is disqualifying. AutomationBench could *adopt* pass^k (its
  determinism makes repeats cheap); τ-bench *needs* it because its variance is intrinsic.

## Shared blind spot

Both are **simulations**: AutomationBench's hand-seeded Pydantic worlds and τ-bench's LLM
user simulator each diverge from production (the sim2real gap). A high score is necessary,
not sufficient — neither replaces a measured online trial.

## Transfer (for the learner's Lina TMR sales agent)

The learner's agent is *conversational* (lean τ-shape) but the **eval engineering** to
borrow is AutomationBench's: end-state assertions over a typed world, must-not-occur guards
against shotgun behavior, deterministic seeded noise, a sanity domain, and no LLM-judge
where a programmatic check exists. Combine: a τ-style user simulator for the conversation +
AutomationBench-style end-state assertions + pass^k for reliability. (ch-10 deliverable.)
