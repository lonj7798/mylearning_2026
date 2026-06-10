<!-- scope: τ-bench / τ²-bench / τ³-bench design, user simulator, pass^k
     see-also: automationbench-overview, benchmark-comparison
-->

# τ-bench (and τ² / τ³) — the Comparison Benchmark

- **Core Insight:** τ-bench measures the *tool-agent-USER* triad — an agent must follow
  domain policy **and** elicit the task from a simulated user over multi-turn dialogue —
  and it introduces **pass^k** to score *reliability*, not just peak accuracy.
- **Guideline:** When deployment reliability matters, report **pass^k** (succeed on all k
  i.i.d. trials), not pass@k (succeed on at least one).
- **Source:** arXiv 2406.12045 (τ-bench, Yao et al./Sierra); 2506.07982 (τ²-bench);
  taubench.com (τ³ task fixes); github.com/sierra-research/{tau-bench, tau2-bench}.
- **Relevant chapters:** ch-01, ch-09.

## Structure

- **Domains**: retail (115 tasks; 500 users / 50 product types / 1,000 orders) and airline
  (50 tasks; 500 users / 300 flights / 2,000 reservations).
- **A task = 4 parts**: (1) a user *goal* held by the simulator, not given to the agent;
  (2) a mutable JSON **database** state; (3) a **policy document** (wiki) of domain rules;
  (4) a fixed set of Python-backed **tools** (read vs write; a no-op `think` scratchpad).
- The agent must *elicit* the goal through dialogue — it never sees the ground truth.

## The user simulator (the defining feature)

A separate LLM plays the customer: it is given the goal + persona, withholds information
not asked for, and emits `###STOP###` when done. Strategies: `llm`/`react`/`verify`/
`reflection`. This creates the tool-agent-user triad — and injects **stochastic variance**:
the same task varies run to run because the simulated user volunteers different info and
terminates differently. High σ at n=115 → practitioners run 10+ repeats. The "sim2real
gap" (arXiv 2603.11245) shows LLM users diverge from real users in turn-taking, error
recovery, and persona consistency.

## Tool-call mechanics & grading

Native function-calling via `litellm`. **Read** tools are free; **write** tools mutate the
DB and are **irreversible within a task** (one wrong arg corrupts state). Policy requires
confirming key params with the user before destructive writes. Grading = **final
database-state compared exactly to the annotated goal state** (not the agent's text);
binary reward `r∈{0,1}`, no partial credit.

## pass^k (the metric to remember)

Reliability over k i.i.d. trials, averaged over tasks:
`pass^k = (1/|T|) Σ_i p̂_i^k`, with unbiased estimator
`ρ(n,c,k) = 1 − C(n−c,k)/C(n,k)`. A task at `p=0.9` contributes only `0.9^8 ≈ 0.43` to
pass^8 — reliability collapses fast. `pass^1 = pass@1 = mean success`. Rationale: an agent
that fails 30% of the time on the *same* task is not deployable, however well it does on
unique attempts; pass@k rewards breadth, pass^k rewards consistency.

## Headline numbers & failure modes

Original (2024): GPT-4o ~**61% pass^1 retail**, ~**35% airline**; **pass^8 retail < 25%**.
2026 leaderboard: best retail ~0.86 pass^1, airline ~0.70 (Claude Sonnet 4.5); reproduced
scores swing 10+ pts with prompt changes. Failures: wrong policy branch, write-before-
confirm, under-elicitation → wrong args, irreversibility compounding.

## Limitations & the τ²/τ³ follow-ups

- **Limits**: only retail+airline (narrow, transactional); user-simulator noise conflates
  with agent variance; binary reward hides partial progress; single conversation per task.
  τ³ audit found ~75 defective tasks (airline pass^1 rose 14–20 pts after fixes).
- **τ²-bench** (2506.07982): **dual-control** Dec-POMDP — the *user* also has tools (e.g.
  telecom troubleshooting: reboot router, read config) and the agent must *guide* user
  actions. GPT-4 drops from 56–74% (single-control) to ~34% (dual-control). Compositional
  task generator for controlled difficulty.
- **τ³-bench**: incorporates the task fixes, adds a `banking` RAG/doc-tool-discovery domain,
  and voice/full-duplex metrics — the current recommended baseline.

## Connections

- Head-to-head with AutomationBench in [[benchmark-comparison]]: no-user vs user-simulator,
  fixed tools vs generic discovery, DB-state hash vs assertion rubric, pass^k vs pass-rate+cost.
