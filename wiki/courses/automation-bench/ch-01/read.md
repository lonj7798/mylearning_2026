<!-- chapter: ch-01
     track: landscape
     kind: content
     title: The Agentic Tool-Call Benchmark Landscape
     deps: []
     sources: [[automationbench-overview]], [[taubench]], [[benchmark-comparison]]
-->

# Chapter 01 — The Agentic Tool-Call Benchmark Landscape

> **Core insight.** The field spent 2022–2024 evaluating whether models could *select* the right function from a small candidate set. That bar is now cleared by fine-tuned 7B models. What remains hard — and what AutomationBench is designed to measure — is whether an agent can *discover* which of 400 tools to use, coordinate across multiple applications, follow a policy buried in a seeded artifact, and leave the right final state in each system. No prior benchmark combined all three requirements, which is why frontier models score below 20% here.

> **Guideline.** When selecting a benchmark for an agentic system, match the benchmark's *structure* to the agent's deployment shape: a high score on the wrong-shaped benchmark predicts little. AutomationBench measures the back-office automator (single trigger, no user, many apps, find-your-own-tools). τ-bench measures the customer-service conversationalist (multi-turn user you must interrogate, one app, given tools). Using AutomationBench scores to predict τ-bench performance — or vice versa — is a category error.

---

## 1. The Bar Moved: From Function Selection to Agentic Orchestration

The phrase "tool-call benchmark" changed meaning between 2022 and 2026. The early generation — Berkeley Function-Calling Leaderboard (BFCL), API-Bank, ToolBench — asked a narrow question: given a natural-language instruction *and a pre-supplied set of candidate tools*, can the model choose the correct one and emit a syntactically valid call? The task was single-turn, the tool set was handed over, and the ground truth was a function name plus argument schema.

That question still has diagnostic value for measuring raw function-calling reliability, but it tells you almost nothing about deployment-grade agentic behavior. In production, an agent:

1. Does not receive a pre-filtered candidate set — it must search a catalog.
2. Does not finish in one step — it chains calls, reads intermediate results, branches.
3. Does not work inside a single application — enterprise workflows span CRM, email, calendar, spreadsheets, ticketing, and messaging in one task.
4. Must apply policies that are not in the prompt — they live in a spreadsheet row, an inbox message, a document the agent has to find.

Each of these requirements renders the original BFCL-style benchmark insufficient as a proxy for the harder problem. The field's response has been a series of escalating benchmarks, each unlocking one or two more axes while leaving others fixed. AutomationBench's contribution is to combine all four requirements in a single eval, at scale, with programmatic grading.

### The escalation timeline in brief

| Year | Benchmark | What it added |
|------|-----------|---------------|
| 2022–23 | BFCL, API-Bank, ToolBench | Function selection from a candidate set; single-turn or shallow multi-step |
| 2023 | AppWorld | End-state grading, realistic multi-step within a single simulated environment |
| 2023–24 | WebArena, Mind2Web | Web browsing as the tool substrate; UI grounding; real web pages |
| 2024 | τ-bench | Multi-turn user simulator (the triad: tool + agent + user); pass^k reliability metric; domain policy compliance |
| 2024–25 | τ²/τ³ | Dual-control tasks; task-quality fixes; RAG/doc discovery (τ³ banking domain) |
| 2026 | **AutomationBench** | Cross-application coordination + autonomous API discovery from a ~400-tool catalog + policy-in-artifacts, all in one task |

The column "What it added" is not "what it measures exclusively." BFCL still matters for raw function-calling precision diagnostics. τ-bench still matters for conversational reliability. AutomationBench fills the gap for the specific shape of work that dominates enterprise automation at scale.

---

## 2. The Benchmark Family in Detail

### 2.1 BFCL / API-Bank / ToolBench

These benchmarks share a structural assumption: the candidate tool set is given. BFCL evaluates whether the model calls the right function with correct arguments; it grades the *output* (the call text), not the *outcome* (a world state). API-Bank and ToolBench extended this to multi-step chains with retrieved tool documentation, but the retrieval is provided or search-assisted — the agent is not evaluated on whether it can discover the right tool from a large undifferentiated catalog.

These are correctly used as fast, cheap regression tests for function-calling mechanics. They are not valid proxies for cross-application orchestration.

### 2.2 AppWorld

AppWorld introduced end-state grading inside a single simulated multi-app environment: the agent must complete a task and the evaluator checks whether the resulting world state matches the expected state. This is the same philosophical move AutomationBench makes (grade outcomes, not output text), and AppWorld deserves credit for that design. The constraint is that tasks are single-environment: the agent is not required to coordinate across applications with independent schemas, and tools are pre-supplied. [[automationbench-overview]] summarizes the contrast in its Table 1 entry (AppWorld row: cross-app ✗, API discovery ✗, end-state ✓, business rules ✗).

### 2.3 WebArena / Mind2Web

WebArena and Mind2Web use real or replayed web browsers as the action substrate — clicking, form-filling, navigation. The grading is partly outcome-based (did the agent land on the right page, fill the right field?). These benchmarks measure *UI-grounded* tool use, not programmatic API use. The failure modes are different (HTML parsing, element localization, session management) and the skills transfer imperfectly to the API-invocation world that AutomationBench occupies.

### 2.4 τ-bench / τ² / τ³

τ-bench ([[taubench]]) is the most structurally distinct from AutomationBench in the current landscape, and the most important to understand as a contrast.

A τ-bench task consists of four parts: (1) a user *goal* held by an LLM user simulator, never shown to the agent; (2) a mutable JSON database state; (3) a posted policy wiki the agent may read; (4) a fixed set of Python-backed tools. The agent must *elicit* the goal through multi-turn dialogue, follow domain policy, and leave the database in the correct final state. Grading is `final_DB_state == goal_state`, exact, binary — again an outcomes-not-output philosophy, shared with AutomationBench.

The defining feature is the **user simulator**: a separate LLM plays the customer, withholds information not asked for, and terminates with `###STOP###`. This injects stochastic variance that is intrinsic to the design, not a bug — the same task varies run to run because the simulated user volunteers different information. That variance is why τ-bench introduced **pass^k**:

```
pass^k = (1/|T|) Σ_i p̂_i^k
```

with unbiased estimator `ρ(n,c,k) = 1 − C(n−c,k)/C(n,k)`. At `p=0.9` per task, `pass^8 ≈ 0.43` — reliability collapses fast. A model that succeeds 90% of the time on the same task would score only 43% on pass^8: nearly undeployable by that metric even though it looks capable on a single-shot evaluation. `pass^1 = pass@1 = mean success`.

τ²-bench (arXiv 2506.07982) added **dual-control**: the *user* also has tools (e.g. reboot router, read config) and the agent must guide user actions rather than just elicit information. GPT-4 drops from 56–74% (single-control) to ~34% (dual-control). τ³-bench incorporated the ~75 defective task fixes identified by an audit (airline pass^1 rose 14–20 points after fixes) and added a `banking` domain with RAG/doc-tool discovery.

---

## 3. AutomationBench's Three-Part Gap Thesis

AutomationBench's motivation is captured in one sentence from [[automationbench-overview]]:

> *Real enterprise agentic work is cross-application coordination + autonomous API discovery + policy adherence, all at once — and no prior tool-call benchmark combined the three, so frontier models score <20%.*

The paper's Table 1 (reproduced from [[automationbench-overview]]) makes the gap explicit:

| Benchmark | Cross-app | API discovery | End-state grading | Business rules |
|-----------|-----------|---------------|-------------------|----------------|
| WebArena / Mind2Web | ✗ | ✗ | ✗ | ✗ |
| ToolBench / API-Bank | ✗ | retrieval-assisted | varies | ✗ |
| AppWorld | ✗ (single env) | ✗ | ✓ | ✗ |
| τ³-bench | ✗ (single app) | partial (banking) | ✓ | ✓ |
| **AutomationBench** | **✓** | **✓ (BM25)** | **✓** | **✓** |

Each column is a design choice, not an accident. Let's trace why each one exists.

### 3.1 Cross-application coordination

Zapier's production catalog handles ~2 billion monthly tasks. The overwhelming majority of high-value workflows are multi-app by nature: a new Salesforce opportunity triggers a Gmail routing notification whose content depends on a Google Sheets policy row and a Google Drive account-hierarchy document. No single-application benchmark can test whether an agent coordinates across independently-typed state objects that share no schema.

The multi-hop sales task in [[automationbench-tasks-grading]] (example_id 501) illustrates this concretely. The trigger says:

```
"We just closed the Meridian Corp Platform Deal! Mark it won and route the win notice per
our routing policy. Confirm the account tier from the 'Account Hierarchy' sheet, convert
currency (see 'FX Rates'), and check for open support escalations."
```

Solving this requires: Salesforce (mark Closed Won) → Google Drive (find two sheets) → Google Sheets (resolve conflicting dated rows for tier and FX rate, taking the most recent) → Salesforce again (check escalations on account *and* parent account) → Gmail (route to the correct team per policy). The routing policy is not in the prompt — it is in a seeded inbox message the agent must discover.

### 3.2 Autonomous API discovery

In the default `zapier` toolset mode, the agent receives exactly two tools: `search_tools(query, top_k)` and `execute_tool(tool_name, arguments)`. The underlying catalog has ~400 named endpoints. The agent must issue BM25 queries to identify which tools are relevant before it can call them. This tests a skill that all prior benchmarks either avoided (by handing the tool set over) or reduced to retrieval-assistance (by giving the agent a pre-filtered candidate list).

The `limited_zapier` mode gives the agent the full named tool set filtered to the per-task allowlist — this is the ablation that isolates execution skill from discovery skill. The `api` mode gives the agent three REST-shaped tools (`api_search`, `api_fetch`, `base64_encode`) and the agent must construct valid URLs. These three modes make discovery a first-class measured capability while allowing controlled ablation of it.

### 3.3 Policy adherence

The system prompt instructs the agent not to ask clarifying questions. Policies are embedded in seeded artifacts — a spreadsheet row, an inbox message, a document. The agent must find the policy, parse it, and apply it correctly. This is structurally harder than τ-bench's posted policy wiki (which the agent is told exists and may read), and it is untested by any prior benchmark.

Examples from [[automationbench-tasks-grading]]: a scope-creep trap where the user says "also process severance" but the policy artifact says HR must NOT (only Payroll); a compliance-hold trap where noise contacts carry "do not enroll — pending compliance review" in their description field; recency-conflict rows where the agent must identify the most recent FX rate among multiple dated rows in the same sheet.

---

## 4. "Outcomes, Not Output": End-State Grading as a Philosophy

The README states this principle explicitly:

> **Verifiability** — All tasks must be programmatically verifiable. If we can't automatically check whether a task was completed correctly, it doesn't belong in the benchmark.

```
Every run reports two per-task metrics:
- partial_credit (0.0 - 1.0) - fraction of assertions satisfied.
- task_completed_correctly (0.0 or 1.0) - strict pass/fail; 1.0 only if every assertion passes.
```

This is a principled rejection of two cheaper alternatives:

**Output-text grading** (e.g., "did the model say the right company name?") misses the actual effect. A model that calls the wrong Salesforce API, corrupts the deal stage, *and* sends the right email subject line would pass an output check and fail the outcome check. Output grading systematically under-detects partial failures and over-rewards text-fluent hallucination.

**LLM-as-judge** replaces programmatic checks with a second model call. AutomationBench uses pure Python field comparison on Pydantic models with strict types. This is not a resource constraint — it is a deliberate design choice for reproducibility, cost, and resistance to judge-model biases. The assertion registry checks `world.salesforce.opportunities[id].stage_name == "Closed Won"` directly. There is no ambiguity and no judge variance.

The two-tier scoring structure (binary `task_completed_correctly` for the official pass-rate; floating `partial_credit` for RL reward signals and per-stage debugging) reflects a second principle: *the metric must match the use case*. Deploying an agent to business users requires binary reliability — a task that partially completed is a failed task. Training an agent requires a dense reward signal — binary 0/1 is too sparse for gradient flow. AutomationBench provides both from the same assertion rubric.

### Why Zapier built it this way

Zapier operates 9,000+ app integrations and ~2 billion monthly tasks. Their internal decision problem was which models to deploy in production. They found no public benchmark adequate:

> *Zapier built it internally to decide which models to deploy in production, found no public benchmark adequate, and open-sourced it. Its substrate is Zapier's real catalog (9,000+ app integrations, 66,000+ triggers/actions, ~2B monthly tasks) abstracted into 47 simulated apps and ~500 endpoints across six high-frequency business domains.* — [[automationbench-overview]]

The "outcomes, not output" philosophy follows directly from the production context. In production, the outcome is what matters: did the deal get marked Closed Won, did the correct team get notified, did the compliance-hold contact get excluded? The benchmark inherits this standard from its deployment context.

---

## 5. The Dataset: Scope, Shape, and Synthetic Construction

The public task set is 606 tasks across six domains, plus a 200-task `simple` sanity set:

| Domain | Tasks | Coverage |
|--------|-------|----------|
| Sales | 106 | CRM, lead management, cross-app workflows |
| Marketing | 100 | Campaigns, ad performance, content ops, brand monitoring |
| Operations | 100 | Facility management, project tracking, vendor workflows, compliance |
| Support | 100 | Ticket routing, SLA monitoring, knowledge base, multi-platform helpdesk |
| Finance | 100 | AP/AR, expenses, reporting, bookkeeping |
| HR | 100 | Recruitment, employee onboarding, time off, payroll |
| **Simple** | **200** | Single- and two-step harness validation (not scored) |

The 600+ private tasks held out for the official leaderboard follow the same distribution. Scores on the public set are directionally predictive but not identical to leaderboard scores.

Tasks are synthetically generated from the *shapes* of real customer workflows and hardened with negative feedback from Zapier's Agents service — no PII, no raw customer data. The generation code is not public; tasks appear as handcrafted Python constructor dicts in `domains/*/tasks.py`. Each task has a unique `example_id` that seeds the deterministic noise injection — the same task always has the same distractors, which is why run-to-run variance is <1%.

---

## 6. Why the Field Scores Below 20%: Difficulty by Construction

SOTA pass-rate on the official leaderboard as of mid-2026 is approximately 12–17%. The paper's headline at submission was "all SOTA models score below 10%." These numbers have moved modestly upward as stronger models were evaluated, but the ceiling is still very low.

This is not a sign of a broken benchmark — it is a sign of a well-calibrated one. The evidence comes from the **`simple` domain**: even small models hit ~97% there. [[automationbench-overview]] states:

> *Even small models hit ~97% on the `simple` domain, confirming low main-benchmark scores reflect genuine orchestration difficulty, not a broken harness.*

The simple tasks are single- and two-step operations on individual apps. The model knows what to do and the world is not adversarially seeded. 97% pass rate confirms the harness works and the model can use basic tools. The 12–17% on main tasks measures the marginal cost of:

1. **Discovery overhead**: before doing anything, the agent must find the right tools via BM25 queries. A failed search means the agent either hallucinates a tool name or proceeds with the wrong one. The dominant failure mode reported in [[automationbench-overview]] is "false confidence in incorrect tool calls" — 72–91% of failures fall here.

2. **Cross-application coordination**: the agent must maintain context across multiple apps, carry intermediate results from one tool call into the arguments of another, and not confuse entities that appear in multiple app states.

3. **Policy discovery and correct application**: the agent must notice that a policy exists (it is not flagged in the prompt), retrieve the right artifact, parse the policy, and apply it correctly even when the natural-language instruction seems to suggest a different action.

4. **Adversarial seed data**: near-match entity traps (`acme-corp.com` vs `acmecorp.com`), compliance-hold descriptions on noise contacts, recency conflicts, and negative assertion guards that punish shotgun behavior all raise the floor of what a "lucky" model can achieve.

### What a low ceiling tells you about headroom

A benchmark where SOTA scores 97% has exhausted its measurement range — you cannot distinguish future improvements because everything clusters at the top. A benchmark where SOTA scores 15% has a lot of headroom: the difference between 15% and 30% is a meaningful capability jump that the benchmark can measure cleanly. The low ceiling on main AutomationBench tasks is a feature, not a problem.

The sanity control (simple domain) is the critical complement. Without it, you cannot tell whether a 15% score reflects genuine task difficulty or harness bugs. With it — and with the knowledge that the same models score 97% on simple — you can confidently attribute low main-benchmark scores to the hard requirements the benchmark was designed to test.

---

## 7. The Comparison Axis: AutomationBench vs. τ-bench

The structural contrast between the two most important benchmarks in this landscape deserves explicit treatment here, before diving into internals. [[benchmark-comparison]] develops this in detail; the thumbnail version:

| Axis | AutomationBench | τ-bench (τ²/τ³) |
|------|-----------------|-----------------|
| Interaction | No user — single NL trigger; agent runs to completion | Multi-turn LLM user simulator; agent must elicit the hidden goal |
| Apps per task | Many (cross-application is the point) | One app/domain per task |
| Tools given? | No — generic Search+Execute over ~400 tools | Yes — fixed per-domain tool set |
| Policy location | Buried in seeded world artifacts | Posted policy wiki the agent may read |
| Grading | End-state assertion rubric (must-pass + must-not-occur) | Final DB-state == goal-state, exact |
| Partial signal | `partial_credit` (RL use); official score binary | Binary r∈{0,1}, no partial |
| Headline metric | pass-rate + cost/task | pass^k (reliability over k trials) |
| Run-to-run noise | <1% (deterministic seeded world) | High (LLM user stochasticity; 10+ repeats recommended) |
| Hardest sub-skill | Tool discovery + cross-app coordination + policy-under-noise | Information elicitation + user coordination + consistency |

The two metric philosophies express two different deployment questions. **pass-rate + cost** answers "capability at what price, right now?" — apt when each workflow trigger is independent and you care about throughput economics. **pass^k** answers "will it succeed every time a customer asks the same thing?" — apt when the same task recurs and a 30% failure rate is disqualifying regardless of peak performance.

[[benchmark-comparison]] makes a point worth quoting:

> *They measure different agents. AutomationBench measures a back-office automator (one trigger, no human, many apps, find-your-own-tools, follow buried policy); τ-bench measures a customer-service conversationalist (multi-turn user you must interrogate, one app, given tools, follow a posted policy).*

The practical consequence: if you are building a workflow automation agent, AutomationBench scores are predictive of production behavior in a way τ-bench scores are not, and vice versa for a customer-service agent. This chapter's goal is to establish the landscape so that the internals chapters can explain *how* AutomationBench enforces each of its four requirements at the code level.

---

## 8. Running AutomationBench: The Minimal Footprint

The README describes the entry point verbatim:

```bash
# Clone the repo
git clone https://github.com/zapier/AutomationBench.git
cd AutomationBench

# Install dependencies
uv sync

# Set your API key (or create a .env file)
export OPENAI_API_KEY=sk-...

# Run evaluation
uv run auto-bench --model gpt-5-mini

# Run specific domains
uv run auto-bench --model gpt-5-mini --domains sales

# Anthropic models — auto-detected via `claude-*` prefix
export ANTHROPIC_API_KEY=sk-ant-...
uv run auto-bench --model claude-haiku-4-5-20251001
```

Key CLI parameters that matter for benchmark interpretation:

- `--toolset`: `api` | `zapier` | `limited_zapier`. Changing this changes what you are measuring (see §3.2). The default `api` mode uses REST-shaped tools; `zapier` is the headline discovery-required mode.
- `--max-steps 50`: the README and CLI default say 50; [[automationbench-harness]] reports the code default as `max_turns=25`. This discrepancy (docs vs code) is resolved in ch-02.
- `--domains`: run only sales, or only simple, etc. Running `--domains simple` first is a fast harness-validity check.
- Cost metrics are emitted per task alongside the binary pass/fail — this is reported by default, not opt-in.

The benchmark also runs on the Prime Intellect Environments Hub as a hosted environment, which means it can serve as an RL training environment directly:

```bash
prime env install zapier/AutomationBench
prime eval run zapier/AutomationBench
```

The `partial_credit` score that the harness computes alongside `task_completed_correctly` is the dense reward signal for RL — a detail that ch-10 makes central.

---

## Where This Course Goes

This chapter established the landscape: why the benchmark bar moved, where AutomationBench sits in the family, and why its three-part gap thesis (cross-app coordination + autonomous discovery + policy adherence) produces a genuinely hard eval. Everything from here forward is mechanistic.

Ch-02 opens the execution engine: `AutomationBenchEnv`, the `WorldState` Pydantic root model, the episode lifecycle, and the three toolset modes — including the doc-vs-code discrepancies on tool count and step limits. Ch-03 and Ch-04 go deep on the two-phase execution model: BM25 tool discovery then in-process simulated execution. Ch-05 walks a full task dict anatomy and the six business domains. Ch-06 covers the assertion grading engine — how must-pass and must-not-occur assertions are evaluated, and the free-assertion exclusion logic that prevents reward-hacking by doing nothing. Ch-07 covers hardening: seeded noise, near-match traps, compliance-hold patterns, and anti-shotgun negative assertions. Ch-08 closes the internals phase with metrics, cost accounting, and reproducibility.

Ch-09 consolidates the [[benchmark-comparison]] thread that has been seeded across the internals chapters — a full structural head-to-head with the τ-bench family and a decision framework for choosing the right benchmark for a given agent shape. Ch-10 is the lab: run a real evaluation, extend the benchmark with a new task and assertion, and design your own end-state-graded agent benchmark using the principles extracted across the course.
