<!-- chapter: ch-09
     track: comparison
     kind: content
     title: AutomationBench vs the τ-bench Family: A Structural Comparison
     deps: [ch-08, ch-05, ch-06]
     sources: [[benchmark-comparison]], [[taubench]], [[automationbench-overview]]
     figures: figures/ab-vs-tau.html
-->

# Chapter 09 — AutomationBench vs the τ-bench Family: A Structural Comparison

> **Core insight.** AutomationBench and τ-bench do not measure the same agent. AutomationBench measures a *back-office automator*: one trigger, no user, many apps, tool discovery required, policy buried in artifacts. τ-bench measures a *customer-service conversationalist*: multi-turn user simulator, one app, fixed tools, posted policy wiki. A high score on the wrong benchmark predicts almost nothing about your agent's real-world failure modes.

> **Guideline.** Match your benchmark's structure to your agent's deployment shape before you run a single evaluation. If your agent never talks to a user, τ-bench reliability numbers mean little. If your agent is purely conversational, AutomationBench's cross-app discovery score means little. The comparison only becomes actionable when you identify which structural features you need to borrow from the other side.

---

## 1. The Core Axis: Interaction Model

The most fundamental difference between the two benchmarks is not tooling, grading, or metrics. It is whether a human user exists in the loop at all.

**AutomationBench: trigger-and-run.** A task begins with a single natural-language event — an email notification, a Slack message, a request — and the agent runs autonomously to completion. There is no user to ask, no clarifying turn allowed. The system prompt enforces this directly:

```python
# automationbench/domains/sales/tasks.py  L31-37
SYSTEM_PROMPT = (
    "You are a workflow automation agent. Execute the requested tasks using the available tools. "
    "Do not ask clarifying questions - use the information provided and make reasonable assumptions when needed. "
    ...
)
```

Policy ambiguities, conflicting data, and missing information are encoded into the initial world state and must be resolved by *reading artifacts*, not asking the user. The routing policy for a win notification lives in a Gmail inbox message the agent must find. FX conversion rates are in a sheet with two conflicting rows — recency is the implicit rule. No one will tell you. You either find it or you fail the assertion.

**τ-bench: tool-agent-USER triad.** A separate LLM plays the customer. It is given a goal and a persona, withholds information not yet asked for, and emits `###STOP###` when satisfied. The agent must *elicit* the full task across multiple turns — it never sees the ground truth goal. From [[taubench]]:

> *"A task = 4 parts: (1) a user goal held by the simulator, not given to the agent; (2) a mutable JSON database state; (3) a policy document (wiki) of domain rules; (4) a fixed set of Python-backed tools."*

The user simulator is the mechanism that makes information elicitation a first-class test. It is also the mechanism that injects stochastic variance: the same task varies run to run because the simulated user volunteers information differently and terminates at different points. This is not a flaw — it models the real variability of human customers. But it forces a different evaluation philosophy, which is where pass^k enters.

---

## 2. Tooling: Discovery vs. Provision

The second structural divide is how tools reach the agent.

**AutomationBench: generic Search+Execute over ~400 tools.** In the default `zapier` mode (see ch-02), the agent receives exactly two meta-tools: `search_tools(query, top_k=5)` and `execute_tool(tool_name, arguments)`. The BM25 index spans every action across every simulated app — roughly 400 endpoints. The agent must formulate queries, retrieve candidates, read schemas, and construct calls. Tool discovery is an explicit tested skill.

The `limited_zapier` ablation mode hands the agent a named per-task subset (e.g. the eight tools in `info.zapier_tools` for `sales.multi_hop_lookup`). Comparing `zapier` vs. `limited_zapier` scores isolates exactly how much of a model's failure is attributable to discovery rather than execution. The gap between those two numbers is the discovery tax.

**τ-bench: fixed per-domain tool set, provided upfront.** The agent receives the full domain tool set at task start — there is no discovery step. Tools divide into two classes: read tools (free, non-mutating) and write tools (mutate the DB, irreversible within a task). The policy requires confirming key parameters with the user before any destructive write. Tool choice and argument construction are tested; *finding* which tools exist is not.

The practical implication is that τ-bench's failure modes cluster around argument errors (wrong field value after insufficient elicitation) and policy violations (write-before-confirm). AutomationBench's failures cluster around *tool selection* (false confidence in the wrong tool, with 72–91% of failures attributable to incorrect tool calls per [[automationbench-overview]]).

---

## 3. Policy and State: Buried vs. Posted

Both benchmarks require policy adherence. They differ sharply in how policy is surfaced to the agent.

**AutomationBench: policy-in-artifacts.** The routing rule, compliance hold, FX convention, account tier — none of these appear in the task prompt. They live in the seeded world state: an inbox email, a spreadsheet note, a Notion page. The agent must *decide to look for the policy*, *know where to look*, and *extract it from unstructured text*. As the guideline from [[automationbench-tasks-grading]] puts it: *"bury the policy in the world so the agent has to find it."*

The `multi_hop_lookup` walkthrough in ch-05 is the canonical illustration: the routing policy lives in `msg_routing_policy`, a Gmail message seeded into `initial_state.gmail.messages`. The agent has no indication the policy is in Gmail vs. Salesforce vs. a spreadsheet. Failing to retrieve it before acting produces a wrong-team email, which triggers a `gmail_message_not_sent_to` negative assertion.

**τ-bench: posted policy wiki.** The agent is given a structured policy document at task start. It knows the wiki exists, knows where it is, and can query it. The challenge is not policy discovery but policy *application*: reading the right rule branch, not writing before confirmation, handling edge cases. This is harder when the user provides inconsistent information — which branch applies when the user claims a return window has passed but the database says it hasn't?

**State scope: multi-app vs. single-app.** AutomationBench tasks span multiple independent app states simultaneously — ch-05's Salesforce+Google Drive+Google Sheets+Gmail chain is the norm, not the exception. τ-bench tasks live inside one application per task (retail CRM, airline reservation system, banking). τ³-bench adds a `banking` domain with limited RAG/doc discovery, but the core loop remains single-app. This is the sharpest gap in the landscape table from [[automationbench-overview]]:

| | Cross-app | API discovery | End-state grading | Business rules |
|---|---|---|---|---|
| τ³-bench | ✗ (single app) | partial (banking) | ✓ | ✓ |
| **AutomationBench** | **✓** | **✓ (BM25)** | **✓** | **✓** |

---

## 4. Grading: Assertion Rubric vs. DB-State Equality

Both benchmarks grade on outcomes, not agent text. Both use pure Python with no LLM judge. The mechanisms differ.

**AutomationBench: must-pass AND must-not-occur assertion rubric.** Assertions are typed dicts dispatched through `AssertionRegistry` (ch-06). A task may carry:

```python
# must-pass
{"type": "salesforce_field_equals", "collection": "opportunities",
 "record_id": "006xx000004MER1", "field": "stage_name", "value": "Closed Won"}

# must-not-occur (anti-shotgun)
{"type": "gmail_message_not_sent_to", "to": "vp-sales@example.com",
 "subject": "Deal Closed Notification"}
```

The negative assertions are the anti-reward-hacking mechanism. An agent that sends the deal notification to all five mailboxes "to be safe" fails three must-not-occur checks. An agent that marks multiple opportunities won fails a `salesforce_field_equals` guard on the untargeted records. Free-assertion exclusion prevents reward for doing nothing: if an assertion is already satisfied in the initial world state, it is dropped from the denominator — but *breaking* a pre-passing guard still counts as a failure.

The official score is binary (`task_completed_correctly = 1 iff all assertions pass`). `partial_credit = passed/total` exists but only as an RL reward signal and diagnostic — it does not appear in the headline leaderboard number.

**τ-bench: final DB-state equality check.** Grading compares the terminal database state to an annotated goal state, exact match, binary reward `r∈{0,1}`, no partial credit. The mechanism is simpler because the task is scoped to a single application: "does the DB look exactly like this after the conversation ends?" There are no negative assertions, but irreversibility of write tools serves the same anti-gaming function — a wrong write cannot be undone within the task, so shotgun behavior is punished by state corruption rather than by explicit guards.

---

## 5. The Two Metric Philosophies

This is the deepest contrast, and worth understanding precisely.

### 5a. pass-rate + cost (AutomationBench)

AutomationBench reports `pass-rate` (fraction of tasks completed correctly) alongside `cost/task` (dollars of LLM inference per completed or attempted task). The pair answers: *"what can this agent do, and at what price, right now?"*

Each task is a unique, independent episode. The benchmark has 606 public tasks across six distinct business domains — no task recurs. The evaluation question is capability breadth under budget. Cost appears because AutomationBench was built by Zapier to answer a production deployment question: not just "does it work?" but "can we afford to run it at scale?" SOTA pass-rate as of 2026-06 is ~12–17%; frontier models were under 10% at paper submission ([[automationbench-overview]]).

Pass-rate is appropriate here because AutomationBench's in-process Pydantic world is essentially deterministic: run-to-run variance is below 1%. You do not need to repeat trials to get a stable estimate. Each task run once gives you reliable signal.

### 5b. pass^k (τ-bench)

τ-bench reports `pass^k` — reliability over k i.i.d. trials, averaged across tasks. From [[taubench]], the formula and its unbiased estimator:

```
pass^k = (1/|T|) Σ_i p̂_i^k

unbiased estimator:  ρ(n, c, k) = 1 − C(n−c, k) / C(n, k)
```

where for task i, n trials were run and c succeeded.

**Why pass^k and not pass@k?** pass@k (succeed on *at least one* of k attempts) rewards breadth. An agent that fails a task 80% of the time but succeeds 20% of the time has a high pass@k for large k. pass^k requires *every* trial to succeed — it measures consistency. For a customer-service agent where the same transaction recurs thousands of times daily, a 30% failure rate on any single task type is disqualifying even if the agent performs brilliantly on other tasks. pass^1 = pass@1 = mean success rate.

**Numeric example.** Suppose a task has per-trial success probability p = 0.9 — the agent succeeds 90% of the time. Its contribution to pass^8 is:

```
p^8 = 0.9^8 = 0.43
```

That task contributes only 0.43 to the pass^8 average, even though it succeeds 9 times out of 10. For p = 0.7: `0.7^8 ≈ 0.06`. Reliability collapses fast as k increases. The original 2024 τ-bench results illustrate this concretely: GPT-4o scored ~61% pass^1 on retail but below 25% pass^8 — the model looks usable on a single run and falls apart at the standard a real deployment would require ([[taubench]]).

**Can AutomationBench adopt pass^k?** Yes, and cheaply: its near-zero variance means repeating trials costs inference compute but adds no new environmental noise. The benchmark *could* report pass^k as a secondary metric. The reason it does not today is that its primary question is capability-at-a-price, not reliability-across-deployments. τ-bench *needs* pass^k because its user simulator introduces intrinsic stochasticity — without repeated trials you cannot distinguish agent variance from simulator noise.

---

## 6. Side-by-Side Comparison Table

| Axis | AutomationBench | τ-bench (τ²/τ³) |
|------|-----------------|-----------------|
| **Interaction model** | No user. Single NL trigger; agent runs to completion | Multi-turn LLM user simulator; agent must *elicit* the hidden goal |
| **Apps per task** | Many (cross-app is the point; 5+ apps in canonical tasks) | One app/domain per task (τ³ banking adds limited doc discovery) |
| **Tools given to agent?** | No — generic Search+Execute over ~400; discovery is tested | Yes — fixed per-domain tool set, handed upfront |
| **Where policy lives** | Buried in seeded world (inbox message, sheet row, Notion page) | Posted policy wiki; agent knows it exists and where |
| **Grading mechanism** | Assertion rubric: must-pass + must-not-occur, pure Python | Final DB-state == annotated goal-state, exact match, pure Python |
| **Partial signal** | `partial_credit` for RL reward; official score binary | Binary `r∈{0,1}`, no partial credit at all |
| **Anti-reward-hacking** | Negative assertions + count-locks + free-assertion exclusion | Irreversible writes; confirm-before-write policy |
| **Headline metric** | pass-rate + cost/task | pass^k (reliability over k trials) |
| **Run-to-run variance** | < 1% (in-process Pydantic, seeded determinism) | High (LLM user stochasticity); 10+ repeats needed |
| **Hardest sub-skill tested** | Tool discovery + cross-app coordination + policy-under-noise | Information elicitation + user coordination + consistency |
| **Benchmark size (public)** | 606 tasks (6 domains × ~100 tasks) | 165 tasks (retail 115 + airline 50); τ³ adds banking |
| **SOTA ceiling (2026-06)** | ~12–17% pass-rate | ~0.86 pass^1 retail, ~0.70 airline (Claude Sonnet 4.5) |

---

## 7. What Each Tests That the Other Cannot

The comparison table risks making the two benchmarks look like overlapping feature sets. They are not. Each tests skills the other's architecture structurally cannot reach.

**Only AutomationBench can test:**

- *Autonomous tool discovery.* τ-bench hands tools to the agent. AutomationBench requires the agent to figure out that `google_drive_find_multiple_files` exists before it can use it. Whether a model can reason about tool *availability* — not just tool *arguments* — is invisible to any benchmark that provisions tools upfront.
- *Cross-application coordination.* The multi-hop Salesforce → Google Drive → Google Sheets → Gmail chain in `multi_hop_lookup` (ch-05) requires maintaining coherent state across independent app boundaries, with no shared session and no API that bridges them. τ-bench's single-app scope cannot model this.
- *Policy discovery under noise.* Finding a routing policy in an inbox among decoy messages, with near-match entity names, conflicting sheet rows, and a scope-creep trap in the prompt is not a skill a single-app benchmark can probe. AutomationBench's hardening mechanisms (ch-07) encode this explicitly: ID-namespaced decoy pools, compliance-hold traps, recency conflicts.
- *Cost as a first-class axis.* AutomationBench reports dollars per task alongside pass-rate. There is no equivalent in τ-bench.

**Only τ-bench (τ²/τ³) can test:**

- *Conversational information elicitation.* An agent that cannot ask the right questions across multiple turns will fail τ-bench regardless of how good its tool calls are. The user simulator withholds information not explicitly requested — under-elicitation produces wrong-argument writes.
- *User coordination (τ²-bench).* In the dual-control Dec-POMDP formulation, the user also has tools — the customer troubleshooting their router must reboot it themselves when the agent instructs them to. The agent must not just elicit information but *guide user actions*. GPT-4 drops from ~56–74% (single-control) to ~34% (dual-control). That gap is invisible to any benchmark where the agent acts unilaterally.
- *Consistency across a conversation.* A τ-bench agent that contradicts itself mid-conversation — confirming a return on turn 3, then denying eligibility on turn 7 — fails the final DB-state check even if the last write is correct. Conversational coherence is not a property a trigger-and-run benchmark can measure.
- *Reliability as a deployment property.* pass^k is most meaningful when the same task recurs at scale. Customer service transactions are inherently repetitive; enterprise automation workflows are often one-off. τ-bench's metric is better matched to the deployment reality of the agent it measures.

---

## 8. The τ² and τ³ Extensions

τ-bench has evolved. The extensions are relevant because they close some (not all) of the gap with AutomationBench.

**τ²-bench (arXiv 2506.07982):** introduces the dual-control Dec-POMDP — the user also has agency, not just information. The telecom troubleshooting domain requires the agent to guide a customer through rebooting hardware and reading config values. This extends the benchmark toward multi-party orchestration, though still within a single application context. A compositional task generator enables controlled difficulty scaling.

**τ³-bench (taubench.com):** corrects ~75 defective tasks identified in the original τ-bench (airline pass^1 rose 14–20 points after fixes — a sobering benchmark hygiene reminder). Adds a `banking` domain with RAG and document-tool discovery, which partially closes the policy-discovery gap. Adds voice/full-duplex metrics. As of 2026, τ³ is the recommended baseline for reporting new results.

What τ³ does *not* close: cross-application coordination remains absent. The banking domain's doc discovery is a query over a local document corpus, not BM25 over a ~400-tool catalog spanning five independent SaaS backends. The fundamental single-app constraint is architectural, not an oversight.

---

## 9. Shared Blind Spot: The Sim2Real Gap

Both benchmarks simulate their worlds. Neither is exempt from the gap between simulation and production.

AutomationBench's in-process Pydantic world is deterministic and fast, but it hand-seeds every record. Real enterprise workflows have undocumented edge cases, legacy data inconsistencies, and API rate limits that no benchmark task encodes. A model that scores 17% on AutomationBench might score lower in production (genuine tasks are noisier) or higher (the benchmark's engineered traps are denser than real workflows).

τ-bench's LLM user simulator diverges from real users in documented ways. From arXiv 2603.11245 (cited in [[taubench]]): simulated users differ from real users in turn-taking patterns, error recovery, and persona consistency. An agent trained to handle the τ-bench simulator may be over-specialized to the simulator's particular failure modes.

The sim2real gap is not a reason to dismiss either benchmark — it is a reason to treat a high score as *necessary but not sufficient*. A high AutomationBench score without online measurement is a green light to run a real trial, not a substitute for one. The same applies to τ-bench pass^k.

---

## 10. Choosing a Benchmark: A Worked Example

The comparison becomes most actionable when applied to a concrete agent. Consider the **Lina TMR sales agent** — a conversational agent that qualifies leads, manages deals, and coordinates with internal teams on behalf of a sales rep.

**What deployment shape does Lina have?**

- She talks to customers over multi-turn dialogue. → *τ-shape* for the conversation loop.
- She updates Salesforce, sends Slack notifications, and logs Google Sheets entries autonomously after calls. → *AB-shape* for the back-office execution.
- Her failure modes are: wrong-tier routing, duplicate notifications, hallucinated deal amounts, premature escalation. → *Both* benchmarks capture relevant failure signals.

**Which benchmark to run first?**

τ-bench is the better *primary* benchmark for Lina because her core loop is conversational: eliciting the customer's true situation, following a policy playbook, and avoiding destructive writes before confirmation. pass^k is directly interpretable as "will Lina succeed on the next 8 similar calls?"

**What to borrow from AutomationBench:**

The eval *engineering* is where AutomationBench contributes most. Specifically:

1. **End-state assertions over a typed world.** After each simulated Lina conversation, run assertions against a `WorldState`-equivalent: did Salesforce get the right stage, did the right team receive the Slack notification, did the amount match? This is cleaner than LLM-judged rubrics and prevents the scoring gap AutomationBench found in benchmarks that grade on text output.

2. **Must-not-occur guards.** Did Lina *not* notify the VP-Sales team for an SMB deal? Did she *not* create a duplicate opportunity? Negative assertions are cheap to write and catch reward-hacking behaviors that positive assertions miss.

3. **Deterministic seeded noise.** Seed Lina's test conversations with conflicting pricing rows, near-match company names, and a compliance-hold contact in the CRM. AutomationBench's hardening philosophy (ch-07) applies directly: difficulty should be engineered into the data, not the prompt.

4. **Pass^k as the reliability bar.** Borrow from τ-bench. A Lina that fails 30% of deal-close conversations is not deployable regardless of average pass@1.

The resulting eval stack: a τ-style LLM user simulator for the conversation + AutomationBench-style end-state assertions for the back-office writes + pass^k for the reliability verdict. This is the ch-10 deliverable — a custom eval harness that triangulates where a conversational-automator hybrid actually fails.

---

## 11. Why This Chapter Exists

The structural comparison developed in this chapter is the payoff of everything built in ch-02 through ch-08. The toolset modes (ch-02) illuminate the discovery/execution separation. Task anatomy (ch-05) reveals the policy-in-artifacts design. The assertion rubric (ch-06) shows why must-not-occur guards catch what DB-state equality cannot. The metrics chapter (ch-08) sets up the pass-rate-vs-pass^k contrast.

τ-bench is not a competitor to AutomationBench. It is a benchmark for a different class of agent — one that is conversational at its core, not autonomous. The two benchmarks are complementary: a production agent deployment that combines both a user-facing chat layer and a back-office execution layer requires both benchmark shapes to evaluate correctly.

The field's present gap is that no single benchmark covers the agent that has both. Building one — or composing existing infrastructure from each side — is the live research problem this chapter points toward. See ch-10 for a concrete attempt.

A secondary implication of the structural difference: the two benchmarks are complementary *training signals* for RL, not just evaluation tools. AutomationBench's `partial_credit` score (ch-06) gives a dense reward signal across the full assertion rubric — useful for shaping tool-selection and multi-hop coordination. τ-bench's binary `r∈{0,1}` reward on a stochastic MDP is a noisier but richer curriculum for elicitation and conversational coherence. An agent trained exclusively on one signal will overfit its respective interaction shape. The productive research direction is a joint training curriculum that uses AutomationBench partial credit for the execution backbone and τ-bench pass^k as the reliability gate for the conversational layer.

See also: [[benchmark-comparison]], [[taubench]], [[automationbench-overview]], ch-05 (task anatomy), ch-06 (grading), ch-08 (metrics), [interactive comparison](figures/ab-vs-tau.html).
