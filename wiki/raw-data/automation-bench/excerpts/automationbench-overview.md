<!-- scope: AutomationBench public framing, motivation, landscape position, results
     see-also: automationbench-harness, automationbench-tasks-grading, benchmark-comparison
-->

# AutomationBench — Overview, Motivation, and Landscape Position

- **Core Insight:** Real enterprise agentic work is *cross-application coordination +
  autonomous API discovery + policy adherence*, all at once — and no prior tool-call
  benchmark combined the three, so frontier models score <20%.
- **Guideline:** Grade on **outcomes (final world state), not output (text)**, and make the
  agent *discover* which tools to use rather than handing it the right ones.
- **Source:** README.md; Zapier blog "Introducing AutomationBench"; arXiv 2604.18934
  (Shepard & Salimans, 2026-04-21); leaderboard zapier.com/benchmarks; CC BY 4.0.
- **Relevant chapters:** ch-01, ch-08, ch-09.

## What it is

A benchmark for **AI agents on realistic business workflows**: given a natural-language
trigger (a Slack message, an email, a request), the agent must autonomously discover REST
endpoints, chain multi-step calls across multiple SaaS apps, follow business-policy
constraints, and leave the correct final state in the right systems. Zapier built it
internally to decide which models to deploy in production, found no public benchmark
adequate, and open-sourced it. Its substrate is Zapier's real catalog (9,000+ app
integrations, 66,000+ triggers/actions, ~2B monthly tasks) abstracted into 47 simulated
apps and ~500 endpoints across six high-frequency business domains.

## The gap thesis (paper Table 1)

| Benchmark | Cross-app | API discovery | End-state grading | Business rules |
|-----------|-----------|---------------|-------------------|----------------|
| WebArena / Mind2Web | ✗ | ✗ | ✗ | ✗ |
| ToolBench / API-Bank | ✗ | retrieval-assisted | varies | ✗ |
| AppWorld | ✗ (single env) | ✗ | ✓ | ✗ |
| τ³-bench | ✗ (single app) | partial (banking) | ✓ | ✓ |
| **AutomationBench** | **✓** | **✓ (BM25)** | **✓** | **✓** |

The tripartite gap — *cross-application coordination + autonomous API discovery + policy
adherence* — is the benchmark's reason to exist. The sharpest stated contrast is to
τ³-bench: τ tasks live inside a single application; AutomationBench tasks span several.

## Dataset shape

- **606 public tasks** across Sales / Marketing / Operations / Support / Finance / HR
  (sales 106, the others 100 each) + a **200-task `simple`** sanity set (harness control,
  not scored). **600+ private tasks** are held out for the official leaderboard.
- Tasks are **synthetically generated from the *shapes* of real customer workflows**
  (plus negative feedback on Zapier's Agents service) — no PII, no raw customer data — then
  hardened with distractors, ambiguous data, decoy records, and strict policy constraints.
  (No generation code ships in the repo; tasks appear as handcrafted Python dicts.)

## Results (as of 2026-06)

- SOTA pass-rate is **~12–17%** (leaderboard) — the paper's headline at submission was
  "all SOTA models score **below 10%**." Cost-per-task is reported alongside score.
- Even small models hit **~97% on the `simple` domain**, confirming low main-benchmark
  scores reflect genuine orchestration difficulty, not a broken harness.
- Dominant failure mode: **false confidence in incorrect tool calls** (72–91% of failures),
  plus incomplete data retrieval and acting before collecting all list items.

## Practical facts

CC BY 4.0; Python (uv-managed); runs via `uv run auto-bench --model <m>`; supports
OpenAI/Anthropic/custom endpoints; ships a web visualizer; built on the `verifiers`
library and the Prime Intellect Environments Hub. Run-to-run variance typically <1%.

## Connections

- Mechanism detail → [[automationbench-harness]] (how a run executes) and
  [[automationbench-tasks-grading]] (how a task is defined and scored).
- The "outcomes not output" + end-state grading philosophy is the hinge of the
  [[benchmark-comparison]] with [[taubench]].
