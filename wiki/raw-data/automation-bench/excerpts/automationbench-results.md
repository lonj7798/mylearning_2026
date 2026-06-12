<!-- scope: AutomationBench paper + leaderboard quantitative results — SOTA scores,
     toolset ablation (api / zapier / limited_zapier), failure modes, cost, efficiency
     deps: automationbench-overview, automationbench-harness, benchmark-comparison
     see-also: automationbench-tasks-grading
     fetched: 2026-06-12 from arXiv 2604.18934v1 (HTML) + zapier.com/benchmarks via web search
-->

# AutomationBench — Quantitative Results & Toolset Ablation

- **Core Insight:** The toolset ablation orders **api < zapier < limited_zapier** for every
  model — confirming tool *discovery* is a real cost. BUT the gap is **modest** (zapier→limited
  ≈ +1.5–2 pts) and even with tools handed over, the best shown is only **~14%**. So discovery
  is *not* the dominant wall; the **core orchestration** (cross-app coordination + policy +
  adversarial traps + multi-hop) is what keeps scores low. Removing search barely helps.
- **Guideline:** When a benchmark stays brutally hard even after you remove a sub-skill via
  ablation, the residual score tells you where the *real* difficulty lives. Here: hand the agent
  the right tools and it still scores ~14% → the bottleneck is applying them correctly under
  policy/noise, not finding them.
- **Source:** arXiv 2604.18934v1 "AutomationBench" (Shepard & Salimans, April 2026);
  zapier.com/benchmarks live leaderboard; Prime Intellect case study. Fetched 2026-06-12.
- **Relevant chapters:** ch-01 (why <20%), ch-02 (toolset modes), ch-08 (metrics + cost).

## Headline SOTA (paper, api/default toolset)

| Model | Score | Cost/task |
|-------|-------|-----------|
| Opus 4.7 (max) | **9.9%** | $1.80 |
| Gemini 3.1 Pro (high) | 9.6% | $0.54 |
| GPT 5.4 (high) | 7.6% | $1.93 |
| Sonnet 4.6 (max) | 5.3% | $1.81 |
| Haiku 4.5 | 1.5% | $0.18 |
| GPT 5.4 (no reasoning) | 1.2% | $0.19 |

Best in paper = **Opus 4.7 @ 9.9%** (api mode). Cost is reported alongside score: Gemini 3.1 Pro
matches Opus at ~1/3 the cost ($0.54 vs $1.80) — the cost axis changes the "winner".

## Toolset ablation (the answer to "how does limited_zapier do?")

| Model | api/default | zapier | **limited_zapier** |
|-------|-------------|--------|--------------------|
| Gemini 3.1 Pro | 9.6% | 12.8% | **14.3%** |
| Haiku 4.5 | 1.5% | 2.0% | **3.8%** |

- Direction confirmed: **limited_zapier is highest** (discovery removed). api is lowest (raw REST
  schemas are hardest to digest); zapier's curated schemas help; handing the filtered tool list
  (limited) helps most.
- **Two jumps, both small:** api→zapier (schema digestibility) +0.5–3.2 pts; zapier→limited
  (remove search entirely) only **+1.5–1.8 pts**. Removing discovery buys surprisingly little.
- Even limited_zapier tops out at **~14%** — nowhere near the 97% sanity ceiling. The hard part
  survives the ablation.

## Live leaderboard (June 2026, newer models than the paper)

GPT-5.5 (XHigh) tops Sales & Marketing; **Gemini 3.5 Flash (Medium)** tops Operations at **20.0%**.
So the live ceiling (~20% per-domain) has risen above the April paper (~10% api / ~14% limited) as
stronger models landed. Cost (USD/run) is a first-class leaderboard axis.

## Dominant failure mode (% of each model's failures)

**"Declared success while actually failing" (false confidence):** Opus **72%**, GPT 5.4 **84%**,
Gemini **91%**. Plus: data-location persistence, incomplete list processing without verification,
instruction non-compliance. This is exactly the *semantic* wrong-call that the runtime error-retry
loop cannot catch (a valid-but-wrong action returns no error) — and that limited_zapier does NOT
fix (handing over tools doesn't tell you which contact/template is correct).

## Efficiency (avg per task)

| Model | Steps | Tool calls |
|-------|-------|------------|
| Opus 4.7 | 12.6 | 29.8 |
| Gemini 3.1 Pro | 21.8 | 35.4 |
| GPT 5.4 (high) | 15.4 | 43.9 |

## Dataset & scale (paper figures)

- **600 public** (100 / domain × 6) + **600+ private** held out. **47 applications, ~500 endpoints.**
- `simple` sanity domain: **Haiku ~97%** — the harness-validity control.
- *Discrepancy note (code authoritative):* the repo's public set is 606 (sales 106) + a 200-task
  `simple` set; the paper rounds to "100 each / 600". The 47-app count **matches the code**
  (`schema/world.py`), resolving the ch-02 read.md "44" undercount.
