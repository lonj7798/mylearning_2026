<!-- scope: coverage checklist + doc-vs-code reconciliation + gap log for course/automation-bench
     deps: [[LIBRARY]]
     see-also: [[insights]], [[wiki/courses/automation-bench/outline]]
-->

# AutomationBench — Collection Plan

Target: enough source coverage to teach **how the benchmark works end-to-end** and to
**compare it structurally with τ-bench**. Status as of 2026-06-10: research pass complete,
excerpts written, outline drafted. No open must-read gaps for v1.

## Coverage checklist

| Area | Source | Excerpt | Status |
|------|--------|---------|--------|
| Public framing, motivation, leaderboard | README, Zapier blog, arXiv 2604.18934, leaderboard | [[automationbench-overview]] | ✅ |
| Landscape positioning (Table 1 vs prior benchmarks) | arXiv 2604.18934 | [[automationbench-overview]] | ✅ |
| Episode lifecycle / runner | `automationbench/runner.py` | [[automationbench-harness]] | ✅ |
| Toolset modes (zapier / limited_zapier / api) | `runner.py`, `tools/zapier/meta.py`, `tools/api/` | [[automationbench-harness]] | ✅ |
| Tool discovery + BM25 | `utils/bm25.py`, `tools/zapier/meta.py`, `tools/api/search.py` | [[automationbench-harness]] | ✅ |
| Execute → simulated world | `tools/api/fetch.py`, `schema/world.py`, `tool_wrapper.py` | [[automationbench-harness]] | ✅ |
| Cost / pricing / usage | `pricing.py`, `usage.py` | [[automationbench-harness]] | ✅ |
| Task anatomy + domains | `domains/*/tasks.py` | [[automationbench-tasks-grading]] | ✅ |
| App/endpoint data model | `schema/*.py`, `tests/test_api_impl_*.py` | [[automationbench-tasks-grading]] | ✅ |
| Grading / assertions | `rubric/__init__.py`, `rubric/registry.py`, `tests/test_assertions.py` | [[automationbench-tasks-grading]] | ✅ |
| Noise / hardening | `domains/*/_noise.py`, `tests/test_noise.py` | [[automationbench-tasks-grading]] | ✅ |
| τ-bench / τ² / τ³ design + pass^k | arXiv 2406.12045, 2506.07982; sierra-research repos; taubench.com | [[taubench]] | ✅ |
| Head-to-head comparison | synthesis | [[benchmark-comparison]] | ✅ |

## Doc-vs-code reconciliation (code is authoritative)

| Claim in public docs | What the code actually shows | Resolve in |
|----------------------|------------------------------|------------|
| "Agents get 2 tools (Search + Execute)" | THREE toolset modes: `zapier` (2 meta-tools), `limited_zapier` (named tools filtered per task), `api` (api_search/api_fetch/base64_encode) | ch-02, ch-03 |
| "Max 50 steps" | `max_turns` default **25** in `runner.py`; the per-task system prompt mentions a 50-turn budget — the two are not the same knob | ch-02 |
| "~47 apps, ~500 endpoints" | 44 app sub-states in `schema/world.py` (~47 incl. subpackage apps); ~500 endpoints is the API-mode index order of magnitude | ch-04 |
| "600 public tasks (100/domain)" | **606** public task dicts (sales 106 + 5×100) + a separate **200**-task `simple` sanity set; 600+ private tasks held out for the leaderboard | ch-05, ch-08 |
| "No partial credit" | TRUE for the official score (`task_completed_correctly`), but `partial_credit = passed/total` is computed and used as the RL reward (rubric weight 1.0) | ch-06, ch-08 |
| "Built by Zapier" (provenance) | tasks are handcrafted Python dicts with inline hardening comments; **no synthetic-generation code ships in the repo** — provenance is the paper | ch-05 |

## Gap log

- **Private task set** is not in the repo (leaderboard-only). The public 606 + 200-simple
  are sufficient for teaching mechanics; note the split when discussing leaderboard numbers.
- **No task-generation code** in-repo — if the course wants to teach task *authoring*,
  it must reconstruct the recipe from the paper + the handcrafted examples (ch-05/ch-10).
- **Leaderboard numbers drift** (new models added continuously). Cite as "as of 2026-06"
  and prefer the paper's framing ("SOTA < 10–20%") over exact ranks.
