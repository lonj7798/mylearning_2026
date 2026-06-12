<!-- qa for ch-01 — The Agentic Tool-Call Benchmark Landscape — see [[read]]
     Index of clarifying questions raised while reading. Kernel answers only;
     full causal chains live in read.md / discuss transcript. Append-only. -->

# ch-01 Q&A

## Q1. Does AutomationBench provide all ~400 tools to the model every single turn?

**No.** The ~400 is the size of the *hidden catalog*, not the per-turn tool schema. What
the model sees each turn is set by `setup_state` (`runner.py`) and depends on `--toolset`:

| Mode | Tools in the model's schema per turn |
|------|--------------------------------------|
| `zapier` (default, discovery-required) | **2 meta-tools** only: `search_tools` + `execute_tool`. The 400 named tools are *never* registered as model-facing tools (`# NOT in ALL_TOOLS`); they live in a `ToolRegistry` singleton, BM25-indexed, reachable only by name via `execute_tool`. `search_tools` itself returns only `top_k` (default 20) hits per query — a ranked slice, never the full catalog. |
| `limited_zapier` | The task's own allowlist `info["zapier_tools"]` (a handful); empty array if the task names none. The ablation that isolates *execution* from *discovery*. |
| `api` | **3** generic REST tools: `api_search`, `api_fetch`, `base64_encode`. |

**Why it matters (the design point):** putting all 400 schemas in context every turn would
(1) collapse *discovery* — the benchmark's core thesis (read.md §3.2) — into mere selection-
from-a-candidate-set, which is the already-cleared BFCL bar (§1); and (2) benchmark context
window instead of reasoning. The harness even *compresses* old `search_tools` results once
`execute_tool` is called (`runner.py:240`) to keep the per-turn tool surface minimal. The
model is given a *way to find* tools, never the tools themselves. This is the single choice
separating AutomationBench from every "function selection" benchmark in the §1 timeline.

Source: `automationbench/runner.py` (setup_state L159-181, use_meta_tools L73-88),
`tools/zapier/meta.py` (ToolRegistry + make_search_tools), `tools/__init__.py:1921`.
Confirms read.md §3.2 ("the agent receives exactly two tools").

## Q2. In `api` mode, is the model *creating* tools? Is tool-search returning two formats?

**Not creating tools** — it still uses 3 fixed harness tools (`api_search`/`api_fetch`/
`base64_encode`). `zapier` and `api` are two *façades over the same in-process world-mutation
functions*: `api_fetch` is "a thin routing layer over existing Zapier tools... no logic is
duplicated" (`tools/api/fetch.py:4-8`) — it parses the URL and calls the same `route_*`
function `execute_tool` would. What differs is the **abstraction skin**: named-function
(`execute_tool(name, args)`) vs raw-HTTP (`api_fetch(method, url, body)` — model builds the URL).

The two search returns are matched to their paired executor's input: `search_tools` →
`execute_tool` needs a **name**, so it returns names; `api_search` → `api_fetch` needs a **URL**,
so it returns a constructed `url`. They are *different functions over different indexes* (tool
docstrings vs REST endpoint schemas), mode-locked per run — not one function with two outputs.
`limited_zapier` has **no** search at all (named tools handed directly = the discovery-off
control). `api_fetch`'s huge routing table even tolerates hallucinated domains
(`# models hallucinate this domain`) on purpose — to grade *endpoint discovery*, not URL-string
memorization. Source: `runner.py` L56-88/159-181, `tools/api/{fetch,search}.py`.

## Q3. read.md §3.3 says the system prompt tells the agent "not to ask clarifying questions" — is that a tool-search method?

**No — orthogonal to BM25/search.** BM25 = *which tool to call*; "don't ask" = *how to resolve
ambiguity: by investigating the world, not asking a human*. The real prompt
(`domains/*/tasks.py`, e.g. `sales/tasks.py:31-37`):

> *"Do not ask clarifying questions - use the information provided and make reasonable
> assumptions when needed."*

AutomationBench has **no user** (unlike τ-bench's user simulator); a single trigger, agent runs
to completion alone — so there is no one to ask. This is *essential* for policy adherence
because policy is **buried in the seeded world** (e.g. `hr/tasks.py:1125` "ONLY the HR Director
can authorize PTO cap resets"; `sales/tasks.py:26724` "DO NOT CONTACT POLICY"). Forbidding
clarifying questions stops the agent from offloading the hard part (find+apply buried policy)
onto a human → makes adherence a *discovery + application* test, not conversational elicitation.
"Reasonable assumptions" is a trap: scope-creep / compliance-hold / recency seeds make the
surface-level assumption wrong, forcing world-grounded reasoning. Design payoff: no user → full
determinism (<1% variance) + closes the "ask to dodge" escape hatch. The prompt's tail
("handle exclusions silently in the action, not narratively") is the same anti-gaming spine as
ch-06 free-assertion exclusion: grading reads world state, not prose — *saying ≠ doing*.
