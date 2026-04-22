<!-- scope: tool-calling eval — BFCL benchmark methodology for function-calling LLMs (V1 → V4 2025)
     deps: [[apigen]]
     see-also: [[toolace]], [[xlam]], [[apigen-mt]]
-->

# Berkeley Function-Calling Leaderboard (BFCL)
- **Core Insight:** Evaluating function calling requires a multi-category benchmark that separately scores (a) whether the model picks the right function, (b) whether it fills parameters correctly, (c) whether it handles multi-step/parallel calls, and (d) whether it refuses irrelevant calls; BFCL formalizes this and has gone through four versions (V1 single-turn → V4 agentic long-horizon).
- **Guideline:** When training a function-calling model, evaluate on at least BFCL-V2 Live (real user queries with ground-truth gold calls) and BFCL-V3 Multi-turn; single-turn-only evals over-report capability.
- **Authors:** Berkeley Sky Computing Lab / Gorilla team (Shishir G. Patil, Tianjun Zhang, Charlie Cheng-Jie Ji, Fanjia Yan, Huanzhi Mao, Joseph E. Gonzalez, Ion Stoica)
- **Year:** 2024 (V1 Feb), V2 Aug, V3 Sep, V4 (2025 agentic)
- **URL:** https://gorilla.cs.berkeley.edu/leaderboard.html ; https://gorilla.cs.berkeley.edu/blogs/13_bfcl_v3_multi_turn.html
- **Relevant topics:** function-calling evaluation, tool use, agentic evaluation, Gorilla

## Abstract
BFCL is the standard leaderboard and methodology for evaluating function-calling in LLMs, introduced Feb 2024 by the Berkeley Gorilla team. V1 covers simple single-turn function calls. V2 introduces "Live" data from real user queries and a relevance-detection category. V3 (Sep 2024) adds multi-turn and multi-step. V4 (2025) extends to agentic long-horizon with web/memory tools. BFCL is both a benchmark (data) and a methodology (scoring categories + executors).

## Key Contributions
- **7 core evaluation categories** (V1): simple, parallel, multiple, parallel-multiple, relevance-detection, chat, Java/JS-specific.
- **AST-based call matching:** compares predicted tool call against gold by normalizing whitespace, argument order, and literal representations — tolerates equivalent formulations.
- **Live dataset (V2):** real user queries sourced from the Gorilla community — eliminates synthetic-data overfitting concerns.
- **Multi-turn evaluation (V3):** stateful environments where each call updates environment state.
- **Agentic V4 (2025):** long-horizon tasks with web and memory tools, measuring pass^k consistency.

## Evaluation methodology (REQUIRED — tool-calling)

### Scoring categories
- **Simple:** 1 call to 1 function.
- **Multiple:** 1 call to 1 function chosen from ≥2 candidates.
- **Parallel:** ≥2 calls to same function in same turn.
- **Parallel-Multiple:** ≥2 calls across multiple functions.
- **Relevance-Detection:** user query is irrelevant to offered tools → model must refuse / not call.
- **Live (V2+):** real user data in above categories.
- **Multi-Turn (V3+):** sequence of turns with state mutation.
- **Multi-Step:** single task requires several sequential calls.

### AST matcher
Call matching uses an AST comparator:
1. Parse predicted call and gold call into (name, kwargs).
2. Normalize kwargs: sort by key, strip whitespace, canonicalize literals (e.g., `1.0` ≡ `1`, `"red"` ≡ `'red'`).
3. Name must match exactly; kwargs must be equivalent; possible args may be absent if default.

### Executable evaluation
Subset of categories has live executable APIs — BFCL runs the predicted call and checks the returned value against gold.

### Relevance detection
Model is penalized for calling any tool when query is unrelated; only "no call" or text response is correct.

## Dataset size
- **V1:** ~2,000 test cases.
- **V2 Live:** +adds ~1,500 real user cases (total ~3,500).
- **V3 Multi-Turn:** +adds multi-turn tasks across retail/travel/airline domains.
- **V4 Agentic:** +long-horizon tasks with web search, memory, and multiple tool servers.

## Modality-specific technical details (REQUIRED — tool-calling)
- **API registry size:** BFCL covers ~2,000 unique API signatures spanning general-purpose tools.
- **Exact verification rules:** AST equivalence + (where applicable) executable check. Relevance detection checks absence of tool call.
- **Hallucination-rate measurement:** relevance-detection sub-score directly measures false-positive tool calls.
- **Pass^k metric:** from V3 onward, key agentic metric — model must succeed on all k independent trials of the same task.
- **Language coverage:** Python, Java, JavaScript function definitions.

## Current leaderboard snapshot (2025)
- Top proprietary: GPT-4o-class, Claude 3.7 Sonnet.
- Top open < 13B: ToolACE-8B, xLAM-2-8B, Hammer 2.1.
- Top open overall: xLAM-2-70B-fc-r, Llama-4-class derivatives.
- Relevance-detection gap: even frontier models still call tools on ~10% of irrelevant queries.

## Risks + gotchas
- **Benchmark-specific fine-tuning:** some labs train directly on BFCL-style data → inflated scores. V2 Live mitigates by using unseen real queries.
- **AST matcher is lenient on argument order but strict on value canonicalization** — edge cases (list-vs-tuple) cause spurious failures.
- **V1 overfit risk:** V1 ceiling has saturated; V2/V3/V4 are the meaningful evals in 2025.
- **Not a safety eval:** BFCL does not score harmful-tool refusal.

## Connections
- Upstream model training: [[apigen]], [[apigen-mt]], [[toolace]], [[xlam]], [[hammer]].
- Gorilla lineage: [[gorilla]] (original API-calling model + retriever from Berkeley).
- Multi-turn complement: τ-bench (Sierra + Stanford) — often reported alongside BFCL-V3.
