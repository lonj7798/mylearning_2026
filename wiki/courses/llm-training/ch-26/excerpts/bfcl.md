---
chapter: ch-26
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/bfcl.md
source_url: https://proceedings.mlr.press/v267/patil25a.html
created_at: "2026-04-23"
revised: "2026-09-15 (rewritten against the verified card for the ICML 2025 paper; the earlier category list, matcher rules, version table, and pass^k attribution were wrong)"
---

# Excerpt: BFCL — categories, matching rules, and the contamination check

**Source library:** `wiki/raw-data/llm-training/papers/bfcl.md` (verified 2026-09-14)
**Paper:** Patil, Mao, Yan, Ji, Suresh, Stoica et al., "The Berkeley Function Calling Leaderboard (BFCL): From Tool Use to Agentic Evaluation of Large Language Models", ICML 2025 (PMLR 267). The V3 multi-turn blog is dated 2024-09-19.

## Categories

- Single-turn (§3.1, App. C.2), with F the candidate function set: Simple (|F| = 1, one call), Multiple (|F| > 1, one call), Parallel (|F| = 1, several calls), Parallel Multiple (|F| > 1, several calls), Irrelevance (no call expected), Relevance (at least one call expected). Python, Java, JavaScript, REST, and SQL functions.
- Crowd-sourced ("live"): 2,251 entries (§1) from queries collected through the hosted endpoint, 2024-02-26 to 2024-04-01; Nexus leaderboard test queries excluded (App. E.2).
- Multi-turn (§3.3): Base, Missing Parameters, Missing Functions, Long Context; eight API domains; 1,000 queries (§1).
- Agentic (§3.4): Web Search, Memory, SQL.
- Total: 5,551 question–function–answer pairs (§1).

## Scoring

- AST substring matching (§4.1, App. H): exact function name; each parameter value must be in a set of possible answers; Python accepts an int for a float, Java and JavaScript do not; a float for an int is invalid everywhere; strings are case-insensitive with whitespace and listed punctuation removed; lists order-sensitive unless all permutations are listed; parallel calls matched without order, all-or-nothing.
- Multi-turn (§4.4): after each turn a state-based check and a response-based check (minimal viable call path); an entry is correct only if both pass in all turns.
- Irrelevance and missing-information entries score a clarification or no call as correct; any call counts as a hallucination (App. B, App. D.2).
- Modes (§5.1): FC mode passes definitions in the API tools field; prompting mode uses a system prompt requiring `[func(param=value), ...]`.

## Findings

- Mode sensitivity: GPT-4-turbo-2024-04-09 irrelevance 83.8 (FC) vs 35.6 (prompting) (Table 1).
- Single-turn vs stateful: gpt-4o-2024-11-20 (Prompt) 95.5 / 94.0 on AST multiple / parallel, 59.0 on multi-turn base, 6.0 on memory (Table 1).
- Contamination and overfitting check (§5.5, Tables 2–3): 6 of 7 open models have lower perplexity on crowd-sourced than on single-turn data; xLAM-7B rises from 3.67 to 5.09. The authors read a rise as a sign of memorization or tuning to the static test distribution (Interpretation).
- Real queries contain more multiple-function and fewer parallel cases than the curated set (§5.2).

## Not in the paper

pass^k (it is defined in [[tau-bench]]); a "Multi-Step" single-turn category; version release dates other than the V3 blog. Format-sensitivity results are in the BFCL V4 blog ([[bfcl-v4-format-sensitivity]]).
