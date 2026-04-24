---
chapter: ch-47
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/bfcl.md
source_url: https://gorilla.cs.berkeley.edu/leaderboard.html
created_at: "2026-04-23"
---

# Excerpt: BFCL — AST matcher, executable eval, and four version generations

**Source library:** `wiki/raw-data/llm-training/papers/bfcl.md`
**Artifact:** Berkeley Function-Calling Leaderboard V1–V4 (2024–2025). AST-based call matching, live-API executable subset, multi-turn state, agentic long-horizon, pass^k consistency — a concrete ledger of how a harness evolves when its leaderboard saturates.

---

## Why this source grounds §4 (matchers) and §6 (versioning) of ch-47

Ch-47 §4 lists five matcher families; **AST / semantic equivalence** and **executable / unit-test** are the two families BFCL invented and operationalised for tool-calling. Ch-47 §6 argues that a harness must be version-attributable; BFCL's V1→V2→V3→V4 progression is the canonical attested example of release-generation discipline, each version bumping the matcher or the task shape in a named way.

---

## AST matcher — the §4 "semantic equivalence" family

Source §Evaluation methodology / AST matcher:

> Call matching uses an AST comparator:
> 1. Parse predicted call and gold call into (name, kwargs).
> 2. Normalize kwargs: sort by key, strip whitespace, canonicalize literals (e.g., `1.0` ≡ `1`, `"red"` ≡ `'red'`).
> 3. Name must match exactly; kwargs must be equivalent; possible args may be absent if default.

Notice the canonicalisation list — `1.0 ≡ 1`, `"red" ≡ 'red'`. These are not cosmetic; they are the difference between a 60% pass rate and an 85% pass rate for the same model on the same task, because regex / exact-match matchers count `'red'` vs `"red"` as a miss. Ch-47 §4 attributes to BFCL the claim that **AST matchers eliminate format spuriousness at the cost of language-specific parsers** — the quote above is the concrete evidence.

---

## Executable eval — the §4 "unit-test" family

Source §Evaluation methodology / Executable evaluation:

> Subset of categories has live executable APIs — BFCL runs the predicted call and checks the returned value against gold.

This is what ch-47 §4 calls **executable / unit-test**. Notice the subset-of-categories caveat: not every call can be executed (side effects, rate limits, auth). The honest harness ships *both* matchers — AST where execution is not safe, execution where it is. Ch-47's agentic-harness row in §2 ("native env state predicates") is the same pattern one level up: WebArena executes, BFCL-live executes, HumanEval executes.

---

## Relevance detection — abstention as a first-class category

Source §Evaluation methodology / Relevance detection:

> Model is penalized for calling any tool when query is unrelated; only "no call" or text response is correct.

This is a task-shape the other harnesses rarely name explicitly. It measures **hallucination at the tool-call boundary**: does the model call a tool when it shouldn't? Source §Current leaderboard snapshot attests: "even frontier models still call tools on ~10% of irrelevant queries." Ch-47 does not surface this directly, but ch-51 (capability-specific harness) will — the relevance-detection sub-score is the tool-calling analog of a *refusal* label in safety, and [[wildguard-data]]'s three-label split is the cousin pattern.

---

## pass^k — the agent-reliability metric

Source §Modality-specific technical details:

> **Pass^k metric:** from V3 onward, key agentic metric — model must succeed on all k independent trials of the same task.

Ch-47 §1 draws the pass@k vs pass^k distinction — pass@k averages, pass^k demands *every* rollout succeed. BFCL V3 introduced pass^k precisely because multi-turn agentic trajectories have high variance; a 60% pass@1 can be 10% pass^5. Ch-47's guidance "reporting pass@1 when you mean pass^k conflates capability with consistency" is a restatement of BFCL's own framing.

---

## Version generations — §6 versioning as attested release discipline

Source §Dataset size + §Abstract:

> **V1:** ~2,000 test cases.
> **V2 Live:** +adds ~1,500 real user cases (total ~3,500).
> **V3 Multi-Turn:** +adds multi-turn tasks across retail/travel/airline domains.
> **V4 Agentic:** +long-horizon tasks with web search, memory, and multiple tool servers.

Four releases, each widening the task-shape envelope. Source §Risks + gotchas adds the crucial line: "V1 ceiling has saturated; V2/V3/V4 are the meaningful evals in 2025." This is the exact scenario ch-47 §6 calls out: when a version bumps, a model's score can drop 3–8 points without any weight change, and the release note must attribute the delta to the harness, not the model.

Notice: BFCL did *not* deprecate V1. It renamed the contract so "BFCL V2 Live" is a distinct, citable number from "BFCL V1." The version id is a contract. This is the OLMES task-id discipline in a different surface.

---

## Benchmark-specific fine-tuning — the contamination gotcha

Source §Risks + gotchas:

> **Benchmark-specific fine-tuning:** some labs train directly on BFCL-style data → inflated scores. V2 Live mitigates by using unseen real queries.

Ch-47 does not cover contamination in depth (that is ch-53), but this gotcha is why V2 exists as a separate version id rather than a V1 patch — the benchmark needed a distinct, uncontaminated held-out evaluation surface. Versioning is the defence.

---

## What ch-47 keeps, changes, drops from BFCL

| BFCL design choice | Ch-47 normative claim | Reason |
|---|---|---|
| AST call matching with literal canonicalisation | AST matcher family eliminates format spuriousness | §4 matcher taxonomy |
| Executable subset on live APIs | Unit-test / executable is strongest fidelity, highest infra cost | §4 hierarchy |
| Relevance detection as first-class | Abstention is a separate capability | §5 slicing (inferred) |
| pass^k from V3 | pass@k averages, pass^k demands consistency | §1 metric choice |
| V1→V4 release progression | Version id is a contract; attribute drift to harness diff | §6 release discipline |
| AST is lenient on arg order, strict on literal form | Matchers have bias profiles you must own | §4 caveat |

---

## Connections

- **[[ch-47]]** — this excerpt grounds §1 (pass@k vs pass^k), §4 (AST + executable matchers), §6 (V1→V4 as versioning template).
- **[[excerpts/olmes]]** — OLMES task-id discipline is the static-task analog of BFCL's V1→V4 version-generation discipline.
- **[[excerpts/harmbench-data]]** — HarmBench's held-out classifier matcher is the safety analog; both treat matcher as first-class, pinned artefact.
- **[[webarena-data]] (raw-data)** — WebArena predicate matcher is the agentic analog; Docker digest is the version primitive.
- **[[ch-51]]** (downstream) — capability-specific harness deep-dive; tool-calling track reads BFCL as primary.
