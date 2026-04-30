---
chapter: ch-26
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/bfcl.md
source_url: https://gorilla.cs.berkeley.edu/leaderboard.html
created_at: "2026-04-23"
---

# Excerpt: BFCL — the benchmark whose taxonomy became the data taxonomy

**Source library:** `wiki/raw-data/llm-training/papers/bfcl.md`
**Authors:** Berkeley Sky Computing Lab / Gorilla team (Patil, Zhang, Ji, Yan, Mao, Gonzalez, Stoica). V1 Feb 2024 → V4 2025.

---

## Why this source anchors ch-26

BFCL is the benchmark the whole chapter secretly optimises against. Its **ability-axis decomposition** (simple / multiple / parallel / parallel-multiple / relevance / multi-turn / multi-step) became the data-generation taxonomy for every pipeline that followed.

Ch-26 §6 argues this explicitly: APIGen's four data types, ToolACE's five complexity levels, Granite's seven capabilities are all rediscoveries of BFCL's categories. This excerpt expands the AST matcher details, the version progression, and the `pass^k` consistency metric — three technical pieces of BFCL that shape how data is made.

---

## The seven categories and how they shape data

From source lines 24–34:

> ### Scoring categories
> - **Simple:** 1 call to 1 function.
> - **Multiple:** 1 call to 1 function chosen from ≥2 candidates.
> - **Parallel:** ≥2 calls to same function in same turn.
> - **Parallel-Multiple:** ≥2 calls across multiple functions.
> - **Relevance-Detection:** user query is irrelevant to offered tools → model must refuse / not call.
> - **Live (V2+):** real user data in above categories.
> - **Multi-Turn (V3+):** sequence of turns with state mutation.
> - **Multi-Step:** single task requires several sequential calls.

Three observations about how data pipelines respond to this taxonomy.

**APIGen maps directly.** Its four data types are exactly BFCL's first four categories. Parallel-Multiple is the hardest APIGen type; it's also one of the harder BFCL categories. The generator was designed against the benchmark.

**Relevance-Detection is not covered by APIGen or ToolACE as primary targets.** Hammer ([[hammer]]) is the pipeline that explicitly addresses this with function-name masking (30%) and irrelevance augmentation (30%). The result: Hammer-7B ~90% on relevance, vs xLAM-7B ~80%.

**Multi-Turn appears only in V3+.** This is what motivated APIGen-MT in 2025. Before V3, multi-turn was under-incentivised; after V3, every serious release has a multi-turn data recipe.

---

## The AST matcher — why call representation is load-bearing

From source lines 37–41:

> ### AST matcher
> Call matching uses an AST comparator:
> 1. Parse predicted call and gold call into (name, kwargs).
> 2. Normalize kwargs: sort by key, strip whitespace, canonicalize literals (e.g., `1.0` ≡ `1`, `"red"` ≡ `'red'`).
> 3. Name must match exactly; kwargs must be equivalent; possible args may be absent if default.

This is where the chapter's "match the call template to the AST matcher" rule comes from (ch-26 §8). Two concrete implications:

- **Trained templates that need translation at inference time lose points.** Glaive V2's `<functioncall>{"name": ..., "arguments": ...}</functioncall>` XML wrapper ([[glaive-function-calling]]) requires an inference-time converter to produce the OpenAI JSON the AST matcher canonicalises directly. That conversion layer introduces edge cases (escape handling, enum quoting) that silently cost 5–10 points vs training directly on OpenAI JSON.
- **List-vs-tuple gotcha.** The matcher is lenient on argument order but strict on value canonicalisation; the paper's risk section notes that "list-vs-tuple edge cases cause spurious failures." Training data that mixes `[1,2]` and `(1,2)` for the same arg slot produces models that occasionally emit the wrong form. xLAM's tightly-controlled JSON template avoids this.

---

## The version progression and induced data targets

From source lines 48–52 (dataset size) and lines 54–59 (technical details):

> - **V1:** ~2,000 test cases.
> - **V2 Live:** +adds ~1,500 real user cases (total ~3,500).
> - **V3 Multi-Turn:** +adds multi-turn tasks across retail/travel/airline domains.
> - **V4 Agentic:** +long-horizon tasks with web search, memory, and multiple tool servers.

Ch-26 §6 reads this as a sequence of induced data targets. Each V-bump changed what data pipelines had to produce.

| Version | Key addition | Induced data recipe |
|---|---|---|
| V1 (Feb 2024) | 7 categories | APIGen four data types |
| V2 Live (Aug 2024) | Real user queries | ToolACE complexity sampler; Hammer irrelevance aug |
| V3 Multi-turn (Sep 2024) | State-mutating dialogs | APIGen-MT blueprint-then-rollout |
| V4 Agentic (2025) | Long-horizon + web/memory | SWE-Gym-style agent data ([[ch-27]]) |

The V2 Live addition is the point that killed naive "train on BFCL-style synthetic data" overfitting. V1 could be gamed by generating data that matched its templates closely. V2 Live's real-user queries don't match any single template, so a model that had memorised V1 patterns drops materially on V2.

---

## pass^k and the consistency argument

From source line 58:

> **Pass^k metric:** from V3 onward, key agentic metric — model must succeed on all k independent trials of the same task.

This is the metric APIGen-MT's blueprint-then-rollout is optimised for. A model with 70% single-trial success and low consistency lands 30% on `pass^4`; a model with 60% single-trial success and high consistency lands 45%. Blueprint-anchored training produces *structurally regular* trajectories — the training data has fewer "lucky trajectory" artefacts — which translates to smaller pass^1 → pass^4 drops.

xLAM-2-70B's reported τ-bench numbers expose the consistency gap directly: **pass^1 = 56.2%, pass^4 = 39.4%** (a 30% relative drop). Frontier proprietary models have similar or smaller drops, suggesting consistency is now the primary axis separating the top tier from the second tier.

---

## Risks — benchmark-specific overfitting

From source lines 67–71:

> - **Benchmark-specific fine-tuning:** some labs train directly on BFCL-style data → inflated scores. V2 Live mitigates by using unseen real queries.
> - **AST matcher is lenient on argument order but strict on value canonicalization** — edge cases (list-vs-tuple) cause spurious failures.
> - **V1 overfit risk:** V1 ceiling has saturated; V2/V3/V4 are the meaningful evals in 2025.
> - **Not a safety eval:** BFCL does not score harmful-tool refusal.

The V1 overfit risk is ch-26 §6's framing of "the benchmark shaped the data." ToolACE's complexity-distribution being hand-tuned to BFCL's category distribution is the closest any paper in the chapter comes to explicit overfit; V2 Live's unseen queries are the community's check on this.

**For 2025 reporting the rule is: lead with V2 Live and V3 multi-turn scores.** V1 single-turn is saturated and no longer diagnostic.

---

## The current snapshot (2025) — what the leaderboard tells you

From source lines 61–65:

> - Top proprietary: GPT-4o-class, Claude 3.7 Sonnet.
> - Top open < 13B: ToolACE-8B, xLAM-2-8B, Hammer 2.1.
> - Top open overall: xLAM-2-70B-fc-r, Llama-4-class derivatives.
> - Relevance-detection gap: even frontier models still call tools on ~10% of irrelevant queries.

The three sub-13B models are the three canonical specialists of the chapter — ToolACE (breadth via self-evolution), xLAM-2 (APIGen + APIGen-MT), Hammer (relevance via masking). None dominates on all axes, which is the argument for Granite's multi-source mix ([[granite-function-calling]]). The relevance gap — 10% hallucinated tool calls on irrelevant queries even at frontier scale — is the chapter's named open problem.

---

## Connections

- Upstream model training: [[apigen]], [[apigen-mt]], [[toolace]], [[xlam]], [[hammer]].
- Gorilla lineage: [[gorilla]] is the same Berkeley team's original API-calling model; BFCL supersedes Gorilla's APIBench evaluation.
- Ability-axis ancestor: [[api-bank]] (three-axis Call/Retrieve/Plan evaluation predates BFCL by ~10 months).
- Multi-turn complement: τ-bench (Sierra + Stanford) — often reported alongside BFCL-V3.
