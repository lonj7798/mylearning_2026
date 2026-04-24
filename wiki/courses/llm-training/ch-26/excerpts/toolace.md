---
chapter: ch-26
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/toolace.md
source_url: https://arxiv.org/abs/2409.00920
created_at: "2026-04-23"
---

# Excerpt: ToolACE — self-evolution + multi-agent dialog + dual verification

**Source library:** `wiki/raw-data/llm-training/papers/toolace.md`
**Paper:** Liu, Huang, Zeng et al. 2024, "ToolACE: Winning the Points of LLM Function Calling" (Huawei Noah's Ark + collaborators, ICLR 2025).

---

## Why this source anchors ch-26

ToolACE pushes the orthogonal axis to APIGen. APIGen optimises *verifier strictness* on a narrow 3,673-API substrate. ToolACE optimises *API diversity* — 26,507 evolved APIs across 390 domains — by giving up full executability and substituting a dual-layer verifier. The result at 8B scale is 91.41% BFCL-V1, beating xLAM-7B's 88.24% and matching GPT-4.

Ch-26 §5 frames the three modules (TSS / MAI / dual verification) as a response to APIGen's executable-API bottleneck. This excerpt expands the complexity-controller design and the ablation that shows each module earns its keep.

---

## Module 1 — Tool Self-Evolution Synthesis

From source lines 27–34:

> ### Module 1 — Tool Self-Evolution Synthesis (TSS)
> - **Seed APIs:** ~3K real-world APIs from public directories.
> - **Evolution operators:** an LLM mutates seed APIs into new APIs by:
>   - **Parameter extension:** add new input/output parameters.
>   - **Domain transfer:** port an API's pattern to a new domain (e.g., weather→stock).
>   - **Functionality refinement:** specialize a generic API into narrower sub-APIs.
> - **Filtering:** generated API schemas must parse; names must be unique; LLM-judge scores novelty and utility.
> - **Output:** 26,507 APIs covering 390 domains.

TSS is Evol-Instruct-for-APIs. The seed-to-corpus multiplier is ~9× (3K → 26K), which is the design budget: the paper chose the evolution rounds to stop when novelty/utility scores plateaued. The tradeoff is explicit — most evolved APIs have no real implementation; responses during dialog generation are produced by a tool-simulator LLM, not by executing real code.

This is the break from APIGen's constraint. APIGen refused to use an API unless it had an executable reference; ToolACE relaxes that in exchange for 10× coverage. The dual verifier (Module 3) tries to catch the resulting simulated-response errors.

---

## Module 2 — multi-agent dialog with complexity control

From source lines 36–40:

> ### Module 2 — Multi-Agent Interactive Dialog (MAI)
> - **Roles:** user-LLM, assistant-LLM, tool-simulator-LLM; each role has a distinct prompt.
> - **Complexity evaluator:** classifies target dialog into one of 5 difficulty levels (simple / multi-call / parallel / nested / info-incomplete) and conditions generation to hit a balanced distribution.
> - **Dialog generation:** role-play proceeds turn-by-turn; tool-simulator generates realistic (but not always executed) responses.
> - **Output:** ~11K dialogs spanning the 5 complexity classes.

The architectural contribution is the **complexity evaluator**. Three LLM role-players alone produce an unbalanced distribution — dialogs collapse to whatever's easiest to generate (usually simple single-call). The evaluator conditions generation on a target complexity bin and resamples until the output matches, producing the distribution ch-26 §5 lists:

| Level | Share |
|---|---|
| Simple single-call | ~30% |
| Multiple (choose right from list) | ~25% |
| Parallel (multiple calls same turn) | ~20% |
| Nested / multi-turn | ~15% |
| Info-incomplete (needs clarification) | ~10% |

**This distribution is hand-tuned to match BFCL's category distribution.** It is the explicit answer to "whoever sets the eval taxonomy sets the data-generation taxonomy." ToolACE is training against the eval, not coincidentally matching it.

---

## Module 3 — dual-layer verification

From source lines 42–46:

> ### Module 3 — Dual-Layer Verification
> - **Rule-based checks:** JSON schema validation; required parameters present; parameter types match; enum values within range; deterministic executable sanity check on a subset where simulators have Python implementations.
> - **Model-based checks:** LLM judge (GPT-4) evaluates (a) whether the user query is ambiguous, (b) whether the assistant's call satisfies the query, (c) whether the simulated tool response is consistent with the schema.
> - **Acceptance:** must pass both layers.
> - **Output shape:** final 11,300 dialogs after filtering (~40% rejection rate).

Compare to APIGen's three layers:

| Layer | APIGen | ToolACE |
|---|---|---|
| 1. Format | ✓ schema + types | ✓ schema + types + enum |
| 2. Execution | ✓ Python sandbox on all 3,673 APIs | Partial (subset with Python mocks only) |
| 3. Semantic | ✓ GPT-4 judge | ✓ GPT-4 judge, 3-way verdict |

The missing piece is full executability. APIGen runs every call; ToolACE runs only a subset. The 3-way LLM judge (query-clarity / call-correctness / response-consistency) adds a clarity dimension APIGen's Yes/No judge lacks — but does not fully compensate for the lack of execution on 90%+ of samples. Rejection rate is still ~40%, the same as APIGen, but the composition is different (more rejected by semantic, fewer by execution).

---

## Ablation — every module earns its keep

From source line 66:

> Ablation: remove TSS → –4.3% BFCL; remove MAI complexity controller → –3.1%; remove model-judge → –5.2%; remove rule-checks → –2.8%.

Totals are non-additive (the removals interact), but the take-home is that no single module dominates. Execution-equivalent of APIGen's 11-point lift — the biggest APIGen ablation result — does not appear here because ToolACE has no equivalent full-execution layer to remove. Instead the model-judge (5.2) and TSS (4.3) are the two biggest contributors, reflecting that ToolACE's quality floor is largely set by the LLM judge and the API diversity.

---

## Honest tradeoff vs APIGen

This is the framing ch-26 §5 settles on. APIGen and ToolACE optimise different axes:

- **APIGen:** 3,673 APIs, full execution verification, 60K samples, 88.24% BFCL-V1 at 7B.
- **ToolACE:** 26,507 APIs, partial execution + model judge, 11K dialogs, 91.41% BFCL-V1 at 8B.

The 3-point BFCL gap at similar model scale is real but the causal attribution is contested. Is it ToolACE's broader API coverage? Its complexity-balanced distribution? The Llama-3.1 base vs Mistral-7B base difference? The ablations suggest all three contribute.

The **production recommendation** from ch-26 §8: use both. APIGen's narrow-but-executable corpus covers the single-turn categories with ground-truth; ToolACE's broader-but-simulated corpus adds nested and info-incomplete coverage where APIGen is thin. Granite ([[granite-function-calling]]) is the explicit instance of this "blend complementary sources" recipe, and its ablation ("naive equal-mix loses 5–10 points vs tuned") confirms neither source dominates the other.

---

## Risks the paper names

From source lines 69–73:

> - **Tool-simulator hallucination:** without full executable endpoints, simulated tool responses can be unrealistic; the LLM judge partially catches this but not reliably.
> - **API "evolution" can drift:** mutated APIs sometimes reference non-existent conventions.
> - **Proprietary teacher dependency:** GPT-4 for generation AND judging — circular quality ceiling.
> - **Mostly single-turn:** multi-turn is underweighted vs [[apigen-mt]].

The simulator-hallucination risk is the structural weakness. The dual verifier catches some but not all; the 91.41 BFCL number is achieved in spite of this, and arguably could be higher if the execution layer were full rather than partial.

---

## Connections

- Contemporary sibling: [[apigen]] — stricter verification, narrower coverage.
- Multi-turn sibling: [[apigen-mt]] — blueprint-then-rollout; different approach to "verification before realism."
- Lineage of evolution operators: [[evol-instruct]] — TSS applies the same mutation philosophy to APIs rather than instructions.
- Broader mix recipe: [[granite-function-calling]] — uses ToolACE-style data alongside APIGen and Glaive.
- Evaluation target: [[bfcl]] — the category distribution ToolACE's complexity controller explicitly matches.
