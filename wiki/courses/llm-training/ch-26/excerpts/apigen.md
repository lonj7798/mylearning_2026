---
chapter: ch-26
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/apigen.md
source_url: https://arxiv.org/abs/2406.18518
created_at: "2026-04-23"
---

# Excerpt: APIGen — the three-layer verifier

**Source library:** `wiki/raw-data/llm-training/papers/apigen.md`
**Paper:** Liu, Hoang, Zhang, Zhu, Lan, Kokane et al. 2024, "APIGen: Automated Pipeline for Generating Verifiable and Diverse Function-Calling Datasets" (Salesforce AI Research, NeurIPS 2024).

---

## Why this source anchors ch-26

APIGen is the chapter's verifier-first thesis made concrete. The paper's claim — synthetic function-calling data is trustable *iff* every sample clears three independent checks — is both a methodological stance and a working pipeline that produced `xLAM-function-calling-60k`, the dataset on which Salesforce trained xLAM-7B to the #1 BFCL-V1 slot among <13B models.

Ch-26 §3 reconstructs the three-layer ablation table as the chapter's single most-cited empirical result. This excerpt walks through each layer's operational definition and the rejection-rate composition that produces the ~40% end-to-end filter.

---

## The pipeline in sequence

From source lines 23–41:

> **Step 1 — API curation:** start from ToolBench's 16K APIs but keep only the 3,673 APIs with executable reference implementations (Python mock or real endpoints under Salesforce control).
>
> **Step 2 — Seed sampling:** for each generation, sample (k=1–3) functions from the 3,673 pool. Diversity sampler weights rare API categories higher.
>
> **Step 3 — Query + solution generation:** prompt DeepSeek-Coder-V2-Instruct or GPT-4 (authors ablate both) with the sampled functions and ask for:
>   - A natural-language user query.
>   - The gold function-call sequence as structured JSON.
>
> **Step 4 — 3-layer verification:**
>   - **Format check:** JSON must parse, fields must match function schema, types enforced.
>   - **Execution check:** run the call(s) against the reference implementations; must not raise.
>   - **Semantic check:** LLM-as-judge (GPT-4) is shown the query + the call + the execution result, and must answer "yes" to "does the call correctly fulfill the query?"
>
> **Step 5 — Dedup:** MinHash on (query, call) pairs.

The 3,673-API floor is the pipeline's hard ceiling. APIGen only accepts APIs with executable reference implementations because the execution check is load-bearing (the −11 BFCL-point ablation below). This is why APIGen is narrower than ToolBench's 16K: the verifier gates the corpus size.

---

## The ablation table that justifies each layer

From source lines 52–54:

> Ablation: removing semantic check → –6% BFCL-V1; removing execution → –11%; removing format → –18%. All three layers are load-bearing.

Ch-26 §3.2 reads this table as a composition argument.

| Verifier config | BFCL-V1 overall | Δ |
|---|---|---|
| Full 3-layer (format + execution + semantic) | **88.24** | — |
| Remove semantic check | 82.2 | −6.0 |
| Remove execution check | 77.3 | −10.9 |
| Remove format check | 70.1 | −18.1 |

Three observations.

- **Format-only lands at the Glaive ceiling.** Glaive V2 ([[glaive-function-calling]]) applies only format validation and produces derivative models that top out around BFCL 70. The 70.1 "format only" APIGen number matches Glaive-trained Hermes-2-Pro (~70% BFCL-V1). *The 2023 format-only ceiling is real and measurable.*
- **Execution is the biggest single-layer lift.** Removing it costs 11 points, more than format (which was already validated) and more than semantic. This is why APIGen's 3,673-API subset is the binding constraint — Salesforce could not add more APIs without sacrificing executability, and execution is the largest single verifier contribution.
- **Semantic catches the residual 6%.** After format and execution, the surviving errors are "parses and runs but answers the wrong question" — wrong unit, wrong target, right function with wrong arg semantics. The GPT-4 judge is the only cheap way to catch these.

**The stacking order matters.** Format runs first because it's free (pure schema validation on generated JSON). Execution runs second because it's cheap (~$0.001/call in a Python sandbox). Semantic runs last because it's expensive (~$0.01/call GPT-4 judge). Cumulative acceptance decays across the stack: ~75% pass format, of those ~90% pass execution (cumulative ~68%), of those ~88% pass semantic (cumulative ~60%). Total rejection rate ≈ 40%.

---

## What the pipeline produces

From source lines 34–47:

> **Output shape:** 60,000 samples covering four data types:
>   - Simple (1 call, 1 function).
>   - Multiple (1 call, multiple candidate functions — correct one must be chosen).
>   - Parallel (≥2 calls in same turn to same function).
>   - Parallel-multiple (≥2 calls across multiple functions).
> - **Teacher model:** DeepSeek-Coder-V2-Instruct (primary) and GPT-4 (comparison).
> - **Cost / compute:** ~$8K in teacher API + ~10K GPU-hours for execution sandbox.
> - **API registry size:** 3,673 executable APIs (21 categories).
> - **Exact verification rules:**
>   - **Format:** valid JSON; required params present; types match schema (int / str / bool / enum / list).
>   - **Execution:** Python sandbox with 5 sec timeout; call must return without exception.
>   - **Semantic:** GPT-4 judge prompt requires "Yes" / "No" verdict with reasoning; only "Yes" accepted.

The four data types map directly to BFCL-V1's first four scoring categories ([[bfcl]]). This is the explicit sense in which "whoever sets the eval taxonomy sets the data-generation taxonomy" — APIGen's four types are BFCL-V1 simple / multiple / parallel / parallel-multiple.

---

## Hallucination: the number that sold xLAM

From source line 48:

> **Hallucination-rate measurement:** the 3-layer filter rejects ~40% of raw generations; post-filter, hallucination rate on BFCL-V1 is <3% for xLAM-7B (vs ~15% for ToolLLaMA).

A 5× reduction in hallucination rate is the result that moved practitioners from ToolBench/ToolLLaMA to APIGen/xLAM in the second half of 2024. The <3% number is achievable only because the execution layer removes calls that parse but reference non-existent APIs or take impossible arguments, and the semantic layer removes "right function, wrong intent" cases.

---

## Limits the paper names explicitly

From source lines 57–62:

> - **Executable-API requirement** limits scale — the pipeline is bottlenecked on having reference implementations.
> - **LLM-judge blind spots:** GPT-4 judge occasionally accepts semantically-close-but-wrong calls (e.g. wrong unit passed to a conversion function).
> - **No multi-turn:** APIGen generates single-turn function calls only. Addressed in [[apigen-mt]].
> - **License:** CC-BY-NC-4.0 — non-commercial.

Each of these maps to a later paper in the chapter. ToolACE ([[toolace]]) addresses the executable-API bottleneck with TSS self-evolution (26K APIs, mostly LLM-simulated responses). APIGen-MT ([[apigen-mt]]) addresses multi-turn. The LLM-judge blind-spot is an open problem: no paper in the chapter claims to fix it, which is why relevance-detection rates even for frontier models top out around 90%.

---

## Connections

- Direct predecessor: [[toolllm]] — provides the 16K-API substrate that APIGen curates to 3,673.
- Ablation baseline: [[glaive-function-calling]] — the format-only pipeline whose ceiling the APIGen ablation table matches at 70%.
- Downstream consumer: [[xlam]] — the model family trained on xLAM-FC-60k.
- Multi-turn extension: [[apigen-mt]] — blueprint-then-rollout adds verifiability to multi-turn.
- Broader-coverage alternative: [[toolace]] — 26K APIs via self-evolution, with a dual-layer (rule + model) verifier instead of APIGen's three.
- Evaluation target: [[bfcl]] — the benchmark APIGen's four data types are explicitly matched against.
