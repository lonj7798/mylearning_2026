---
chapter: ch-18
course: llm-training
phase: read
excerpt_of: "APIGen: Automated Pipeline for Generating Verifiable and Diverse Function-Calling Datasets (Liu et al., Salesforce, 2024)"
source_url: https://arxiv.org/abs/2406.18518
created_at: "2026-04-23"
---

# Excerpt: APIGen — the cleanest stage-4 articulation in the literature

**Authors:** Zuxin Liu, Thai Hoang, Jianguo Zhang, Ming Zhu, Tian Lan, Shirley Kokane, Juntao Tan, Weiran Yao, Zhiwei Liu, Yihao Feng, Rithesh Murthy, Liangwei Yang, Silvio Savarese, Juan Carlos Niebles, Huan Wang, Shelby Heinecke, Caiming Xiong (Salesforce AI Research)
**Year:** 2024
**Venue:** NeurIPS 2024
**arXiv ID:** 2406.18518
**Raw-data source:** [[raw-data/apigen]]

---

## Why this paper is the reference implementation of stage 4

If Self-Instruct defines the loop's shape and Nemotron-4 shows what stage 4 looks like at industrial scale, APIGen is the paper that *proves stage 4 is load-bearing* with a clean ablation. The raw-data file states the core insight:

> "Synthetic function-calling data can be trusted for SFT iff every sample passes a **3-layer verification**: (1) format check against the function schema, (2) executable check that the call runs, and (3) semantic check that an LLM-judge agrees the call satisfies the query — this eliminates the noisy-trajectory problem plaguing ToolBench."

Three layers, stacked. Every accepted sample must pass all three. Every layer has a specific job and a measured contribution. This is the template every later verification-heavy pipeline copies.

---

## The three verifier layers, and the ablation that makes them load-bearing

From the raw-data extract:

> "**Format check:** valid JSON; required params present; types match schema (int / str / bool / enum / list).
> **Execution check:** Python sandbox with 5 sec timeout; call must return without exception.
> **Semantic check:** GPT-4 judge prompt requires 'Yes' / 'No' verdict with reasoning; only 'Yes' accepted."

These three are increasingly expensive and increasingly semantic. Format is regex-cheap; execution costs a few seconds of sandbox CPU; semantic costs an LLM-judge API call. But each is checking something the others cannot.

The ablation — the single most-cited table in the paper:

> "Ablation: removing semantic check -> -6% BFCL-V1; removing execution -> -11%; removing format -> -18%. All three layers are load-bearing."

Notice: the *cheapest* layer (format) gives the largest drop when removed. This is the stage-4 lesson in miniature — cheap structural checks catch the largest volume of errors, but they do not catch *semantic* errors, and you still need the expensive layers for the last 6 points of accuracy. Removing any one degrades the downstream model.

If anyone tells you "our pipeline has verification" and means only one layer, ask them which layer and what it catches. APIGen's ablation is the empirical weapon you bring to that conversation.

---

## Mapping APIGen to the six-stage loop

- **(1) Generate.** "Sample (k=1-3) functions from the 3,673 pool. Diversity sampler weights rare API categories higher." Then prompt DeepSeek-Coder-V2-Instruct or GPT-4 for (query, gold function-call) pair.
- **(2) Filter.** Layer 1 of the 3-layer verifier: structural JSON + schema check. This is *format*, not *correctness*.
- **(3) Dedup.** "MinHash on (query, call) pairs." Cross-corpus, applied after generation.
- **(4) Verify.** Layers 2 + 3: execution + semantic judge. The heart of the paper.
- **(5) Select.** Four call-shape buckets (simple, multiple, parallel, parallel-multiple); balance by bucket at the end.
- **(6) Mix.** Single-stage SFT on Mistral-7B / Mixtral-based xLAM models.

Notice: stage 4 occupies two of the three "layers," and the third is really stage 2. The paper's framing collapses filter and verify into one "verification stack," but our six-stage decomposition separates them by cost and semantic depth.

---

## The executable-API bottleneck

APIGen's most important design constraint is one the paper is honest about:

> "Step 1 — API curation: start from ToolBench's 16K APIs but keep only the 3,673 APIs with executable reference implementations (Python mock or real endpoints under Salesforce control)."

Stage 4's execution layer is only possible because stage 1 was constrained to APIs you can *actually run*. 16K -> 3,673 is a ~77% reduction of the API pool. This is the upstream cost of downstream verifiability, and it generalises: **a modality that cannot be cheaply verified forces the pipeline to either restrict scope (APIGen's move) or move up the verifier stack to LLM-as-judge (expensive, calibration-fragile)**.

The raw-data file's "Risks + gotchas" section calls this out:

> "**Executable-API requirement** limits scale — the pipeline is bottlenecked on having reference implementations."

For ch-18: stage 4's choice of verifier propagates upstream into stage 1. You cannot just "add a verifier" to an existing pipeline; the verifier defines what stage 1 is allowed to generate.

---

## Yield rate and what it tells you about pipeline health

From the raw-data file:

> "The 3-layer filter rejects ~40% of raw generations; post-filter, hallucination rate on BFCL-V1 is <3% for xLAM-7B (vs ~15% for ToolLLaMA)."

- **Rejection rate:** 40% of raw generations fail at least one verifier layer.
- **Residual hallucination:** <3% on BFCL-V1 for the trained model.
- **Comparison:** ToolLLaMA (ToolBench-trained, no execution verification) shows ~15% hallucination — 5x higher.

Notice: APIGen's 40% rejection is not waste; **it is the dataset**. Self-Instruct's ROUGE filter also rejected ~50% of raw generations at stage 3; the difference is that APIGen's rejection is at stage 4 (correctness) and Self-Instruct's was at stage 3 (redundancy). Both rejection rates are pipeline-defining.

A healthy synthetic-data pipeline has a rejection rate at stage 4. If your stage 4 accepts everything, your verifier is broken or your generator is already aligned to the verifier's failure modes.

---

## Scale, cost, and output shape

From the raw-data extract:

> "Output shape: 60,000 samples covering four data types: Simple (1 call, 1 function); Multiple (1 call, multiple candidate functions); Parallel (>=2 calls in same turn to same function); Parallel-multiple (>=2 calls across multiple functions)."
> "Cost / compute: ~$8K in teacher API + ~10K GPU-hours for execution sandbox."

- 60K samples after 3-layer filtering (from ~100K raw, given 40% rejection).
- Each of the 3,673 APIs appears on average 16x with different argument combinations.
- Downstream: xLAM-7B hits 88.24% on BFCL-V1 (#1 among <13B models at release); xLAM-8x7B hits 88.9% (close to GPT-4).

Notice: 60K samples is a small dataset compared to Nemotron's ~1.4M or OMI-2's 14M. APIGen's bet is that **rigorously verified data beats high-volume noisy data** — the same bet as LIMA (1K examples) but for a specific, verifiable modality.

---

## What APIGen leaves to successors

The raw-data file's "Risks + gotchas" section:

> "- **LLM-judge blind spots:** GPT-4 judge occasionally accepts semantically-close-but-wrong calls (e.g. wrong unit passed to a conversion function).
> - **No multi-turn:** APIGen generates single-turn function calls only. Addressed in [[apigen-mt]]."

Two axes for future work: judge calibration (addressed in general by ch-26) and multi-turn (addressed by APIGen-MT in ch-23). Both are extensions of the same stage-4 machinery — a better judge or a trajectory-level verifier — not new loops.

---

## Why stacking verifiers in series beats one strong verifier

APIGen's three layers are in series, each of increasing semantic depth. An alternative design would be a single very expensive verifier (e.g. a large LLM-as-judge that simultaneously checks format, executability, and semantic intent). Why the stacked design?

Two reasons the paper's approach is the better engineering:

- **Early-exit economics.** Most generations fail at the cheap format layer. Running a GPT-4 judge on a generation that has malformed JSON is wasted spend. Stacking the cheap layer first lets ~18 points' worth of garbage be rejected before an expensive judge is ever called. This is standard pipeline-engineering: cheap filters first, expensive verifiers last.
- **Orthogonal error modes.** Format errors, executable errors, and semantic errors are uncorrelated — a call can be JSON-valid but throw an exception, or run cleanly but answer the wrong question. An LLM judge trained to look for "is this correct?" may notice semantic mismatch but miss a silent executor exception. Separate layers dedicated to separate error modes are more auditable than a combined verdict.

For ch-18: when designing a stage 4 for a new modality, think about **error-mode decomposition first, verifier implementation second**. APIGen's ablation is the reason — each layer catches a distinct class of errors, and the evidence is that all three are load-bearing.

## Diversity sampling: the quiet stage-1 mechanism

Most of APIGen's reputation is at stage 4, but the paper has a non-trivial stage-1 design worth highlighting:

> "Diversity sampler weights rare API categories higher."

Naive sampling from a 3,673-API pool would concentrate generations on whichever APIs the teacher finds easy to invoke. A diversity-weighted sampler re-balances this by upweighting rare API categories. The downstream measurement:

> "Dataset diversity: each of the 3,673 APIs appears on average 16x with different argument combinations."

16x average coverage across a 3,673-API pool is roughly what you need for the student to generalise to held-out APIs with similar signatures (BFCL's test APIs are held out from the 3,673 training pool). Without the diversity sampler, head categories would appear hundreds of times and tail categories zero times, producing a student that learns specific APIs rather than generalisable function-calling.

For ch-18: stage 1 and stage 3 (dedup) are dual — stage 1 upsamples rare categories, stage 3 downsamples redundant outputs. Together they produce a flat coverage distribution. This pairing is under-discussed in most pipelines but is pervasive wherever the underlying domain has a skewed prior (API categories, programming-language distribution, math-topic distribution).

## Connections

- [[excerpts/self-instruct]] — the Self-Instruct-era baseline had no stage-4 executor; APIGen is the proof of what stage 4 should look like when it can be done.
- [[excerpts/nemotron-4]] — different stage-4 philosophy: one RM instead of three independent layers. Complementary design decisions.
- [[excerpts/openmathinstruct-2]] — stage 4 via SymPy, a free and cheap modality-specific verifier. Shows that APIGen's 3-layer cost is not always necessary.
- [[excerpts/nathan-lambert-synth]] — "verifiable tasks compound" — APIGen is the cleanest empirical case.
- [[ch-18]] — parent. The 3-layer ablation is ch-18's pedagogical anchor for "stage 4 is load-bearing."
