# GraphGen: Enhancing Supervised Fine-Tuning for LLMs with Knowledge-Driven Synthetic Data Generation

<!-- scope: GraphGen: Enhancing Supervised Fine-Tuning for LLMs with Knowledge-Driven Synthetic Data Generation; scope-limited research extract
     deps: [[insights-long-tail]]
     see-also: [[long-tail-papers]]
-->

- **Core Insight:** Source-grounded knowledge graphs can guide synthetic QA toward estimated knowledge gaps.
- **Guideline:** Ground rare factual examples in documents instead of expecting unconditioned generation to discover them.
- **Authors:** Zihong Chen et al.
- **Year:** 2025
- **URL:** [Primary source](https://arxiv.org/abs/2505.20416)
- **Relevant topics:** long-tail coverage; synthesis; evaluation

## Abstract

The method builds a knowledge graph from source text, estimates knowledge gaps, and produces atomic, aggregated, and multi-hop QA.

## Key Contributions

- Result (single study): Source-grounded knowledge graphs can guide synthetic QA toward estimated knowledge gaps.
- Interpretation: Ground rare factual examples in documents instead of expecting unconditioned generation to discover them.

## Key Figures/Tables to Study

Figures/tables not extracted. The Technical Details section identifies verified method sections where available; numerical outcome comparisons require the primary text.

## Technical Details

§4 Step 2, Eq. 4 assesses comprehension using cross-entropy on true and negated statements; the abstract describes ECE. Detailed estimator thresholds are not extracted.

## Limits

Closed-book knowledge tasks; no demonstrated sales conversation or spoken-style coverage.

## Connections

[[insights-long-tail]]; [paper catalog](../long-tail-papers.md).

## Verification

Primary abstract and §4 checked on 2026-10-06; first release 2025-05-26. Numerical training recipe not extracted.

## Full-text follow-up (2026-10-07)

§5: Qwen2.5-72B synthesizer and Qwen2.5-7B-Instruct student; SeedEval agriculture atomic QA, PQArefEval medical aggregation, HotpotEval multi-hop. Figure 3 ROUGE-F gains over best baseline: 1.08/2.70/4.73 points, not factual accuracy. Table 2 GraphGen 54,287 examples average 657.9 tokens vs LongForm 57,854 average 357.1 tokens, so length is a potential confound.

Checked primary arXiv HTML full text at the cited sections; these facts replace the earlier abstract-only scope for this extract.
