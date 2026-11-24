# User-Oriented Multi-Turn Dialogue Generation with Tool Use at Scale

<!-- scope: User-Oriented Multi-Turn Dialogue Generation with Tool Use at Scale; scope-limited research extract
     deps: [[insights-long-tail]]
     see-also: [[long-tail-papers]]
-->

- **Core Insight:** A dedicated user simulator can separate incremental user interaction from task generation.
- **Guideline:** Generate user disclosure and feedback trajectories, not only complete one-shot requests.
- **Authors:** Jungho Cho, Minbyul Jeong, Sungrae Park
- **Year:** 2026
- **URL:** [Primary source](https://arxiv.org/abs/2601.08225)
- **Relevant topics:** long-tail coverage; synthesis; evaluation

## Abstract

The framework separates task generation and user simulation for multi-turn tool-use dialogue.

## Key Contributions

- Result (single study): A dedicated user simulator can separate incremental user interaction from task generation.
- Interpretation: Generate user disclosure and feedback trajectories, not only complete one-shot requests.

## Key Figures/Tables to Study

Figures/tables not extracted. The Technical Details section identifies verified method sections where available; numerical outcome comparisons require the primary text.

## Technical Details

§5.1 uses Nemotron tools and τ2 database task seeds; cross-domain generalization remains open. Limitations report higher generation cost and partial-database brittleness.

## Limits

Synthetic interactions do not establish real customer-frequency coverage; no verified universal simulator fidelity.

## Connections

[[insights-long-tail]]; [paper catalog](../long-tail-papers.md).

## Verification

Primary abstract, §5.1 and Limitations verified on 2026-10-06; first release 2026-01-13.
