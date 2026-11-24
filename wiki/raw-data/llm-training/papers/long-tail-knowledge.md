# Large Language Models Struggle to Learn Long-Tail Knowledge

<!-- scope: Large Language Models Struggle to Learn Long-Tail Knowledge; scope-limited research extract
     deps: [[insights-long-tail]]
     see-also: [[long-tail-papers]]
-->

- **Core Insight:** Relevant-document frequency predicts closed-book factual QA accuracy.
- **Guideline:** Separate rare facts from rare interaction patterns and test retrieval for missing facts.
- **Authors:** Nikhil Kandpal, Haikang Deng, Adam Roberts, Eric Wallace, Colin Raffel
- **Year:** 2023
- **URL:** [Primary source](https://proceedings.mlr.press/v202/kandpal23a.html)
- **Relevant topics:** long-tail coverage; synthesis; evaluation

## Abstract

QA accuracy is stratified by relevant pretraining documents; counterfactual corpus removal and retrieval probe frequency dependence.

## Key Contributions

- Result (single study): Relevant-document frequency predicts closed-book factual QA accuracy.
- Interpretation: Separate rare facts from rare interaction patterns and test retrieval for missing facts.

## Key Figures/Tables to Study

Use the primary source for experiment tables; this card makes no numerical outcome claim.

## Technical Details

Relevant documents contain question and answer entities. See §2–§4 and Figs. 5–9; the scaling extrapolation is not a measured giant-model result.

## Limits

Factual QA, not conversation style, turn-taking, or rare customer behavior.

## Connections

[[insights-long-tail]]; [paper catalog](../long-tail-papers.md).

## Verification

Existing verified ch-12a/ch-23 excerpts of arXiv:2211.08411v2 (2026-09-15); bibliographic link is ICML2023.
