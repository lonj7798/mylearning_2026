# ConsistentChat: Building Skeleton-Guided Consistent Multi-Turn Dialogues for Large Language Models from Scratch

<!-- scope: ConsistentChat: Building Skeleton-Guided Consistent Multi-Turn Dialogues for Large Language Models from Scratch; scope-limited research extract
     deps: [[insights-long-tail]]
     see-also: [[long-tail-papers]]
-->

- **Core Insight:** Intent and information-flow skeletons guide coherent multi-turn synthesis.
- **Guideline:** Plan conversation structure while preserving causal inference-time inputs.
- **Authors:** Jiawei Chen et al.
- **Year:** 2025
- **URL:** [Primary source](https://arxiv.org/abs/2506.03558)
- **Relevant topics:** long-tail coverage; synthesis; evaluation

## Abstract

The method generates a sequence of user queries from a skeleton, then generates responses in one pass.

## Key Contributions

- Result (single study): Intent and information-flow skeletons guide coherent multi-turn synthesis.
- Interpretation: Plan conversation structure while preserving causal inference-time inputs.

## Key Figures/Tables to Study

Use the primary source for experiment tables; this card makes no numerical outcome claim.

## Technical Details

§3: intent/flow scaffold; §4: SFT and consistency evaluation. The generator can see future queries during response generation.

## Limits

Future-query access differs from deployment; generator also participates in judging. No measured population rarity.

## Connections

[[insights-long-tail]]; [paper catalog](../long-tail-papers.md).

## Verification

Existing ch-29b excerpt checked against arXiv v2 on 2026-09-15; ingested on 2026-10-06.

## Full-text follow-up (2026-10-07)

§4 Tables 1–2 (arXiv v2): approximately 15,000 training dialogues; Qwen2.5-7B consistency 7.32 vs ShareGPT 7.10, MT-Eval multi-turn 8.38 vs 7.86. These are judged scores, not accuracy percentages. Verified full text on 2026-10-07.
