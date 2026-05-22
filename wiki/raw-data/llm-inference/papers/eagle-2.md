<!-- scope: dynamic draft tree improvement for EAGLE speculative sampling
     deps: eagle
     see-also: medusa, lookahead-decoding
-->

# EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees
- **Core Insight:** Draft-token acceptance is context-dependent, so speculative decoding should allocate draft-tree width dynamically instead of using a fixed tree.
- **Guideline:** Use dynamic draft trees when the proposer exposes calibrated confidence scores and verification cost should be spent on likely-to-be-accepted branches.
- **Authors:** Yuhui Li, Fangyun Wei, Chao Zhang, Hongyang Zhang
- **Year:** 2024
- **URL:** https://arxiv.org/abs/2406.16858
- **Relevant topics:** EAGLE, dynamic draft trees, context-aware speculation, acceptance prediction, lossless acceleration

## Abstract
EAGLE-2 builds on EAGLE by replacing static draft trees with context-aware dynamic trees. The paper observes that token acceptance rates vary by context, not just by tree position. Because EAGLE's draft model confidence approximates acceptance probability, EAGLE-2 uses confidence to allocate tree expansion toward more promising nodes. It reports 3.05x-4.26x speedups and 20%-40% improvement over EAGLE-1.

## Key Contributions
- Shows empirically that draft acceptance depends strongly on context.
- Uses draft-model confidence as a proxy for acceptance likelihood.
- Dynamically expands the draft tree to spend verification budget on better candidates.
- Preserves the target distribution as a lossless speculative acceleration method.
- Improves speed over EAGLE without requiring a different base target model.

## Key Figures/Tables to Study
- Acceptance-rate analysis: motivates why fixed trees waste work.
- Dynamic draft tree algorithm: explains expansion and pruning decisions.
- Confidence calibration plots: support using draft confidence as acceptance proxy.
- Main speedup table across model families/tasks.

## Technical Details
Static tree methods allocate the same structure at every step, even when the next token is obvious or highly uncertain. EAGLE-2 constructs a tree online. Nodes with higher draft confidence receive more expansion, while low-value branches are pruned or not expanded.

The verification still happens through the target LLM in parallel over tree candidates. The speedup improvement comes from increasing accepted tokens per unit of verification work and reducing wasted target computation on unlikely branches.

## Connections
- [[eagle]] supplies the feature-level drafter and baseline static tree.
- [[medusa]] also uses tree attention but with multi-head token proposers.
- [[lookahead-decoding]] is another exact parallel-decoding method, but without a trained auxiliary drafter.
- [[hf-assisted-generation]] is the simpler assistant-model API that dynamic tree methods try to outperform.
- [[speculative-decoding]] provides the older draft/verify framing.
