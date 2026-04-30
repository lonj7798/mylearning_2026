<!-- scope: scaling under finite-data constraints and repeated-token reuse
     see-also: [[the-pile]], [[scaling-laws-data-quality]]
-->

# Data-Constrained Scaling
- **Core Insight:** Once model size or compute outpaces fresh high-quality data, repeated exposure and data reuse become unavoidable, and scaling behavior changes.
- **Guideline:** Treat token budget and unique-example budget separately; when fresh data is scarce, measure the tradeoff between more epochs over existing data and adding noisier new data.
- **Authors:** Thomas Muennighoff and collaborators
- **Year:** 2023
- **URL:** https://arxiv.org/abs/2305.16264
- **Relevant topics:** data scarcity, token reuse, scaling laws, repeated training examples

## Abstract
Data-constrained scaling studies how language-model training changes when the amount of unique high-quality data becomes a bottleneck. The core finding is that the value of repeating existing data versus adding lower-quality new data depends on where the model sits in the compute-data regime.

## Key Contributions
- Reframed scaling laws around unique-data scarcity, not just total tokens.
- Analyzed the tradeoff between seeing more unique examples and repeating old ones.
- Helped motivate later work on data quality, synthetic augmentation, and dedup.

## Technical Details
- Compares regimes with more unique data versus more repeated passes over fixed corpora.
- Practical implication: “more tokens” is ambiguous if many are repeats.
- Important for frontier training because unique high-quality corpora are finite.

## Connections
- Pairs naturally with [[the-pile]], [[dolma]], and [[fineweb]].
- Conceptually adjacent to synthetic-data scaling debates in [[phi-textbooks]] and [[hf-cosmopedia]].

