<!-- scope: instruction tagging for analyzing and selecting SFT data
     see-also: [[instag-diversity]], [[deita]]
-->

# InsTag: Instruction Tagging for Analyzing Supervised Fine-Tuning of Large Language Models
- **Core Insight:** Diversity and complexity in SFT data become measurable if you first tag instructions with a large open-ended tag vocabulary.
- **Guideline:** Before selecting or pruning SFT data, tag it semantically and by intent; use the tag distribution to reason about diversity and complexity rather than raw prompt count.
- **Authors:** Keming Lu, Hongyi Yuan, Zheng Yuan, Runji Lin, Junyang Lin, Chuanqi Tan, Chang Zhou, Jingren Zhou
- **Year:** 2023
- **URL:** https://arxiv.org/abs/2308.07074
- **Relevant topics:** instruction tagging, diversity measurement, SFT selection

## Abstract
InsTag proposes an open-set tagging framework for SFT datasets and uses those tags to analyze instruction diversity and complexity. The paper shows that selecting a small but diverse and complex subset can outperform much larger raw SFT datasets.

## Key Contributions
- Built a fine-grained instruction tagger.
- Defined diversity and complexity in tag space rather than by dataset size alone.
- Showed strong results from small selected subsets.

## Technical Details
- Tag vocabulary spans thousands of semantic and intent categories.
- Analyze open SFT datasets through tag coverage and tag complexity.
- Use the tagger as a selector to build stronger small training subsets.

## Connections
- The analysis layer behind [[instag-diversity]].
- Closely related to [[deita]] and other quality/diversity selection work.

