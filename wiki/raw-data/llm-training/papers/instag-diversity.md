<!-- scope: InsTag diversity and complexity as a data-selection signal
     deps: [[instag]]
     see-also: [[deita]], [[cherry-llm]]
-->

# InsTag Diversity
- **Core Insight:** Measured diversity in tag space is a better proxy for SFT usefulness than raw dataset size.
- **Guideline:** Use instruction tags to track coverage and oversampled modes; downselect toward wide tag coverage plus higher complexity rather than indiscriminate scale.
- **Authors:** Same line as [[instag]]
- **Year:** 2023
- **URL:** https://arxiv.org/abs/2308.07074
- **Relevant topics:** tag-space diversity, SFT data selection, complexity metrics

## Abstract
The diversity-focused reading of InsTag is that instruction datasets should be judged by tag coverage and task complexity, not only by number of examples. This gives a practical way to reason about why small curated SFT sets sometimes outperform much larger pools.

## Key Contributions
- Operationalized diversity through fine-grained tags.
- Connected tag coverage to downstream SFT quality.
- Provided a practical selector for building stronger small subsets.

## Technical Details
- Tag the instruction pool.
- Score candidate subsets by coverage and complexity.
- Fine-tune on selected subsets rather than the full raw corpus.

## Connections
- Directly related to [[instag]].
- Sits beside [[deita]] and [[ifd]] in the data-selection lineage.

