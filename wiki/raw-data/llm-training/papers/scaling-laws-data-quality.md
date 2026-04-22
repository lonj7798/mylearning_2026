<!-- scope: scaling-law treatment of data quality in LM pretraining
     see-also: [[data-constrained-scaling]], [[fineweb]]
-->

# Scaling Laws Revisited: Modeling the Role of Data Quality in Language Model Pretraining
- **Core Insight:** Data quality can be treated as an explicit scaling variable, not just an anecdotal curation benefit.
- **Guideline:** When comparing data pipelines, model effective sample size and noise/deficiency explicitly instead of treating all tokens as equal.
- **Authors:** Anirudh Subramanyam, Yuxin Chen, Robert L. Grossman
- **Year:** 2025
- **URL:** https://arxiv.org/abs/2510.03313
- **Relevant topics:** scaling laws, data quality, effective sample size, noisy corpora

## Abstract
This paper extends standard language-model scaling-law thinking by adding a formal data-quality term. The central claim is that model loss should be understood as a function of model size, token count, and data quality jointly, with quality affecting the effective value of the data budget.

## Key Contributions
- Introduces an explicit quality-aware extension to Chinchilla-style scaling.
- Provides practical proxies for data quality rather than only dataset size.
- Gives a framework for trading curation effort against raw token count.

## Technical Details
- Models quality through effective sample size / deficiency-style terms.
- Evaluates how corruption or redundancy changes the useful training signal.
- Practical lesson: two corpora with the same token count can sit on different scaling curves if quality differs enough.

## Connections
- Formal counterpart to empirical filtering results in [[fineweb]], [[dolma]], and [[ccnet]].
- Pairs with [[data-constrained-scaling]] as the “token count is not enough” argument.

