<!-- scope: knowledge-capacity scaling in the Physics of Language Models line
     see-also: [[data-constrained-scaling]], [[scaling-laws-data-quality]]
-->

# Physics of Language Models: Part 3.3, Knowledge Capacity Scaling Laws
- **Core Insight:** Scaling laws should track how much factual knowledge a model can store and retrieve, not only loss or benchmark score.
- **Guideline:** When reasoning about data budgets, ask not only “how much loss drops” but “how much distinct knowledge the model can actually absorb.”
- **Authors:** Zeyuan Allen-Zhu, Yuanzhi Li
- **Year:** 2024
- **URL:** https://arxiv.org/abs/2404.05405
- **Relevant topics:** knowledge capacity, scaling laws, factual storage, model capacity

## Abstract
The Physics of Language Models line studies language-model behavior through controlled synthetic and factual settings. Part 3.3 focuses on knowledge-capacity scaling laws, arguing that model parameters constrain how much factual information can be stored and later extracted.

## Key Contributions
- Reframes scaling around stored knowledge rather than only perplexity.
- Studies factual tuple storage and retrieval in controlled settings.
- Connects model size to knowledge capacity more directly than benchmark aggregates do.

## Technical Details
- Uses factual knowledge representations rather than generic next-token loss alone.
- Measures how much knowledge survives storage and can be flexibly queried.
- Practical implication: data usefulness depends on the model’s capacity to absorb distinct facts, not only exposure volume.

## Connections
- Complements [[data-constrained-scaling]] by focusing on model-side capacity.
- Useful conceptual background for interpreting data saturation and repetition effects in pretraining.

