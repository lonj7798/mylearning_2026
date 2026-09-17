---
chapter: ch-09
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/model-reports/llama-3.md and llama-3-recipe.md (cards verified 2026-09-14)
source_url: https://arxiv.org/abs/2407.21783
primary_version: arXiv:2407.21783v3 (2024-11-23; v1 2024-07)
created_at: "2026-04-23"
revised_at: "2026-09-15"
---

# Excerpt: The Llama 3 Herd of Models — pretraining data mix

Quotations from the Llama 3 report used by ch-09 `read.md`, matching the verified library cards. Byline: Llama Team, AI @ Meta.

## Web data filtering (§3.1.1)
> "These include using fast classifiers such as fasttext (Joulin et al., 2017) trained to recognize if a given text would be referenced by Wikipedia (Touvron et al., 2023a), as well as more compute-intensive Roberta-based classifiers (Liu et al., 2019a) trained on Llama 2 predictions."

> "Similar to DeepSeek-AI et al. (2024), we build domain-specific pipelines that extract code and math-relevant web pages."

## Determining the data mix (§3.1.2)
> "Our main tools in determining this data mix are knowledge classification and scaling law experiments."

> "Knowledge classification. We develop a classifier to categorize the types of information contained in our web data to more effectively determine a data mix. We use this classifier to downsample data categories that are over-represented on the web, for example, arts and entertainment."

> "Scaling laws for data mix. To determine the best data mix, we perform scaling law experiments in which we train several small models on a data mix and use that to predict the performance of a large model on that mix (see Section 3.2.1). We repeat this process multiple times for different data mixes to select a new data mix candidate. Subsequently, we train a larger model on this candidate data mix and evaluate the performance of that model on several key benchmarks."

> "Data mix summary. Our final data mix contains roughly 50% of tokens corresponding to general knowledge, 25% of mathematical and reasoning tokens, 17% code tokens, and 8% multilingual tokens."

## Annealing as data evaluation (§3.1.3)
> "We measure the value of such datasets by annealing the learning rate of a 50% trained Llama 3 8B model linearly to 0 on 40B tokens. In those experiments, we assign 30% weight to the new dataset and the remaining 70% weight to the default data mix."

## Mix changes during training (§3.4.1)
> "We made a several adjustments to the pre-training data mix during training to improve model performance on particular downstream tasks. In particular, we increased the percentage of non-English data during pre-training to improve the multilingual performance of Llama 3. We also upsample mathematical data to improve the model's mathematical reasoning performance, we added more recent web data in the later stages of pre-training to advance the model's knowledge cut-off, and we downsampled subsets of the pre-training data that were later identified as being lower quality."

## Scale
Llama 3.1 405B was pre-trained on 15.6T tokens with 3.8 × 10^25 FLOPs (Abstract, §1; card).

## Not reported
Per-source shares, the definition of "general knowledge", the list of knowledge-classifier categories beyond the example given, the results of the mix scaling-law experiments, and the percentages of the mix adjustments during training.

## Correction to earlier ch-09 material
The April 2026 version of ch-09 stated that the Llama 3 report gives "token total only; no mix". §3.1.2 gives the final category mix and the method used to choose it.
