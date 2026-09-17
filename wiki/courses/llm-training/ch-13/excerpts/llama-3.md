---
chapter: ch-13
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/model-reports/llama-3.md and llama-3-recipe.md (verified cards, 2026-09-14)
source_url: https://arxiv.org/abs/2407.21783
primary_version: arXiv:2407.21783v3 (2024-11-23); v1 2024-07
created_at: "2026-04-23"
revised_at: "2026-09-15"
---

# Excerpt: The Llama 3 Herd of Models — mixture decisions

Quotations used by ch-13 `read.md` §4, §5, §6, and Recipe, re-read in the v3 PDF text on 2026-09-15. Byline: Llama Team, AI @ Meta.

## How the pretraining mix was chosen (§3.1.2)
> "Our main tools in determining this data mix are knowledge classification and scaling law experiments."

> "Knowledge classification. We develop a classifier to categorize the types of information contained in our web data to more effectively determine a data mix. We use this classifier to downsample data categories that are over-represented on the web, for example, arts and entertainment."

> "Scaling laws for data mix. To determine the best data mix, we perform scaling law experiments in which we train several small models on a data mix and use that to predict the performance of a large model on that mix (see Section 3.2.1). We repeat this process multiple times for different data mixes to select a new data mix candidate. Subsequently, we train a larger model on this candidate data mix and evaluate the performance of that model on several key benchmarks."

> "Data mix summary. Our final data mix contains roughly 50% of tokens corresponding to general knowledge, 25% of mathematical and reasoning tokens, 17% code tokens, and 8% multilingual tokens."

No table of candidate mixes, classifier accuracy, or per-candidate results is printed.

## Changes during 405B pretraining (§3.4.1)
> "We made a several adjustments to the pre-training data mix during training to improve model performance on particular downstream tasks. In particular, we increased the percentage of non-English data during pre-training to improve the multilingual performance of Llama 3. We also upsample mathematical data to improve the model's mathematical reasoning performance, we added more recent web data in the later stages of pre-training to advance the model's knowledge cut-off, and we downsampled subsets of the pre-training data that were later identified as being lower quality."

Percentages are not printed.

## Annealing data (§3.1.3)
> "We do not include any training sets from commonly used benchmarks in our annealing data. This enables us to assess the true few-shot learning capabilities and out-of-domain generalization of Llama 3."

> "We find that annealing improved the performance of a pre-trained Llama 3 8B model on the GSM8k and MATH validation sets by 24.0% and 6.4%, respectively. However, the improvements on the 405B model are negligible, suggesting that our flagship model has strong in-context learning and reasoning capabilities and does not require specific in-domain training samples to obtain strong performance."

> "We measure the value of such datasets by annealing the learning rate of a 50% trained Llama 3 8B model linearly to 0 on 40B tokens. In those experiments, we assign 30% weight to the new dataset and the remaining 70% weight to the default data mix."

The report does not say whether 24.0% and 6.4% are absolute points or relative changes.

## Annealing and long-context stages (§3.4.2, §3.4.3)
> "We assess successful adaptation by measuring whether (1) model performance on short-context evaluations has recovered completely and (2) the model perfectly solves 'needle in a haystack' tasks up to that length. In Llama 3 405B pre-training, we increased context length gradually in six stages, starting from the original 8K context window and ending in the final 128K context window. This long-context pre-training stage was performed using approximately 800B training tokens."

> "During pre-training on the final 40M tokens, we linearly annealed the learning rate to 0, maintaining a context length of 128K tokens. During this annealing phase, we also adjusted the data mix to upsample data sources of very high quality"

No long/short ratio and no annealing mix percentages are printed.

## SFT mixture (Table 7, §4.3.4; values from [[llama-3-recipe]])
- Shares of examples: General English 52.66%, Code 14.89%, Multilingual 3.01%, Exam-like 8.14%, Reasoning and tools 21.19%, Long context 0.11%; 4.7 turns and 846.1 tokens per example on average.
- The mix is adjusted each round "to tune performance across a wide range of benchmarks"; some sources are epoched multiple times.
- Long context: SFT with only short-context data caused significant long-context regressions; 0.1% synthetic long-context data, bucketed at 16K, 32K, 64K, and 128K, optimized short- and long-context benchmarks in the authors' ablations (no table printed).
