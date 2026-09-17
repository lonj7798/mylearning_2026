---
chapter: ch-13
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/regmix.md (planned card; not present on 2026-09-15; values below are taken from the primary source)
source_url: https://arxiv.org/abs/2407.01492
primary_version: arXiv:2407.01492v2 (2025-01-23); v1 2024-07; ICLR 2025
created_at: "2026-09-15"
---

# Excerpt: RegMix: Data Mixture as Regression for Language Model Pre-training

Authors: Qian Liu, Xiaosen Zheng, Niklas Muennighoff, Guangtao Zeng, Longxu Dou, Tianyu Pang, et al. (Sea AI Lab, SMU, Contextual AI, Stanford University, SUTD). Source type: paper. Read in the v2 PDF text on 2026-09-15 for ch-13 `read.md` §3, §4, §7, and Recipe.

## Main claim (Abstract)
> "we train 512 models with 1M parameters for 1B tokens to fit the regression model and predict the best data mixture. Using this mixture we train a 1B parameter model for 25B tokens (i.e. 1000× larger and 25× longer) which we find performs best among 64 candidate 1B parameter models with other mixtures. Furthermore, RegMix consistently outperforms human selection in experiments involving models up to 7B models trained on 100B tokens, while matching or exceeding DoReMi using just 10% of the computational resources."

Key assumption (§1): "rank invariance of data mixtures, which posits that the relative ranking of data mixtures in terms of their impact on model performance is consistent across different model sizes and numbers of training tokens."

## Method (§3)
1. Sample mixtures from a Dirichlet distribution whose hyperparameter is the token distribution multiplied by a value from 0.1 to 5.0 (§3.1).
2. Fit linear (ridge) regression or LightGBM from domain weights to a target value (§3.2).
3. Predict over simulated mixtures ("running prediction for 1,000,000 data mixtures takes less than 10 CPU seconds") (§3.3).
4. "we select the top 100 mixtures and average them as the data mixture for large-scale training" (§3.4).
Data: the 17 available Pile domains (Table 1). Target y: validation loss on Pile-CC (§4.1).

## Rank prediction (Table 2; fit on 512 × 1M models, 1B tokens)
| Method | 1M ρ | 1M MSE | 60M ρ | 1B ρ |
|---|---|---|---|---|
| Linear | 90.08 | 0.13 | 89.26 | 88.01 |
| LightGBM | 98.45 | 0.04 | 98.64 | 97.12 |
Evaluated on 256 unseen mixtures (1M, 60M, 1B tokens each) and 64 unseen mixtures (1B models, 25B tokens). §4.2: "increasing the training tokens of the proxy models saturates after approximately 0.25B tokens"; 512 models on 0.2B tokens beat 128 models on 0.8B tokens (Figure 4).

## Spread across 64 mixtures (Table 3; 1B models, 25B tokens, average of 0- to 5-shot)
Worst vs best model per task, e.g. HellaSwag 33.0 vs 43.4, Lambada 18.9 vs 33.5, QQP 48.0 vs 59.7; average 43.7 vs 47.9.

## Web corpora (§5.2)
The validation loss on Pile-CC showed the strongest correlation with downstream performance, against the authors' prior hypothesis that Wikipedia (en) would.

## Comparison of mixtures (Table 4; 1B, 25B tokens)
| | Human | DoReMi | PPL | ODM | Pile-CC only | RegMix |
|---|---|---|---|---|---|---|
| Average performance | 45.1 | 46.8 | 46.2 | 45.0 | 46.8 | 47.3 |
| Best on | 2/14 | 0/14 | 1/14 | 0/14 | 5/14 | 7/14 |
| Estimated FLOPs to find mixture | 0 | 3.7e19 | 1.8e19 | 0 | 0 | 3.5e18 |
"Human refers to the weights put forth in The Pile". For ODM and DoReMi, "we obtain the data mixture directly from their reported best domain weights and re-normalize it across the available 17 domains. This may result in sub-optimal performance for them compared to the originally reported results." (§5.3)

## Mixing laws (§5.5)
The authors plotted 1M training logs and found a near log-log linear relationship for DM Mathematics but "more complex patterns" for most domains such as Pile-CC, and state that "data mixture effects transcend scaling laws".

## Note from a later source
[[olmix]] footnote 1: "Investigating the public RegMix code, we found their 1M implementation is closer to 15M, which our results suggest is a good proxy size."
