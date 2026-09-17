---
chapter: ch-09
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/code-pretraining-task-performance.md (library card not present on 2026-09-15; quotations below are taken from the primary source)
source_url: https://arxiv.org/abs/2409.04556
primary_version: arXiv:2409.04556v2 (2025-02-25; v1 2024-09)
created_at: "2026-09-15"
---

# Excerpt: How Does Code Pretraining Affect Language Model Task Performance?

Verbatim quotations used by ch-09 `read.md`, with loci. Authors: Jackson Petty, Sjoerd van Steenkiste, Tal Linzen (NYU, Google Research). Checked against the v2 PDF on 2026-09-15.

## Settings (§3)
> "Each dataset, which we refer to as a 'code mixture,' is parameterized by a single value m ∈ [0, 1] representing the percentage of code in the training data [...] N_code = m · N_total, N_lang = (1 − m) · N_total."

> "competitive, in which the total amount of data is held constant while m varies, reducing the number of language tokens as the number of code tokens increases; and additive, in which the number of language tokens is held constant while the number of code tokens increases proportional to m"

> "Additive: Here, N_lang is held constant while m varies between 0% and 50%. In order to keep N_lang fixed while m varies, we increase the number of total tokens proportionally: N_total = N_lang × 1/(1 − m)."

Data: English C4 and "a version of the code portion of The Pile" from GitHub (§3).

## Models (§4.1)
> "resulting in models with roughly 374 M parameters [...] We pretrain these models with a base natural language data volume of 132 B tokens. [...] models in the additive setting were trained with N_lang = 132 B tokens, and hence N_total varying between 132 B tokens and 264 B tokens depending on the mixture [...] For each combination of code mixture and setting, we pretrain models from five different random seeds."

## Results (§5)
> "The effect is most pronounced for the structural generalization examples from COGS-vf in the competitive and additive settings (regression coefficients β̂ = 0.147 and β̂ = 0.165, respectively; this indicates that the best-fit line predicts an accuracy increase of 14.7% as the proportion of code increases from 0% to 100%)"

> "In the competitive setting, performance peaks at a code mixture between 40% and 50% and thereafter tends to decrease, though the overall trend remains positive" (multi-digit arithmetic)

> "These tasks include ones which involve purely linguistic knowledge (such as the English Passivization compositional generalization task as well as the Implicatures and Common Morpheme BigBench tasks) as well as those which involve reasoning or world-knowledge (such as the General Knowledge and Fantasy Reasoning BigBench tasks)."

## Permutation tests (§5.1, Fig. 8; competitive, multiple choice)
> "We find a statistically significant difference of variance (p = 0.0002) and upper-quartiles (p = 0.006) at a significance level of α = 0.05 [...] Other statistics measured were not significant at this significance level."

Fig. 8 prints p = 0.1038 for the difference of means.

## Discussion (§6)
> "We hypothesize that the deleterious impact of code on tasks involving linguistic or real-world knowledge comes from a reduction in linguistically-relevant inductive biases as models see less natural language data (either in an absolute sense in the competitive setting or a relative sense in the additive setting)."

> "we do not explore how post-training objectives like instruction tuning (Wei et al., 2022) or RLHF (Ouyang et al., 2022) interact with code mixture in pretraining"

## Limitations (§6.1)
> "We survey relatively small models (374 M parameters) [...] We also only consider pretraining corpora of between 132 B and 264 B tokens."
