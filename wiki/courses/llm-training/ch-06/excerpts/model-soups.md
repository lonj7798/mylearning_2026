---
chapter: ch-06
course: llm-training
phase: read
excerpt_of: "Model soups: averaging weights of multiple fine-tuned models improves accuracy without increasing inference time (Wortsman et al.)"
source_url: https://arxiv.org/abs/2203.05482
created_at: "2026-09-17"
note: "No library card exists for the slug `model-soups` as of 2026-09-17. Every number below was read in the cached full text of arXiv:2203.05482v3."
---

# Excerpt: Model soups

- **Authors:** Mitchell Wortsman, Gabriel Ilharco, Samir Yitzhak Gadre, Rebecca Roelofs, Raphael Gontijo-Lopes,
  Ari S. Morcos, et al.
- **Year:** arXiv v1 2022-03; read at v3 (2022-07-01); ICML 2022
- **Source type:** paper. All experiments are image classification or text classification, not language modelling.

## Definitions (§2, Recipe 1)
- θ_i = FineTune(θ0, h_i): the model obtained by fine-tuning a shared pre-trained initialization θ0 under
  hyperparameter configuration h_i (learning rate, weight decay, epochs, augmentation, label smoothing, seed).
- **Uniform soup:** θ_S = (1/|S|) · Σ_{i∈S} θ_i over all trained models. Inference cost is that of one model.
- **Greedy soup (Recipe 1):** sort the models by held-out validation accuracy, descending; add θ_i to the
  ingredient set only if the accuracy of the averaged ingredients does not decrease. Because the validation
  set is disjoint from training and test data, the greedy soup is by construction no worse on that validation
  set than the best single model.

## Results used by ch-06
**CLIP ViT-B/32, 72 models from a random search over learning rate, weight decay, epochs, label smoothing,
and augmentation, ImageNet top-1 % (Table 3):**

| Method | ImageNet | Avg. of 5 distribution shifts |
|---|---|---|
| Best individual model | 80.38 | 47.83 |
| Uniform soup | 79.97 | 51.45 |
| Greedy soup | 81.03 | 50.75 |
| Greedy soup, random order (3 orders) | 80.79 (0.05) | 51.30 (0.16) |

- The uniform soup is 0.41 points **below** the best individual model on ImageNet while being 3.62 points above
  it under distribution shift. The greedy soup is above the best model on both.
- ViT-G/14 pre-trained on JFT-3B, 58 fine-tuned models: greedy soup (14 ingredients) reaches 90.94 ImageNet and
  85.02 shift average, against 90.78 and 84.68 for the best individual model (Table 1, §3.3.2).
- Text classification (Table 5), 32 models per dataset: BERT-base greedy soup minus best model = +0.0 (MRPC),
  +0.7 (RTE), +0.0 (CoLA), +0.5 (SST-2); T5-base = +0.6, +0.8, +0.4, +0.1. The authors describe these NLP
  experiments as preliminary (§3.3.3).

## Conditions and limits stated by the source
- A uniform soup beats the best individual model only when the individual models are all accurate; an error
  barrier can exist between fine-tuned models, mainly at high learning rates, and the greedy procedure excludes
  those ingredients (§3.3.1, App. J.1).
- All ingredients must share the initialization θ0. Interpolating models trained from independent random
  initializations does not work (§3.2, citing prior work).
- Soups do not improve calibration, unlike output-space ensembles (§5, Fig. B.2).
- With ImageNet-22k pre-training instead of CLIP or ALIGN, the greedy-soup gains are smaller (§5, App. G).

## Verification
- Read on 2026-09-17 in the cached full text of arXiv:2203.05482v3, §1–§5 and Tables 1, 3, 5.
