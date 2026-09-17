---
chapter: ch-30c
course: llm-training
phase: read
excerpt_of: "Model soups: averaging weights of multiple fine-tuned models improves accuracy without increasing inference time (Wortsman et al.)"
source_url: https://arxiv.org/abs/2203.05482
created_at: "2026-09-15"
---

# Excerpt: Model soups

This excerpt stands in for the library card `model-soups`, which did not exist when ch-30c was written.
Every number below was read in arXiv:2203.05482v3 at the stated locus.

- **Authors:** Mitchell Wortsman, Gabriel Ilharco, Samir Yitzhak Gadre, Rebecca Roelofs, Raphael Gontijo-Lopes, Ari S. Morcos, et al.
- **Year:** arXiv v1 2022-03; v3 2022-07-01; ICML 2022 (39th ICML, per the PDF footer)
- **Source type:** paper

## Definitions (§2, Table 2)
- θ_i = FineTune(θ0, h_i): model fine-tuned from pre-trained θ0 with hyperparameter configuration h_i
  (optimizer, augmentation, iterations, and a seed that sets data order).
- Soup: θ_S = (1/|S|)·Σ_(i∈S) θ_i. Uniform soup: S = all models. Inference cost O(1); a logit ensemble costs O(k).
- Greedy soup (Recipe 1): sort models by held-out validation accuracy, descending; add θ_i to the ingredient set
  only if ValAcc(average(ingredients ∪ {θ_i})) ≥ ValAcc(average(ingredients)). The validation set is disjoint
  from training and test sets, so the greedy soup is no worse than the best single model on that validation set.
- Learned soup: interpolation weights optimized by gradient descent; needs all models in memory (App. I).

## Results
**CLIP ViT-B/32, 72 models from a random search over LR, weight decay, epochs, label smoothing, augmentation (Table 3):**

| Method | ImageNet | Avg distribution shifts |
|---|---|---|
| Best individual model | 80.38 | 47.83 |
| Uniform soup | 79.97 | 51.45 |
| Greedy soup | 81.03 | 50.75 |
| Greedy soup (random order, 3 orders) | 80.79 (0.05) | 51.30 (0.16) |
| Ensemble (logit ensemble) | 81.19 | 50.77 |

- The greedy soup selects 5 models for CLIP and for ALIGN (12 ALIGN models from a grid) and improves over the best
  model by 0.7 and 0.5 pp (§3.3.1).
- ViT-G/14 pre-trained on JFT-3B, 58 models: greedy soup (14 models) 90.94 ImageNet and 85.02 shift average vs best
  individual 90.78 and 84.68 (Table 1, §3.3.2).
- Text classification, 32 models per dataset from a random search over LR, batch size, epochs, seed (Table 5):
  BERT-base greedy soup vs best model: MRPC +0.0, RTE +0.7, CoLA +0.0, SST-2 +0.5; T5-base: +0.6, +0.8, +0.4, +0.1.
  The authors call these "preliminary" NLP experiments (§3.3.3).

## Conditions and limits
- A uniform soup beats the best individual model only when all individual models have high accuracy; an error
  barrier can exist between fine-tuned models, mainly at high learning rates, and the greedy soup excludes those
  models (§3.3.1; App. J.1).
- Error-landscape plots suggest that models whose displacements from θ0 form an angle closer to 90° gain more on the
  interpolation path; the authors test this correlation over seeds, learning rates and augmentations (§3.2, Figs. 2-3).
- With ImageNet-22k pre-training the greedy soup gains are smaller than with CLIP and ALIGN (§5, App. G).
- Soups do not improve calibration, unlike ensembles (§5, Fig. B.2, 20 models differing only in seed).
- The authors relate the soup-versus-ensemble gap to loss flatness and prediction confidence (§4; Interpretation).

## Verification
- Read on 2026-09-15 in the cached full text of arXiv:2203.05482v3 (PDF), §1-§5 and appendix captions cited above.
