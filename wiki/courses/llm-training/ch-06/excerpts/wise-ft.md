---
chapter: ch-06
course: llm-training
phase: read
excerpt_of: "Robust fine-tuning of zero-shot models (WiSE-FT; Wortsman, Ilharco, Kim, Li, Kornblith, Roelofs, et al.)"
source_url: https://arxiv.org/abs/2109.01903
created_at: "2026-09-17"
note: "No library card exists for the slug `wise-ft` as of 2026-09-17. Every number below was read in the cached full text of arXiv:2109.01903v3."
---

# Excerpt: WiSE-FT — weight-space interpolation between the pre-fine-tuning and fine-tuned models

- **Authors:** Mitchell Wortsman, Gabriel Ilharco, Jong Wook Kim, Mike Li, Simon Kornblith, Rebecca Roelofs, et al.
- **Year:** arXiv v1 2021-09; read at v3 (2022-06-21); CVPR 2022
- **Source type:** paper. All experiments are image classification (§7 Limitations names this as a limit).

## Method (§3, Eq. 1)
1. Start from a model θ0 that is broadly capable without target-task training (here, zero-shot CLIP).
2. Fine-tune it on the target data, giving θ1.
3. Use the interpolated weights θ_α = (1 − α) · θ0 + α · θ1 for α ∈ [0, 1].

`wse(x, α) = f(x, (1 − α)·θ0 + α·θ1)`, where f is the network, x the input, and α the mixing coefficient.
α = 0 is the original model, α = 1 the fine-tuned model.

## Results used by ch-06 (Table 1; CLIP ViT-L/14@336px; top-1 %)

| Model | Reference distribution (ImageNet) | Avg. of 5 distribution shifts |
|---|---|---|
| Zero-shot | 76.6 | 73.4 |
| Fine-tuned end-to-end | 86.2 | 68.6 |
| WiSE-FT, α = 0.5 | 86.8 | 76.9 |
| WiSE-FT, optimal α | 87.1 | 77.4 |

Fine-tuning raises target accuracy by 9.6 points and lowers shift accuracy by 4.8 points relative to the
zero-shot model; interpolation at α = 0.5 recovers the shift accuracy and exceeds the fine-tuned model on the
target distribution as well.

- Six further shifts (WILDS-FMoW, WILDS-iWildCam, CIFAR-10.1, CIFAR-10.2, ImageNet-Vid-Robust, YTBB-Robust):
  α = 0.5 improves shift accuracy by 3.5, 6.2, 1.7, 2.1, 9.0 and 23.2 points over the fine-tuned model, with
  reference accuracy dropping by at most 0.3 points (§4).
- Learning rate matters more under shift than on target: for CLIP ViT-B/16, 10 epochs at LR 3·10⁻⁵ versus
  3·10⁻⁶ differ by 0.3 points on ImageNet and by up to 8 points under shift (§4, Fig. 3).
- The optimal α is 0 to 0.4 points better than α = 0.5 on average; the authors recommend α = 0.5 when there is
  no domain knowledge about the target shift (§4, App. B).

## Conditions and limits stated by the source
- Observation 1 (§5.2, Eq. 2): accuracy along the interpolation path is at or above the linear interpolation of
  the endpoint accuracies, for ImageNet and the five shifts. This holds because θ0 and θ1 share an
  initialization; interpolating independently initialized networks does not (§5.2).
- No language-model experiments, and no method for choosing α per target distribution (§7).

## Verification
- Read on 2026-09-17 in the cached full text of arXiv:2109.01903v3, §1–§7, Table 1, Figs. 3 and 6.
