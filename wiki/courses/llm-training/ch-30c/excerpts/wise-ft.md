---
chapter: ch-30c
course: llm-training
phase: read
excerpt_of: "Robust fine-tuning of zero-shot models (Wortsman, Ilharco, et al.)"
source_url: https://arxiv.org/abs/2109.01903
created_at: "2026-09-15"
---

# Excerpt: WiSE-FT (weight-space ensembles for fine-tuning)

This excerpt stands in for the library card `wise-ft`, which did not exist in `wiki/raw-data/llm-training/`
when ch-30c was written. Every number below was read in arXiv:2109.01903v3 at the stated locus.

- **Authors:** Mitchell Wortsman, Gabriel Ilharco, Jong Wook Kim, Mike Li, Simon Kornblith, Rebecca Roelofs, et al.
- **Year:** arXiv v1 2021-09; v3 2022-06-21; CVPR 2022
- **Source type:** paper. All experiments are image classification (§7 Limitations).

## Method (§3, Eq. 1; App. A)
1. Fine-tune the zero-shot model (parameters θ0) on the target data, giving θ1.
2. For a mixing coefficient α ∈ [0, 1], use the weights (1 − α)·θ0 + α·θ1.

`wse(x, α) = f(x, (1 − α)·θ0 + α·θ1)` (Eq. 1). When only a linear classifier is fine-tuned, this equals an
output-space ensemble (§3). App. A pseudocode:

```python
def wse(model, zeroshot_checkpoint, finetuned_checkpoint, alpha):
    theta_0 = torch.load(zeroshot_checkpoint)["state_dict"]
    theta_1 = torch.load(finetuned_checkpoint)["state_dict"]
    assert set(theta_0.keys()) == set(theta_1.keys())
    theta = {
        key: (1-alpha) * theta_0[key] + alpha * theta_1[key]
        for key in theta_0.keys()
    }
    model.load_state_dict(theta)
```

## Linear mode connectivity (§5.2)
- Observation 1: Acc((1 − α)·θ0 + α·θ1) ≥ (1 − α)·Acc(θ0) + α·Acc(θ1) for all α, on ImageNet and five shifts
  (Eq. 2, Fig. 6). When endpoint accuracies are similar, Eq. 2 is the linear mode connectivity definition of
  Frankle et al.
- Interpolating two networks trained from random initialization gives no better than random accuracy (§5.2,
  citing [25]). Neyshabur et al. observed a low-error linear path when two models are fine-tuned from a shared
  initialization (§5.2).
- Observation 2: some α exceeds both endpoints (Fig. 6).

## Main table (Table 1; CLIP ViT-L/14@336px; top-1 %)
| Model | ImageNet | Avg of 5 shifts | Avg (ref, shifts) |
|---|---|---|---|
| Zero-shot (PyTorch) | 76.6 | 73.4 | 75.0 |
| Fine-tuned end-to-end (E2E) | 86.2 | 68.6 | 77.4 |
| Fine-tuned linear classifier (LC) | 85.2 | 72.6 | 78.9 |
| WiSE-FT E2E, α = 0.5 | 86.8 | 76.9 | 81.8 |
| WiSE-FT E2E, optimal α | 87.1 | 77.4 | 81.9 |

"For optimal α, we choose the single mixing coefficient that maximizes the column" (Table 1 caption).
The five shifts are ImageNet-V2, ImageNet-R, ImageNet Sketch, ObjectNet, ImageNet-A (§2).

## Other results
- Six further shifts (WILDS-FMoW, WILDS-iWildCam, CIFAR-10.1, CIFAR-10.2, ImageNet-Vid-Robust, YTBB-Robust):
  α = 0.5 improves accuracy under shift by 3.5, 6.2, 1.7, 2.1, 9.0 and 23.2 pp over the fine-tuned model,
  while reference accuracy decreases by at most 0.3 pp (§4).
- Seven transfer datasets: relative error reduced by 4 to 49% versus standard fine-tuning (§4, Table 2).
- Hyperparameters (CLIP ViT-B/16, Fig. 3): 10 epochs at LR 3·10⁻⁵ vs 3·10⁻⁶ differ by 0.3 pp on ImageNet but up
  to 8 pp under shift; LR 10⁻⁷ → 3·10⁻⁵ raises ImageNet by 5 pp and lowers shift accuracy by 8 pp (§4).
- Mixing coefficient: optimal α is 0 to 0.4 pp better than α = 0.5 on average; the authors recommend α = 0.5 when
  no domain knowledge is available (§4; App. B, Table 3).
- Beyond CLIP: for BASIC-L, α = 0.5 improves the five-shift average by over 7 pp and ImageNet by 0.4 pp (§4).

## Not reported / limits
- No language-model experiments (§7). No method for choosing α per target distribution (§7).

## Verification
- Read on 2026-09-15 in the cached full text of arXiv:2109.01903v3 (PDF, 2022-06-21), §1-§7 and App. A-B.
