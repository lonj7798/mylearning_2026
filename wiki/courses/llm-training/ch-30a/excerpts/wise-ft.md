---
chapter: ch-30a
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/wise-ft.md (planned card; not present on 2026-09-15)
source_url: https://arxiv.org/abs/2109.01903
created_at: "2026-09-15"
---

# Excerpt: Robust fine-tuning of zero-shot models (WiSE-FT)

**Authors:** Mitchell Wortsman, Gabriel Ilharco, Jong Wook Kim, Mike Li, Simon Kornblith, Rebecca Roelofs, et al. (University of Washington; OpenAI; Columbia University; Google Research; Allen Institute for AI; Toyota Research Institute)
**Version read:** arXiv:2109.01903v3 (21 Jun 2022); v1 September 2021.
**Status:** no library card existed for this slug on 2026-09-15; the equation and Table 1 were checked against the v3 PDF text. The full treatment of weight averaging is in ch-30c.

## Method (§3, Eq. 1)
"For a mixing coefficient α ∈ [0, 1], we consider the weight-space ensemble between the zero-shot model with parameters θ_0 and the model obtained via standard fine-tuning with parameters θ_1":
wse(x, α) = f(x, (1 − α) · θ_0 + α · θ_1).
"we find that the zero-shot and fine-tuned models are connected by a linear path in weight-space along which accuracy remains high."

## Table 1 (CLIP ViT-L/14@336px; ImageNet reference and average over five shifts)
| Model | ImageNet | Avg. shifts |
|---|---|---|
| Zero-shot (PyTorch) | 76.6 | 73.4 |
| Fine-tuned end-to-end (ours) | 86.2 | 68.6 |
| WiSE-FT end-to-end, α = 0.5 | 86.8 | 76.9 |
| WiSE-FT end-to-end, optimal α | 87.1 | 77.4 |

## Statements used in ch-30a
- §4: "α=0.5 yields close to optimal performance across a range of experiments. Hence, we recommend α=0.5 when no domain knowledge is available."
- §4, hyperparameters (Fig. 3, which shows CLIP ViT-B/16, not the ViT-L/14@336px model of Table 1): "while training for 10 epochs with learning rate 3 · 10⁻⁵ and 3 · 10⁻⁶ lead to a small accuracy difference on ImageNet (0.3 pp), accuracy under distribution shift varies by as much as 8 pp."
- §4: "moving from small to moderate learning rates (10⁻⁷ to 3 · 10⁻⁵) improves performance on ImageNet by 5 pp, it also deteriorates accuracy under distribution shift by 8 pp."
- Abstract: WiSE-FT "improves accuracy under distribution shift by 4 to 6 percentage points (pp) over prior work while increasing ImageNet accuracy by 1.6 pp."

## Limit relevant to LLMs
All experiments are image classification with CLIP, ALIGN, BASIC, and a JFT-pretrained ViT. The paper does not test language models. LLM evidence for interpolation toward the pre-fine-tuning weights comes from [[mitigating-alignment-tax-rlhf]].

## How ch-30a uses it
§1 (target-distribution validation does not reveal robustness changes), §5.2 (learning rate), §5.5 (weight interpolation), Recipe row.
