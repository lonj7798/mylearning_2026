---
chapter: ch-01
course: llm-training
phase: read
excerpt_of: primary source, arXiv:2402.19449 (no library card as of 2026-09-15)
source_url: https://arxiv.org/abs/2402.19449
created_at: "2026-09-15"
---

# Excerpt: Heavy-Tailed Class Imbalance and Why Adam Outperforms Gradient Descent on Language Models

**Authors:** Frederik Kunstner, Robin Yadav, Alan Milligan, Mark Schmidt, Alberto Bietti (UBC, Flatiron Institute)
**Version read:** arXiv:2402.19449v2 (2024-07-12); v1 2024-02. Source type: paper.

## Claim (Abstract, §1.1)
"When trained with gradient descent, the loss of infrequent words decreases more slowly than the loss of frequent ones. This leads to a slow decrease on the average loss as most samples come from infrequent words. On the other hand, Adam and sign-based methods are less sensitive to this problem." The authors "do not claim that class imbalance is the only reason Adam outperforms SGD" (§1.1).

## Experiments (§2; settings from App. A Table 1)
- Figure 1: GPT2-Small on WikiText-103 (sequences of 1,024 tokens, batch 512). Classes (next tokens) are sorted by frequency and split into groups of about 10% of the samples each. "SGD makes little to no progress on low-frequency classes while Adam makes progress on all groups." The metric is training loss per group.
- Figure 2: CNN on MNIST plus about 10k added "barcoded" classes with 5 samples each (relative frequency difference 1000). GD makes almost no progress on the low-frequency half; Adam makes progress on both.
- Figure 3: ResNet18 on an ImageNet subset with class frequencies π_k ∝ 1/k; SGD and Adam are similar on a uniform subset and differ on the imbalanced one. Appendix B: the same pattern for vision transformers.
- Figure 4: softmax linear model on inputs drawn uniformly from [0, 1]^d with heavy-tailed random labels reproduces the gap.
- §2.3: the gap appears with full-batch (deterministic) GD, so it is not only a noise effect. Sign descent behaves like Adam; normalization and momentum help less than changing the direction (Figure 5). Upweighting the loss of low-frequency classes improves SGD (Appendix F).

## Mechanism on a weighted quadratic (§3.1)
For per-class losses f_k(w) = ½‖w‖² weighted by class frequency π_k (π_1 ≥ … ≥ π_c, Σπ_k = 1), gradient descent with step α gives
w_k^(t) = w_k^(t−1) − απ_k f_k′(w_k^(t−1)) = (1 − απ_k)^t w_k^(0).
"This slow convergence on functions with low weights cannot be fixed by increasing the step-size, as increasing it beyond 1/π_1 would cause instabilities on the highest-frequency 'class' f_1." Sign descent: w_k^(t) = w_k^(t−1) − α sign(f_k′(w_k^(t−1))), independent of π_k.

## Scope and limits
- All measurements are training loss; no downstream or held-out capability evaluation.
- Largest language model: GPT2-Small.

## Verification
- Checked on 2026-09-15 against arXiv:2402.19449v2 (Abstract, §1.1, §2, §3.1, App. A Table 1).
