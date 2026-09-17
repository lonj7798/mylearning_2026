---
chapter: ch-30a
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/rls-razor.md (planned card; not present on 2026-09-15)
source_url: https://arxiv.org/abs/2509.04259
created_at: "2026-09-15"
---

# Excerpt: RL's Razor: Why Online Reinforcement Learning Forgets Less

**Authors:** Idan Shenfeld, Jyothish Pari, Pulkit Agrawal (Improbable AI Lab, MIT)
**Version read:** arXiv:2509.04259v1 (4 Sep 2025).
**Status:** no library card existed for this slug on 2026-09-15; quotes checked against the v1 PDF text. The full treatment is in ch-38a.

## Claim (Abstract, §1)
"the degree of forgetting is determined by the distributional shift, measured as the KL-divergence between the fine-tuned and base policy evaluated on the new task." The predictor is E_{x∼τ}[KL(π_0 ‖ π)], computed on inputs from the new task τ; the paper calls this the forward KL. "on-policy RL is implicitly biased towards KL-minimal solutions among the many that solve the new task, whereas SFT can converge to distributions arbitrarily far from the base model."

## Settings named in §3.1
LLM experiments use Qwen 2.5 3B-Instruct trained on math reasoning, on the Chemistry L-3 subset of SciKnowEval, and on ToolAlpaca. A toy setting in §4 (ParityMNIST with FashionMNIST as the prior task, 3-layer MLP) replicates the SFT-RL gap.

## Fit quality
- ParityMNIST: "A quadratic fit achieves R2 = 0.96 in this setting" (§4).
- LLM experiments: "a quadratic fit achieving R2 = 0.71 (Figure 11). While weaker, the residuals are mean-zero and can be attributed to noise from approximate KL and accuracy estimation" (§4).
- Table 1 (§6, MNIST task, R² of a 2nd-degree polynomial): forward KL 0.96 ± 0.01; reverse KL 0.93 ± 0.01; total variation 0.80 ± 0.01; weight change L1 0.34 ± 0.02; Fisher-weighted L2 0.58 ± 0.02.

## Limit stated by the authors (§7)
"we still lack a mechanistic account of why larger KL shifts on the new task disrupt prior knowledge."

## Conflict to record
[[retaining-by-doing]] reports that the KL-forgetting connection "does not always hold in our setting." ch-30a labels the KL predictor an **Open question** for LLM post-training.

## How ch-30a uses it
§5.7 (preview of on-policy data as a forgetting control), Checklist item on logging KL to the base model on target prompts.
