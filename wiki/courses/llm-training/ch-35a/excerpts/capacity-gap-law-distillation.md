---
chapter: ch-35a
course: llm-training
phase: read
excerpt_of: primary source (no library card at wiki/raw-data/llm-training/papers/capacity-gap-law-distillation.md on 2026-09-15)
source_url: https://arxiv.org/abs/2311.07052
source_version: arXiv v4 (2025-07-30); v1 2023-11
created_at: "2026-09-15"
---

# Excerpt: Towards the Law of Capacity Gap in Distilling Language Models (Zhang, Li, Song, Ye, Gao, Hu; Beijing Institute of Technology, University of Copenhagen, The Open University, Xiaohongshu)

Facts used by [[read]], read in the arXiv v4 PDF text on 2026-09-15.

## Claim
- "Curse of capacity gap": for a fixed student scale, student quality "does not improve along the increased teacher scale" and often degrades as the teacher grows (§1, Fig. 1: GPT2 and Pythia distilled on OpenWebText, evaluated by WikiText2 perplexity).
- Pilot method (§3): take a teacher LM, prune it to a target sparsity to form the student, distill from teacher to student, and record the teacher scale that gives the best student.
- Distillation loss (§3.2, Eq. 4): token-level cross-entropy, half against the teacher's distribution over the vocabulary and half against the ground-truth token. Sequence-level objectives "are not taken into consideration due to a relatively low efficiency".
- Law 1 (§3.3, Eq. 5-6): T* ≈ α·S + β with α = 2.498 and β = −11.498, fitted on (student scale, best teacher scale) pairs in millions of parameters (Fig. 4 axes "MParams"), R² = 0.9957; rounded as T* ≈ 2.5·S.
- Corollary 1: a student of scale 0.4·T* is matched to a teacher of scale T*.

## Extrapolation (§4)
- LLaMA2-7B (adapted with Chinese vocabulary) and LLaMA3.1-8B distilled to about 3B students (0.4 × 7 ≈ 3B), named MiniMA; then instruction-tuned on 1.1M examples (MiniChat).
- Distillation settings: sequences of 4,096 tokens; batch 1,024 (about 4M tokens); LR 3e-4; weight decay 0.1; 1 epoch; 1% linear warmup.

## Scope stated by the authors (§2.1)
- The study distills pretrained LMs on pretraining-style data with the token-level loss above (task-agnostic distillation). For pseudo (black-box) distillation on teacher-generated text, "the concern of capacity gap might be rather waived along the reduction of teacher knowledge from informative distributed probabilities to one-hot labels".

## Not tested
Reasoning-trace SFT; students above about 3B in the fitting data; tasks measured by accuracy on reasoning benchmarks.
