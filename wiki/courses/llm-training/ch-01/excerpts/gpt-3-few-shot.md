---
chapter: ch-01
course: llm-training
phase: read
excerpt_of: primary source, arXiv:2005.14165 (no library card as of 2026-09-15)
source_url: https://arxiv.org/abs/2005.14165
created_at: "2026-09-15"
---

# Excerpt: Language Models are Few-Shot Learners (GPT-3), training details

**Authors:** Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, et al. (OpenAI)
**Version read:** arXiv:2005.14165v4 (2020-07-22); v1 2020-05. Source type: official technical report / paper (OpenAI trained GPT-3).

## Appendix B, "Details of Model Training" (quoted)
"To train all versions of GPT-3, we use Adam with β1 = 0.9, β2 = 0.95, and ε = 10⁻⁸, we clip the global norm of the gradient at 1.0, and we use cosine decay for learning rate down to 10% of its value, over 260 billion tokens (after 260 billion tokens, training continues at 10% of the original learning rate). There is a linear LR warmup over the first 375 million tokens. We also gradually increase the batch size linearly from a small value (32k tokens) to the full value over the first 4-12 billion tokens of training, depending on the model size. Data are sampled without replacement during training (until an epoch boundary is reached) to minimize overfitting. All models use weight decay of 0.1 to provide a small amount of regularization [LH17]."

[LH17] is Loshchilov & Hutter, "Decoupled weight decay regularization".

## How ch-01 uses it
- Earliest report in ch-01's source set with β2 = 0.95, ε = 1e-8, global-norm clipping at 1.0, and weight decay 0.1 for LLM pretraining (2020).
- No ablation of these values is reported in the paper.

## Verification
- Checked on 2026-09-15 against arXiv:2005.14165v4, Appendix B.
