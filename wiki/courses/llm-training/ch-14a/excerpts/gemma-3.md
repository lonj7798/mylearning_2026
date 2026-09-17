---
chapter: ch-14a
course: llm-training
phase: read
excerpt_of: arXiv:2503.19786v1 (no library card as of 2026-09-15; chapter-local verified extract)
source_url: https://arxiv.org/abs/2503.19786
created_at: "2026-09-15"
---

# Excerpt: Gemma 3 Technical Report — pre-training budgets and distillation

- **Authors:** Gemma Team, Google DeepMind (byline)
- **Year:** 2025 (arXiv v1 2025-03-25; report dated 2025-03-12)
- **Source type:** official technical report

## Parameter counts (Table 1)
| Model | Vision encoder | Embedding parameters | Non-embedding parameters |
|---|---|---|---|
| 1B | 0 | 302M | 698M |
| 4B | 417M | 675M | 3,209M |
| 12B | 417M | 1,012M | 10,759M |
| 27B | 417M | 1,416M | 25,600M |

Table 1 caption: "Our vocabulary has 256k entries." §2.2 (Tokenizer): "The resulting vocabulary has 262k entries."

## Pre-training (§2.2)
- "We follow a similar recipe as in Gemma 2 for pre-training with knowledge distillation."
- "we train on 14T tokens for Gemma 3 27B, 12T for the 12B version, 4T for the 4B, and 2T tokens for the 1B. The increase in tokens accounts for the mix of images and text used during pre-training."
- "We also increase the amount of multilingual data to improve language coverage."
- "Distillation. We sample 256 logits per token, weighted by teacher probabilities. The student learns the teacher's distribution within these samples via cross-entropy loss. The teacher's target distribution is set to zero probability for nonsampled logits, and renormalized."
- §2.2 does not name the teacher model or its size. Post-training (§3) uses "knowledge distillation ... from a large IT teacher".

## Small versus large teacher (§5.4, Figure 8)
- "A common finding is that, to train a small model, it is preferable to distill from a smaller teacher. We suspect this is because these studies are often performed in settings where the regularization effect of using a worse teacher surpasses the benefit of using a better teacher. We train a student with 2 teachers of different sizes, one large and one small, for different training horizons. In Fig. 8, we observe that for short training horizons, the smaller teacher is better, but the trend is reversed for longer training."
- Figure 8 plots the relative perplexity difference against total training tokens in billions (log axis, ticks at 10^1 and 10^2); student and teacher sizes are not printed.

## Verification
- Checked on 2026-09-15 against arXiv:2503.19786v1 PDF text: Table 1, §2.2, §3, §5.4, Figure 8 caption.
- Not reported: pre-training learning rate, batch size, schedule, and mixture percentages; teacher identity for pre-training; whether the 256 logits are sampled with or without replacement; an ablation of the number of sampled logits.
