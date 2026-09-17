---
chapter: ch-35
course: llm-training
phase: read
excerpt_of: arXiv:2503.19786v1 (no library card at the time of writing; chapter-local verified extract)
source_url: https://arxiv.org/abs/2503.19786
created_at: "2026-09-15"
---

# Excerpt: Gemma 3 Technical Report

- **Authors:** Gemma Team, Google DeepMind (byline)
- **Year:** 2025 (arXiv v1 2025-03)
- **Source type:** official technical report
- **Used in:** ch-35 §1, §6.5, Recipe

## Pre-training with distillation (§2.2)
- "We follow a similar recipe as in Gemma 2 for pre-training with knowledge distillation."
- Token budgets: 14T tokens for Gemma 3 27B, 12T for 12B, 4T for 4B, 2T for 1B.
- "Distillation. We sample 256 logits per token, weighted by teacher probabilities. The student learns the teacher's distribution within these samples via cross-entropy loss. The teacher's target distribution is set to zero probability for nonsampled logits, and renormalized."
- The teacher model is not named in §2.2.

## Post-training (§3)
"Our post-training approach relies on an improved version of knowledge distillation (Agarwal et al., 2024; Anil et al., 2018; Hinton et al., 2015) from a large IT teacher, along with a RL finetuning phase based on improved versions of BOND, WARM, and WARP."

## Small versus large teacher (§5.4, Figure 8)
"We train a student with 2 teachers of different sizes, one large and one small, for different training horizons. In Fig. 8, we observe that for short training horizons, the smaller teacher is better, but the trend is reversed for longer training." Figure 8 plots the relative perplexity difference against total training tokens in billions on a log axis (tick marks at 10¹ and 10²); the student and teacher sizes and exact values are not printed in the text.

## Verification
- Checked on 2026-09-15 against https://arxiv.org/abs/2503.19786 (v1): §2.2, §3, §5.4, Figure 8 caption.
- Not reported by the source: teacher identity and size for pre-training distillation; ablation numbers for the 256-logit choice; post-training distillation settings.
