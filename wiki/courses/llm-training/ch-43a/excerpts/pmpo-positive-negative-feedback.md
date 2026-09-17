---
chapter: ch-43a
course: llm-training
phase: read
excerpt_of: primary source (no library card at wiki/raw-data/llm-training/papers/pmpo-positive-negative-feedback.md on 2026-09-15)
source_url: https://arxiv.org/abs/2410.04166
source_version: arXiv v3 (2025-03-07); v1 2024-10-05; ICLR 2025
created_at: "2026-09-15"
---

# Excerpt: Learning from Negative Feedback, or Positive Feedback or Both (Abdolmaleki, Piot, Shahriari, Springenberg, Hertweck, Joshi, et al.; Google DeepMind)

Facts used by [[read]], read in the arXiv v3 PDF on 2026-09-15.

## Objective (§3.3, Eq. 10)
`J_ar(π_θ; x) = α E_{y∼D_a}[log π_θ(y|x)] − (1 − α) E_{y∼D_r}[log π_θ(y|x)] − β KL(π_ref, π_θ; x)`
- D_a: accepted (positive) samples; D_r: rejected (negative) samples; α: trade-off between the two terms; β: weight of a KL term towards the reference policy. Paired preferences are not required, so the method can run with only positives (α = 1) or only negatives (α = 0).
- The KL term appears in the derivation only for the negative-sample estimate: it comes from the reparameterisation of the variational distribution (§3.2), and the authors describe the negative objective as "modifying the reference distribution such that the negative examples are removed".

## Evidence that the KL term is what makes negative-only training stable
- §5.2, Figure 3 (Control Suite tasks, β swept over 0.0, 0.5, 1.0, 1.5, 2.0): with negatives only (α = 0) performance "is highly sensitive to β" and needs β > 1.0; with positives only (α = 1) the algorithm is insensitive to β; with both (α = 0.5) a β above 0.5 is generally needed. "As predicted by the theory, not using a KL can quickly lead to collapse when using only dis-preferred samples" (§5.2).
- Offline RL on RGB Stacking (§5.3, Table 1; average reward over 100 evaluation episodes): BC 24, Accept+BC 26, Accept 27, Reject+BC 77, Accept+Reject+BC 93. The value function is used only to label transitions as accept (positive advantage) or reject (negative advantage), and the KL term is computed on all 140k episodes.
- Language alignment (§5.4): Gemma 2B pre-trained model, reward model from human preferences, prompts from LMSYS-chat-1M, one epoch over 500k prompts in about 4,000 learner steps, 128 prompts per batch with 4 generations each; the top two generations are labelled preferred and the bottom two dis-preferred. Using both signals (PMPO-AR) learns faster than either alone and is competitive with DPO; PMPO-AR is also "the quickest to hack the reward", with the GPT-4-judged win rate against the base Gemma checkpoint dropping in mid-training while the reward-model score keeps rising (Figure 4). A sufficiently high β > (1 − α) is again needed for negative-only training (§5.4, Figure 5).
