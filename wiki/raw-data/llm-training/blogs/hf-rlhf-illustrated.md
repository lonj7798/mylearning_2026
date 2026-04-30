<!-- scope: HuggingFace Illustrated RLHF — the canonical visual RLHF explainer
     deps: [[README]]
     see-also: [[rlhf-instructgpt]], [[ppo]], [[dpo]]
-->

# HuggingFace — Illustrated Reinforcement Learning from Human Feedback (RLHF)
- **Core Insight:** Three-stage RLHF — SFT, reward model, PPO against the RM with a KL penalty — can be communicated with four diagrams; the diagrams are the single best intuition primer for the whole field.
- **Guideline:** Use this post as the onboarding reference for anyone new to RLHF before showing them InstructGPT or Llama 2.
- **Author:** Nathan Lambert, Louis Castricato, Leandro von Werra, Alex Havrilla (HuggingFace)
- **Year:** 2022
- **URL:** https://huggingface.co/blog/rlhf
- **Relevant topics:** RLHF pipeline, reward model, PPO, KL regularization, illustrated tutorial

## Summary
"Illustrated RLHF" is HuggingFace's flagship RLHF tutorial post. It explains the three-stage pipeline — SFT, reward model training on pairwise human preferences, PPO against the RM with KL-to-reference regularization — using a small set of clear diagrams that have been re-used across hundreds of downstream tutorials. The post predates Llama 2 (and therefore most modern post-training) but remains the best single entry point to the field because its conceptual framing maps cleanly onto every subsequent paper.

## Key Contributions
- Canonical three-stage diagram reused field-wide.
- Clear derivation of the KL-regularized reward signal `r_total = r(x,y) - beta * KL(pi || pi_ref)`.
- Concrete pointers to open-source tools (TRL, trlx) at publication time.
- Framing of the pipeline as "LM + classifier + RL" that makes the three components orthogonal in the reader's mental model.

## Key Figures/Tables to Study
- **Figure 1 (Pipeline overview):** the often-reproduced left-to-right three-box diagram.
- **Figure 2 (Reward Model training):** the pairwise preference diagram with Bradley-Terry loss.
- **Figure 3 (PPO loop):** policy sampling -> RM scoring -> PPO update with reference KL.
- **Figure 4 (KL penalty effect):** qualitative showing how KL keeps policy near reference.

## Technical Details

### Stage 1: Supervised Fine-Tuning (SFT)
- Initialize chat/instruct behavior from a pretrained LM.
- Train on high-quality (prompt, response) pairs with completion-masked cross-entropy.
- Output: pi_SFT — the reference policy for subsequent stages.

### Stage 2: Reward Model (RM)
- Architecture: same as LM but with a scalar output head.
- Data: pairs (x, y_w, y_l) where y_w is preferred over y_l.
- Loss: `-log sigma(r(x, y_w) - r(x, y_l))` — pairwise logistic (Bradley-Terry).
- Initialized from pi_SFT; head is a linear layer on the final hidden state of the last token.

### Stage 3: PPO against RM with KL penalty
- Policy pi_theta initialized from pi_SFT.
- For each sampled (x, y):
  - Scalar reward = r(x, y) minus a KL penalty term at each token.
  - Per-token reward: `r_t = -beta * log(pi_theta(y_t | ...) / pi_ref(y_t | ...))` for t < |y|, plus `r(x, y)` added at t = |y| (end-of-sequence).
- PPO update with clipped ratio, value head, advantage normalization.
- Reference policy pi_ref = pi_SFT held fixed.

### Why KL regularization
Without it, PPO exploits the RM (reward hacks) by producing outputs outside the distribution the RM was trained on. The KL penalty bounds the policy to stay close to pi_SFT, where the RM is well-calibrated.

### Beta typical values
- beta ~ 0.01–0.2 in the InstructGPT / Llama 2 range (applied to token-level log-ratios).
- Too small -> reward hacking.
- Too large -> policy never moves.

### What this post predates
- DPO (2023) — the closed-form alternative.
- RLVR (2024) — the verifier-based alternative.
- GRPO (2024) — the critic-free variant.
- Iterative multi-round RLHF (Llama 3).
The post remains pedagogically valuable because all those variants are best understood as modifications of the baseline it introduces.

## Connections
- [[rlhf-instructgpt]] — the Ouyang 2022 paper the post visualizes.
- [[ppo]] — the RL algorithm.
- [[dpo]] — the post-DPO variant landscape (see [[hf-dpo-zoo]]).
- [[costa-huang-ppo-details]] — implementation-details companion.
- [[lilianweng-rlhf]] — tutorial-text companion at higher mathematical rigor.
