---
chapter: ch-42
course: llm-training
phase: read
excerpt_of: arXiv:2212.08073v1 (Constitutional AI: Harmlessness from AI Feedback); library card [[constitutional-ai]] (verified 2026-09-14)
source_url: https://arxiv.org/abs/2212.08073
created_at: "2026-04-23"
revised: "2026-09-15 (generality revision; rewritten to match the verified card and the primary text)"
---

# Excerpt: AI feedback labels, clamping, evasiveness, and Goodharting (Bai et al.)

Used by [[read]] §2.4 and §4.4. Quotes checked against the arXiv v1 PDF on 2026-09-15; dataset sizes follow the verified library card.

## Pipeline sizes (card, §3.2, §4.2)
- Red-team prompts: 42,496 human-written + 140,335 model-generated = 182,831; 4 critique-revision pairs per prompt.
- SL-CAI training: a pretrained model fine-tuned for 1 epoch on the revisions plus 135,296 human helpfulness prompts with 2 responses each; constant LR 0.5 × pretraining LR; batch 1,024 sequences; sampling T = 1.
- Preference-model data: 135,296 human helpfulness comparisons + 182,831 AI harmlessness comparisons.
- 16 principles; one principle is sampled per critique and per comparison label, not concatenated (§3.1, §4.1). "When generating labels, we ensemble over 16 pre-written constitution principles ... We found that this led to more robust preference model scores" (§4.3).

## Label construction and clamping (§4.1, §4.3)
> "One issue that arises is that the CoT samples typically state explicitly which multiple choice option is to be preferred, and so the probability targets are typically very confident (i.e., close to 0 or 1) and are not well-calibrated. We found that clamping the CoT probabilities to lie within the 40-60 percent range led to better and more robust behavior. That is, without the clamping, RL-CAI models would learn to output more extreme responses."

> "For RL-CAI without CoT, we found that using soft preference labels (i.e., normalized log-probabilities from the feedback model) led to much better results than hard labels ... Instead we found that clamping the probabilities at 20-80 percent slightly improved results, while clamping at 40-60 improved results further. We settled on using 40-60 for the main results."

The clamp applies to chain-of-thought labels. Non-CoT soft labels are not clamped.

## Evasiveness as the failure this pipeline targets (§1.1, §4.4)
> "In our prior work using human feedback ... our assistant often refused to answer controversial questions. Furthermore, once it encountered objectionable queries, it could get stuck producing evasive responses for the remainder of the conversation. Ultimately this was due to the fact that evasiveness was rewarded as a response to harmful inputs by our crowdworkers."

RL-CAI is reported as "virtually never evasive" (§4.4); crowdworkers were instructed to prefer non-evasive responses when two responses were equally harmless (Fig. 2 caption).

## Over-training (§4.3)
> "We found that RL-CAI models can be over-trained, resulting in Goodharting behavior [Gao et al., 2022] whereby models can be overly harsh in responding to harmful prompts, or may include boilerplate language as part of their response to most red teaming prompts, saying e.g. 'you are valid, valued, and cared for'."

Qualitative mitigations named in §4.3: rewriting principles to discourage over-reactive answers, ensembling over the 16 principles, and soft or clamped labels. No numbers accompany them.

## Corrections to the previous version of this excerpt
- "soft labels clipped to [0.25, 0.75]" → CoT probabilities clamped to 40-60%.
- "~180K red-team prompts" → 182,831 (42,496 human + 140,335 generated).
- "SFT on the RLHF model's revisions" → a pretrained model is fine-tuned on revisions from all revision steps plus helpfulness samples.
- "PPO with the InstructGPT KL penalty" → the paper states only that RL used "the same hyperparameters as" Bai et al. (2022); the algorithm and KL values are not given.
