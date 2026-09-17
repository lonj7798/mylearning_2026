---
chapter: ch-42
course: llm-training
phase: read
excerpt_of: arXiv:2506.10947v2 (Spurious Rewards: Rethinking Training Signals in RLVR); library card [[spurious-rewards-rlvr]]
source_url: https://arxiv.org/abs/2506.10947
created_at: "2026-04-23"
revised: "2026-09-15 (generality revision; rewritten from the primary source because the library card has no Verification section and omits the model-family limit)"
---

# Excerpt: spurious rewards and the model-family limit (Shao, Li, Xin, Geng et al.)

Used by [[read]] §5.1 and the detection section. Checked against arXiv v2 (2026-02-25) on 2026-09-15.

## Reward conditions (§2.2)
1. Ground-truth reward. 2. Majority-vote pseudo-labels (majority of 64 samples from the pre-RL model). 3. Format reward: any response containing a non-empty `\boxed{}`. 4. Random reward: reward 1 with probability γ (main runs γ = 0.5). 5. Majority-voted **incorrect** reward: reward only answers matching a label known to be wrong.

> "We emphasize that spurious rewards—particularly random and incorrect rewards—are proposed purely for analytical purposes and should not be interpreted as a recommended approach for developing true model capabilities."

## Results on Qwen2.5-Math-7B (§2.2, Abstract)
- MATH-500 absolute gains: ground truth +29.1, incorrect labels +24.1, random +21.4.
- AMC: format +13.8, incorrect +24.1, random +21.4, against about +27 to +29 for majority-voted and ground-truth labels.
- AIME 2024: ground truth +15.3, format +10.3, incorrect +10.2, random +10.2. On AIME 2025 (written after the models' knowledge cutoff) ground truth has "a clear advantage" and the others give −0.4 to +4.5.
- Qwen2.5-Math-1.5B gains from random rewards arrive later (after 100 steps) and are smaller on AMC (+4.9).

## The limit that must be carried with the result (§3)
- Models tested beyond Qwen2.5-Math: Qwen2.5-7B, Qwen2.5-1.5B, Llama3.1-8B(-Instruct), Llama3.2-3B(-Instruct), OLMo2-7B, OLMo2-7B-SFT.
- "Across Qwen2.5, all non-random rewards (even spurious incorrect) improve MATH-500, whereas OLMo stays flat under spurious rewards and gains mainly with ground-truth rewards." "Each weak or spurious reward fails to help at least one other model and can be flat or even harmful."
- Practical warning printed in the paper: "Proposed RLVR reward signals should be tested on diverse models!" Two published methods (test-time training; one-shot RL) reproduced the pattern: strong gains on Qwen, little or none elsewhere (§3, App. E).
- Models that were already RL post-trained see minimal gains under nearly all rewards (App. J).

## Mechanism (§4, §5)
- The GRPO clip term biases updates toward tokens that already have high probability under the base model; disabling clipping removes the gains from random rewards (§4, Fig. 6).
- Case study: "code reasoning" — reasoning written as Python without execution — rises from 65% to over 90% of Qwen2.5-Math-7B answers under spurious rewards, and is predictive of accuracy (§5).

## Use in this chapter
The random-reward run is a control: it measures how much of a reported RLVR gain is attributable to the reward signal rather than to amplification of the base model's priors. The control is informative only on the model family being trained.
