<!-- scope: Lilian Weng's RLHF survey blog post — canonical reference on RLHF algorithmic structure
     deps: [[README]]
     see-also: [[ppo]], [[dpo]], [[rlhf-instructgpt]]
-->

# Lil'Log — "The Transformer Family" lineage: RLHF posts
- **Core Insight:** RLHF decomposes cleanly into (1) preference data collection, (2) reward model training with Bradley-Terry loss, (3) KL-regularized policy optimization against that RM — and the bottleneck is usually (1) or (2), not the RL algorithm.
- **Guideline:** Debug RLHF from the bottom up: verify RM calibration on held-out preferences before touching PPO hyperparameters.
- **Author:** Lilian Weng (formerly OpenAI Head of Safety Systems)
- **Year:** 2023 series ("LLM Powered Autonomous Agents", RLHF-focused posts)
- **URL:** https://lilianweng.github.io/tags/rlhf/
- **Relevant topics:** RLHF pipeline, Bradley-Terry reward model, KL-regularized PPO, preference data, reward hacking

## Summary
Lil'Log is the single most-cited tutorial-grade source on RLHF. Weng's RLHF-lineage posts ("Reinforcement Learning from Human Feedback," "Reward Hacking in RL," "LLM Powered Autonomous Agents") lay out the InstructGPT-style three-stage pipeline with enough mathematical rigor to serve as a reference. She frames RLHF as preference-data -> Bradley-Terry RM -> KL-regularized policy gradient, with each stage's failure modes (annotator disagreement, RM overoptimization, entropy collapse in PPO) documented separately.

## Key Contributions
- Unified tutorial on RLHF stages with consistent notation across posts.
- "Reward Hacking in RL" (2024) surveys Goodhart's law as it appears in RLHF specifically — with examples from sycophancy, mode collapse, and length bias.
- Clear derivation of why KL regularization is required (PPO to reference policy) and how it interacts with entropy bonuses.
- Pointers to canonical implementation choices (value head, reward whitening, advantage normalization).

## Key Figures/Tables to Study
- **Three-stage RLHF pipeline diagram** (SFT -> RM -> PPO-against-RM) — the reference schematic every subsequent RLHF paper reuses.
- **Bradley-Terry derivation:** P(y_w > y_l | x) = sigmoid(r(x, y_w) - r(x, y_l)).
- **KL-regularized PPO objective:** reward = r(x,y) - beta * KL(pi || pi_ref).
- **Reward-hacking taxonomy:** length hacking, sycophancy, specification gaming.

## Technical Details
Weng's posts cover:
- **Bradley-Terry model:** the pairwise preference probability is sigmoid of reward difference; training loss is the negative log-likelihood of observed preferences.
- **KL penalty implementation:** beta is the KL coefficient; the per-token reward becomes `r_total = r(x,y) 1[y=EOS] - beta * log(pi(y|x) / pi_ref(y|x))`.
- **Reward normalization:** whitening (subtract mean, divide by std within batch) is standard to stabilize PPO.
- **GAE lambda:** typical 0.95 for RLHF; gamma = 1.0 (undiscounted, because rewards concentrate at EOS).
- **Value head:** shares the trunk with policy or uses separate network; Weng notes the tradeoff.
- **Entropy bonus:** often dropped in RLHF because KL-to-reference already regularizes exploration; re-introduced in some GRPO variants.
- **Reward hacking vulnerabilities:** Weng catalogs (a) length bias in RMs, (b) sycophancy (RM rewards agreement), (c) specification gaming (model finds prompt exploits that score high).

## Connections
- [[rlhf-instructgpt]] — the Ouyang 2022 paper Weng's pipeline mirrors.
- [[ppo]] — algorithmic foundation.
- [[dpo]] — the DPO post is a companion explaining the closed-form alternative.
- [[reward-hacking-taxonomy]] — Weng's reward-hacking post is one of the primary references for this topic.
- [[lilianweng-reasoning-llms]] — follow-up post "Why We Think" (2025) extends to reasoning RL.
