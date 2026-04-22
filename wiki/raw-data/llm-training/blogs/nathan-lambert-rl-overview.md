<!-- scope: Nathan Lambert Interconnects — RL-for-LLMs overview posts
     deps: [[README]]
     see-also: [[ppo]], [[grpo]], [[dpo]], [[rlvr-tulu3]]
-->

# Interconnects — RL for LLMs Overview
- **Core Insight:** The RL-for-LLMs stack now has three algorithm families in production use (PPO-RLHF, DPO, GRPO/RLVR) and the question is which reward-signal source you can afford, not which algorithm is "best."
- **Guideline:** Pick the algorithm based on the reward signal you have — verifiable -> RLVR/GRPO with rule rewards; learned preferences -> DPO or PPO-RLHF; no preferences -> SFT + self-play.
- **Author:** Nathan Lambert
- **Year:** 2024–2025 (running series on Interconnects)
- **URL:** https://www.interconnects.ai/ (multiple RL-overview posts, notably "RLHF 201" and "Recent reasoning research: GRPO tweaks, base model RL, and data curation")
- **Relevant topics:** PPO, DPO, GRPO, RLVR, base-model RL, reward model role, reasoning RL

## Summary
Lambert's RL-overview posts synthesize the current state of RL for LLMs across multiple algorithm families. The posts emphasize that the field has converged on a small set of algorithmic templates (PPO, DPO, GRPO) and that the main axis of innovation has shifted from algorithm design to reward-signal sourcing: who defines correctness (human, RM, verifier, model-self-critique), at what granularity (outcome vs process), and on which prompts (reasoning-only vs general chat).

A secondary thread: base-model RL — running RL on a pretrained base without a supervised reasoning SFT cold-start — is now the most generative research direction after R1-Zero's success. The post tracks Kimi k1.5, OpenReasonerZero, and other follow-ups.

## Key Contributions
- Consolidated taxonomy of RL-for-LLM algorithms with production-use status.
- Analysis of why GRPO's length normalization causes subtle biases (rewarding shorter correct responses and under-penalizing repetition).
- Running coverage of base-model RL lineage post-R1.
- Framing of RLVR as the open-community parallel to DeepSeek's rule-reward RL.

## Key Figures/Tables to Study
- **Algorithm-to-reward-signal matrix** (PPO+RM, DPO, RLVR, GRPO+rule).
- **GRPO tweaks comparison** across Kimi k1.5, DeepSeek-R1, and recent follow-ups.
- **Reasoning benchmark trajectory plots** from base-model-RL runs.

## Technical Details

### Algorithm family quick-reference
- **PPO-RLHF:** classic actor-critic with RM reward + KL-to-reference. Expensive (needs value net + RM + reference policy + policy = 4 models in GPU memory). Canonical for dialogue RLHF.
- **DPO:** closed-form solution under Bradley-Terry preference, no RM needed at RL time. Cheap, reproducible, but harder to continue-train once pairwise data plateaus.
- **GRPO:** PPO minus the critic; advantages are group z-scores. Doubles effective batch size at fixed memory. Required reward: any scalar (RM, rule, verifier).
- **RLVR:** special case of PPO/GRPO where the reward is a deterministic verifier output. No RM trained. Requires verifier-tractable prompts (math, code, constraints).

### GRPO length-normalization issue
Lambert's X thread + blog highlight: the output-length normalization in DeepSeek's original GRPO formulation (loss divided by |o_i|) means shorter correct responses are preferred, and repetitive-yet-correct responses are not penalized as hard as they would be under PPO. Dr. GRPO (2025) proposes the bias correction.

### Base-model RL (post-R1 lineage)
- DeepSeek-R1-Zero: RL directly on V3-Base, no SFT cold-start. Emergent long CoT.
- Kimi k1.5: similar recipe, different reward blend.
- OpenReasonerZero / SimpleRL-Zoo: community replications on 7B-scale base models.
- Open question: how much of the base model's latent reasoning is unlocked by RL, vs genuinely new skill.

### Reward-signal source taxonomy (Lambert's framing)
1. Human pairwise preference (expensive, noisy).
2. Learned reward model (scales but overoptimizes).
3. Verifier / rule (cheap, narrow domain).
4. Model self-critique (flexible, risk of drift).
5. Process reward model (expensive to label, promising but unproven at scale).

## Connections
- [[ppo]], [[grpo]], [[dpo]], [[rlvr-tulu3]] — the algorithm pages these posts synthesize.
- [[deepseek-r1]], [[kimi-k2]], [[tulu-3]] — frontier reports Lambert covers in detail.
- [[dr-grpo]] — technical follow-up on the length-normalization bias.
- [[lilianweng-rlhf]] — complementary tutorial-style overview.
- [[nathan-lambert-interconnects]] — the parent lab-index page.
