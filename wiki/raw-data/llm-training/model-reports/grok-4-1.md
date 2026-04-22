<!-- scope: xAI Grok 4.1 — RL with agentic reasoning models AS reward models
     deps: [[llama-3]]
     see-also: [[qwen-3]], [[gemini-2.5-deep-research]]
-->

# Grok 4.1
- **Core Insight:** Use a frontier agentic reasoning model as the reward model — an LLM-judge that can itself reason, tool-use, and verify — to autonomously score and iterate on responses at RL scale.
- **Guideline:** When classical RMs saturate, upgrade the RM's reasoning capability rather than the policy's; reasoning-grade RMs unlock new post-training axes (hallucination reduction, EQ, personality).

- **Authors / Lab:** xAI
- **Year:** 2025 (released Nov 17, 2025)
- **URL:** https://x.ai/news/grok-4-1 — https://data.x.ai/2025-11-17-grok-4-1-model-card.pdf
- **Relevant topics:** reasoning-model-as-RM, hallucination reduction, style/personality RL, silent-rollout evaluation, pairwise-preference eval

## Abstract
Grok 4.1 is a post-training-only upgrade of Grok 4 — xAI re-used the Grok 4 base + RL infra and focused the 4.1 cycle on reducing hallucinations, improving style/personality/EQ, and sharpening helpfulness. Post-training is driven by "frontier agentic reasoning models as reward models" — an RL loop where the RM itself is a reasoning-capable model that can autonomously evaluate and iterate on responses. Evaluation relied on a two-week silent rollout with continuous blind pairwise comparisons on live production traffic; Grok 4.1 was preferred 64.78% of the time vs the prior production model.

## Key Contributions
- **Reasoning-model-as-RM** — xAI's key methodological claim: frontier agentic reasoning models serve as the reward model, autonomously evaluating responses at scale.
- **Hallucination reduction** for info-seeking queries as an explicit RL objective; reductions reported on sampled production info-seeking prompts.
- **Silent rollout evaluation** — two-week gradual traffic ramp with blind pairwise eval on live grok.com / X / mobile traffic — replaces offline static evals for style/personality decisions.
- Uses the **same large-scale RL infra as Grok 4**, applied to style, personality, helpfulness, and alignment rather than raw reasoning gains.
- 64.78% pairwise preference over the prior production Grok version.

## Post-training pipeline
- **SFT data:** not disclosed.
- **Preference / RL algorithm:** not disclosed (xAI does not name PPO / GRPO / DPO publicly). Described generically as "large scale reinforcement learning."
- **Reward model:** **frontier agentic reasoning model used as RM** — the central methodological statement. No architecture / training-data details public. Likely a heavyweight reasoning model (possibly Grok 4 itself or a specialized variant) used to score policy outputs.
- **KL / entropy handling:** not disclosed.
- **Rollout scale:** not disclosed.
- **Hyperparameters:** none disclosed.
- **Verifiable rewards:** partial — hallucination reduction presumably uses factuality verification, but mechanism not public.
- **Self-improvement / iterative:** RM "autonomously evaluates and iterates on responses at scale" — implies an iterative online-RL loop with the reasoning RM continuously scoring fresh rollouts.

## Innovations vs predecessors
Changes from **Grok 4 → Grok 4.1**:
- Post-training-only release on top of the Grok 4 base (no new pretraining).
- Reasoning-model-as-RM shifted from secondary tool to primary reward signal.
- Explicit hallucination-reduction objective added to post-training.
- Production-traffic silent rollout used as the decisive evaluation vehicle.

vs industry 2025 norms:
- Most labs still use either rule-based verifiers (Magistral, Phi-4-reasoning) or trained scalar RMs (Tülu 3, Nemotron GenRM); xAI's reasoning-RM posture is closer to Constitutional-AI-style LLM judges but scaled up to agentic reasoning capability.

## Key Figures/Tables to Study
- Model card: factuality/hallucination reduction chart on sampled info-seeking prompts.
- Pairwise-preference rate (64.78%) vs production predecessor — the headline metric.

## Connections
- [[llama-3]] — contrast: explicit PPO + DPO pipeline vs Grok 4.1's opaque "RL at scale."
- [[qwen-3]] — contrast: Qwen's Stage 4 General RL shares the "20+ domains" ethos, but uses rule + judge mix rather than pure reasoning RM.
- [[constitutional-ai]] — conceptual ancestor: LLM-as-judge.

## Gaps / what the report does NOT disclose
Almost everything quantitative is absent. Not disclosed: RL algorithm family, reward-model architecture / size / training data, SFT data, KL, LR, batch size, clip ε, rollouts per prompt, RL step count, hallucination-reward shape, exact pairwise-eval sample size for the 64.78% figure, whether reasoning RM = Grok 4. xAI has historically disclosed only blog + release-note-level detail; no arXiv tech report exists for Grok 4.1.
