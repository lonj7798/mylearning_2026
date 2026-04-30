<!-- scope: Nvidia Nemotron 3 (Nano/Super/Ultra) — multi-environment RL + GenRM
     deps: [[nemotron]]
     see-also: [[llama-3]], [[qwen-3]]
-->

# Nemotron 3 (Ultra / Super / Nano)
- **Core Insight:** "Multi-environment reinforcement learning" — one RL run across reasoning, tool-use, and agentic environments with a GenRM reward model — beats single-environment RL stages for agentic generalization.
- **Guideline:** Ship your reward model alongside the policy; Nvidia's open GenRM release lets downstream users resume RLHF without retraining the RM.

- **Authors / Lab:** NVIDIA (Nemotron team)
- **Year:** 2025 (Nemotron 3 Nano + white paper Dec 24, 2025; Super and Ultra releases to follow)
- **URL:** https://arxiv.org/abs/2512.20856 — https://research.nvidia.com/labs/nemotron/Nemotron-3/
- **Relevant topics:** multi-environment RL, GenRM, agentic reasoning, reasoning budget control, open RLHF datasets

## Abstract
Nemotron 3 is NVIDIA's 2025 open family: Nano (3.2B active / 31.6B total — MoE), Super, Ultra. Nano is released first, with white paper + tech report + GenRM reward model + curated training datasets. Post-training philosophy centers on **multi-environment RL** — a single RL run spanning reasoning, multi-step tool use, and agentic environments — with a granular "reasoning budget control" lever letting users trade tokens for accuracy at inference.

## Key Contributions
- **Nemotron 3 Nano:** 3.2B active (3.6B w/ embeddings), 31.6B total MoE — <50% of activated params vs Nemotron 2 Nano at better accuracy.
- **Multi-environment RL:** a unified RL stage across reasoning, tool use, and agentic task environments rather than sequential reasoning-then-tools RL.
- **GenRM release:** a generative reward model trained on NVIDIA-curated preference data, released as an open asset for downstream RLHF.
- **Reasoning budget control:** inference-time parameter for number of thinking tokens — similar in spirit to Qwen 3's thinking budget.
- Open artifact bundle: weights, GenRM, training recipes, curated datasets — intended as a reproducible stack.

## Post-training pipeline
- **SFT data:** NVIDIA-curated mix covering reasoning, agentic, multi-step tool use; size not publicly itemized in the Nano release summary. Uses NVIDIA's prior Nemotron-4 data pipeline as a base.
- **Preference / RL algorithm:** Multi-environment RL; specific algorithm (PPO vs GRPO vs DPO) not stated in Nano release notes — white paper details expected. Nemotron-4 used an iterative RLHF approach, so PPO-family is likely.
- **Reward model:** **GenRM** — generative reward model. Released as an open asset alongside the policy. Architecture and training corpus summarized in the white paper; details beyond "generative" not in the release blog.
- **KL / entropy handling:** Not disclosed in summary.
- **Rollout scale:** Not disclosed.
- **Hyperparameters:** Not disclosed in search-surfaced summary — LR, batch, clip ε, group size all reserved.
- **Verifiable rewards:** Multi-environment RL includes environments with verifiable signals (tool-use success, code execution) alongside GenRM-scored outputs.
- **Self-improvement / iterative:** Nemotron-4 used iterative SFT↔RLHF loops; the same template presumably carries forward but is not explicitly described for Nemotron 3.

## Innovations vs predecessors
Changes from **Nemotron-4 340B → Nemotron 3**:
- Shifted to MoE for Nano (Nemotron-4 was dense).
- Multi-environment RL supersedes the sequential RLHF stages used in Nemotron-4.
- GenRM released publicly — Nemotron-4's RM was not open.
- Granular reasoning-budget control introduced; absent in Nemotron-4.
- Smaller active-parameter footprint (3.2B vs 340B dense) reflects the 2025 MoE efficiency shift.

## Key Figures/Tables to Study
- Multi-environment RL diagram — conceptual contrast with sequential reasoning-then-tool RL.
- Nano accuracy-vs-activated-params plot — the efficiency-frontier claim.
- GenRM training overview — the only public window into NVIDIA's RM recipe at this scale.

## Connections
- [[nemotron]] — Nemotron-4 340B; direct ancestor with published RM recipe.
- [[llama-3]] — comparison point for dense RLHF pipeline.
- [[qwen-3]] — thinking budget concept shared; different algorithmic approach.

## Gaps / what the report does NOT disclose
Nano white-paper summary is thin on hyperparameters. Not disclosed: exact RL algorithm, KL β, LR, batch size, clip ε, group size G, rollouts per prompt, RL step counts, GenRM loss form, preference-data sizes, multi-environment reward mixing weights. Super and Ultra tech reports are not yet released at time of writing; their post-training deltas vs Nano are unknown.
