<!-- scope: Zhipu GLM-4.5 — slime RL infra, hybrid sync/async training
     deps: []
     see-also: [[qwen-3]], [[deepseek-v3]]
-->

# GLM-4.5
- **Core Insight:** The RL infrastructure choice (sync-colocated vs async-disaggregated) should be made *per task type* — synchronous for math/code, asynchronous for agentic — not globally.
- **Guideline:** If you have both reasoning and agentic RL workloads, run them on different infra modes; slime's hybrid design is the open template.

- **Authors / Lab:** Z.ai (Zhipu AI)
- **Year:** 2025 (released Aug 2025)
- **URL:** https://arxiv.org/abs/2508.06471 — https://github.com/zai-org/GLM-4.5
- **Relevant topics:** Agentic-Reasoning-Coding (ARC) foundation, slime RL framework, hybrid sync/async RL, two-stage difficulty curriculum, 64K output length RL

## Abstract
GLM-4.5 is an open MoE (355B total / 32B active) trained on 23T tokens with 7T focused on code + reasoning. The post-training contribution is **slime**, an SGLang-native open-source RL framework supporting a hybrid training architecture: colocated-synchronous mode for reasoning (math / code) and disaggregated-asynchronous mode for agentic tasks. Two-stage difficulty curriculum for reasoning RL; single-stage 64K-output-length RL outperformed progressive length extension.

## Key Contributions
- **slime RL framework** (open-source) — SGLang-native, supports colocated-synchronous and disaggregated-asynchronous modes.
- **Task-type-driven infra selection:** reasoning ≈ sync; agentic ≈ async. Sync is more effective for math/code where rollouts are similar-length; async handles long-horizon agentic workloads where data generation is slow.
- **Two-stage difficulty curriculum** — moderate → extremely difficult problems as the model improves.
- **Fixed 64K output length RL** outperformed progressive length extension — a notable negative result on curriculum length scheduling.
- Hybrid thinking / direct-response modes in the final model.
- Benchmark results: 70.1% TAU-Bench, 91.0% AIME 24, 64.2% SWE-bench Verified.

## Post-training pipeline
- **SFT data:** expert-model iteration + SFT before RL; concrete sizes not disclosed in extracts.
- **Preference / RL algorithm:** RL via slime; specific algorithm (GRPO / PPO / custom) not specified in available extracts — slime supports multiple.
- **Reward model:** not disclosed in available extracts. Reasoning RL relies on rule-based verifiers (math, code); agentic RL relies on environment feedback.
- **KL / entropy handling:** not disclosed.
- **Rollout scale:** 64K max output length for reasoning RL.
- **Hyperparameters:** not disclosed in available extracts.
- **Verifiable rewards:** yes — for reasoning and coding domains.
- **Self-improvement / iterative:** two-stage curriculum implements implicit self-curriculum; expert-model iteration before RL is an iterative distillation loop.

## Innovations vs predecessors
Changes from **GLM-4 → GLM-4.5**:
- First GLM generation with open RL framework (slime).
- Explicit hybrid sync/async infra — prior GLM generations ran monolithic RL.
- Two-stage difficulty curriculum formalized.
- ARC positioning (Agentic, Reasoning, Coding) as unified objective — GLM-4 targeted conversational + reasoning without explicit agentic leg.
- 23T pretraining tokens (7T code/reasoning-focused) — up from GLM-4's smaller corpus.

## Key Figures/Tables to Study
- slime architecture diagram — colocated-sync vs disaggregated-async branches.
- Reasoning RL output-length ablation — single-stage-64K vs progressive-length curve.
- Two-stage difficulty curriculum timeline.

## Connections
- [[qwen-3]] — peer 2025 hybrid-thinking MoE open release.
- [[deepseek-v3]] — comparable MoE scale (355B/32B vs 671B/37B).
- [[kimi-k2]] — agentic long-horizon RL peer; slime's async mode is conceptually similar to Kimi's partial-rollout infra.

## Gaps / what the report does NOT disclose
Extracts available do not specify RL algorithm family, reward model, KL β, LR, batch size, clip ε, group size G, rollouts per prompt, SFT data size, exact agentic-task environments, or the agentic reward composition. 64K-vs-progressive ablation is described qualitatively without numbers. Full hyperparameters presumably in the full paper body; chapter authors should fetch arxiv 2508.06471 directly.
