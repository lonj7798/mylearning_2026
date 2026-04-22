<!-- scope: Tencent Hunyuan-Large MoE post-training — SFT + single-stage online+offline DPO
     deps: []
     see-also: [[deepseek-v3]], [[qwen-2.5]]
-->

# Hunyuan-Large
- **Core Insight:** A critique model (70B dense) + single-stage online+offline DPO with an SFT-loss term on chosen responses and EMA-stabilized policy is enough — no PPO, no separate RM-based RL needed.
- **Guideline:** For MoE alignment, pair DPO with an SFT auxiliary loss on the chosen response and an EMA of the policy; this combats reward-hacking and alignment tax at low complexity.

- **Authors / Lab:** Tencent (Hunyuan Team)
- **Year:** 2024 (Nov 2024; v3 arxiv)
- **URL:** https://arxiv.org/abs/2411.02265
- **Relevant topics:** MoE post-training, critique-model SFT filtering, single-stage hybrid DPO, SFT-loss-augmented DPO, EMA policy stabilization

## Abstract
Hunyuan-Large is Tencent's 389B-parameter / 52B-active open MoE. Post-training runs a two-phase pipeline: (1) SFT on >1M high-quality examples filtered by a 70B-dense critique model across four quality tiers, with 3 epochs and LR 2e-5 → 2e-6; (2) DPO in a single-stage setup combining pre-compiled offline preference data with on-policy rollouts scored by a learned reward model, stabilized by an SFT-loss term on chosen responses and an exponential moving average of the policy.

## Key Contributions
- **Critique-model filtering:** a 70B dense Hunyuan model trained to score instruction samples on a 4-tier quality scale (accuracy, relevance, completeness, usefulness, clarity); followed by human annotation.
- **Single-stage hybrid DPO:** offline preference data + online rollouts scored by an RM in the same stage — not sequential offline-then-online.
- **SFT-loss augmented DPO:** the DPO loss is combined with an SFT loss on the chosen responses to prevent degradation and alignment-tax.
- **EMA-stabilized DPO:** exponential moving average of the policy used to mitigate reward hacking during online preference tuning.
- Full dropout at SFT: attention dropout 0.1, hidden dropout 0.2 — atypically high for post-training.

## Post-training pipeline
- **SFT data:** >1M high-quality instructions, critique-model-filtered + human-annotated. 3 epochs.
- **Preference / RL algorithm:** **DPO only** — hybrid single-stage offline + online. No PPO, no GRPO.
- **Reward model:** 70B-dense Hunyuan-based RM, used in the online branch to pick preferred responses among on-policy samples.
- **KL / entropy handling:** implicit via DPO's reference-anchoring (β not surfaced in extracts).
- **Rollout scale:** not disclosed in extracts.
- **Hyperparameters:**
  - SFT LR 2e-5 → 2e-6 over 3 epochs.
  - Attention dropout 0.1, hidden dropout 0.2.
  - DPO β, LR, batch size not explicitly surfaced.
- **Verifiable rewards:** not the primary signal — RM-scored preference is the main reward.
- **Self-improvement / iterative:** single-stage hybrid offers a light iterative element (online branch) without a full multi-round RLHF loop.

## Innovations vs predecessors
Changes from **Hunyuan-7B / 13B → Hunyuan-Large**:
- First Hunyuan generation at 389B MoE scale.
- Critique-model filtering introduced at SFT stage (prior models used rule-only filters).
- Single-stage hybrid DPO — novel combination of offline + online in one pass.
- SFT-loss + EMA stabilization specific to this release.

vs 2024 RLHF norms:
- No separate PPO stage — unusual for models of this scale (most peers ran DPO → PPO or PPO directly).
- Explicit critique model at filter-time rather than as part of the RL reward.

## Key Figures/Tables to Study
- 4-tier critique-scoring rubric (accuracy/relevance/completeness/usefulness/clarity) — useful data-filtering template.
- Single-stage hybrid-DPO ablation vs sequential offline+online DPO.
- MoE expert-utilization plot (for context on why dropout is set high).

## Connections
- [[deepseek-v3]] — peer MoE generation; V3 uses SFT + GRPO + R1-distillation, different algorithmic route.
- [[qwen-2.5]] — contemporary SFT + DPO+PPO peer; contrasts with Hunyuan's DPO-only.
- [[dpo]] — base algorithm.

## Gaps / what the report does NOT disclose
DPO β, DPO LR, DPO batch size, EMA decay rate, exact online-branch rollouts per prompt, RM training-data size, RM architecture beyond "70B dense Hunyuan series," quantitative ablations isolating EMA vs SFT-loss contributions. Multilingual post-training details thin.
