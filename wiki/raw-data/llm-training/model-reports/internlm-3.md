<!-- scope: Shanghai AI Lab InternLM3 — 4T-token pretrain + hybrid thinking post-training
     deps: []
     see-also: [[qwen-3]], [[deepseek-r1]]
-->

# InternLM3
- **Core Insight:** 4T high-quality tokens + a hybrid deep-thinking / normal-response post-training beat larger-token budgets at similar scale — efficiency per token matters more than absolute count.
- **Guideline:** At 7B–8B scale, don't chase 15T tokens; invest in quality and post-training-mode unification instead.

- **Authors / Lab:** Shanghai AI Laboratory (InternLM team)
- **Year:** 2025 (Jan 15, 2025)
- **URL:** https://huggingface.co/internlm — https://github.com/InternLM/InternLM-techreport
- **Relevant topics:** hybrid deep-thinking / normal mode, IQPT metric, 4T-token pretraining, open release

## Abstract
InternLM3 is Shanghai AI Lab's January 2025 release focused on data efficiency: 4T training tokens (>75% less than same-scale peers) for an 8B model that integrates deep-thinking mode (long-CoT) and normal response mode in a single checkpoint. Positioned around "Intelligence Quality per Token" (IQPT) as a new efficiency metric. Post-training details are relatively thin in public surfaces compared to peer open releases.

## Key Contributions
- **Hybrid thinking / normal mode** in one model — preceded Qwen3's public rollout of the same pattern.
- **IQPT framing** — intelligence per training token — as a positioning metric.
- **4T-token training budget** vs 15T+ in contemporaries.
- Open weights + tech repo on GitHub; used as a base for the Intern-S1 scientific multimodal follow-up (which adds Mixture-of-Reward).

## Post-training pipeline
- **SFT data:** not publicly itemized.
- **Preference / RL algorithm:** not named publicly in surfaced extracts. Intern-S1 (scientific successor) uses offline + online RL with a **Mixture-of-Reward (MoR)** framework unifying diverse feedback forms into a single scalar — same methodology likely underlies InternLM3 but is not explicitly documented.
- **Reward model:** not disclosed for InternLM3 proper.
- **KL / entropy handling:** not disclosed.
- **Rollout scale:** not disclosed.
- **Hyperparameters:** not disclosed.
- **Verifiable rewards:** implied by thinking-mode training on math / code but no explicit recipe disclosed.
- **Self-improvement / iterative:** not disclosed.

## Innovations vs predecessors
Changes from **InternLM2 / 2.5 → InternLM3**:
- First InternLM release with explicit deep-thinking mode toggle.
- 4T-token training — smaller than prior generations' token counts relative to performance.
- Release predates Qwen3 for hybrid-thinking (Jan 2025 vs May 2025) but with less detailed post-training publication.

## Key Figures/Tables to Study
- Benchmark vs peers at 4T vs 15T+ tokens — the IQPT evidence.
- Hybrid-mode ablation (if published in the repo tech-report).

## Connections
- [[qwen-3]] — shares hybrid thinking pattern; Qwen3 publishes recipe details InternLM3 does not.
- [[deepseek-r1]] — thinking-mode peer.
- Intern-S1 (scientific follow-up, not in this library) — introduces Mixture-of-Reward framework.

## Gaps / what the report does NOT disclose
Most post-training detail is absent from the surfaced public material. Not disclosed: RL algorithm, RM, KL, LR, batch size, clip ε, group size, rollouts per prompt, step count, SFT data size, thinking-mode fusion procedure, verifier inventory. Full technical recipe reportedly documented in the InternLM-techreport GitHub — chapter authors should pull directly from that repo when writing.
