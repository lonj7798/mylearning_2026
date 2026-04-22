<!-- scope: Llama 4 (Scout/Maverick/Behemoth) post-training pipeline
     deps: [[llama-3]]
     see-also: [[deepseek-v3]], [[qwen-3]]
-->

# Llama 4
- **Core Insight:** Continuous online RL alternating policy updates with prompt-difficulty refiltering replaces classical single-pass RLHF — and curriculum via pass@k analysis replaces fixed datasets.
- **Guideline:** Throw away ≥50% of your SFT data (the "easy" half) before training; keep only prompts where the policy can still learn.

- **Authors / Lab:** Meta AI (Llama team)
- **Year:** 2025 (Llama 4 announced April 2025; Scout + Maverick open weights; Behemoth still training at release)
- **URL:** https://ai.meta.com/blog/llama-4-multimodal-intelligence/
- **Relevant topics:** mixture-of-experts post-training, curriculum RL, co-distillation, pass@k-driven curriculum, online RL

## Abstract
Llama 4 is Meta's first natively multimodal and first MoE open-weight generation. Scout = 17B active / 16 experts (109B total); Maverick = 17B active / 128 experts (400B total); Behemoth (teacher) = 288B active / 16 experts (~2T total). Post-training is the explicit focus: Meta redesigned the recipe around (1) aggressive difficulty filtering before SFT, (2) a continuous online-RL loop with curriculum, and (3) a new soft+hard distillation loss from Behemoth → Maverick. No separate paper has been published — the blog post is the primary technical source.

## Key Contributions
- **Lightweight SFT → online RL → lightweight DPO** replaces the Llama 3 "heavy SFT + DPO" pipeline; SFT is minimized to avoid over-fitting the model to easy distribution.
- **Easy-prompt pruning:** ≥50% of SFT candidates marked "easy" by Llama judges are removed.
- **Continuous online RL loop:** alternates between RL training and using the current policy to refilter prompts, keeping only medium-to-hard remaining.
- **Pass@k curriculum** (used for Behemoth): sample hard prompts by running pass@k on the current policy; only prompts where the policy sometimes-but-not-always succeeds stay.
- **Co-distillation from Behemoth → Maverick** using a "novel distillation loss that dynamically weights the soft and hard targets through training."
- First open-weight natively multimodal MoE; 10M-token context for Scout (claimed).

## Post-training pipeline
- **SFT data:** Not publicly sized; filtered by Llama-as-judge; >50% of the initial pool discarded as "easy." Multimodal SFT included.
- **Preference / RL algorithm:** Online RL (algorithm not named publicly — assumed PPO-family based on Llama 3 lineage, but not confirmed). Final stage is lightweight DPO on top of the RL policy.
- **Reward model:** Meta has NOT disclosed whether a classical RM is used or whether rewards are verifier-driven; blog implies model-as-judge for prompt filtering plus domain-specific signals (math/code/logic verifiers) inside RL.
- **KL / entropy handling:** Not disclosed.
- **Rollout scale:** Not disclosed. Behemoth-scale RL used "extensive asynchronous online RL" — Meta rebuilt their RL infra for Behemoth, ~10× throughput vs prior gen (per blog, no absolute numbers).
- **Hyperparameters:** Not disclosed (LR, batch size, clip ε, group size, number of RL steps).
- **Verifiable rewards:** Implied for math / logic / code domains — "pass@k evaluation and curriculum sampling to strengthen performance in math, logic, and coding."
- **Self-improvement / iterative:** Yes — the continuous RL loop is self-improving in that the model filters its own next-round training set via pass@k.

## Innovations vs predecessors
Changes from **Llama 3**:
- First-ever MoE generation for Llama (Llama 3 was dense).
- First natively multimodal — early-fusion vision tokens through all layers vs Llama 3's bolted-on adapter.
- Single-pass RLHF (Llama 3: SFT → RM → DPO + PPO) replaced by the continuous online-RL curriculum loop.
- Explicit "easy prompt pruning" — Llama 3 training mix was broader; Llama 4 prunes aggressively.
- Behemoth-as-teacher co-distillation — Llama 3 did not distill across model sizes at this scale.
- Infrastructure: fully asynchronous RL for Behemoth; moved away from the synchronous generation/training stack used for Llama 3.

## Key Figures/Tables to Study
- Curriculum diagram (blog) showing policy-driven prompt refiltering loop — explains why "continuous" matters.
- MoE architecture diagram — expert count / active-param split; context for why Maverick was chosen as the distillation target (128 experts enable capacity without activating them all).

## Connections
- [[llama-3]] — prior dense generation; Llama 4 post-training explicitly departs from Llama 3's SFT + DPO+PPO template.
- [[deepseek-v3]] — comparison point for MoE scale (Maverick 400B total vs V3 671B total).
- [[qwen-3]] — fellow 2025 MoE frontier release with published post-training.

## Gaps / what the report does NOT disclose
Substantially more opaque than Llama 3's 73-page paper. Not disclosed: specific RL algorithm (PPO? GRPO? custom?), RM architecture or whether one is trained, KL coefficient, learning rate, batch size, clip ε, rollouts per prompt, RL step counts, SFT token counts, exact pass@k parameters, distillation-loss formula, MoE routing details, pretraining data mix, exact multimodal SFT size. Meta pivoted to blog-only disclosure for Llama 4 — no tech report exists at time of writing. Behemoth weights not released.
