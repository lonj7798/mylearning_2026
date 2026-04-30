<!-- scope: Tulu 3 fully-open post-training pipeline from Allen AI
     deps: [[README]]
     see-also: [[olmo-2]], [[llama-3]], [[rlvr-tulu3]], [[dpo]]
-->

# Tulu 3: Pushing Frontiers in Open Language Model Post-Training
- **Core Insight:** RL with Verifiable Rewards (RLVR) — skip the reward model and use task verifiers (math checker, code exec, IFEval) — is the missing third leg of open post-training.
- **Guideline:** Follow SFT -> DPO -> RLVR; RLVR lifts reasoning benchmarks without requiring a reward model.
- **Authors:** Nathan Lambert, Jacob Morrison, Valentina Pyatkin, et al. (Allen AI)
- **Year:** 2024
- **URL:** https://arxiv.org/abs/2411.15124
- **Relevant topics:** Fully open recipe, SFT data curation, DPO tuning, RLVR / verifiable rewards, PPO for LLMs

## Abstract
Tulu 3 is a family of fully-open post-trained models built on Llama 3.1 base. The recipe covers three stages — SFT, DPO, and RLVR (Reinforcement Learning with Verifiable Rewards). Tulu 3 matches or beats Llama 3.1 Instruct, Qwen 2.5 Instruct, and closed models (GPT-4o-mini, Claude 3.5 Haiku) on standard benchmarks while releasing all data, code, and evaluation tooling. The report is the most detailed open disclosure of a modern post-training pipeline to date.

## Key Contributions
- RLVR: PPO driven by task-specific binary verifiers (exact match, code-test pass, IFEval constraint check) — no reward model.
- Fully public SFT mix drawn from 939,344 prompts (57% public, 43% in-house synthetic) and the preference pool for DPO.
- Detailed ablations: per-mixture SFT contribution, DPO vs RLVR gain, which verifiers help which benchmark.
- Safety-specific DPO slice built from red-team prompts.
- Released 8B, 70B, and 405B instruct checkpoints, evaluation suite, training code (`open-instruct`).

## Key Figures/Tables to Study
- **Figure 1** (Three-stage pipeline): SFT -> DPO -> RLVR, with per-stage benchmark gains.
- **Table of SFT mixture:** per-source counts across 939K total prompts.
- **RLVR config block:** all PPO hyperparameters in one place.
- **Figure on reward over training:** for RLVR with verifiable vs learned reward.

## Technical Details — Post-Training Pipeline

### SFT
- **Total prompts:** 939,344 (57% public sources incl. WildChat/OpenAssistant, 43% synthetic/in-house).
- **Model sizes:** 8B / 70B / 405B (Llama 3.1 base).
- Train for 2 epochs; standard completion-masked loss.

### DPO
- **Preference data size:** hundreds of thousands of pairs, curated from on-policy sampling of the SFT model + reward model ranking.
- **Beta:** 5.0 (length-normalized DPO) for 8B; different values per size.
- **Learning rate:** 5e-7 for 8B DPO.
- **Objective variant:** length-normalized DPO (helps avoid length hacking).

### RLVR (the signature contribution)
RLVR is PPO whose scalar reward is a deterministic verifier output (1 if correct, 0 otherwise) plus a KL penalty to the reference SFT/DPO model. No reward model trained.

- **Algorithm:** PPO (not GRPO).
- **Learning rate:** 3e-7
- **Beta (KL coeff):** 0.05
- **Clip epsilon:** 0.2
- **PPO update epochs (K):** 4
- **Mini-batches per update (N_mb):** 1
- **GAE lambda:** 0.95; **gamma:** 1.0
- **Local mini batch size:** 32; **local rollout batch size:** 32.
- **Total episodes:** 10,000,000.
- **Verifiers used:**
  - GSM8K / MATH: exact-match / sympy equivalence.
  - IFEval: constraint-satisfaction checker.
  - Code tasks: unit-test execution.

### What RLVR buys
Measured gains relative to DPO-only checkpoint: +5–10pp on GSM8K, +~4pp on IFEval, neutral-to-positive on other evals. No reward hacking observed because the verifier is ground-truth.

### Scale
Tulu 3 8B post-training fits on a single 8xH100 node; 70B requires multi-node; 405B requires FSDP + sequence parallel. All releases include exact training configs.

## Connections
- [[olmo-2]] — applies the Tulu 3 recipe unchanged to Allen AI's OLMo 2 base models.
- [[rlvr-tulu3]] — dedicated methodology page for the RLVR component.
- [[llama-3]] — contrast: Llama 3 uses DPO-only (no RLVR), no reward model either; Tulu 3 adds verifiable RL as a third stage.
- [[deepseekmath]] — GRPO alternative to PPO for similar verifiable-reward setup.
- [[dpo]] — DPO is stage 2 of the Tulu recipe.
- [[wildchat]] — key SFT data source (real user logs).
