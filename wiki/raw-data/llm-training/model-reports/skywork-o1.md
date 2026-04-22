<!-- scope: Skywork-OR1 open reasoning model — MAGIC (GRPO variant) with entropy scheduling
     deps: [[grpo]]
     see-also: [[deepseek-r1]], [[magistral]]
-->

# Skywork-OR1 (Open Reasoner 1)
- **Core Insight:** Preventing premature entropy collapse — via high rollout temperature (τ=1.0) plus adaptive entropy control — is the critical-path factor for long-CoT RL convergence at 32B scale.
- **Guideline:** Log entropy every step; if it's dropping below target, raise rollout temperature and adjust the adaptive α before anything else.

- **Authors / Lab:** Skywork AI (Kunlun Inc.)
- **Year:** 2025 (May 2025)
- **URL:** https://arxiv.org/abs/2505.22312
- **Relevant topics:** MAGIC (Multi-stage Adaptive entropy scheduling for GRPO In Convergence), entropy collapse, high-temperature rollouts, offline+online filtering

## Abstract
Skywork-OR1 is an open long-CoT reasoning model family (7B, 32B) with a published recipe. The 32B variant lifts AIME24+AIME25+LiveCodeBench average from 57.8% → 72.8% (+15.0%); the 7B variant from 43.6% → 57.5% (+13.9%). Outperforms DeepSeek-R1 and Qwen3-32B on math (AIME24 82.2, AIME25 73.3, LiveCodeBench 63.0). The core algorithmic contribution is **MAGIC** — a GRPO variant with length-normalization removed, adaptive entropy control, KL removed, and high-temperature rollouts.

## Key Contributions
- **MAGIC (Multi-stage Adaptive entropy scheduling for GRPO In Convergence):**
  - Token-level policy loss averaged across batch tokens; removes the 1/|y_ij| length-normalization term.
  - Adaptive entropy coefficient α_k adjusted based on current-vs-target entropy.
  - **KL loss omitted entirely**.
- **High-temperature rollouts:** τ = 1.0 (vs typical 0.6) — prevents premature entropy collapse.
- **Multi-stage context extension:** 8K → 16K → 32K with online filtering removing problems solved in the prior stage.
- **Data filtering:** offline removes 0%- and 100%-correct problems; online discards in-stage-solved problems.
- **Rejection of zero-advantage groups:** excluded from batches; keeps gradients informative.

## Post-training pipeline
- **SFT data:** not specified in extracts; starts from a cold-start-SFT checkpoint.
- **Preference / RL algorithm:** **MAGIC** (GRPO variant).
- **Reward model:** rule-based (math verifier, code execution) — no learned RM reported.
- **KL / entropy handling:** KL = 0 (removed). Entropy controlled by adaptive α_k against a **target entropy = 0.2**; high rollout temperature τ = 1.0 to maintain exploration.
- **Rollout scale:**
  - Group size M = 16.
  - Batch size 64–256 (varies by stage).
  - Mini-batch 32–128.
  - Context length stages: 8K → 16K → 32K.
- **Hyperparameters:**
  - Clip ε = 0.2.
  - Rollout temperature τ = 1.0.
  - Target entropy = 0.2.
- **Verifiable rewards:** yes — math verification + code execution.
- **Self-improvement / iterative:** online filtering between stages is an implicit self-curriculum; solved problems are pruned in the next stage.

## Innovations vs predecessors
vs **DeepSeek-R1 / Qwen3-32B / standard GRPO**:
- KL term removed (R1 keeps KL; Magistral also removes it — convergent finding).
- Length-normalization term removed (contra original GRPO).
- Adaptive entropy scheduling formalized — R1 and Qwen3 do not publish this explicitly.
- τ = 1.0 rollouts vs τ = 0.6 standard — a deliberate entropy-preservation choice.
- Multi-stage online filtering with zero-advantage group rejection — a dedicated batch-composition recipe.

## Key Figures/Tables to Study
- Entropy-collapse comparison: τ = 0.6 vs τ = 1.0 rollouts — the headline empirical claim.
- Benchmark progression over stages (8K → 16K → 32K).
- AIME24/25 + LiveCodeBench curves vs DeepSeek-R1 / Qwen3-32B.

## Connections
- [[grpo]] — base algorithm.
- [[deepseek-r1]] — the benchmark to beat.
- [[magistral]] — independent convergent finding that KL can be removed.
- [[entropy-mechanism-llm-rl]] — formal treatment of the entropy phenomena MAGIC engineers around.

## Gaps / what the report does NOT disclose
Learning rate. Exact adaptive-α update rule. SFT dataset composition / size. Number of RL steps per stage. GPU count / wall-clock. Whether a PRM is used anywhere (unlikely from extracts but not explicitly excluded). Full formal derivation of MAGIC's loss is in the paper body; extraction here is from abstract + review notes.
