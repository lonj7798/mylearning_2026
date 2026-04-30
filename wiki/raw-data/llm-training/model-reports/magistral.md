<!-- scope: Mistral Magistral reasoning model — GRPO without KL, async RL infra
     deps: [[grpo]]
     see-also: [[deepseek-r1]], [[qwen-3]]
-->

# Magistral (Mistral)
- **Core Insight:** GRPO with the KL term removed entirely and asymmetric clipping (ε_low ≈ 0.2, ε_high ≈ 0.26–0.28) is more compute-efficient and remains stable when coupled with strict advantage-normalization and zero-group filtering.
- **Guideline:** Don't default to KL=0.001. Measure it — if reference model is a good SFT start, KL=0 saves compute and works.

- **Authors / Lab:** Mistral AI
- **Year:** 2025 (Jun 12, 2025)
- **URL:** https://arxiv.org/abs/2506.10910
- **Relevant topics:** GRPO, asymmetric clipping, KL-free RL, async RL infra, format + length + language-consistency reward, SymPy verifier

## Abstract
Magistral Small (24B, Apache 2.0) and Medium are Mistral's first reasoning models, built from Mistral Small 3 / Medium 3 via a ground-up RL pipeline with no distillation from external reasoning models. Paper's core claims: (1) KL penalty can be removed, (2) asymmetric clipping (ε_high > ε_low) enables stable exploration, (3) asynchronous infra with continuous generator updates + in-flight weight broadcasts scales RL efficiently. Yields ~50% AIME-24 gain over baseline.

## Key Contributions
- **GRPO modifications:**
  - KL divergence **eliminated** (β = 0).
  - Asymmetric clipping: ε_low ≈ 0.2, ε_high ∈ [0.26, 0.28] (tuned during training).
  - Advantage normalization at minibatch level.
  - Zero-advantage groups filtered out before batch formation.
  - Loss normalized by total token count across all generations in the group.
- **Async RL infrastructure:** generators produce completions continuously; weight updates broadcast via NCCL without pausing in-flight sequences; KV caches retained from prior weight versions; greedy-by-size collation reduces padding by 19%.
- **Reward design (shaped):**
  - Formatting (0 / 0.1): requires `<think>/</think>`, `\boxed{}` for math, markdown code blocks.
  - Correctness (+0.9 if correct): math via SymPy normalization; code via C++/Python execution (10s compile timeout, 4s per test, 300MB mem, 20 randomly-selected test cases).
  - Length penalty (soft, two thresholds l_max and l_cache, up to −0.1).
  - Language consistency bonus (+0.1) via fastText classifier.
- **Negative results documented:** partial/proportional code rewards cost 2% final perf; entropy bonuses unstable — replaced with ε_high tuning.

## Post-training pipeline
- **SFT data:** not detailed publicly in the paper summary; small SFT-only stage exists for Magistral Small.
- **Preference / RL algorithm:** **GRPO variant** as described — KL-free, asymmetric-clip, zero-group-filtered.
- **Reward model:** rule-based only (no learned RM) — SymPy for math, execution for code, fastText for language, format-regex for structure, length-threshold for verbosity.
- **KL / entropy handling:** β = 0 (KL eliminated). Entropy bonuses tried and rejected; exploration controlled by ε_high instead.
- **Rollout scale:**
  - Magistral Medium: batch size schedule 8k → 4k → 2k (reduced as max completion length grew).
  - Magistral Small: RL batch 2048 sequences; ε_high 0.3 (higher to encourage exploration from cold start).
  - Max completion length 16k → 24k → 32k (progressively increased).
- **Hyperparameters:**
  - ε_low ≈ 0.2, ε_high ≈ 0.26–0.28 (Medium) / 0.3 (Small).
  - Temperature: 0.7 for math/GPQA, 0.95 for code.
  - Non-penalized length (l_max − l_cache) 16k → 24k → 32k.
  - LR and group size G not explicitly disclosed in extracts.
- **Verifiable rewards:** primary signal; no learned RM in the main RL loop.
- **Self-improvement / iterative:** progressive context-length extension drives a self-curriculum.

## Innovations vs predecessors
Changes from **Mistral Small 3 / Medium 3 → Magistral**:
- First Mistral reasoning models — prior generations did not ship a thinking-mode.
- **KL = 0** — atypical; most contemporary GRPO pipelines retain a small KL.
- **Asymmetric clipping** (ε_high > ε_low) — deliberate exploration-bias, distinct from PPO/GRPO defaults.
- Fully async RL infra with NCCL weight broadcasts, KV-cache retention, greedy padding reduction.
- Ground-up RL, no distillation from o1/R1/Claude — differentiates from phi-4-reasoning, DeepSeek-R1-Distill.

## Key Figures/Tables to Study
- AIME-24 pass@1 curve vs baseline — the +~50% headline gain.
- Batch-size / max-length schedule table — shows the progressive-context curriculum.
- Reward-shape breakdown figure — format + correctness + length + language weights.
- Async-infra diagram — rollout/train/weight-broadcast overlap.

## Connections
- [[grpo]] — algorithm baseline; Magistral modifies it.
- [[deepseek-r1]] — the pure-RL reasoning peer; R1 keeps KL.
- [[qwen-3]] — Stage-2 GRPO comparison point.
- [[kimi-k1-5]] — contrast: Kimi uses mirror descent + learned CoT RM; Magistral uses GRPO + rule-based rewards.

## Gaps / what the report does NOT disclose
Group size G not specified. Learning rate not published in extracts. KL=0 ablation vs small β not explicitly run. SFT dataset size + composition. AdamW betas. Number of RL steps. How ε_high was "carefully tuned during training" (no schedule). Multimodal / tool-use details — Magistral is text-reasoning-only at this release.
