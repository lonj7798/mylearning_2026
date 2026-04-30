---
chapter: ch-20
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/model-reports/deepseek-r1.md
source_url: https://arxiv.org/abs/2501.12948
created_at: "2026-04-23"
---

# Excerpt: DeepSeek-R1 — the 800K RS-SFT corpus that became the distill template

**Source library:** `wiki/raw-data/llm-training/model-reports/deepseek-r1.md`
**Paper:** DeepSeek-AI, *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning*, 2025 (Nature 645:633–638).

---

## Why this source anchors ch-20

R1 is a *pair* of artifacts: the 671B reasoning model and the 800K reasoning-trace corpus derived from it. For ch-20, the model is almost a sidebar — the chapter on RL-for-reasoning is ch-24 — and **the corpus is the main subject**. Ch-20 §3 walks through the pipeline that produced the 800K; this excerpt extracts the concrete numbers that make that walkthrough possible.

From source line 6–7:

> **Core Insight:** Pure RL on a base model, with only rule-based rewards for correctness + format, causes reasoning behaviors (reflection, verification, backtracking) to emerge without any SFT on reasoning traces.
>
> **Guideline:** For verifiable domains (math/code), skip the reasoning-SFT bootstrap; start RL directly from the base model and let long CoT emerge.

These are the RL-side claims; ch-24 unpacks them. The ch-20 claim is downstream: **the RL-trained model's trace distribution, filtered by RS-SFT, is the strongest public distillation corpus of 2025**.

---

## The four-stage pipeline (source lines 31–49)

```
Stage 0: Base = DeepSeek-V3-Base (671B MoE, inherited from V3 pretraining)

Stage 1: Cold-start SFT
  ~thousands of hand-cleaned long-CoT examples with human-readable format
  purpose: fix R1-Zero's readability / language-mixing problems

Stage 2: Reasoning RL (GRPO + rule-based reward)
  Learning rate   : 3e-6
  KL coefficient  : 0.001
  GRPO clip ε     : 10 (intentionally loose — tight clipping destroys exploration)
  Rollout T       : 1.0
  Group size G    : 16 samples per prompt
  Max gen length  : 32,768 tokens
  Batch size      : 32 unique prompts/step → 32 × 16 = 512 samples/step

Stage 3: Rejection-Sampling SFT (the corpus-production stage)
  Stage-1 RL model generates multiple traces per prompt
  V3 judge filters for readability + correctness
  Output: ~600K reasoning + ~200K non-reasoning = 800K total
  This is the corpus used to train the 6 distilled students.

Stage 4: Stage-2 Alignment RL
  Second RL run with helpfulness + harmlessness preference rewards
  (used for the final R1 model only; not part of the distill corpus)
```

Two things are worth calling out because they are not obvious from reading about R1 through press coverage:

- **The distill corpus comes from Stage 3, not Stage 2.** People say "R1 traces" — what they mean is "traces sampled from the stage-1 RL model and filtered by V3." The stage-1 model is a *snapshot* of R1 after reasoning RL but before alignment RL. It's the maximally-reasoning-capable version; the final R1 has been softened by alignment.
- **The loose clip ratio ε = 10 is not a typo.** Standard PPO/GRPO uses ε ∈ [0.1, 0.3]. DeepSeek argues (and ch-24 will cover) that tight clipping in the reasoning-RL regime kills exploration — the policy needs to make large moves to explore long-CoT trajectories — so they use ε = 10, which in practice means the clip rarely triggers and GRPO behaves more like reward-weighted regression with KL stabilization.

---

## The 800K composition

Source line 48:

> **Rejection-sampling SFT:** Use stage-1 RL model to generate data; filter via V3 judge; ~600K reasoning + 200K non-reasoning.

The report does not give a per-source breakdown of the 600K reasoning half. Inference from the distill students' evaluation profile (strong on MATH, AIME, Codeforces, GPQA; decent on MMLU) and the verifier stack suggests roughly:

| Slice              | Approx. size | Verifier                       |
|--------------------|--------------|--------------------------------|
| Math (competition) | 200–300K     | SymPy on \boxed{} answer       |
| Code               | 200–300K     | Unit-test execution            |
| Logic / science    | 50–100K      | V3-judge (LLM)                 |
| Non-reasoning SFT  | 200K         | V3-judge for quality           |

This is the single most important ambiguity for anyone reproducing R1. Bespoke-Stratos and Open-R1 are in part efforts to *disambiguate* this split by building analogous corpora with publicly-documented compositions.

---

## The distillation line (source lines 51–54)

```
6 distilled students, all SFT-only (no RL on student):
  Qwen-2.5-Math  1.5B / 7B / 14B / 32B
  Llama-3.1-8B
  Llama-3.3-70B

Training:  pure SFT on the 800K corpus (one pass; no multi-epoch reported)
```

The lesson the R1 report explicitly states (and ch-20 §3 quotes): **dense students benefit more from copied reasoning structure than from rediscovering it via their own RL**. This is a budget argument — a dense 32B model doing RL from scratch on reasoning problems would cost multiple orders of magnitude more than SFT-on-R1-traces, and produce a weaker model because the 32B dense base doesn't have R1's 671B-MoE capacity to discover long-CoT trajectories. The distillation transfers a behavior the dense student could not learn directly.

---

## Benchmark numbers (source lines 60–64)

```
AIME 2024 pass@1   :  R1 79.8%,  R1-Zero 71.0% (majority-vote pass@1 86.7%),
                     V3 39.2%
MATH-500           :  R1 97.3%
Codeforces         :  Elo 2029, 96.3rd percentile
MMLU               :  90.8%
```

R1-Distill-Qwen-32B (the flagship distilled checkpoint, trained on the 800K):
- AIME24 ~72%, MATH500 ~94%, LiveCodeBench ~57% — beats o1-mini on MATH and AIME.

This is the target the open reproductions (Stratos, Open-R1, Sky-T1) aim at. Stratos-32B reaches within 2–3 points with 1/47 the data volume (17K vs 800K); Sky-T1-32B is ~20 points behind on AIME because its teacher (QwQ) is weaker than R1.

---

## How ch-20 cites this

The read uses this paper as the anchor for §3 (R1-distill pipeline) and §4 (the reference point the open reproductions are compared against). The §7 licensing table cites the MIT weight license — the specific clause that permits redistribution of R1 outputs — which is the reason the open-reproduction community exists at all. A proprietary-weight reasoning model of comparable quality (o1, Claude-3.5-Sonnet) would have produced zero redistributable distill corpora under current API terms of service.
