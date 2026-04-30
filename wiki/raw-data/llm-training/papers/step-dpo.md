<!-- scope: reasoning-trace synthesis — step-level preference synthesis for DPO on reasoning
     deps: [[dpo]], [[math-shepherd]]
     see-also: [[omegaprm]], [[prm800k]], [[rstar-math]]
-->

# Step-DPO: Step-wise Preference Optimization for Long-chain Reasoning of LLMs
- **Core Insight:** Applying DPO to *individual reasoning steps* — not full trajectories — gives a large signal-to-noise gain because errors in long CoTs are localized; an automated pipeline can produce ~10K step-preference pairs that beat full-trajectory DPO by wide margins.
- **Guideline:** For reasoning alignment via DPO, identify the **first erroneous step** in a wrong trajectory, pair it with a corrected step produced by a stronger model, and optimize only that step — not the entire sequence.
- **Authors:** Xin Lai, Zhuotao Tian, Yukang Chen, Senqiao Yang, Xiangru Peng, Jiaya Jia (CUHK / SmartMore)
- **Year:** 2024
- **URL:** https://arxiv.org/abs/2406.18629
- **Relevant topics:** preference optimization, reasoning, step-level DPO, process supervision

## Abstract
Step-DPO argues that vanilla DPO on long reasoning chains is noisy because the loss averages rewards over many correct and few incorrect tokens. Step-DPO instead constructs a preference dataset of (prefix, correct-step, incorrect-step) triplets by locating the first wrong step in sampled trajectories and replacing it with a corrected step from a stronger model. A 10K-pair step-DPO run on Qwen2-7B-Instruct lifts MATH from 53.0% → 58.6% and GSM8K from 85.5% → 87.9% — larger than full-trajectory DPO on 10× more pairs.

## Key Contributions
- **Step-DPO loss:** DPO where the completion is a single step given a fixed prefix, not a full answer.
- **Automated pipeline** for constructing step-preference pairs using a stronger model + a gold-answer checker.
- **~10K-pair Step-DPO dataset** (Step-DPO-10K) release.
- Empirical gains on MATH, GSM8K, AIME, compatible with any DPO-capable base.

## Synthesis pipeline (REQUIRED — concrete, modality-specific)
- **Seed input:** math problem set with gold answers (MATH train + GSM8K train + additional competition problems).
- **Step 1 — collect incorrect trajectories:** sample K CoT completions from the policy model; keep those whose final answer is wrong.
- **Step 2 — locate first erroneous step:** prompt a stronger model (GPT-4 / Qwen2-72B) with the problem and the wrong trajectory segmented into steps; ask it to identify the index of the first incorrect step.
- **Step 3 — generate corrected step:** given the (problem, prefix-up-to-error), prompt the stronger model to produce a correct next step. Verify by continuing the trajectory and checking the final answer — if correct, accept the corrected step.
- **Step 4 — form Step-DPO pair:** `(prefix_i, step_correct, step_incorrect)` where both steps share the same prefix.
- **Filtering:**
  - Reject pairs where stronger model's continuation also fails.
  - Reject pairs where the "incorrect" step actually still leads to the gold answer (false positive).
- **Output shape:** ~10K triplets; each step averages 30–120 tokens.
- **Teacher model(s):** GPT-4 / Qwen2-72B for step localization and correction.
- **Cost / compute:** ~$5K–$10K in GPT-4 API for the 10K-pair build.

## Step-DPO loss (REQUIRED)
Given step-preference triplet (x, y_w, y_l) where y_w / y_l share prefix x:

```
L_StepDPO = -log σ( β · log[π_θ(y_w|x) / π_ref(y_w|x)]  -  β · log[π_θ(y_l|x) / π_ref(y_l|x)] )
```

Same functional form as vanilla DPO; the distinction is granularity — x is a multi-step prefix, y_w/y_l are single reasoning steps.

## Modality-specific technical details (REQUIRED — reasoning-trace)
- **Reasoning length distribution:** prefixes 200–1000 tokens, chosen steps 30–120 tokens.
- **Trace style:** standard CoT with explicit step segmentation (`Step 1: …  Step 2: …`).
- **Correctness verifier:** trajectory continuation + gold-answer exact-match validates whether corrected step is truly correct.
- **Why step-level beats trajectory-level DPO:** under KL-constrained optimization, gradient is dominated by tokens with largest log-prob gap; when most of a long trajectory is identical between chosen and rejected, the effective signal is diluted. Step-DPO concentrates signal on the actual disagreement.

## Quality / diversity evaluation
- Qwen2-7B-Instruct: MATH 53.0 → **58.6**; GSM8K 85.5 → **87.9** with Step-DPO-10K.
- Full-trajectory DPO on 100K pairs: MATH 54.3 — worse than Step-DPO with 10× less data.
- Scales to Qwen2-72B: MATH 70.8% → 79.5%.

## Risks + gotchas
- **Stronger-teacher dependency:** step-localization + correction requires GPT-4-class teacher; the quality of Step-DPO data is bounded by it.
- **Step-segmentation ambiguity:** "first wrong step" is sometimes ill-defined when multiple steps jointly err; authors use teacher judgement.
- **Not a process reward model:** Step-DPO is pairwise preference, not a scalar step-value — complementary to [[math-shepherd]], [[omegaprm]].

## Connections
- Preference-optimization ancestry: [[dpo]], [[kto]], [[simpo]].
- Step-level process supervision lineage: [[prm800k]], [[math-shepherd]], [[lets-verify]], [[omegaprm]].
- Contrasts trajectory-level preference: [[ultrafeedback]], [[west-of-n]].
- Complements MCTS-based step-preference extraction: [[rstar-math]].
