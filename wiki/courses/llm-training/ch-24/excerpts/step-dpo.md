---
chapter: ch-24
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/step-dpo.md
source_url: https://arxiv.org/abs/2406.18629
created_at: "2026-04-23"
---

# Excerpt: Step-DPO — DPO applied at the step granularity

**Source library:** `wiki/raw-data/llm-training/papers/step-dpo.md`
**Paper:** Lai et al. 2024, "Step-DPO: Step-wise Preference Optimization for Long-chain Reasoning of LLMs"

---

## Why this source anchors ch-24 §6

Step-DPO is the cleanest demonstration that **moving the preference-optimization granularity from trajectory to step** produces a large signal-to-noise win on reasoning tasks. It is also the paper that justifies ch-24's insistence that DPO-style preference data for reasoning must be *curated at the step level*, not sampled as full-response pairs.

---

## The data-construction pipeline

From the source (§Synthesis pipeline):

1. **Collect incorrect trajectories**: sample K CoTs from the policy; keep ones whose final answer is wrong.
2. **Locate first erroneous step**: prompt a stronger model (GPT-4 or Qwen2-72B) with (problem, wrong-trajectory-segmented-into-steps); ask for the index of the first incorrect step.
3. **Generate corrected step**: prompt the stronger model to produce a correct next step given (problem, prefix-up-to-error). Verify by continuing the trajectory and checking final answer; accept only if correct.
4. **Form triplet**: `(prefix_i, step_correct, step_incorrect)` — both steps share the same prefix.

Filters from the same section:
- Reject pairs where the stronger model's continuation itself fails.
- Reject pairs where the "incorrect" step actually still leads to gold (false-positive error localization).

Output: **~10K triplets**, step lengths 30-120 tokens each. GPT-4 API cost ~$5-10K.

---

## The Step-DPO loss — identical form, different granularity

From the source (§Step-DPO loss):

Given step-preference triplet (x, y_w, y_l) where y_w and y_l share prefix x:

```
L_StepDPO = -log σ( β · log[π_θ(y_w|x) / π_ref(y_w|x)]
                   - β · log[π_θ(y_l|x) / π_ref(y_l|x)] )
```

Same functional form as vanilla DPO. The distinction is entirely in **x and y_w/y_l**:

- **Vanilla DPO**: x = problem, y_w/y_l = full answers (hundreds of tokens each).
- **Step-DPO**: x = problem + first k-1 steps (200-1000 tokens), y_w/y_l = single reasoning steps (30-120 tokens).

Ch-24 §6 carries the loss verbatim.

---

## The gradient-dilution argument — why granularity matters

From the source (§Modality-specific):

> Under KL-constrained optimization, gradient is dominated by tokens with largest log-prob gap; when most of a long trajectory is identical between chosen and rejected, the effective signal is diluted. Step-DPO concentrates signal on the actual disagreement.

Worked case: suppose a 1000-token wrong trajectory and a 1000-token correct trajectory share the first 800 tokens. Vanilla DPO computes the log-prob ratio over all 1000 tokens per side; 800 of those are identical (or near-identical) between the two completions, so the token-level log-prob gap averages near zero over 80% of the sequence. The gradient signal from the 200 tokens that *do* differ is diluted by a factor of ~5.

Step-DPO extracts only the disagreeing step. **x is the 800-token shared prefix**, **y_w / y_l are the ~80-token divergent steps**. The loss gradient concentrates on the actual disagreement; the shared prefix contributes to x (conditioning) but not to the log-prob ratio.

The empirical consequence, from the source (§Quality evaluation):

- Qwen2-7B-Instruct: MATH **53.0 → 58.6**, GSM8K **85.5 → 87.9** with Step-DPO-10K.
- Full-trajectory DPO on 100K pairs: MATH 54.3 — **worse than Step-DPO with 10× less data**.
- Qwen2-72B: MATH **70.8 → 79.5**.

Scale-invariance of the gradient-dilution argument: the 72B numbers shift linearly, so the step-level advantage does not diminish with model size.

---

## Where Step-DPO sits relative to PRM-based methods

From the source (§Risks + gotchas):

> Not a process reward model: Step-DPO is pairwise preference, not a scalar step-value — complementary to math-shepherd, omegaprm.

Step-DPO and OmegaPRM solve the same problem (step-level supervision for reasoning) via different machinery:

- **OmegaPRM** fits a scalar function r_φ(step, prefix) → [0,1] via MC rollout regression. Used at inference time for weighted best-of-N.
- **Step-DPO** fits a pairwise ranker implicitly by modifying the policy's log-prob surface. Used to directly improve the policy; no separate reward head.

Ch-24 §6 treats them as complementary: OmegaPRM-labeled step-values can be converted into step-preference pairs (take Q-gap > δ siblings) and consumed by Step-DPO. rStar-Math's PPM is effectively the formalized version of this combination.

---

## Caveats

From the source (§Risks + gotchas):

- **Stronger-teacher dependency**: step-localization + correction requires GPT-4-class teacher. Step-DPO data quality is teacher-capped.
- **Step-segmentation ambiguity**: "first wrong step" is ill-defined when multiple steps jointly err; authors rely on teacher judgment.
- No process-reward-model head — Step-DPO trains only the policy, so downstream best-of-N or RL with a separate reward must still provide its own signal.

The teacher dependency is the sharpest practical constraint. Without a strong teacher that can reliably identify step-level errors and emit corrections, Step-DPO reduces to vanilla trajectory DPO.

---

## Connections

- [[excerpts/omegaprm]] — scalar PRM route; step-level Monte-Carlo labels.
- [[excerpts/rstar-math]] — MCTS-native step-preference pairs; the PPM is pairwise like Step-DPO but trained as a separate head.
- [[ch-24]] §6 (step-level supervision), Track 4 (RL with step-preference inputs).
