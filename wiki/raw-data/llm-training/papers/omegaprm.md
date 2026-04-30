<!-- scope: reasoning-trace synthesis — automated MC rollout step-labeling for PRM training
     deps: [[math-shepherd]], [[prm800k]]
     see-also: [[rstar-math]], [[step-dpo]]
-->

# OmegaPRM: Improve Mathematical Reasoning in LMs by Automated Process Supervision
- **Core Insight:** Step-level correctness labels for PRM training can be generated automatically by running Monte-Carlo completions from every intermediate step of a CoT; a step is labeled "correct" iff a sufficient fraction of its MC rollouts reach the gold answer — no human annotation required.
- **Guideline:** To build a PRM without PRM800K-style human labels, take a pool of CoT trajectories, run K=8–32 completions from each prefix, and use the empirical rollout success rate as the step's soft target; use the **divide-and-conquer MCTS** variant to save compute on long trajectories.
- **Authors:** Liangchen Luo, Yinxiao Liu, Rosanne Liu, Samrat Phatale, Harsh Lara, Yunxuan Li, Lei Shu, Yun Zhu, Lei Meng, Jiao Sun, Abhinav Rastogi (Google DeepMind)
- **Year:** 2024
- **URL:** https://arxiv.org/abs/2406.06592
- **Relevant topics:** process reward models, automated step labeling, Monte-Carlo rollouts, math reasoning

## Abstract
OmegaPRM proposes a divide-and-conquer Monte-Carlo Tree Search to automate step-level labeling for process reward models. Rather than asking humans to mark each step correct/incorrect (as in PRM800K) or using single-rollout estimates (as in Math-Shepherd), OmegaPRM binary-searches the trajectory for the first incorrect step by evaluating MC completion success at tree nodes. The resulting 1.5M step-labeled dataset trains a PRM that, when used for weighted best-of-N sampling, lifts Gemini Pro's MATH score by 69.4% relative.

## Key Contributions
- **Divide-and-conquer MC** — O(log L) completion cost to find the first error in an L-step trajectory, vs O(L) for Math-Shepherd.
- **1.5M-step-label dataset** generated fully automatically for PRM training.
- **Per-step soft label** = fraction of MC rollouts from that step that yield the correct final answer (MC-value).
- Strong best-of-N downstream: Gemini Pro 1.0 MATH 51% → 69.4% with PRM-weighted selection.

## Synthesis pipeline (REQUIRED — concrete, modality-specific)
- **Seed input:** math problem with gold answer + a seed trajectory (sampled from the generator policy).
- **Divide-and-conquer MC tree search:**
  1. Split trajectory T of length L into halves at midpoint m.
  2. Run K Monte-Carlo completions from the prefix T[0..m] with the policy; compute p_m = fraction correct.
  3. If p_m ≈ p_0 (prefix correctness preserved → no error in first half), recurse on second half. Else recurse on first half.
  4. Binary-search down to the first step where MC success probability drops sharply — this is the first error.
- **Step label:** each step t gets label `MC_t = fraction of K completions from prefix T[0..t] that reach gold answer`, used as the regression target for the PRM.
- **Filtering:**
  - Exclude trajectories whose first step already fails.
  - Exclude problems with no trajectory reaching the gold answer in K tries.
- **Output shape:** 1.5M step-level labels over ~80K problems; trajectories average ~10 steps.
- **Teacher / policy model:** Gemini Pro 1.0 as the rollout policy.
- **Cost / compute:** ~100K TPU-hours for rollouts.

## MC-value formal definition (REQUIRED)

For a step s_t in trajectory (s_1, …, s_L):

```
MC(s_t) = (1/K) · Σ_{i=1}^{K} 𝟙[rollout(policy | s_1..s_t) yields gold answer]
```

PRM is trained to regress onto MC(s_t) via MSE or cross-entropy on soft labels. A step is "wrong" if MC(s_t) falls below a threshold τ (authors use τ ≈ 0.2) AND its parent's MC is above τ.

## Modality-specific technical details (REQUIRED — reasoning-trace)
- **Reasoning length distribution:** seed trajectories ~500–1500 tokens, 8–15 steps.
- **Trace style:** standard CoT.
- **Correctness verifier:** gold-answer exact-match at trajectory level; step label = statistical MC.
- **Efficiency gain vs Math-Shepherd:** Math-Shepherd uses a single completion per step; OmegaPRM's divide-and-conquer reduces total rollout count from O(L·K) to O(K·log L) without losing label fidelity.

## Quality / diversity evaluation
- Trained PRM used with **weighted best-of-N** (PRM score × policy log-prob): Gemini Pro MATH 51.0 → **69.4** (+18.4 absolute).
- Beats Math-Shepherd PRM by ~5 MATH points at equal compute.
- MC-value labels more stable than single-rollout labels, especially for deep trajectories.

## Risks + gotchas
- **K too small → noisy labels:** authors recommend K ≥ 16 for reliable MC estimates.
- **Gold-answer dependence:** still requires ground-truth final answers; zero-supervision extensions are future work.
- **Threshold τ sensitivity:** PRM quality varies with binary vs soft labeling; authors use soft MC regression to avoid threshold tuning.
- Compute remains high despite divide-and-conquer.

## Connections
- Direct ancestor: [[math-shepherd]] (single-rollout MC); direct successor approach: [[rstar-math]] PPM (pairwise instead of scalar MC).
- Contrasts human-labeled process supervision: [[prm800k]], [[lets-verify]].
- Complementary to [[step-dpo]] (preference-pair extraction from step labels).
