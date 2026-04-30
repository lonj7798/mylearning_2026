---
chapter: ch-37
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/rloo.md
source_url: https://arxiv.org/abs/2402.14740
created_at: "2026-04-23"
---

# Excerpt: RLOO — why LLM-RL kills most of PPO's machinery

**Source library:** `wiki/raw-data/llm-training/papers/rloo.md`
**Artifact:** *Back to Basics: Revisiting REINFORCE-Style Optimization for Learning from Human Feedback in LLMs* — Ahmadian, Cremer, Gallé, Fadaee, Kreutzer, Pietquin, Üstün, Hooker, 2024. The paper that reopened the "is PPO necessary?" question and answered *no* for online RLHF.

---

## Why this source anchors ch-37

Ch-37 §4 argues that LLM post-training is structurally simpler than classical RL in four specific ways — deterministic dynamics, terminal-only rewards, long episodes, large discrete action space. Those four arguments are first assembled as a coherent whole in the RLOO paper, which then derives the consequence: the value network, GAE, clip, and K>1 epochs are all unnecessary or counterproductive. RLOO is the "minimum viable LLM-RL method" ch-37 describes in §4.

---

## The attested claim — what PPO brings that LLMs don't need

From the source (line 15, §Abstract):

> PPO has been positioned as the canonical RL algorithm for RLHF. We revisit the alignment of LLM RLHF with classical RL assumptions and show that many PPO components — value networks, GAE, multiple PPO epochs, clipping — are unnecessary or counterproductive in the LLM setting.

From the source (line 18, §Key Contributions):

> Analyzes which PPO assumptions break in LLM RLHF (deterministic transitions, full-trajectory rewards, long episodes).

These three are exactly the structural properties ch-37 §4 enumerates (plus the large discrete action space, which ch-37 adds from [[reinforce-plus-plus]]).

> Notice: the paper is not arguing that PPO is wrong. It is arguing that PPO's complexity is *insurance against properties the LLM setting does not have*. Removing insurance you don't need gives you a simpler, faster, more memory-efficient method with strictly better empirical Pareto curves. The generalisation is: when porting RL algorithms to LLMs, start by listing what the source algorithm protects against, then subtract the protections that are no-ops.

---

## The RLOO estimator

From the source (lines 34–37, §RLOO gradient estimator):

> `∇_θ J ≈ (1/k) Σ_{i=1..k} [ R(y_i, x) − (1/(k−1)) Σ_{j≠i} R(y_j, x) ] · ∇_θ log π_θ(y_i | x)`
> Baseline `b_i = (1/(k−1)) Σ_{j≠i} R(y_j, x)` is unbiased (independent of y_i given x).
> Reduces variance relative to a moving-average baseline `b_MA = (1/S) Σ_s R(x^s, y^s)`.

The estimator is the per-sequence form from ch-37 §2 with a specific, rollout-based baseline. Unbiasedness comes from Williams 1992's identity (because `b_i` conditions only on `(y_j){j≠i}` and `x`, it is independent of `y_i` given `x`). Variance reduction is the whole point: a moving-average baseline is a scalar computed over all past rollouts, which has high bias against the current policy and high variance when the reward distribution drifts; the leave-one-out mean is computed *on the current rollouts* so it tracks the policy.

> Notice: RLOO is literally three lines of code on top of SFT. Sample k completions, compute per-completion reward, subtract leave-one-out mean, run weighted cross-entropy. The simplicity is itself an argument — anything simpler than SFT is by definition more reproducible, and RLOO is *only one extra addition* on top of SFT (the leave-one-out subtraction).

---

## What is removed vs PPO

From the source (lines 45–51, §What is removed vs PPO):

| Component | PPO | RLOO |
|-----------|-----|------|
| Value network | required | removed |
| GAE | yes | no (full-sequence reward) |
| Clip ε | yes | no |
| Epochs per rollout K | 4 | 1 |
| Baseline | learned V | leave-one-out across k |

Four components out of five are removed. Memory footprint drops by ~50% (the value network is typically a copy of the policy with a scalar head). Compute drops because there's no critic-loss term and no K-fold minibatch loop. Ch-37 §4's claim — "what survives: the score-function estimator, the baseline, and the KL-to-reference regulariser" — is a direct paraphrase.

> Notice: K=1 (single gradient step per rollout) is surprising at first. PPO uses K=3–10 because the clipped ratio gives it a trust region that lets it reuse the same rollout K times. RLOO drops clip because `π_θ/π_θ_old = 1` exactly at the start of each step (rollouts are on-policy), and drops K>1 because without clip you don't have a trust region to sit inside. The consequence is that RLOO does more gradient steps per GPU-second than PPO but fewer per rollout — the net compute-per-quality gain is RLOO's Pareto win.

---

## The empirical Pareto

From the source (line 20, §Key Contributions):

> Empirically beats PPO on TL;DR summarization and HH-RLHF by 5–20% win rate at comparable KL.

From the source (line 26, §Key Figures/Tables to Study):

> Figure 3: TL;DR win rate vs KL Pareto frontier — RLOO dominates PPO and DPO at every KL.
> Figure 5: k=2 vs k=4 vs k=8 — diminishing returns beyond k=4.

The k=4 sweet spot is the attested recommendation. Ch-37 §2's guidance table lists `k ∈ {2, 4}` explicitly.

> Notice: "dominates at every KL" is a strong claim — the Pareto frontier does not cross. If RLOO beats PPO at every KL budget, there is no regime where PPO is preferred on quality grounds. The only remaining argument for PPO over RLOO on LLMs is engineering inertia — the existing trainer has a value head and GAE, changing it is friction.

---

## KL regularisation as a per-token shaped reward

From the source (lines 39–42, §KL regularization):

> Applied as a per-token shaped reward, identical to InstructGPT:
> `R̃(x, y) = R(x, y) − β · KL(π_θ(·|x) || π_ref(·|x))`
> No explicit KL term inside the loss, no entropy bonus.

Two design choices worth noting: (1) KL is added to the reward, not the loss — this is the InstructGPT canon ([[lilianweng-rlhf]] attests the same formula). (2) No entropy bonus — KL-to-reference already regularises the policy, as ch-37 §5 argues at length.

> Notice: whether to place KL inside the reward (RLOO / PPO-InstructGPT / REINFORCE++) or inside the loss (GRPO's k3 estimator) is not a cosmetic difference. KL-in-reward is *propagated through advantages* and gets baseline-normalised; KL-in-loss is applied at the final gradient step and does not interact with advantage estimation. These produce measurably different training dynamics at the same β — ch-40's GRPO treatment must re-establish its own β schedule because the k3-in-loss variant has different scaling than k1-in-reward.

---

## Relationship to GRPO

From the source (lines 64–65, §Relationship to GRPO):

> GRPO's advantage `(r_i − mean(r))/std(r)` over a group of G is equivalent (up to scaling) to RLOO's leave-one-out when G is large; GRPO additionally clips the ratio, normalizes by std, and computes KL in the loss.

For ch-37's template equation, RLOO and GRPO live in the same row (critic-free, per-prompt baseline) and differ only in: clip yes/no, std normalisation yes/no, KL location. This is the kind of nearly-equivalent-under-scaling relationship that ch-37 §6's knob-table captures.

---

## What ch-37 keeps from this source

- The four structural properties of LLM-RL (deterministic, terminal, long, discrete) from §4.
- The "value network / GAE / clip / K>1 are unnecessary" argument that structures §4.
- The leave-one-out baseline row of the §2 menagerie table.
- The per-token KL-in-reward regularisation form, which ch-37 §5 contrasts against in-loss KL.
- The k ∈ {2, 4} sampling count guidance.

---

## Connections

- **ch-37 §2 / §4** — menagerie and LM-RL specificity.
- **ch-41** — RLOO-specific chapter with full algorithm and hparam treatment.
- [[ppo]] — the method RLOO argues against for LLM RLHF.
- [[reinforce-plus-plus]] — small-k successor with global-batch baseline.
- [[grpo]] / [[rloo-vs-grpo]] — group-normalised cousin; systematic comparison.
- [[excerpts/reinforce-plus-plus]] — companion excerpt; when RLOO's leave-one-out is inadequate (k=1) and global-batch takes over.
- [[vanilla-pg]] — the score-function estimator RLOO instantiates.
