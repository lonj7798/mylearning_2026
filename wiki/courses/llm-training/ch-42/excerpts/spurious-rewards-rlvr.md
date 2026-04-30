---
chapter: ch-42
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/spurious-rewards-rlvr.md
source_url: https://arxiv.org/abs/2506.10947
created_at: "2026-04-23"
---

# Excerpt: Shao 2025 — Spurious Rewards: Rethinking Training Signals in RLVR

**Source library:** `wiki/raw-data/llm-training/papers/spurious-rewards-rlvr.md`
**Paper:** Shao et al., *Spurious Rewards: Rethinking Training Signals in RLVR*, 2025.

---

## Why this source anchors ch-42

This paper forces a reinterpretation of every RLVR / RL post-training gain claim. On Qwen2.5-Math-7B, GRPO with **randomly assigned rewards** still improves MATH-500 by 21.4 points — close to the 29.1-point gain from ground-truth rewards. The proposed explanation is a clipping bias in GRPO that amplifies high-prior pretrained behaviors, including a specific "code reasoning" mode.

Ch-42 §1 uses this to argue that RLHF and RLVR gains must be audited against *reward informativeness*, not just end-of-training benchmark score. A run that gains +20 points with a random reward is not a run that proved its reward design. Ch-42 §7's pre-deployment checklist includes the **random-reward control** — re-run training with a spurious reward and compare.

Raw-data header:

> **Core Insight:** RLVR can improve reasoning even with random or negatively correlated rewards because GRPO's clipped objective can preferentially amplify strong pretrained behaviors without truly informative training signals.

## The headline result

On **Qwen2.5-Math-7B**:

- Ground-truth reward GRPO: **+29.1** MATH-500 points.
- Random reward GRPO: **+21.4** MATH-500 points.
- Anti-correlated (negatively aligned) reward: also substantial gains.

The random-reward run is ~73% of the ground-truth gain. If you had seen only the random-reward curve, you could have written a paper titled "Our Novel Reward Function Adds 21 Points to MATH-500."

## The proposed mechanism — GRPO clipping bias

GRPO computes per-token advantages via group normalization and clips policy-ratio updates (standard PPO-style clip). The key asymmetry:

- When the model is already confident in a behavior (high prior probability), the ratio clip prevents the update from pushing that token's probability below a floor.
- When the model is not confident (low prior), the clip cuts positive updates but is less binding on negative updates.
- Net effect: under a noisy reward, confident-prior behaviors get *reinforced* more than uncertain-prior behaviors get *discouraged*. The clip term becomes a filter that preferentially amplifies high-prior modes.

In the limit where the reward carries no information, this still produces a non-zero update in the direction of the prior's already-strong behaviors. The policy collapses onto its pretrained modes rather than learning from the reward signal.

## Code reasoning — a concrete behavior

The authors identify a behavior they call **code reasoning**: the model reasons in code-like form (Python pseudocode with variable assignments, function definitions, explicit step-by-step) without actually executing code. On Qwen2.5-Math-7B, this mode is present ~65% of the time in the base model. After RLVR with spurious rewards, it rises to **above 90%**.

This is what the clipping bias is amplifying: a pretrained reasoning mode that correlates with MATH-500 performance in the base distribution. RL doesn't teach new reasoning; it concentrates the output distribution on whichever pretrained mode happens to score well.

## Why this matters for ch-42

### Reward informativeness is not measured by benchmark gain

A +21 point improvement on MATH-500 is not evidence that the reward function is doing useful work. The right experiment is a matched control: run the same training with a random reward. If the gain is within ~30% of the real run, the reward is not informative — the algorithm is amplifying priors.

Ch-42 §7 incorporates this as the "prior-vs-signal audit" in the pre-deployment checklist.

### RLHF faces the same risk

[[echo-chamber-rl-post-training]] makes the parallel claim for PPO / GRPO / Expert Iteration: RL fine-tuning converges on behaviors already latent in the pretrained model's distribution. RL does not invent; it sharpens.

The joint implication for ch-42: benchmark gains are a necessary but not sufficient condition for good reward design. The sufficient condition is that a random-reward control shows substantially less gain.

### Implications for RLHF length bias

Length bias and code-reasoning bias are the same kind of phenomenon: a pretrained behavior that the RL objective amplifies because it correlates with reward. The length-bias mitigation in ch-42 §2 (length-residualized reward) is the structural correction. The spurious-reward audit generalizes this to arbitrary biases.

## Practical auditing prescription

The paper's implicit prescription, restated for the chapter:

1. **Baseline the prior.** Measure the frequency of target behaviors in the base model before RL.
2. **Random-reward control.** Run the same algorithm with a random reward; compare gains.
3. **Objective-bias audit.** Compute GRPO's (or PPO's) update under the actual and the spurious reward; if they produce similar updates in distribution, the reward is not driving the policy.
4. **Cross-base verification.** Run the same RL recipe on a base model that does *not* have the code-reasoning prior; if gains disappear, the recipe was prior-dependent.

## Caveats the paper does not cover

- The result is demonstrated on Qwen2.5-Math-7B + GRPO + MATH-500. Other base models, other algorithms (PPO with different clip settings), and other benchmarks may behave differently.
- [[prorl]] is a useful counter-weight: with longer and more diverse RL, the reasoning frontier can genuinely expand. Spurious-reward dominance is a finite-training phenomenon.
- The paper does not claim *all* RLVR gains are spurious — only that a large share on one setup is, and that reward informativeness must be measured explicitly.

## Takeaways for the chapter

1. Random-reward GRPO can produce most of the gain of ground-truth GRPO — the reward was not doing the work.
2. The mechanism is GRPO clipping bias: confident-prior behaviors get reinforced; uncertain-prior behaviors do not get equally discouraged.
3. Code reasoning on Qwen2.5-Math is the concrete behavior being amplified from 65% to 90%.
4. Benchmark gain is not evidence of reward informativeness; a random-reward control is the required audit.
5. Ch-42 §7's pre-deployment checklist includes this audit as the "prior-vs-signal" gate.
