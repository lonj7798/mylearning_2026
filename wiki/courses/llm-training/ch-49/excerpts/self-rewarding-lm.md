---
chapter: ch-49
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/self-rewarding-lm.md
source_url: https://arxiv.org/abs/2401.10020
created_at: "2026-04-23"
---

# Excerpt: Self-Rewarding LMs — judge drift and the iter-3 plateau

**Source library:** `wiki/raw-data/llm-training/papers/self-rewarding-lm.md`
**Authors:** Weizhe Yuan, Richard Yuanzhe Pang, Kyunghyun Cho, Xian Li, Sainbayar Sukhbaatar, Jing Xu, Jason Weston (Meta AI / NYU)
**Year:** 2024

---

## Why this source matters for ch-49

Self-Rewarding is the clearest empirical demonstration that a judge can **improve** and then **drift** under iterative training. The trajectory — judge Spearman rising 0.62 → 0.71 across three iterations, then saturating and regressing — is the best-documented judge-drift curve in the 2024 literature. Ch-49 §8 (the anchor protocol) exists because of this failure mode.

---

## The judge improves — then plateaus

Source §Key Contributions:

> "Showed the judge signal improves with each iteration -- an emergent property absent from fixed-RM pipelines."
> "Demonstrated 3 iterations of Iterative DPO on Llama-2-70B lifts AlpacaEval 2.0 win-rate from 9.94% (SFT) -> 15.38% -> 20.44% -> 20.8%, passing GPT-4 (June 2023) at iter 2."
> "Showed the judge's Spearman correlation with held-out human preference improves from 0.62 (iter 0) to 0.71 (iter 3) on the same Open-Assistant rubric."

Two signals: actor improvement (AlpacaEval) and judge improvement (Spearman). Both are monotone through iter 3, then flatten. The paper explicitly notes iter 4 regresses on reward-bench, which it attributes to reward hacking. Ch-49 §8 uses this as the archetype: without an external anchor, even an apparently-improving judge will eventually start drifting.

---

## The stopping criterion — hard-coded and fragile

Source §Technical Details:

> "Stopping: 3 iterations -- the paper notes iter 4 regresses on reward bench (likely reward hacking)."

"3 iterations" is not a principled cap; it is the first iteration the authors observed regression. A production stack cannot rely on a magic number — it needs the anchor-correlation check ch-49 §8 specifies.

---

## The judge prompt — fixed across iterations

Source §Technical Details:

> "Judge prompt: a 5-point rubric ('Additive scoring (1-5) of helpfulness, relevance, depth, clarity, and completeness') appended to every completion -- identical across iterations."

This is a clean example of what ch-49 §7 calls "rubric versioning matters". Self-Rewarding keeps the rubric *fixed* precisely so the iteration signal is interpretable. If the rubric had changed per iteration, the AlpacaEval and Spearman trajectories would be uninterpretable.

---

## Why policy-as-judge is interesting even though it drifts

Source §Key Contributions:

> "Introduced the Self-Rewarding training loop: one model plays both Actor and Judge, with the Judge role invoked via a fixed evaluation prompt template."

The single-model setup is what makes the iter-3 plateau interesting: actor and judge share weights, so actor improvement improves judge, and judge improvement improves actor. Ch-49 §6 uses this loop as the counter-example to the RL-time-≠-eval-time rule: for *actor training*, policy-as-judge is a feature, not a leak. For *eval*, it is the worst possible leak.

---

## The per-prompt cost structure

Source §Technical Details:

> "Preference-pair construction per iteration:
>   1. Sample 4 responses per prompt from the current policy at T=0.7, top-p=0.9.
>   2. Score all 4 with the policy-as-judge (pairwise or 5-point, averaged over 3 judge samples).
>   3. Take the highest-scored response as chosen, lowest as rejected.
>   4. Run DPO with beta = 0.1 for 1 epoch from the previous iteration's checkpoint."

4 generations + 4 × 3 judge calls + DPO = ~16 forward passes per prompt per iteration. The judge cost dominates — half of compute goes to scoring, not generating. This is why ch-49 §6 treats "judge compute" as a first-class budget line item.

---

## The connection that makes it part of the synthetic-judge line

Source §Connections:

> "Related to rlaif-scaling: both remove the human preference bottleneck, but RLAIF uses a separate frozen judge model; Self-Rewarding uses the policy itself."

Ch-49 §5 places Self-Rewarding alongside Con-J / STE / J1 even though it takes a different architectural path (weight-sharing instead of training a dedicated judge). The common thread is "judge is not an external API." Self-Rewarding and Meta-Rewarding are the in-stack variants; Self-Taught Evaluators / J1 are the dedicated-model variants. Both live in the synthetic-judge era.

---

## The handoff to Meta-Rewarding

Source §Connections:

> "Direct precursor to meta-rewarding-lm (adds a meta-judge to regulate judge quality)."

Meta-Rewarding's meta-judge is the in-stack answer to the iter-3 plateau. Ch-49 §5's Meta-Rewarding row reads this as "self-correcting via a third role"; the Self-Rewarding row reads as "monotone then drift, which motivates the third role."

---

## Connections

- `read.md` §3 judge-drift row: Spearman 0.62 → 0.71 trajectory quoted.
- `read.md` §5 method-contrast table: Self-Rewarding / Meta-Rewarding row.
- `read.md` §6: policy-as-judge as the extreme case of RL-time = eval-time overlap.
- `read.md` §8 anchor protocol: Self-Rewarding's iter-3 regression is the archetype this section mitigates.
- [[meta-rewarding-lm]]: the structural fix that pushes the plateau to iter 5.
- [[direct-judgement-preference]]: the dedicated-judge-model counterpart to in-stack self-rewarding.
