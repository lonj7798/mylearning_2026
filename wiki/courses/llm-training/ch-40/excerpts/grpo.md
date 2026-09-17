---
chapter: ch-40
course: llm-training
phase: read
excerpt_of: arXiv:2402.03300v3 (DeepSeekMath); library card [[grpo]] (verified 2026-09-14)
source_url: https://arxiv.org/abs/2402.03300
created_at: "2026-04-23"
revised: "2026-09-15 (generality revision; aligned with the verified card and the v3 text)"
---

# Excerpt: GRPO as published — objective, KL estimator, results

Used by [[read]] §4, §5, §9, and the Recipe. Shao et al. (DeepSeek-AI, Tsinghua, Peking University), arXiv v1 Feb 2024, v3 27 Apr 2024.

## Advantage and objective (§4.1.1–§4.1.2, Eqs. 3–4)
```
Â_{i,t} = (r_i − mean(r_1 … r_G)) / std(r_1 … r_G)        (outcome supervision; same value for every token of o_i)

J_GRPO(θ) = E[q, {o_i}] (1/G) Σ_i (1/|o_i|) Σ_t { min[ ρ_{i,t} Â_{i,t}, clip(ρ_{i,t}, 1−ε, 1+ε) Â_{i,t} ] − β D_KL[π_θ || π_ref] }
ρ_{i,t}   = π_θ(o_{i,t} | q, o_{i,<t}) / π_θold(o_{i,t} | q, o_{i,<t})
D_KL      = π_ref/π_θ − log(π_ref/π_θ) − 1                 (Eq. 4, cited to Schulman 2020, "guaranteed to be positive")
```
The paper does not use the label "k3"; that name comes from [[john-schulman-kl-tricks]]. PPO adds its KL penalty to the reward (Eq. 2); GRPO adds it to the loss, "avoiding complicating the calculation of Â" (§4.1.1). Stated motivation for dropping the value model: it is "typically another model of comparable size as the policy model", and a reward at the last token only "may complicate the training of a value function that is accurate at each token".

Process supervision (§4.1.3) normalizes step rewards by the mean and std of all step rewards in the group and takes the undiscounted sum of normalized rewards of steps ending at or after token `t`. Iterative RL (§4.1.4, Algorithm 1) retrains the reward model on policy samples with 10% replay and resets `π_ref` to the current policy.

## Recipe (§4.2)
Policy LR 1e-6, β = 0.04, G = 64 outputs per question, max length 1024, training batch size 1024 (unit not stated in the paper), one policy update per exploration stage, about 144K chain-of-thought questions related to GSM8K and MATH taken from the SFT set, reward model initialized from DeepSeekMath-Base 7B with LR 2e-5. Clip ε and rollout temperature are not reported.

## Results (Table 5)
DeepSeekMath-Instruct 7B → DeepSeekMath-RL 7B, chain-of-thought: GSM8K 82.9 → 88.2; MATH 46.8 → 51.7; MGSM-zh 73.2 → 79.6; CMATH 84.6 → 88.8. Tool-integrated: GSM8K 83.7 → 86.7; MATH 57.4 → 58.8. Figure 5 (1.3B) compares RFT, Online RFT, GRPO+OS, GRPO+PS as curves with no numbers in the text. No direct PPO-versus-GRPO accuracy comparison is reported at 7B.

## Generality and negatives (§5.2)
- RL raises Maj@K but not Pass@K at temperature 0.7 for K ≤ 64; the authors attribute the gain to "boosting the correct response from TopK rather than the enhancement of fundamental capabilities" (Fig. 7).
- RL used math chain-of-thought prompts only; the authors treat MGSM-zh and CMATH as out of domain, and both improved. Non-math benchmarks after RL are not reported.
- Online RFT has gradient coefficient 1 for correct and 0 for incorrect outputs and "does not penalize incorrect responses"; GRPO's normalized advantage gives below-mean outputs a negative coefficient (§5.2.1, Eq. 10; App. A.1.6 Eq. 21).
- Reward-label noise: PRM800K is cited as containing about 20% incorrect annotations (§5.2.3, footnote 7).

## Claims removed from the earlier version of this excerpt
"MATH 51.7 from PPO's 51.0, RFT's 49.0" (no such comparison exists in the paper), "batch 1024 prompts = 16 × 64", "clip ε = 0.2", "π_ref frozen SFT", "removing the critic halves memory", and "GRPO is the loss DeepSeek used for R1-Zero and R1" (a fact from [[deepseek-r1-recipe]], not from this paper).
