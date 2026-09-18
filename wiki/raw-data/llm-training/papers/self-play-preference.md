<!-- scope: NLHF — alignment as the Nash equilibrium of a pairwise preference model (Nash-MD, Nash-EMA)
     deps: [[dpo]], [[ppo]]
     see-also: [[ipo]], [[spin]], [[self-rewarding-lm]], [[trl-online-dpo]]
-->

# Nash Learning from Human Feedback
- **Core Insight:** Replacing the Bradley-Terry reward model with a pairwise preference model P(y ≻ y′ | x) and seeking the Nash equilibrium π* = argmax_π min_{π′} P(π ≻ π′) gives a solution that a reward model cannot express: in the paper's three-action example with three human types, the Nash solution places probability 1/3 + ε/2, 1/3 + ε/2 and 1/3 − ε on the three actions while the Bradley-Terry optimum selects one action deterministically and flips as ε crosses 0 (§3.2).
- **Guideline:** When training a policy against a learned preference model, play the current policy against the geometric mixture π_θ^β of itself and the SFT reference with β in 0.125-0.375, because on the TL;DR summarization task those values beat both β = 0 (self-play) and β = 1 (best response against SFT) under the PaLM 2 Large judge (§8, Table 1).
- **Authors:** Rémi Munos, Michal Valko, Daniele Calandriello, Mohammad Gheshlaghi Azar, Mark Rowland, Daniel Guo, et al. (Google DeepMind)
- **Year:** 2023 (arXiv v1 2023-12; v4 2024-06-11)
- **URL:** https://arxiv.org/abs/2312.00886
- **Source type:** paper
- **Relevant topics:** preference models, Nash equilibrium, mirror descent, self-play, non-transitive preferences, RLHF alternatives

## Abstract
RLHF usually begins by learning a reward model from pairwise human feedback and then fine-tunes the policy to maximize that reward. A reward model assigns one score per generation, so it cannot fully represent the richness of human preferences and it depends on the sampling distribution used to train it. This paper proposes an alternative pipeline: first learn a pairwise preference model conditioned on two responses given a prompt, then seek a policy whose responses are consistently preferred over those of any competing policy, which is the Nash equilibrium of that preference model. The authors call this Nash learning from human feedback (NLHF). For tabular policies they introduce Nash-MD, a mirror-descent algorithm with last-iterate convergence to the Nash equilibrium of the regularized preference model, and Nash-EMA, a variant based on an exponential moving average of past parameters. They introduce the corresponding deep-learning policy-gradient algorithms Nash-MD-PG and Nash-EMA-PG, and report experiments on a text summarization task using the TL;DR dataset, evaluated by pairwise comparison with a large LLM judge and compared against an RLHF baseline.

## Key Contributions
- Defines NLHF: the objective π* = argmax_π min_{π′} P(π ≻ π′), a two-player antisymmetric constant-sum game whose solution exists by the minimax theorem (§3, Eq. 1).
- Introduces the regularized preference model P_τ(π ≻ π′) = P(π ≻ π′) − τ·KL(π, μ) + τ·KL(π′, μ) and establishes existence and uniqueness of its Nash equilibrium (§4).
- Nash-MD: the policy plays against the geometric mixture π_t^μ(y) ∝ π_t(y)^{1−η_t τ}·μ(y)^{η_t τ} instead of the full mixture of past policies, so no intermediate policies must be stored (§6).
- Theorem 1: KL(π_τ*, π_{t+1}) ≤ (1 − η_t τ)·KL(π_τ*, π_t) + 2η_t², and with η_t = 2/(τ(t+2)), KL(π_τ*, π_T) ≤ 8/(τ²(T+1)) — last-iterate convergence at O(1/T) (§6).
- Theorem 2: if a preference model cannot be perfectly represented by Bradley-Terry, the optimal BT reward model depends on the sampling distribution (App. A, Prop. 2 and Thm. 2).
- Deep-learning versions Nash-MD-PG and Nash-EMA-PG and a summarization study comparing them with SFT, RLHF, self-play and best-response (§7, §8, App. F-G).

## Key Figures/Tables to Study
- **Table 1** — PaLM 2 Large pairwise preferences P*(π_c ≻ π_r) among SFT, RLHF, SP, MD1-MD6, BR, EMA1-2 and the EMA average policies.
- **Table 2 (App. G.5)** — the same comparison scored by the regularized preference model P_τ used during training.
- **Figure 1 (App. G.1)** — preference-model accuracy for T5X-small, T5X-XL and T5X-XXL on the TL;DR train and test sets.
- **Figure 2 (App. G.1)** — preference-model versus reward-model accuracy at equal size.
- **§3.2** — the three-type, three-action non-transitivity example.
- **§6, Theorem 1** — the Nash-MD update and its last-iterate convergence bound.
- **App. F.3** — the definition of the alternative policy π_θ^β and the β = 0 and β = 1 limits.

## Technical Details
- **Preference model:** P_θ(y ≻ y′ | x) ∈ [0,1], initialized by prompting an LLM as a summary rater and passing the last logit of one chosen token through a sigmoid; trained with cross-entropy against human preference labels (App. G.1). It makes no Bradley-Terry assumption (§2).
- **Reward-model baseline:** r_θ(x,y) from the same architecture, trained through P_BT(y ≻ y′|x) = σ(r_θ(x,y) − r_θ(x,y′)) on the same data (App. G.1).
- **Preference versus reward accuracy:** at T5X-XL the preference model peaks around 0.78 test accuracy and the reward model around 0.76 (App. G.1).
- **Nash-MD-PG policy gradient:** ∇_θ log π_θ(y|x)·[P(y ≻ y′|x) − 1/2 − τ·log(π_θ(y|x)/μ(y|x))], with y′ drawn from the alternative policy (§7).
- **Alternative policy:** log π_θ^β(y|x) = (1−β)·log π_θ(y|x) + β·log μ(y|x) + c(x). β = 0 reduces to self-play, β = 1 to best response against the reference μ (App. F.3, Eq. 15).
- **Nash-EMA-PG:** the alternative policy is π_θ̄ with θ̄_t = (1−β)·θ_t + β·θ_0, an exponential moving average of past parameters, inspired by fictitious play (App. F.3).
- **Equivalence of regularizers:** ∇_θ KL(π_θ, π_θ^β) = β·∇_θ KL(π_θ, μ), so regularizing toward the mixture and toward the reference agree for a single gradient step (App. F.3).
- **Judge evaluation:** all pairs are scored by a PaLM 2 Large preference model (§8, Table 1). Table 2's P_τ figures come from 1,000 pairwise comparisons per cell, with a 95% Clopper-Pearson interval width of at most ±0.032 (App. G.5).
- **Reported outcomes under the PaLM 2 judge (Table 1):** P*(MD1 ≻ RLHF) = 0.598, P*(MD1 ≻ SP) = 0.592, P*(BR ≻ RLHF) = 0.148; every method beats SFT with probability 0.94 or higher.

## Recipe ledger

| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| Preference / reward model (TL;DR) | T5X-small 110M, T5X-XL 3B, T5X-XXL 11B | preference-model | sizes compared | 110M / 3B / 11B | arXiv:2312.00886v4 App. G.1 Fig. 1 | verified 2026-09-18 | accuracy rises with size, small gain from 3B to 11B; T5X-XL chosen (App. G.1) |
| Preference model (TL;DR) | T5X-XL (3B) | preference-model | training data | 92,820 examples (Stiennon et al. 2020 TL;DR train set) | App. G.1 | verified 2026-09-18 | n/a |
| SFT policy (TL;DR) | T5X-L | SFT | initialization for all policies and the KL reference μ | T5X-L, supervised fine-tuned on the TL;DR dataset | App. G.2 | verified 2026-09-18 | T5X-L chosen over T5X-XL "for computational efficiency" (App. G.2) |
| RLHF baseline (TL;DR) | T5X-L | RL | steps | 10000 | App. G.3 | verified 2026-09-18 | n/a |
| RLHF baseline (TL;DR) | T5X-L | RL | KL coefficient τ | 0.05 | App. G.3 | verified 2026-09-18 | swept over {0.01, 0.02, 0.05, 0.1, 0.2} (App. G.3) |
| Nash-MD-PG / Nash-EMA-PG (TL;DR) | T5X-L | RL | steps | 10000 | App. G.5 | verified 2026-09-18 | n/a |
| Nash-MD-PG / Nash-EMA-PG (TL;DR) | T5X-L | RL | regularization coefficient τ | 0.008 | App. G.4-G.5 | verified 2026-09-18 | swept over {0.02, 0.01, 0.008, 0.005} (App. G.4) |
| Nash-MD-PG (TL;DR) | T5X-L | RL | mixture coefficient β | swept {0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0}; 0.125-0.375 best | App. G.4; §8 | verified 2026-09-18 | §8: β ∈ [0.125, 0.375] outperforms both SP (β=0) and BR (β=1) under the PaLM 2 judge |
| Nash-EMA-PG (TL;DR) | T5X-L | RL | EMA coefficient β | swept {0, 0.999, 0.9995, 0.9999, 1.0}; reported for 0.999 and 0.9995 | App. G.4-G.5 | verified 2026-09-18 | no ablation beyond the sweep reported |

Optimizer, learning rate, batch size and sequence length are not reported.

## Findings relevant to generality
- The authors state the goal of the experiments is a proof of concept rather than state-of-the-art summarization, and that a fair NLHF-versus-RLHF comparison is difficult because the two rely on different learned models, a preference model versus a reward model (§8; App. G.2).
- Argument for breadth of preference coverage: a preference model conditioned on two responses can represent non-transitive population preferences, which a single-score reward model cannot, and its Nash solution varies smoothly with the population mixture while the BT optimum switches discontinuously (§3.1, §3.2).
- Argument for robustness to data distribution: a reward model's optimum depends on the distribution used to train it, so iterated RLHF requires relearning the reward model, whereas a preference model can be kept across iterations (§3.3, App. A).

## Connections
- [[dpo]] is cited as related work that optimizes a Bradley-Terry objective in closed form; NLHF's objection to BT applies to it (§2).
- [[ipo]] shares the concern that a BT-based objective over-commits to one mode; both are from the same group.
- [[spin]] and [[self-rewarding-lm]] also train a model against its own generations; here the opponent is the geometric mixture of the policy and the SFT reference, and β = 0 (pure self-play) is reported as a weaker setting than β ∈ [0.125, 0.375] (§8).
- [[trl-online-dpo]] implements online preference optimization against a judge; the Nash-MD variant in that framework follows Eq. 15's mixture opponent.
- [[ppo]] is named as a possible substitute for the single-gradient-step implementation of Nash-MD (App. F.3).

## Verification
- Checked on 2026-09-18 against: https://arxiv.org/abs/2312.00886 (arXiv v4, 2024-06-11)
- Corrections to the previous card version:
  - "Year: 2024" → arXiv v1 is 2023-12; v4 is 2024-06.
  - "Nash-MD ... implements this by playing against a slow-moving EMA of itself" → Nash-MD plays against the geometric mixture of the current policy and the reference μ; the EMA-of-parameters opponent is the separate Nash-EMA algorithm (§6; App. F.3).
  - "Nash objective: find π such that max ... = 1/2" → the objective is π* = argmax_π min_{π′} P(π ≻ π′); a policy is said to win against π′ when P(π ≻ π′) ≥ 1/2 (§3, Eq. 1).
  - "Proposition 4.1 (Nash equilibrium existence) and Theorem 5.2 (mirror-descent convergence)" → the results are Theorem 1 (last-iterate convergence of Nash-MD, §6) and Proposition 2 / Theorem 2 (BT reward depends on the sampling distribution, App. A).
  - "Figure 2 (non-transitive preferences toy): shows DPO oscillates while Nash-MD converges" → Figure 2 compares preference-model and reward-model accuracy on TL;DR; no figure compares DPO trajectories. The non-transitivity argument is the worked example in §3.2.
  - "Figure 4 (text summarization human-eval): Nash-MD matches or beats DPO / RLHF-PPO on human pairwise preference" → the evaluation is Table 1, scored by a PaLM 2 Large preference model rather than humans, and the baselines are SFT and a regularized-policy-gradient RLHF run, not DPO (§8; App. G.3).
  - "Practical implementation (Nash-MD-PG): sample two responses from π_t" → the second response is sampled from the alternative policy π_θ^β, not from π_t, except in the β = 0 self-play case (App. F.3).
  - "Introduces Nash-MD ... alternate between self-play sampling and a soft-policy update" → Nash-MD is one mirror-descent update per iteration against π_t^μ; the paper notes a faithful two-timescale version was not implemented (App. F.3).
- Removed as unsupported by the source: "Proves convergence under mild regularity even with non-transitive preference models (where argmax-of-reward fails)" as stated (Theorem 1 is convergence to the regularized Nash equilibrium and does not condition on transitivity); "Directly motivates IPO's L2-on-margin loss"; "Follow-up work extends to decentralized XPO"; the claim that Nash-MD was evaluated against DPO.
- Not reported by the source: optimizer, learning rate, batch size, sequence length; human evaluation of the trained policies; results on tasks other than TL;DR summarization; parameter count of PaLM 2 Large.
