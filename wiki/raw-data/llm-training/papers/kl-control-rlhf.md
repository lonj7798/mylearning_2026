<!-- scope: KL-from-reference as the core RLHF regularizer; which KL, added where
     deps: [[reward-model-overoptimization]]
     see-also: [[entropy-mechanism-llm-rl]], [[constitutional-ai]], [[rlvr-tulu3]]
-->

# KL-Control in RLHF (Jaques 2019 / Korbak 2022 / Stiennon 2020 / Ouyang 2022)
- **Core Insight:** Standard RLHF is not "pure RL" but KL-regularized RL — the objective is `E[r(x,y)] − β · KL(π‖π_ref)`, which is mathematically equivalent to variational inference over a target distribution `π*(y|x) ∝ π_ref(y|x) · exp(r(x,y)/β)`.
- **Guideline:** Add the KL as a per-token penalty to the reward signal (not to the loss), use the k3 unbiased estimator `kl ≈ (r − 1) − log r` with `r = π/π_ref`, and choose β so that typical end-of-training KL sits at the sweet spot identified by your overoptimization curve (often β in 0.01–0.2 of the reward scale).
- **Authors (primary lineage):** Natasha Jaques et al. (2019, "Way Off-Policy"); Nisan Stiennon et al. (2020, "Learning to summarize from human feedback"); Long Ouyang et al. (2022, InstructGPT); Tomasz Korbak et al. (2022, "RL with KL Penalties is Better Viewed as Bayesian Inference")
- **Years:** 2019 / 2020 / 2022 / 2022
- **URLs:**
  - Jaques 2019 — https://arxiv.org/abs/1907.00456
  - Stiennon 2020 — https://arxiv.org/abs/2009.01325
  - Ouyang 2022 — https://arxiv.org/abs/2203.02155
  - Korbak 2022 — https://arxiv.org/abs/2205.11275
- **Relevant topics:** KL penalty, forward vs reverse KL, k1/k2/k3 estimators, reference policy, Bayesian inference view, DPO's implicit KL

## Abstract (synthesized across the lineage)
Jaques 2019 introduced "KL-control" for dialog agents: fine-tune with RL while penalizing KL divergence from a pretrained language prior, so outputs stay fluent. Stiennon 2020 applied the same trick to summarization with a learned RM, formalizing the RLHF template. Ouyang 2022 (InstructGPT) made this canonical. Korbak 2022 showed that this RL-with-KL-penalty objective is exact Bayesian inference over a tilted target distribution, and that the "reverse-KL" nature of the penalty explains mode-seeking behavior.

## Key Contributions
- **RLHF objective (Stiennon / InstructGPT form):**
  `J(φ) = E_{(x,y)~π_φ}[ r_θ(x,y) − β · log( π_φ(y|x) / π_SFT(y|x) ) ] + γ · E_{x~D_pretrain}[log π_φ(x)]`
  — KL is added to the **per-token reward**, then standard PPO is run.
- **β coefficient:** InstructGPT used β ≈ 0.02 (in reward-scale units); practitioners tune in [0.01, 0.5].
- **KL direction:** the penalty is `KL(π_new ‖ π_ref)` (reverse, mode-seeking) — forces the policy to place mass only where the reference also has mass.
- **KL estimators** (Schulman's blog): three unbiased estimators for `log(p/q)`-style KL:
  - `k1 = log(π/π_ref)` — unbiased but high variance, can be negative.
  - `k2 = 0.5 · (log(π/π_ref))^2` — biased but low variance.
  - `k3 = (π_ref/π) − 1 − log(π_ref/π)` — unbiased AND always ≥ 0; **recommended**, used in modern TRL/OpenRLHF.
- **Korbak's reformulation:** `argmax_π E_π[r] − β · KL(π‖π_ref)` has closed-form optimum `π*(y|x) ∝ π_ref(y|x) · exp(r(x,y)/β)`. RLHF is therefore amortized sampling from this tilted posterior — not unconstrained reward maximization.
- **Why reward + not loss:** adding KL to the reward keeps the PPO advantage estimator well-defined per token; adding KL to the loss breaks the advantage-based policy gradient and empirically trains worse.

## Key Figures/Tables to Study
- **InstructGPT Eq. 2** (the full objective with β and γ) — the canonical RLHF equation.
- **Stiennon Fig. 3** (KL vs reward Pareto front) — what tuning β traces out.
- **Schulman's KL blog table** — variance comparison of k1/k2/k3.
- **Korbak Fig. 1** — the tilted posterior interpretation.

## Technical Details
- **Per-token reward used in TRL/OpenRLHF:**
  `r̂_t = r_t − β · (log π_φ(y_t|y_<t,x) − log π_ref(y_t|y_<t,x))`
  with only the sequence-terminal `r_t` being the RM reward (zero elsewhere), and the KL term active on every token.
- **Reference policy:** usually the SFT checkpoint; frozen; identical tokenizer and architecture.
- **Adaptive KL:** some implementations (InstructGPT early ablations, DeepSpeedChat) adapt β to hit a target KL per batch — multiplicatively raise β when KL exceeds target, lower when below.
- **DPO's implicit KL:** DPO replaces the online KL penalty with a closed-form implicit reward `r_θ(x,y) = β · log(π_θ(y|x)/π_ref(y|x))`, so β plays the same role and the same overoptimization laws apply.
- **Failure modes:** β too small → reward hacking and mode collapse; β too large → policy cannot depart from SFT, ignores RM signal.

## Connections
- Provides the regularizer whose "budget" is quantified in **[[reward-model-overoptimization]]**.
- Complementary to **[[entropy-mechanism-llm-rl]]**: KL-to-reference keeps you near SFT; entropy keeps the distribution wide — different axes.
- Replaced in pure-RLVR settings by verifier-grounded reward (**[[rlvr-tulu3]]**, **[[deepseek-r1]]**) but the KL term is still used.
- Blog reference: Schulman's "Approximating KL Divergence" (http://joschu.net/blog/kl-approx.html).
