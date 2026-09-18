<!-- scope: KL-control — penalizing divergence from a pre-trained prior inside an RL objective; the primary artifact is Jaques et al. 2019, with the later RLHF instances of the same regularizer listed and located at the end
     deps: [[reward-model-overoptimization]]
     see-also: [[rlhf-instructgpt]], [[entropy-mechanism-llm-rl]], [[maximum-entropy-rl]], [[dpo]]
-->

# Way Off-Policy Batch Deep Reinforcement Learning of Implicit Human Preferences in Dialog
- **Core Insight:** Adding a KL penalty against a pre-trained language prior to the RL objective makes batch (fully off-policy, no-exploration) RL on dialog work: the KL-control models in the paper's live human evaluation score significantly higher total human ratings than the Batch Q and discrete-BCQ baselines, F = 4.781, p < .05 (§6, Table 1).
- **Guideline:** When RL is run on top of a pre-trained generative model and fresh exploration is unavailable or unsafe, optimize `E_q[r]/c − D_KL(q‖p)` against the pre-trained model as the prior `p`, because the resulting policies stayed fluent while the unregularized batch-Q policies degenerated into repeated and malformed utterances (§3.3, §6, Table 2).
- **Authors:** Natasha Jaques, Asma Ghandeharioun, Judy Hanwen Shen, Craig Ferguson, Agata Lapedriza, Noah Jones, Shixiang Gu, Rosalind Picard
- **Year:** 2019 (arXiv v1 2019-07; v2 2019-07-08 is the version checked here)
- **URL:** https://arxiv.org/abs/1907.00456
- **Source type:** paper
- **Relevant topics:** KL-control, batch / off-policy RL, pre-trained prior, Ψ-learning, entropy regularization, dropout-based uncertainty, implicit human feedback

## Abstract
The paper addresses off-policy batch reinforcement learning: learning from a fixed batch of human interaction data without the ability to explore online. It proposes leveraging models pre-trained on data as a strong prior and using KL-control to penalize divergence from that prior during RL training, and it uses dropout-based uncertainty estimates to lower-bound target Q-values as an alternative to Double Q-Learning. The algorithms are tested on open-domain dialog generation, described as an RL problem with a 20,000-dimensional action space. Multiple reward functions are extracted post hoc from collected human interaction data, and the resulting policies are deployed live to converse with humans. The authors report significant improvements over prior off-policy batch RL methods (Abstract).

## Key Contributions
- Introduces KL-control from a pre-trained prior as the regularizer for batch RL on language, with the trajectory-level objective in Eq. 5 (§3.3).
- Shows the same objective rewritten as an action-level Q-function in which the prior term and an entropy term appear explicitly (Eq. 6, Eq. 7) (§3.3).
- Derives a Ψ-learning variant that replaces the hard max over noisy Q-estimates with a Boltzmann (soft) update, argued to reduce over-estimation in the batch setting (Eq. 8, Eq. 9) (§3.3).
- Uses dropout-based uncertainty estimates and Monte Carlo target sampling to lower-bound target Q-values (§3.2).
- Averages over the priors of many pre-trained dialog models to form a model-averaged prior `p_MA(a|s) = Σ_M S(M) p(a|s; M)` (§3.4).
- Evaluates by live human interaction rather than in the training environment, which the authors argue is a stronger generalization test (§5).

## Key Figures/Tables to Study
- **Eq. 5-7:** the KL-regularized objective and its two rewritings; the source of the "reward plus log-prior plus entropy" decomposition.
- **Table 1:** interactive human evaluation of batch-RL techniques, including the KL-control Q, Ψ, and MA-Ψ variants against Batch Q and DBCQ.
- **Table 2:** sample conversations contrasting Batch Q output with KL-control output.
- **Table 3:** human evaluation across the different reward functions, all trained with KL-control.

## Technical Details
**Objective.** With `τ = {a_1, …, a_{t−1}}` a trajectory of actions, `q(τ) = Π_t π_θ(a_t, s_t)` the policy at trajectory level, and `p(τ) = Π_t p(a_t|s_t)` the prior at trajectory level:

`L(q) = E_{q(τ)}[ r(τ) ] / c − D_KL[ q(τ) ‖ p(τ) ]`  (§3.3, Eq. 5)

`c`: the weight placed on the RL reward relative to the KL term; larger `c` weakens the regularizer. Expanding the KL gives the action-level value function

`Q_π(s_t, a_t) = E_π[ Σ_{t'≥t} r(s_{t'}, a_{t'})/c + log p(a_{t'}|s_{t'}) − log π(a_{t'}|s_{t'}) ]`  (§3.3, Eq. 6)

and, folding the last term into an entropy term, `Q(s_t,a_t) = E_π[ Σ r/c + log p + H(·|s_{t'}) ]` (Eq. 7). The paper states the two added terms have distinct roles: `log p(a|s)` biases the policy toward state-action pairs that are realistic and likely to be in the batch, and `−log π(a|s)` acts as entropy regularization, which matters for generative dialog models known to collapse to a small number of repeated samples (§3.3).

**Ψ-learning.** `Ψ*(s_t,a_t) = r(s_t,a_t)/c + log p(a_t|s_t) + γ log Σ_{a'} exp(Ψ*(s',a'))`, with `π_Ψ(a_t|s_t) = exp(Ψ*(s_t,a_t))` (§3.3, Eq. 8-9). The soft update avoids a hard max over noisy estimates and is argued to reduce optimism under uncertainty in the batch setting (Interpretation stated by the authors, §3.3).

**Model.** A hierarchical seq2seq dialog model; the VHRED variational hierarchical encoder-decoder is selected as the base (§4). Knowledge distillation from a sentence-embedding model is used to improve the context encoder (§4, App. 8.2).

**Data and evaluation.** Over 40 dialog models of differing architectures, trained on movie dialogs and Reddit, were hosted live; a batch of 14,232 pairs of user input and agent response was collected and used to train the RL models, which were then redeployed (§5). 90 Mechanical Turk workers gave 718 seven-point Likert ratings of quality, fluency, diversity, contingency, and empathy after at least 3 turns with each bot (§5).

**Result.** KL-control models score significantly higher than the Batch Q baselines and DBCQ on total human rating, F = 4.781, p < .05 (§6). Table 1 totals: KL-control Q 13.98 ± 1.81, KL-control Ψ 14.67 ± 1.82, KL-control MA Ψ 14.44 ± 1.96.

## Recipe ledger

| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| Jaques 2019 KL-control (VHRED dialog) | not reported as parameter count | RL (batch Q/Ψ-learning) | reward-vs-KL weight `c` | 2 | arXiv:1907.00456v2 App. 8.2 | verified 2026-09-18 | no ablation reported |
| Jaques 2019 KL-control (VHRED dialog) | — | RL | discount γ; target update rate α; learning rate | 0.5; 0.005; 1e-4 | arXiv:1907.00456v2 App. 8.2 | verified 2026-09-18 | no ablation reported |
| Jaques 2019 KL-control (VHRED dialog) | — | RL | batches trained; batch size; Monte Carlo target samples M; grad clip | 800-1000 batches; 32; 5; 1.0 | arXiv:1907.00456v2 App. 8.2 | verified 2026-09-18 | checkpoint chosen by early stopping (App. 8.2) |
| Jaques 2019 KL-control (VHRED dialog) | — | RL | training batch of human interaction data | 14,232 user-input / agent-response pairs | arXiv:1907.00456v2 §5 | verified 2026-09-18 | live human evaluation, 90 raters, 718 ratings (§5) |

## Related artifacts in this lineage (each verified at its own source)
These are separate artifacts, not part of the primary source above. Cite them by their own URLs.
- **Stiennon et al. 2020, "Learning to summarize from human feedback"** (https://arxiv.org/abs/2009.01325). The per-sequence reward is `R(x,y) = r_θ(x,y) − β log[π^RL_φ(y|x) / π^SFT(y|x)]` (§3, "Human feedback policies"). The paper states the KL term acts as an entropy bonus deterring collapse to a single mode and keeps outputs inside the distribution the reward model saw. The main 1.3B and 6.7B runs use KL coefficient 0.05 (App. B.1, Table 9). Figure 5 is the over-optimization curve across KL penalty coefficients.
- **Ouyang et al. 2022, InstructGPT** (https://arxiv.org/abs/2203.02155). Eq. 2 adds the KL term to the reward and a pre-training term with coefficient γ. β = 0.02 for the reported models (App. C.4). App. E.7 reports that both 0 and 2 give poor performance and the optimal KL reward coefficient is around 0.01 to 0.02; App. E.6 reports that raising the coefficient to 2.0 does not fully remove the regressions on DROP and SQuAD (Figure 34). Card: [[rlhf-instructgpt]].
- **Korbak, Perez, Buckley 2022, "RL with KL penalties is better viewed as Bayesian inference"** (https://arxiv.org/abs/2205.11275). The distribution optimizing the KL-regularized objective is `π_KL-RL(x) = (1/Z) π_0(x) exp(r(x)/β)`, with `π_0` the prior and `Z` the normalizer; minimizing KL to it coincides with the KL-regularized RL objective, so the procedure is variational inference against that target (§4, Eq. 5, Eq. 10, Eq. 16). The paper also shows the unregularized RL objective's optimum is the degenerate deterministic distribution `δ_{x*}` (§2, Eq. 2).
- **Schulman, "Approximating KL Divergence"** (http://joschu.net/blog/kl-approx.html), an author's blog post, not a paper. It is the usual reference for the estimators `k1 = log(p/q)`, `k2 = ½(log(p/q))²`, and `k3 = (p/q) − 1 − log(p/q)` with samples drawn from `q`. Applied to `KL(π ‖ π_ref)` with samples from `π`, the ratio inside k3 is `π_ref/π`, giving `k3 = (π_ref/π) − 1 − log(π_ref/π)`. Which estimator a trainer uses is a code fact; see [[openrlhf-entropy-debugging]] and [[openrlhf-ppo]] for the framework defaults.

## Connections
- [[reward-model-overoptimization]] — measures how far the policy can be pushed from the reference before the true objective degrades; the KL distance is the axis of those curves.
- [[rlhf-instructgpt]] — the RLHF instance of this regularizer, with its own β and ablation.
- [[entropy-mechanism-llm-rl]] — KL-to-reference and entropy are separate quantities; Eq. 7 above is where they appear together in this paper.
- [[maximum-entropy-rl]] — the entropy-regularized RL line the paper cites for the soft value function.
- [[dpo]] — replaces the online KL penalty with a closed-form objective in which β plays the analogous role; verify that claim at the [[dpo]] card, not here.
- [[reinforce-plus-plus]], [[prorl]] — later RL methods that cite the KL term's placement in the reward rather than the loss.

## Verification
- Checked on 2026-09-18 against: https://arxiv.org/abs/1907.00456 (arXiv v2, 2019-07-08), Abstract, §3.2-3.4, §4, §5, §6, Tables 1-3, App. 8.2. Related-artifact claims checked against arXiv:2009.01325, arXiv:2203.02155 (v1), and arXiv:2205.11275 (v2) at the loci given in that section.
- Corrections to the previous card version:
  - The card mixed four artifacts under one title and gave no single primary. It is now an extract of Jaques et al. 2019 — the artifact that introduces the term "KL-control" (§1, §3.3) — with the later instances kept as clearly separated, individually located pointers.
  - "the objective is `E[r] − β·KL(π‖π_ref)`" attributed to this lineage generally → Jaques Eq. 5 writes it as `E_q[r]/c − D_KL(q‖p)`, with `c` scaling the reward rather than a `β` scaling the KL; the `β`-on-the-KL form is Stiennon 2020 §3 and InstructGPT Eq. 2.
  - "Guideline … choose β so that typical end-of-training KL sits at the sweet spot … often β in 0.01–0.2 of the reward scale" → replaced with the located values: 0.05 (Stiennon App. B.1), 0.02 with an optimum near 0.01-0.02 (InstructGPT App. C.4, E.7), c = 2 (Jaques App. 8.2).
  - "k3 ≈ (r − 1) − log r with r = π/π_ref" → for `KL(π‖π_ref)` estimated on samples from `π` the ratio is `π_ref/π`; the card's own Technical Details already had the correct form, so the guideline was inconsistent with it.
  - "Korbak … showed that the 'reverse-KL' nature of the penalty explains mode-seeking behaviour" → Korbak's stated results are the variational-inference equivalence and the degeneracy of the unregularized optimum; mode-seeking is not the paper's claim at those loci.
  - "Stiennon Fig. 3 (KL vs reward Pareto front)" → the over-optimization figure is Figure 5.
  - Missing fields "Source type" and a Verification section were added.
- Removed as unsupported by the source: "practitioners tune in [0.01, 0.5]"; "adding KL to the loss breaks the advantage-based policy gradient and empirically trains worse" (no source checked states this); "Adaptive KL: some implementations (InstructGPT early ablations, DeepSpeedChat) adapt β to hit a target KL per batch" (InstructGPT reports a fixed β = 0.02 and a sweep, not an adaptive controller; no DeepSpeedChat locus was available); "k3 … used in modern TRL/OpenRLHF" as a claim of this card (it is a framework fact belonging to [[openrlhf-ppo]] and [[trl-ppo]]); "Reference policy: usually the SFT checkpoint; frozen; identical tokenizer and architecture" as an unattributed general statement; "Korbak Fig. 1 — the tilted posterior interpretation" (no such figure was located); the DPO implicit-KL paragraph, which belongs to [[dpo]].
- Not reported by the source: a parameter count for the dialog models; any language-model-scale experiment; any comparison of KL-in-reward against KL-in-loss; any recommendation for `c` other than the single value used.
