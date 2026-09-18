<!-- scope: REINFORCE Leave-One-Out — k-sample REINFORCE with a per-prompt leave-one-out baseline for RLHF
     deps: [[vanilla-pg]], [[ppo]]
     see-also: [[grpo]], [[reinforce-plus-plus]], [[rloo-vs-grpo]]
-->

# Back to Basics: Revisiting REINFORCE Style Optimization for Learning from Human Feedback in LLMs
- **Core Insight:** In RLHF the policy starts from a strong SFT initialization and the reward arrives once at the end of a full generation, so PPO's variance-reduction machinery is not needed: on TL;DR Summarize and Anthropic-HH, RLOO with k = 4 reaches simulated win rates of 77.9, 43.7 and 64.1, against PPO's 67.6, 29.2 and 32.0 (Table 1).
- **Guideline:** When running online RLHF against a trained reward model, sample k = 2 or 4 completions per prompt and use the mean reward of the other k − 1 as the baseline, because RLOO with k = 2 matches or beats RAFT with k = 4 at half the online-sample budget across all three dataset and model pairings tested (§5.1, Figure 3). When the reward model is noisy or the KL coefficient must be large, prefer RLOO over RAFT, because RAFT's reward drops further at noise σ = 3.0 and 5.0 and deviates more from the reference policy at β ∈ {0.25, 0.5, 1.0} (§5.3, Figures 5 and 6).
- **Authors:** Arash Ahmadian, Chris Cremer, Matthias Gallé, Marzieh Fadaee, Julia Kreutzer, Olivier Pietquin, Ahmet Üstün, Sara Hooker (Cohere / Cohere For AI)
- **Year:** 2024 (arXiv v1 2024-02; v2 2024-02-26; ACL 2024)
- **URL:** https://arxiv.org/abs/2402.14740
- **Source type:** paper
- **Relevant topics:** variance reduction, REINFORCE, online RLHF, critic-free policy gradient, LLM alignment

## Abstract
PPO has been treated as the canonical RL algorithm for RLHF, but it carries high computational cost and sensitive hyperparameter tuning. The authors argue that the motivating assumptions behind PPO are less of a practical concern in RLHF, revisit the formulation of preference alignment as an RL problem, and show that several PPO components are unnecessary in this setting. Simpler REINFORCE-style variants, in particular REINFORCE Leave-One-Out (RLOO), outperform PPO and the "RL-free" methods DPO and RAFT on TL;DR Summarize and Anthropic Helpful & Harmless with Pythia-6.9B and Llama-7B policies.

## Key Contributions
- Argues that PPO is not the right tool for RLHF, and shows Vanilla Policy Gradient REINFORCE outperforming PPO by 3.2 to 20.3 points of win rate across all dataset and base-model pairings (§1 contribution 1; Table 1).
- Applies the **RLOO estimator** of Kool et al. (2019) to LLM preference training: each of k online samples uses the mean reward of the other k − 1 as an unbiased, parameter-free baseline (§2.3).
- Shows that modeling partial completions is unnecessary: full-generation modeling preserves performance and accelerates learning (§1 contribution 3; §5.1).
- Reports robustness of RLOO relative to RAFT under reward noise and under stronger KL regularization (§5.3).
- Reports fluency, diversity, and reward-variance metrics alongside win rate rather than reward alone (§5.2.1, Table 2).

## Key Figures/Tables to Study
- **Figure 1:** PPO reward curves on Llama-7B / HH under varying GAE λ; reward decreases monotonically as λ decreases, that is as variance reduction is traded for bias.
- **Figure 2:** Test reward through training for RLOO, RAFT, REINFORCE with baseline, Vanilla PG, and PPO on three dataset/model pairings.
- **Figure 3 and Figure 4:** Sample efficiency of RLOO vs RAFT at k = 2 and k = 4.
- **Table 1:** Final simulated win rates for all methods on held-out test prompts.
- **Table 2:** Length, perplexity, Diversity-1/2, and reward variance on Anthropic-HH with Llama-based models.
- **Figures 5 and 6:** Sensitivity to the KL weight β and to injected reward noise.
- **§2.3:** The RLOO gradient estimator.

## Technical Details

### Setting
- RLHF is treated as a bandit problem over full generations: the reward model scores a complete completion, and the KL penalty is applied per token as a shaped reward `R(x,y) = r_φ(x,y) − β log(π_θ(y|x)/π_ref(y|x))` (§2.2, Eq. 3 and 4).
- Datasets: TL;DR Summarize (Stiennon et al., 2020) and the preprocessed Anthropic-HH dataset with 112k training preference pairs (§4).
- Policies: Pythia-6.9B on both datasets; Llama-7B additionally on Anthropic-HH (§4).
- Prompts longer than 448 tokens (TL;DR) and 348 tokens (HH) are filtered out to reduce generations without an EOS token (App. C).

### RLOO gradient estimator (§2.3)
`(1/k) Σ_{i=1..k} [ R(y^(i), x) − (1/(k−1)) Σ_{j≠i} R(y^(j), x) ] ∇ log π_θ(y^(i) | x)`, with `y^(1),…,y^(k)` sampled i.i.d. from `π_θ(·|x)`.
- `k` is the number of online samples per prompt; `R` is the KL-shaped reward of Eq. 3.
- The baseline is unbiased because it does not depend on `y^(i)`, and is described as a parameter-free value function re-estimated at each training step (§2.3).
- The cost is increased sampling time during training (§2.3).

### Evaluation
- Win rates are simulated with GPT-4 as the judge, against reference SFT completions for TL;DR and preferred completions for HH; generation at evaluation uses greedy sampling (§4, §5.2, App. D).
- Win rates in Table 1 are reported for the checkpoint with the highest test reward.

## Recipe ledger
All rows verified 2026-09-18 against arXiv:2402.14740v2.

| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| Pythia (TL;DR, HH) | 6.9B | SFT | epochs / initial LR | 2 / 2e-5 | App. C | verified | follows Touvron et al. 2023b and Bai et al. 2022a; no ablation reported |
| Llama (HH) | 7B | SFT | epochs | 1 | App. C | verified | "1 epoch was sufficient"; no ablation reported |
| Pythia / Llama | 6.9B / 7B | reward-model | epochs / initial LR / schedule | 1 / 1e-5 / cosine decay, 0.03 warmup ratio | App. C | verified | no ablation reported |
| Pythia (TL;DR) | 6.9B | RL | steps / rollout batch / step batch / β | 600 / 512 / 256 / 0.03 | App. C | verified | no ablation reported |
| Pythia (HH) | 6.9B | RL | steps / rollout batch / step batch / β | 393 / 512 / 256 / 0.10 | App. C | verified | β varied to {0.25, 0.5, 1.0} in the sensitivity study (§5.3, Figure 5) |
| Llama (HH) | 7B | RL | rollout batch / step batch / epochs / β | 2048 / 2048 / 2 / 0.10 | App. C | verified | setup follows Dong et al. 2023 |
| all RL runs | — | RL | learning rate / warmup / gradient steps per batch | constant 1e-6 / linear over 3% of steps / 2 | App. C | verified | swept {1e-6, 1e-5, 2e-5} for RAFT and RLOO, {1e-6, 1e-5} for PPO and Vanilla PG |
| all RL runs | — | RL | samples per prompt k | 2 and 4 | §5.1, Table 1 | verified | Figure 3: RLOO k=2 matches or beats RAFT k=4 at half the sample budget |
| RLOO | — | RL | value network / GAE / ratio clipping | none | §2.3, §3 | verified | §3.1, Figure 1: reward decreases monotonically as GAE λ decreases |

## Findings relevant to generality and negative feedback
- **Fluency and diversity are not sacrificed for reward.** On Anthropic-HH with Llama-based models, RLOO k=4 gives length 60.6, PPL 27.6, Diversity-1 0.10, Diversity-2 0.43; PPO gives length 16.5, PPL 40.4, Diversity-1 0.34, Diversity-2 0.60; DPO gives length 104.4, PPL 33.8, Diversity-1 0.08 (Table 2). The authors note Diversity-2 decreases slightly for the methods that optimize reward most, and attribute this to generation length (§5.2.1).
- **Reward variance.** RLOO has slightly lower reward variance than RAFT at the same k (3.1 vs 3.2 at k=4); Vanilla PG is highest at 3.7 and PPO lowest at 2.3 (Table 2). The authors frame low reward variance as desirable for safety applications (§5.2.1).
- **Negative signal.** RLOO's only use of below-average samples is the sign of `R(y^(i),x) − b_i` in the gradient, which decreases the likelihood of samples scoring below their peers (§2.3). The paper does not analyze what this does to probability mass, entropy, or pass@k; those questions are not addressed by this source.
- **Robustness.** Under injected reward noise `ε ~ N(0, σ²)` added to the classifier logits, RAFT's training reward drops far more than RLOO's at σ = 3.0 and 5.0 (§5.3, Figure 6). Under β ∈ {0.25, 0.5, 1.0}, RAFT both optimizes reward worse and deviates further from the reference policy than RLOO (§5.3).
- **Stated limitation.** The authors note that they report only simulated (GPT-4) win rates and do not measure correlation with human judgments (§7).

## Connections
- Direct ancestor: [[vanilla-pg]] — RLOO is REINFORCE with a leave-one-out baseline.
- Method the paper argues against: [[ppo]].
- Group-baseline relative: [[grpo]].
- Global-normalization relative: [[reinforce-plus-plus]].
- Comparison notes: [[rloo-vs-grpo]].
- The KL-shaped-reward formulation it inherits: [[rlhf-instructgpt]], via Stiennon et al. (2020).
- Framework implementations: [[trl-ppo]], [[openrlhf-ppo]].

## Verification
- Checked on 2026-09-18 against: https://arxiv.org/abs/2402.14740 (arXiv v2, 2024-02-26).
- Corrections to the previous card version:
  - "beats PPO on TL;DR and HH-RLHF by 5–20% win rate at comparable KL" → the 3.2 to 20.3 point range is Vanilla PG REINFORCE over PPO (§1, contribution 1); RLOO k=4 beats PPO by 10.3, 14.5 and 32.1 points on TL;DR, HH (Pythia) and HH (Llama) (§5.2, Table 1). No comparison at matched KL is reported.
  - "Figure 3: TL;DR win rate vs KL Pareto frontier — RLOO dominates PPO and DPO at every KL" → Figure 3 compares sample efficiency of RLOO against RAFT at k = 2 and 4. No Pareto frontier over KL is plotted.
  - "Figure 5: k=2 vs k=4 vs k=8 — diminishing returns beyond k=4" → the paper tests only k = 2 and k = 4; Figure 5 is the KL-weight sensitivity study.
  - "Section 3 / Equation 6" for the RLOO estimator → the estimator is the unnumbered display in §2.3; Eq. 6 is the plain REINFORCE gradient.
  - "KL coef β 0.05 (tuned per task on Pareto curve)" → β = 0.03 for TL;DR and β = 0.10 for Anthropic-HH (App. C).
  - "Learning rate 1e-6 to 3e-6 (AdamW)" → a constant 1e-6 for all RL runs, chosen from a sweep of {1e-6, 1e-5, 2e-5}; the optimizer is not named (App. C).
  - "Batch size (prompts) 32–64" → rollout batch 512 and step batch 256 for Pythia runs; 2048 rollout and step batch for Llama runs (App. C).
  - "Max new tokens 53 (TL;DR), 256 (HH)" → the paper reports prompt-length filters of 448 (TL;DR) and 348 (HH) tokens; generation length limits are not reported (App. C).
  - "Epochs per rollout K: 1" → 2 gradient steps are taken per batch (App. C).
  - "beats PPO at a fraction of the cost … ~50% memory footprint of PPO" → the paper states PPO loads up to 4 models simultaneously (§1) but reports no memory measurement.
- Removed as unsupported by the source: the claim that RLOO uses about 50% of PPO's memory; "Sampling T 1.0" (training temperature is not reported; evaluation uses greedy sampling); "no entropy bonus" as a stated design choice; the "Relationship to GRPO" paragraph asserting equivalence up to scaling (GRPO is not discussed in this paper — see [[rloo-vs-grpo]]); the claim that the paper "provides a one-line algorithm that non-RL practitioners can implement".
- Not reported by the source: training-time sampling temperature; generation length caps; optimizer name and betas; memory or wall-clock measurements; results at k > 4.
