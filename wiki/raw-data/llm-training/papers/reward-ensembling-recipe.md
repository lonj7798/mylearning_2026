<!-- scope: Coste et al. (2023) reward-model-ensemble experiment settings, split out of [[reward-ensembling]] for length
     deps: [[reward-ensembling]]
-->

# Reward Model Ensembles Help Mitigate Overoptimization — Recipe ledger

Split out of [[reward-ensembling]] under the 120-line card limit. All rows read from arXiv:2310.02743v2
(2024-03-10) on 2026-09-18. These are the paper's synthetic-setup hyperparameters, not a production recipe:
the policy is Pythia 1.4B and the gold reward model is the 7B AlpacaFarm preference model.

| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| Pythia 1.4B policy (paper run) | 1.4B | SFT | LR; epochs; batch size | 8e-6; 3; 4 | arXiv:2310.02743v2 App. D.1 Table 1 | verified (2026-09-18) | no ablation reported |
| Pythia-derived proxy RM (paper run) | 7M / 44M / 1.3B | reward-model | LR; epochs; batch size | 1e-5; 5; 32 | arXiv:2310.02743v2 App. D.1 Table 2 | verified (2026-09-18) | no ablation reported |
| Pythia 1.4B policy (paper run) | 1.4B | RL (PPO) | LR; cosine-annealing floor | 1e-6; 1e-7 | arXiv:2310.02743v2 App. D.1 Table 3 | verified (2026-09-18) | no ablation reported |
| Pythia 1.4B policy (paper run) | 1.4B | RL (PPO) | PPO epochs; batch size; rollouts; chunk size | 4; 32; 256; 32 | arXiv:2310.02743v2 App. D.1 Table 3 | verified (2026-09-18) | no ablation reported |
| Pythia 1.4B policy (paper run) | 1.4B | RL (PPO) | clip range (policy and value); GAE λ | 0.2; 0.95 | arXiv:2310.02743v2 App. D.1 Table 3 | verified (2026-09-18) | no ablation reported |
| Pythia 1.4B policy (paper run) | 1.4B | RL (PPO) | KL penalty with WCO/UWO | 0.01 | arXiv:2310.02743v2 §5.3 | verified (2026-09-18) | §5.3, Figure 6: KL alone needs 0.2 (20x larger) to remove overoptimization, with a performance cost |
| Pythia 1.4B policy (paper run) | 1.4B | RL (PPO) | training steps | 3000 (6000 for 1.3B reward models) | arXiv:2310.02743v2 §4.3; Figure 8 caption | verified (2026-09-18) | slower policy optimization with large reward models (App. F.4) |
| Pythia 1.4B policy (paper run) | 1.4B | eval (generation) | max instruction len; max new tokens; top-p; top-k; temperature | 520; 256; 0.9 (1.0 for PPO training); 0; 1.0 | arXiv:2310.02743v2 App. D.1 Table 4 | verified (2026-09-18) | no ablation reported |


## Verification
- Checked on 2026-09-18 against: https://arxiv.org/abs/2310.02743 (v2).
- Corrections to the previous card version: none (new file; rows moved unchanged from [[reward-ensembling]]).
- Removed as unsupported by the source: none.
- Not reported by the source: optimizer name and betas, weight decay, gradient clipping, warmup, KL
  controller type, GPU-hours.
