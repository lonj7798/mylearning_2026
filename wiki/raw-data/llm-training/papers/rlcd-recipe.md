<!-- scope: recipe ledger for [[rlcd]]; values read at the loci given
     deps: [[rlcd]]
-->

# Recipe ledger — RLCD: Reinforcement Learning from Contrastive Distillation for LM Alignment

Split from `rlcd.md` to keep that card at or under 120 lines. Table format follows §5.2 of the
llm-training authoring standard.


| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| LLaMA-7B / LLaMA-30B (generator) | 7B / 30B | preference | precision; sampling temperature; resample retries | 8-bit; T = 1; up to 5 | arXiv:2307.12950v3 App. E | verified 2026-09-18 | no ablation reported |
| LLaMA-7B (aligned policy) | 7B | preference | preference-model and SFT hyperparameters | AlpacaFarm defaults (Dubois et al., 2023) | arXiv:2307.12950v3 App. E | verified 2026-09-18 | no ablation reported |
| LLaMA-7B (aligned policy) | 7B | RL | KL coefficient; PPO steps (512 rollouts per step) | grid search over {0.001, 0.002, 0.004, 0.008, 0.016, 0.032} and {20, 40, 60, 80}; outlining fixed at 20 steps | arXiv:2307.12950v3 App. E | verified 2026-09-18 | too many PPO steps degraded performance, including mode collapse on outlining (App. E) |
| LLaMA-7B (aligned policy) | 7B | RL | checkpoint/hyperparameter selection rule | 1000 validation generations scored by each method's own learned reward model; no stronger evaluator used | arXiv:2307.12950v3 App. E | verified 2026-09-18 | stated as a deliberate constraint (App. E) |

## Verification
- Checked on 2026-09-18 against the primary source cited in each Source location cell.
- Corrections to the previous card version: new file, split out of `rlcd.md`.
- Removed as unsupported by the source: none.
