<!-- scope: recipe ledger for AgentRL (arXiv 2510.04206): RL settings stated in the paper and, separately, values in the released THUDM/AgentRL training configs
     deps: [[agentrl]]
     see-also: [[grpo]], [[verl-rollout]], [[agenttuning]]
-->

# AgentRL — Recipe ledger
- **Core Insight:** The paper states temperature 0.8, 8 samples per rollout, a −0.2 penalty for incomplete trajectories, and "over 1000 steps" (App. E.2), while the learning rate (1.0e-6), KL loss coefficient (1.0e-4), clip ratio (0.2), and batch size (256 prompts × 8) appear only in the released config `qw14b_waodk.yaml`.
- **Guideline:** When reproducing the Table 3 checkpoints, treat config values as released-config facts, because the paper does not state which config file produced which reported model.
- **Authors:** Hanchen Zhang, Xiao Liu, Bowen Lv, Xueqiao Sun, Bohao Jing, Iat Long Iong, et al. (Tsinghua University; Z.AI)
- **Year:** 2025 (arXiv v1 2025-10)
- **URL:** https://arxiv.org/abs/2510.04206 ; https://github.com/THUDM/AgentRL/tree/main/examples/training
- **Source type:** paper + released config/code
- **Relevant topics:** agentic RL hyperparameters, GRPO, KL loss, cross-policy sampling, multi-task RL

## Recipe ledger
Paper locus = arXiv:2510.04206v1. Config locus = github.com/THUDM/AgentRL@main (read 2026-09-14, commit not pinned), `examples/training/configs/<file>`. Trainer locus = `examples/training/agentrl_trainer.py`.

| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| AgentRL w/ Qwen2.5-{3B,7B,14B,32B}-Instruct | 3B-32B | RL | Initialization | Instruct checkpoints; no warm-up SFT | paper §4.1; App. E.2 | verified 2026-09-14 | no ablation reported |
| AgentRL w/ GLM-4-9B-0414 | 9B | SFT | Cold-start SFT before RL | "a limited set of supervised fine-tuning (SFT) data"; amount not reported | paper App. E.2 | not reported (amount; checked body, App. A-H, README, both configs) | needed to adapt to the function-call format (App. E.2); no ablation reported |
| AgentRL (all sizes) | 3B-32B | RL | Algorithm | GRPO baseline + cross-policy sampling + task advantage normalization | paper App. E.2; §3 | verified 2026-09-14 | Table 6 (14B): 65.0 full; 60.7 without cross sampling; 59.4 without task adv. norm |
| AgentRL (all sizes) | 3B-32B | RL | Rollout temperature; samples per rollout | 0.8; 8 | paper App. E.2 | verified 2026-09-14 | no ablation reported |
| AgentRL (all sizes) | 3B-32B | RL | Reward | binary trajectory correctness (range [0, 1]); −0.2 for exceeding max interaction rounds or max response length (abnormal termination) | paper App. E.1, E.2 | verified 2026-09-14 | no ablation reported |
| AgentRL (multi-task) | 3B-32B | RL | Training length | "over 1000 steps" | paper App. E.2 | verified 2026-09-14 | no ablation reported |
| AgentRL (multi-task) | 3B-32B | RL | Task sampling | smaller datasets replicated to about the largest task's size; datasets interleaved one element at a time | paper §4 "Data" | verified 2026-09-14 | no ablation reported |
| AgentRL w/ Qwen2.5-14B-Instruct | 14B | RL | Hardware | H800; minimum 16 GPUs | paper App. E.2 | verified 2026-09-14 | Fig. 4 throughput at 16/32/64 GPUs |
| AgentRL (all sizes) | 3B-32B | RL | Engines | Verl-based asynchronous trainer; SGLang rollout; FSDP | paper App. E.2 | verified 2026-09-14 | Fig. 4 vs synchronous baseline |
| AgentRL (all sizes) | 3B-32B | eval-gate | Evaluation temperature; runs | 0.8; mean of 4 runs | paper App. E.2 | verified 2026-09-14 | no ablation reported |
| AgentRL (all sizes) | 3B-32B | RL | LR, KL, clip ε, batch, max lengths | not in paper | paper body, App. A-H | not reported (paper) | see config rows |
| Config qw14b_waodk (Qwen2.5-14B-Instruct; WebShop, ALFWorld, OS, DB, KG) | 14B | RL | max_steps; val_interval | 1000; 25 | qw14b_waodk.yaml | verified 2026-09-14 | no ablation reported |
| Config qw14b_waodk | 14B | RL | Prompts per step; samples per prompt | train.batch_size 256; train.n 8 (trainer: `real_bsz = batch_size * n` = 2,048 trajectories) | qw14b_waodk.yaml; agentrl_trainer.py `main` | verified 2026-09-14 (2,048 is derived: 256 × 8) | no ablation reported |
| Config qw14b_waodk | 14B | RL | Optimizer LR; weight decay; actor dtype | 1.0e-6; 0; float32 | qw14b_waodk.yaml `actor.optim`, `actor.torch_dtype` | verified 2026-09-14 | no ablation reported |
| Config qw14b_waodk | 14B | RL | KL loss; KL coefficient | use_kl_loss True; kl_loss_coef 1.0e-4 (reference worker) | qw14b_waodk.yaml `loss` | verified 2026-09-14 | no ablation reported |
| Config qw14b_waodk | 14B | RL | Clip ratio (low, high); clip_ratio_c; entropy coef | 0.2 (0.2, 0.2); 3.0; 0.0 | qw14b_waodk.yaml `loss` | verified 2026-09-14 | no ablation reported |
| Config qw14b_waodk | 14B | RL | Loss reduction; max tokens per micro-batch | token-mean; 16,384 | qw14b_waodk.yaml `actor` | verified 2026-09-14 | no ablation reported |
| Config qw14b_waodk | 14B | RL | Sampling temperature; max new tokens per generation | 0.8; 1,024 | qw14b_waodk.yaml `rollout.sampling_params` | verified 2026-09-14 | no ablation reported |
| Config qw14b_waodk | 14B | RL | Incomplete penalty; max total length; max turns | −0.2; 8,192; 20 | qw14b_waodk.yaml `task` | verified 2026-09-14 | no ablation reported |
| Config qw14b_waodk | 14B | RL | Rollout GPU fraction; stale engine fraction; stale sync interval | rollout_ratio 0.5; rollout_stale_ratio 0.25; stale_step 25 | qw14b_waodk.yaml; agentrl_trainer.py `main`, `cross_sampler` | verified 2026-09-14 | Table 6 ablates cross sampling on/off only |
| Config qw14b_waodk | 14B | RL | Task advantage normalization; advantage estimator; FSDP size | True; grpo; 16 | qw14b_waodk.yaml | verified 2026-09-14 | Table 6 (on/off) |
| Config qw3b_ws (Qwen2.5-3B-Instruct; WebShop only) | 3B | RL | Prompts per step; samples per prompt; max_steps | 32; 8; 1000 | qw3b_ws.yaml | verified 2026-09-14 | no ablation reported |
| Config qw3b_ws | 3B | RL | LR; KL coef; clip; temperature; stale ratio; stale_step; penalty; max turns | 1.0e-6; 1.0e-4; 0.2; 0.8; 0.25; 25; −0.2; 20 (same as qw14b_waodk) | qw3b_ws.yaml | verified 2026-09-14 | no ablation reported |

Values that agree between paper and config: temperature 0.8, 8 samples, −0.2 penalty, 16 GPUs for 14B (`fsdp_size: 16`). Training length differs in wording: the paper says "over 1000 steps" and the config sets `max_steps: 1000`. The paper does not name the config behind any Table 3 row, so no row above is marked as the released-checkpoint setting.

## Starting point for a small general-purpose run
For a 14B instruct model trained with GRPO on five function-call agent environments on 16 or more H800 GPUs, the released config uses LR 1.0e-6, KL loss coefficient 1.0e-4, clip ratio 0.2, 256 prompts × 8 samples per step, temperature 0.8, at most 20 turns and 8,192 total tokens, a −0.2 penalty for incomplete trajectories, per-task advantage normalization, and 25% stale rollout engines synced every 25 steps, for 1000 steps. These values come from the verified config rows above; the paper reports no sweep over them.

## Connections
- [[agentrl]]: the main card (method, results, ablations).
- [[grpo]]: advantage estimator used by the config (`adv_estimator: grpo`).
- [[verl-rollout]]: framework the paper's trainer was built from (App. E.2).
- [[agenttuning]]: earlier SFT-based agent training on AgentBench tasks, for contrast with RL from instruct checkpoints.

## Verification
- Created on 2026-09-14 from https://arxiv.org/abs/2510.04206 (arXiv v1) and https://github.com/THUDM/AgentRL (main branch; examples/training/configs/qw14b_waodk.yaml, qw3b_ws.yaml; examples/training/agentrl_trainer.py).
- Audit claims not found in the source: none for the values in this ledger. The batch-level audit lead that GLM-4-9B had no SFT warm-up is contradicted by App. E.2 (recorded in [[agentrl]]).
- Not reported: amount of GLM4-9B cold-start SFT data; configs for 7B, 32B, and GLM-4-9B (only the two files above were listed in the configs directory as displayed); GPU hours; warmup and LR schedule (no schedule key in either config).
