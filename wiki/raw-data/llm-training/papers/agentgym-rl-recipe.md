<!-- scope: Recipe ledger for AgentGym-RL / ScalingInter-RL — paper Appendix B values and released training-script values, kept as separate rows
     deps: [[agentgym-rl]]
     see-also: [[grpo]], [[verl-grpo]]
-->

# AgentGym-RL recipe ledger
- **Core Insight:** The paper's Appendix B and the released `examples/train` scripts disagree on several settings (WebArena policy LR 5e-7 vs 1e-6; Deep Search 8 vs 4 trajectories per query and a 4-turn cap vs 5-10 turns), so paper values and script values are recorded as separate facts.
- **Guideline:** When reproducing a table result, use the paper row for the reported setting and treat the unpinned main-branch script as a separate configuration, because the repository does not state which configuration produced the paper checkpoints.
- **Authors:** Zhiheng Xi, Jixuan Huang, Chenyang Liao, Baodai Huang, Honglin Guo, Jiaqi Liu, et al. (Fudan University, ByteDance Seed, Shanghai Innovation Institute)
- **Year:** 2025 (arXiv v1 2025-09)
- **URL:** https://arxiv.org/abs/2509.08755 ; https://github.com/WooooDyy/AgentGym-RL/tree/main/examples/train
- **Source type:** paper + released config/code
- **Relevant topics:** multi-turn agent RL hyperparameters, horizon schedule, GRPO settings

## Summary
Companion to [[agentgym-rl]]. Rows marked "arXiv App. B" come from arXiv:2509.08755v1 Appendix B. Rows marked "script" come from `examples/train/ScalingInter-RL/<env>_train.sh` (SI) or `examples/train/AgentGym-RL/<env>_train.sh` (fixed) on the main branch, read 2026-09-14 without a pinned commit. All scripts load `Qwen2.5-7B-Instruct`, use `algorithm.adv_estimator=grpo`, `use_kl_loss=True`, `kl_loss_coef=0.001`, `kl_loss_type=low_var_kl`, `algorithm.kl_ctrl.kl_coef=0.001`, and `rollout.max_model_len=32768`. Units: `train_batch_size` is the number of task ids per batch read from `data.train_file`, and `rollout.n` trajectories are sampled per task id (Fig. 3 pseudocode `expand(task_ids, sample_num)`); `rounds` is the maximum number of agent-environment turns, `steps_scaling_inter` the training steps between cap increases, `max_response_length` the total trajectory tokens excluding the first-turn task prompt, and `rollout.max_tokens` the tokens per turn (README "Training" section).

## Recipe ledger
| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| AgentGym-RL / ScalingInter (all envs) | 3B, 7B | RL | backbone | Qwen2.5-3B, Qwen2.5-7B (Table 6 labels: -Instruct) | arXiv §4.1, Table 6 | verified 2026-09-14 | Table 6: 3B vs 7B compared |
| AgentGym-RL (all envs) | 3B, 7B | RL | SFT-baseline LR | 1e-4 | arXiv App. B.1-B.5 | verified 2026-09-14 | no ablation reported |
| AgentGym-RL WebArena | 7B | RL | algorithm; policy LR; KL coef; temperature; trajectories/query; max turns | GRPO; 5e-7; 1e-3; 1.0; 4; 15 | arXiv App. B.1 | conflict (LR: script 1e-6) | no ablation reported |
| AgentGym-RL WebArena | 7B | RL | train / test queries | 372 / 50 (from 812; Content & Config excluded) | arXiv App. B.1 | verified 2026-09-14 | not applicable |
| AgentGym-RL Deep Search | 7B | RL | algorithm; policy LR; KL coef; temperature; trajectories/query; max turns | GRPO; 1e-6; 1e-3; 1.0; 8; 4 | arXiv App. B.2 | conflict (scripts: n=4; rounds 5 or [5,8,10]) | Fig. 7 compares caps 5 and 10, not 4 |
| AgentGym-RL TextCraft | 7B | RL | same settings | GRPO; 1e-6; 1e-3; 1.0; 8; 20 | arXiv App. B.3 | conflict (fixed script rounds=30) | no ablation reported |
| AgentGym-RL BabyAI | 7B | RL | same settings | GRPO; 1e-6; 1e-3; 1.0; 8; 20 | arXiv App. B.4 | conflict (SI script n=4) | no ablation reported |
| AgentGym-RL SciWorld | 7B | RL | same settings | GRPO; 1e-6; 1e-3; 1.0; 8; 20 | arXiv App. B.5 | verified 2026-09-14 | no ablation reported |
| ScalingInter (all envs) | 7B | RL | horizon schedule | h_{t+1} = h_t + δh every Δ steps; transition points set from total optimization steps; numbers not given | arXiv §3.3, §4.2 | not reported (numeric values) | Fig. 7 (Deep Search): fixed cap 10 rises early then collapses; fixed cap 5 plateaus; staged cap ends highest (§4.2) |
| ScalingInter WebArena (script) | 7B | RL | rounds; steps_scaling_inter | [8,12,15]; 80 | script SI/webarena_train.sh | verified 2026-09-14 | no ablation reported |
| ScalingInter WebArena (script) | 7B | RL | LR; n; train batch; mini-batch; PPO epochs; total epochs | 1e-6; 4; 32; 4; 2; 25 | script SI/webarena_train.sh | verified 2026-09-14 | no ablation reported |
| ScalingInter WebArena (script) | 7B | RL | max prompt; max response; tokens/turn | 750; 14098; 512 | script SI/webarena_train.sh | verified 2026-09-14 | no ablation reported |
| AgentGym-RL WebArena (fixed script) | 7B | RL | rounds; LR; n; batch; mini-batch; PPO epochs; total epochs | 15; 1e-6; 4; 32; 4; 2; 25 | script fixed/webarena_train.sh | verified 2026-09-14 | no ablation reported |
| ScalingInter Deep Search (script) | 7B | RL | rounds; steps_scaling_inter; LR; n; batch; mini-batch; PPO epochs; total epochs | [5,8,10]; 100; 1e-6; 4; 32; 8; 2; 20 | script SI/searchqa_train.sh | verified 2026-09-14 | no ablation reported |
| ScalingInter Deep Search (script) | 7B | RL | max prompt; max response; tokens/turn | 1024; 8192; 512 | script SI/searchqa_train.sh | verified 2026-09-14 | no ablation reported |
| AgentGym-RL Deep Search (fixed script) | 7B | RL | rounds; n; batch; mini-batch; PPO epochs; total epochs | 5; 4; 32; 8; 2; 20 | script fixed/searchqa_train.sh | verified 2026-09-14 | no ablation reported |
| ScalingInter TextCraft (script) | 7B | RL | rounds; steps_scaling_inter; LR; n; batch; mini-batch; PPO epochs; total epochs | [10,20,30]; 100; 1e-6; 8; 32; 8; 2; 30 | script SI/textcraft_train.sh | verified 2026-09-14 | no ablation reported |
| ScalingInter TextCraft (script) | 7B | RL | max prompt; max response; tokens/turn | 512; 10240; 512 | script SI/textcraft_train.sh | verified 2026-09-14 | no ablation reported |
| AgentGym-RL TextCraft (fixed script) | 7B | RL | rounds; n; batch; PPO epochs; total epochs | 30; 8; 32; 2; 30 | script fixed/textcraft_train.sh | verified 2026-09-14 | no ablation reported |
| ScalingInter BabyAI (script) | 7B | RL | rounds; steps_scaling_inter; LR; n; batch; mini-batch; PPO epochs; total epochs | [6,13,20]; 100; 1e-6; 4; 32; 8; 2; 20 | script SI/babyai_train.sh | verified 2026-09-14 | no ablation reported |
| ScalingInter BabyAI (script) | 7B | RL | max prompt; max response; tokens/turn | 512; 8192; 512 | script SI/babyai_train.sh | verified 2026-09-14 | no ablation reported |
| AgentGym-RL BabyAI (fixed script) | 7B | RL | rounds; n; batch; mini-batch; PPO epochs; total epochs; max prompt; max response; tokens/turn | 20; 8; 16; 8; 1; 10; 1024; 4096; 200 | script fixed/babyai_train.sh | verified 2026-09-14 | no ablation reported |
| ScalingInter SciWorld (script) | 7B | RL | rounds; steps_scaling_inter; LR; n; batch; mini-batch; PPO epochs; total epochs | [10,20,30]; 100; 1e-6; 8; 16; 8; 1; 10 | script SI/sciworld_train.sh | verified 2026-09-14 | no ablation reported |
| ScalingInter SciWorld (script) | 7B | RL | max prompt; max response; tokens/turn | 1024; 8192; 200 | script SI/sciworld_train.sh | verified 2026-09-14 | no ablation reported |
| AgentGym-RL SciWorld (fixed script) | 7B | RL | rounds; n; batch; mini-batch; PPO epochs; total epochs; max prompt; max response; tokens/turn | 20; 8; 16; 8; 1; 10; 1024; 4096; 200 | script fixed/sciworld_train.sh | verified 2026-09-14 | no ablation reported |
| README example | — | RL | ScalingInter launch flags | rounds=[10,20,30]; steps_scaling_inter=100; fixed baseline rounds=15 | repository README "Training" | verified 2026-09-14 | not tied to a specific run |
| all runs | 3B, 7B | RL | optimizer, warmup, grad clip, clip ε, training steps actually run, compute hours, seeds | not reported | checked arXiv §3-§5, App. A-B; README; ten scripts listed above | not reported | — |

## Connections
- [[agentgym-rl]] — the main card with results and findings.
- [[grpo]], [[verl-grpo]] — GRPO advantage estimator and the veRL trainer whose flags the scripts set.
- [[search-r1]] — Deep Search task construction the paper follows (App. B.2).

## Verification
- Created on 2026-09-14 from https://arxiv.org/abs/2509.08755 (arXiv v1) and github.com/WooooDyy/AgentGym-RL main branch (README and ten `*_train.sh` scripts; no commit hash recorded by the fetch).
- Audit claims not found in the source: "the README example [10, 20, 30] every 100 steps is the ScalingInter-RL schedule" (the per-environment scripts use four different schedules; see rows above).
- Not reported by the source: which script revision produced the paper's Tables 1-6; whether App. B turn caps apply to training, evaluation, or both.
