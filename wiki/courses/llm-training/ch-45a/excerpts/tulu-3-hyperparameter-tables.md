---
chapter: ch-45a
course: llm-training
phase: read
excerpt_of: "Tülu 3 report Tables 18, 20, 21, 34, 35 and the matching open-instruct launch commands; the library cards [[tulu-3]] and [[rlvr-tulu3]] carry preference and RLVR numbers that these tables do not support"
source_url: https://arxiv.org/abs/2411.15124
primary_version: "arXiv:2411.15124 (Tülu 3 report, PDF read 2026-09-15) + github.com/allenai/open-instruct docs/tulu3.md"
created_at: "2026-09-15"
---

# Excerpt: Tülu 3 preference and RLVR settings (Tables 18–21, 34–35, open-instruct commands)

Lambert et al. (Ai2), 2024. ch-45a uses these tables for the Tülu 3 rows of both ledgers.

## Table 18 — the algorithm ablation that selected length-normalized DPO
UltraFeedback, on top of an early Tülu 3 SFT checkpoint; one run per row; "Average Score" is the report's
development average.

| Algorithm | LR | β | Epochs | Batch | Average |
|---|---|---|---|---|---|
| SFT base | — | — | — | — | 55.7 |
| SimPO | 5.0e-7 | 2 (γ−β ratio 0.5) | 1 | 128 | 51.8 |
| SimPO | 5.0e-7 | 10 (γ−β ratio 0.3) | 1 | 128 | 52.9 |
| DPO | 5.0e-7 | 0.1 | 3 | 32 | 55.2 |
| PPO | 1.0e-6 | 0.0325 | 1 | 64 | 54.5 |
| PPO | 1.0e-6 | 0.05 | 1 | 64 | 55.5 |
| DPO-norm | 1.0e-7 | 5 | 3 | 32 | 56.1 |
| DPO-norm | 5.0e-7 | 10 | 3 | 32 | 55.2 |
| DPO-norm | 5.0e-7 | 15 | 3 | 32 | 55.7 |
| DPO-norm | 5.0e-7 | 2 | 3 | 32 | 46.8 |
| DPO-norm | 5.0e-7 | 5 | 3 | 32 | 53.4 |
| DPO-norm | 5.0e-7 | 5 | 1 | 32 | 57.3 |

Report text: "We found that only length-normalized DPO outperformed our base checkpoint overall, and so further
tuned it, resulting in the final hyperparameters shown in Table 20."

## Table 20 — final DPO hyperparameters
| Hyperparameter | 8B | 70B |
|---|---|---|
| Learning rate | 5e-7 | 2e-7 |
| LR schedule | Linear | Linear |
| Batch size (effective) | 128 | 128 |
| Max token length | 2,048 | 2,048 |
| "KL penalty coefficient β" | 5 | 5 |
| Warm-up ratio | 0.1 | 0.1 |
| Epochs | 1 | 1 |

The column label "KL penalty coefficient β" is the report's; the value 5 is the β of the length-normalized DPO
loss, not a KL coefficient added to a reward. Table 19 (70B LR ablation, two mixes): 5.0e-7 gives 72.74 and
71.14, 2.0e-7 gives 71.17 and 74.35, 1.5e-7 gives 71.12, 1.0e-7 gives 71.06.

## Table 21 — PPO hyperparameters (RM column and RLVR column)
Shared: γ = 1.0, GAE λ = 0.95, mini-batches N_mb = 1, clip ε = 0.2, value-function coefficient c1 = 0.1,
gradient-norm threshold 1.0, linear LR schedule, generation temperature 1.0, max token length 2,048, max prompt
length 2,048, penalty reward −10.0 for a response without an EOS token.

| Setting | against a general RM | against the verifiable reward |
|---|---|---|
| Learning rate | 3e-7 | 3e-7 (1e-7 for 70B) |
| Batch size (effective) | 224 | 224 (640 for 70B) |
| PPO update iterations K | 1 | 4 |
| Response length | 1,024 | 2,048 (1,024 for GSM8K only) |
| Total episodes | 300,000 | 100,000 |
| KL penalty coefficient β | [0.05, 0.03, 0.02, 0.01] | [0.1, 0.05, 0.03, 0.01] |
| Warm-up ratio ω | [0.1, 0.0] | [0.0, 0.1] |

Caption: "The final 8B RLVR model used β = 0.05 and ω = 0.0; the final 70B RLVR model used β = 0.07 and ω = 0.07."

§6.4 prose for the same 70B run: "we used the hyperparameters from Table 21, but with a 1 × 10⁻⁷ learning rate,
0.1 warmup ratio, 2048 response length, 400,000 episodes, 640 effective batch size, and β = 0.7". The "β = 0.7"
here disagrees with the caption's 0.07 and with the released launch command (`--beta 0.07`), and "400,000
episodes" disagrees with the table's 100,000 and agrees with the launch command (`--total_episodes 400000`).

Table 22 (verifiable prompt set): GSM8K train 7,473; MATH train 7,500; IF verifiable 14,973; total 29,946.

## Table 34 / Table 35 — 405B
- 405B SFT: LR 2e-6, linear, batch 256, max length 4,096, warm-up 0.03, 2 epochs.
- 405B DPO: LR 2e-7, linear, batch 256, max length 2,048, β = 5, warm-up 0.1, 1 epoch.
- 405B RLVR (Table 35, whose caption reads "optimizing against a general RM" while §8.1 introduces it as the
  RLVR table and its column header reads "405B RLVR"): LR 1e-7, γ 1.0, λ 0.95, N_mb 1, ε 0.2, c1 0.1, batch
  1,856, response length 1,024, total episodes 300,000, β 0.05, ω 0.0, K 1.
- §8.1: GSM8K and IFEval data were dropped for 405B RLVR, leaving MATH only; "even with as few as 25 RLVR
  steps, MATH performance improved by over 5 points"; the run was stopped at 75 steps for compute reasons.

## open-instruct launch commands (docs/tulu3.md)
- `Llama-3.1-Tulu-3-8B-DPO`: `dpo_tune.py --learning_rate 5e-07 --lr_scheduler_type linear --warmup_ratio 0.1
  --num_epochs 1 --max_seq_length 2048 --dpo_loss_type dpo_norm --dpo_beta 5`, 8 GPUs, gradient accumulation 16.
- `Llama-3.1-Tulu-3-8B` (RLVR): `ppo_vllm_thread_ray_gtrl.py --learning_rate 3e-7 --beta 0.05 --temperature 1.0
  --response_length 2048 --local_mini_batch_size 32 --local_rollout_batch_size 32 --actor_num_gpus_per_node 7
  --penalty_reward_value -10.0 --total_episodes 10000000 --apply_verifiable_reward true
  --reward_model_multiplier 0.0`, from `allenai/Llama-3.1-Tulu-3-8B-DPO`.
  7 × 32 = 224 matches Table 21's effective batch; `--total_episodes 10000000` is the script's stopping budget
  and does not match Table 21's 100,000.
  `--reward_model_multiplier 0.0` means the loaded reward model contributes nothing to the reward; it supplies
  the value-model initialization only.
- `Llama-3.1-Tulu-3-70B` (RLVR): same script with `--learning_rate 1e-7 --beta 0.07 --warmup_ratio 0.1
  --total_episodes 400000 --local_mini_batch_size 16`, 40 actor GPUs, so 40 × 16 = 640.
- `Llama-3.1-Tulu-3.1-8B`: `grpo_vllm_thread_ray_gtrl.py --learning_rate 5e-7 --beta 0.01 --kl_estimator kl3
  --number_samples_per_prompt 16 --local_rollout_batch_size 4 --local_mini_batch_size 32 (= 4 × 16 / 2)
  --response_length 2048 --temperature 1.0 --lr_scheduler_type constant --num_epochs 1
  --penalty_reward_value 0.0 --total_episodes 10000000`, 2 nodes (16 GPUs), from
  `allenai/Llama-3.1-Tulu-3-8B-DPO`, data `allenai/RLVR-GSM-MATH-IF-Mixed-Constraints`. The doc marks the
  script as removed from open-instruct and preserved "for historical reference"; the experiments ran at commit
  `745bf58d321c`.

## Verification
- Read on 2026-09-15 from the cached primary text of the Tülu 3 report (scratchpad `sources/tulu-3.txt`,
  §5.4.1, §6.3–§6.4, §8.1) and of open-instruct `docs/tulu3.md` (scratchpad
  `sources/tulu-3.1-openinstruct-doc.txt`).
- Values in the older library cards that these sources do not support: RLVR "learning rate 3e-7, β 0.05, total
  episodes 10,000,000" as one row (the episode count comes from the launch script, not Table 21); "group/batch
  ~128 prompts, 4 rollouts each, lr ~1e-6, β_KL ~0.04, length up to 2k–4k tokens" ([[rlvr-tulu3]] Technical
  Details); "preference data size: hundreds of thousands of pairs" (Table 15's final mixes are stated as "more
  than 270k data points", §5.3).
- Not reported: the number of PPO steps actually run for the 8B RLVR model (the report evaluates "every 100
  training steps, 40 for 70B" and selects on MATH and IFEval); DPO optimizer betas.
