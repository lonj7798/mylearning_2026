<!-- scope: DeepSpeed-Chat (Microsoft DeepSpeed, 2023) — paper arXiv:2308.01320 plus code in DeepSpeedExamples/applications/DeepSpeed-Chat: InstructGPT-style SFT → reward model → PPO pipeline, Hybrid Engine (train/inference mode switching), PPO trainer defaults and documented instabilities
     deps: [[rlhf-instructgpt]], [[ppo]]
     see-also: [[openrlhf-ppo]], [[trl-ppo]], [[verl-rollout]], [[kl-control-rlhf]], [[hh-rlhf]], [[costa-huang-ppo-details]], [[minillm]]
-->

# DeepSpeed-Chat: Easy, Fast and Affordable RLHF Training of ChatGPT-like Models at All Scales
- **Core Insight:** In step-3 PPO, generation of 256 answer tokens is about 20% of the computation but dominates time because it is memory-bandwidth bound; switching the same actor between DeepSpeed inference kernels / tensor parallelism for generation and ZeRO sharding for training (Hybrid Engine) gives up to 9× generation throughput over HuggingFace and 15× over Colossal-AI, and OPT-13B step 3 in 9 hours on one 8×A100-80GB node (§4, §5.2–5.3; Table 1).
- **Guideline:** In this codebase, keep one PPO update per generated batch (`ppo_epochs=1`, `generation_batches=1`, generation batch = training batch), because the maintainers report that multiple actor updates per generated batch were unstable; treat the InstructGPT pretraining-mix coefficient 27.8 as untuned, since it also caused instability (step-3 README, "Instability of RLHF training"; BenckmarkSetting.md).
- **Authors:** Zhewei Yao, Reza Yazdani Aminabadi, Olatunji Ruwase, Samyam Rajbhandari, Xiaoxia Wu, Ammar Ahmad Awan, et al. (Microsoft DeepSpeed; 19 authors, last-listed Yuxiong He)
- **Year:** 2023 (arXiv v1 2023-08-02; only version). Code inspected at deepspeedai/DeepSpeedExamples master, last commit touching DeepSpeed-Chat 43578c4 (2026-09-27).
- **URL:** https://arxiv.org/abs/2308.01320 ; https://github.com/microsoft/DeepSpeedExamples/tree/master/applications/DeepSpeed-Chat (redirects to deepspeedai/DeepSpeedExamples)
- **Source type:** framework (system paper + code)
- **Relevant topics:** RLHF systems, PPO implementation, reward-model training, KL-shaped reward, GAE, EMA checkpoints, pretraining-loss mixing, throughput

## Abstract
DeepSpeed-Chat provides (1) a single script that takes a Hugging Face model through the three InstructGPT steps and an inference API, (2) the DeepSpeed-RLHF pipeline replicating InstructGPT's SFT, reward-model fine-tuning and RLHF with data abstraction and blending, and (3) the DeepSpeed-RLHF system, whose Hybrid Engine (DeepSpeed-HE) combines DeepSpeed training and inference optimizations. The paper reports over 15× speed-up over existing systems and training of up to 175B-parameter models.

## Key Contributions
- End-to-end three-step pipeline with optional EMA checkpoint and "Mixture Training" (PPO + pretraining loss) in step 3 (§3, Fig. 1).
- Hybrid Engine: inference mode uses optimized kernels, KV-cache memory management and tensor parallelism; training mode uses ZeRO and LoRA; repartitioning between modes (§4, Fig. 2).
- Cost/time tables for OPT 1.3B–175B on Azure (Tables 1–6).
- Throughput comparisons: >10× on one GPU; 6–19× over Colossal-AI and 1.4–10.5× over HuggingFace DDP on one node (§5.2, Figs. 3–4).

## Key Figures/Tables to Study
- Table 1 / Table 2: single-node and 64-GPU step-3 time and cost. Table 4: per-step time breakdown (OPT-13B: step 1 2.5 h, step 2 0.25 h, step 3 10.8 h).
- Fig. 5: generation vs training time per iteration. Fig. 6: effective TFLOPs/GPU by model size. Fig. 7: scaling to 64 GPUs.

## Technical Details
**Benchmark setting (paper footnote 1; BenckmarkSetting.md).** Step-3 numbers are one epoch over 135M tokens: 131.9K queries × 256 tokens + 131.9K answers × 256 tokens; max global batch 0.5M tokens (1024 pairs). Data: 40% of six datasets (Dahoas/rm-static, Dahoas/full-hh-rlhf, Dahoas/synthetic-instruct-gptj-pairwise, yitingxie/rlhf-reward-datasets, openai/webgpt_comparisons, stanfordnlp/SHP), 264,292 samples.

**PPO defaults — `dschat/rlhf/ppo_trainer.py` L66–71.**
```python
self.kl_ctl = 0.1
self.clip_reward_value = 5
self.cliprange = 0.2
self.cliprange_value = 0.2
self.gamma = 1.0
self.lam = 0.95
```

**Reward shaping — `compute_rewards`, L181–194.** Per-token reward is −kl_ctl·(log π − log π_ref); the reward-model score, clipped to [−5, 5], is added only at the last generated token:
```python
kl_divergence_estimate = -self.kl_ctl * (log_probs - ref_log_probs)
rewards = kl_divergence_estimate
...
reward_clip = torch.clamp(reward_score, -self.clip_reward_value, self.clip_reward_value)
for j in range(batch_size):
    rewards[j, start:ends[j]][-1] += reward_clip[j]
```
Rewards and values after the sequence end are zeroed before GAE (`train_rlhf`, L213–220). GAE is in `get_advantages_and_returns`, L308–320 (δ = r_t + γV_{t+1} − V_t; returns = advantages + values).

**Losses.** Actor: PPO clipped surrogate, `torch.max(-A·ratio, -A·clamp(ratio, 0.8, 1.2))`, masked mean over answer tokens (`actor_loss_fn`, L282–290). Critic: clipped value loss, 0.5·max((V − R)², (V_clip − R)²) (`critic_loss_fn`, L292–305). Pretraining mix: `DeepSpeedPPOTrainerUnsupervised.train_unsupervised` backpropagates `unsup_coef * loss` as a separate actor step (L362–371).

**Generation filter (L114–115).** Generated answers with ≤ 1 non-pad token are dropped; if a whole batch is invalid, the last valid experience is reused (L60–63, L142–145). The code comment says this happens when a pretrained checkpoint is used without SFT.

**Reward model loss — `dschat/utils/model/reward_model.py` L106–116.** Pairwise loss −log σ(r_chosen − r_rejected) averaged over token positions from the first divergent token to the end of the longer sequence; the scalar used in PPO is the value at the last non-pad token (`chosen_end_scores`).

**Step-3 CLI defaults — `training/step3_rlhf_finetuning/main.py`.** `--data_split 2,4,4` (L61–63); `--unsup_coef 27.8` (L87–89, help text: "gamma in Equation 2 from InstructGPT"); `--generation_batches 1`; `--ppo_epochs 1`; `--max_prompt_seq_len 256`; `--max_answer_seq_len 256`; `--actor_learning_rate 9.65e-6`; `--critic_learning_rate 5e-6`; `--enable_ema` off by default (L319–321). EMA update `moving_average(..., beta=0.992)` in `dschat/utils/utils.py` L267.

## Recipe ledger
| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| DeepSpeed-Chat step 3 (default) | any | RL (PPO) | KL coef; reward clip; ratio clip; value clip; γ; λ | 0.1; ±5; 0.2; 0.2; 1.0; 0.95 | ppo_trainer.py L66–71 @43578c4 | verified 2026-11-24 | code comment "Those value can be changed"; no ablation reported |
| DeepSpeed-Chat step 3 | any | RL (PPO) | PPO epochs; generation batches | 1; 1 | main.py defaults; step-3 README | verified 2026-11-24 | README: >1 epoch or >1 generation batch "very unstable" |
| DeepSpeed-Chat step 3 | any | RL (PPO) | actor LR; critic LR | 9.65e-6; 5e-6 | main.py L142–150 | verified 2026-11-24 | README: script hyperparameters "not based on extensive hyperparameter tuning" |
| DeepSpeed-Chat step 3 | any | RL (PPO) | pretraining-mix coefficient | 27.8 (InstructGPT value) | main.py L87–89 | verified 2026-11-24 | README: caused instability; not tuned |
| DeepSpeed-Chat step 3 | any | RL (PPO) | prompt / answer length | 256 / 256 | main.py; BenckmarkSetting.md | verified 2026-11-24 | benchmark fixed both at 256 |
| DeepSpeed-Chat step 3 | any | RL (PPO) | EMA β | 0.992 (opt-in) | utils.py L267 | verified 2026-11-24 | paper §3 cites InstructGPT that EMA checkpoints give better quality; no own ablation |
| DeepSpeed-Chat pipeline | any | data | step 1/2/3 split | 2,4,4 | main.py L61–63 | verified 2026-11-24 | help text inconsistency, see Verification |
| OPT-13B actor + OPT-350M RM | 13B | RL (PPO) | step-3 time on 8×A100-80GB | 9 h (~$290) | paper Table 1 | verified 2026-11-24 | measured on the 135M-token benchmark setting |

## Findings relevant to generality and negative feedback
- **Forgetting control.** "Mixture Training" mixes the next-token objective into PPO "to prevent regression performance on public benchmarks like SQuAD2.0" (§3), but the maintainers report that the InstructGPT coefficient destabilized training and that they did not tune it (step-3 README). No benchmark-retention numbers are reported in the paper.
- **Evaluation gap.** The step-3 README states there is no standard metric for RLHF or SFT model quality and no metric is provided; the paper shows one conversation example only (§2.1).
- **Negative signal.** Negative feedback enters only through the scalar reward and the per-token KL penalty; there is no rejected-sample or failure-specific term in step 3.
- **Scale behaviour.** Effective throughput is highest for 6.7B–66B; at 175B it drops due to memory limits on batch size (§5.3, Fig. 6). Scaling is super-linear at small GPU counts and near- or sub-linear at larger counts (§5.3, Fig. 7).

## Connections
- [[rlhf-instructgpt]]: the three-step pipeline, EMA and pretraining-mix choices are taken from InstructGPT (§3).
- [[openrlhf-ppo]], [[trl-ppo]], [[verl-rollout]]: other PPO stacks for side-by-side code comparison.
- [[costa-huang-ppo-details]], [[kl-control-rlhf]]: per-token KL reward shaping and PPO implementation details.
- [[hh-rlhf]]: Dahoas/full-hh-rlhf in the benchmark mixture is derived from it.

## Verification
- Checked on 2026-11-24 against https://arxiv.org/abs/2308.01320 (v1 PDF, full text) and raw files at deepspeedai/DeepSpeedExamples master (DeepSpeed-Chat last commit 43578c4, 2026-09-27): `dschat/rlhf/ppo_trainer.py`, `dschat/utils/model/reward_model.py`, `dschat/utils/utils.py`, `training/step3_rlhf_finetuning/main.py`, `training/step3_rlhf_finetuning/README.md`, `training/step3_rlhf_finetuning/BenckmarkSetting.md`. Line numbers refer to that commit and will drift.
- Code inconsistency: `--data_split` help text says `2,4,4` "will use 60% of data for phase 1, 20% for phase 2 and 20% for phase 3"; the proportions 2:4:4 give 20% / 40% / 40%. Which one the code applies was not traced.
- Paper inconsistency: Table 3 lists OPT-13B as the max single-A100-80G model; §5.2 says DeepSpeed-HE runs 6.5B on a single GPU (comparison hardware not named) and 50B on one A100-40G node.
- Not reported: any model-quality evaluation (reward curves, win rates, benchmarks) in the paper.
