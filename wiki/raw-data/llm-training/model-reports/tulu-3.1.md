<!-- scope: Llama-3.1-Tulu-3.1-8B — an 8B checkpoint that changes only the final RL stage of Tülu 3, from PPO with a reward model to GRPO with verifiable rewards
     deps: [[tulu-3]], [[grpo]], [[rlvr-tulu3]]
     see-also: [[open-instruct-allenai-recipes]], [[tulu-3-sft-mix]], [[olmo-3]], [[deepseek-r1]]
-->

# Llama-3.1-Tulu-3.1-8B
- **Core Insight:** Replacing only the final RL stage of the Tülu 3 8B pipeline — PPO with a reward model swapped for GRPO with verifiable rewards and no reward model — raises the reported 11-benchmark average from 64.8 to 66.3, while the 6-task safety average falls from 85.5 to 81.2 (model card, Performance table).
- **Guideline:** When comparing an RL algorithm change, hold the SFT and preference stages fixed and re-report the full evaluation suite rather than the average alone, because in this release the average rose while one skill category fell.
- **Authors / Lab:** Allen Institute for AI (Ai2)
- **Year:** 2025 (Hugging Face model card, last updated 2025-02-10; the underlying pipeline is arXiv:2411.15124, v1 2024-11)
- **URL:** https://huggingface.co/allenai/Llama-3.1-Tulu-3.1-8B
- **Source type:** model/dataset card (official), with a reproduction command; the companion recipe documentation is open-instruct `docs/tulu3.md`
- **Relevant topics:** RLVR, PPO vs GRPO, open post-training, single-stage ablation

## Summary
`allenai/Llama-3.1-Tulu-3.1-8B` is documented by a Hugging Face model card, not by a separate paper. The card states: "The new version of our Tülu model is from an improvement only in the final RL stage of training. We switched from PPO to GRPO (no reward model) and did further hyperparameter tuning." The checkpoint is fine-tuned from `allenai/Llama-3.1-Tulu-3-8B-DPO`, so the SFT and DPO stages are those of Tülu 3. The card publishes the full GRPO hyperparameter list, a benchmark table against the earlier Tülu 3 checkpoints and external 7-9B models, and a reproduction command pinned to an open-instruct commit.

## Key Contributions
- A single-stage replacement of the RL algorithm with everything upstream held fixed, so the PPO-vs-GRPO comparison is against a shared SFT and DPO parent.
- Removal of the learned reward model from the final stage; the released Model Family table lists the 8B reward model as "None with GRPO".
- A published GRPO configuration including the KL estimator, samples per prompt, and the episode index of the released checkpoint.
- A reproduction command pinned to a specific open-instruct commit, and a single-node variant documented in open-instruct `docs/tulu3.md`.

## Key Figures/Tables to Study
- Model card Performance table (8B block): Tülu 3 SFT 8B, Tülu 3 DPO 8B, Tülu 3 8B, and Tülu 3.1 8B side by side on 11 benchmarks.
- Model card "Hyperparamters" list: the GRPO settings for the RLVR stage.
- Model card "Reproduction command": the exact open-instruct invocation.
- open-instruct `docs/tulu3.md` §"(NEW) Llama-3.1-Tulu-3.1-8B Reproduction": node and GPU layout, and the single-node rewrite.

## Technical Details
- Lineage: base `meta-llama/Llama-3.1-8B`; the 3.1 checkpoint is fine-tuned from `allenai/Llama-3.1-Tulu-3-8B-DPO` (model card, "Model description").
- RL data: `allenai/RLVR-GSM-MATH-IF-Mixed-Constraints`, 29,946 rows = GSM8k 7,473 + MATH 7,500 + IF prompts 14,973, where the IF prompts are Tulu 2 SFT mixture prompts with randomly added IFEval constraints (RLVR dataset card, "Dataset Description" and row count).
- The reward is verifiable: the command passes `--apply_verifiable_reward true` and `--reward_model_multiplier 0.0` (model card, Reproduction command).
- Reported 8B averages: Tülu 3 SFT 60.4, Tülu 3 DPO 64.4, Tülu 3 64.8, Tülu 3.1 66.3 (model card, Performance table).
- Per-benchmark deltas, Tülu 3 8B → Tülu 3.1 8B: GSM8K 87.6 → 90.0; MATH 43.7 → 47.8; IFEval 82.4 → 83.9; TruthfulQA 55.0 → 59.9; BBH 66.0 → 68.9; MMLU 68.2 → 69.5; DROP 62.6 → 63.9; PopQA 29.1 → 30.2; HumanEval 84.8 → 86.3; HumanEval+ 79.2 → 80.4; AlpacaEval 2 LC 34.5 → 34.9; Safety (6-task average) 85.5 → 81.2 (model card, Performance table).
- The card adds: "see the updated version of the paper for the latest, fixed evaluations that improve scores for models such as Qwen 2.5 Instruct" (model card, note under the 8B table). Treat the external-model columns as superseded.
- There is no standalone Tülu 3.1 paper. The card links arXiv:2411.15124, which is the Tülu 3 paper and predates this checkpoint.
- Recipe ledger is below; it fits within this card.

## Recipe ledger

| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| Llama-3.1-Tulu-3.1-8B | 8B | RL | algorithm | GRPO, no reward model | HF model card, "Version 3.1 update" | verified 2026-09-18 | no ablation reported |
| Llama-3.1-Tulu-3.1-8B | 8B | RL | starting checkpoint | `allenai/Llama-3.1-Tulu-3-8B-DPO` | HF model card, "Model description"; `--model_name_or_path` in the reproduction command | verified 2026-09-18 | not applicable |
| Llama-3.1-Tulu-3.1-8B | 8B | RL | learning rate, schedule | 5 × 10⁻⁷, constant | HF model card, "Hyperparamters"; `--learning_rate 5e-7 --lr_scheduler_type constant` | verified 2026-09-18 | the command loops over a learning-rate scan (`for learning_rate in 5e-7`); no scan results published |
| Llama-3.1-Tulu-3.1-8B | 8B | RL | KL penalty coefficient β | 0.01 | HF model card, "Hyperparamters"; `--beta 0.01` | verified 2026-09-18 | no ablation reported |
| Llama-3.1-Tulu-3.1-8B | 8B | RL | KL estimator | `kl3` | HF model card, Reproduction command `--kl_estimator kl3` | verified 2026-09-18 | no ablation reported |
| Llama-3.1-Tulu-3.1-8B | 8B | RL | clipping coefficient ε | 0.2 | HF model card, "Hyperparamters" | verified 2026-09-18 | no ablation reported |
| Llama-3.1-Tulu-3.1-8B | 8B | RL | discount factor γ | 1.0 | HF model card, "Hyperparamters" | verified 2026-09-18 | no ablation reported |
| Llama-3.1-Tulu-3.1-8B | 8B | RL | samples per prompt | 16 | HF model card, "Hyperparamters"; `--number_samples_per_prompt 16` | verified 2026-09-18 | no ablation reported |
| Llama-3.1-Tulu-3.1-8B | 8B | RL | unique prompts per training iteration | 48 | HF model card, "Hyperparamters" | verified 2026-09-18 | no ablation reported |
| Llama-3.1-Tulu-3.1-8B | 8B | RL | effective batch (samples) | 48 × 16 = 768 | HF model card, "Hyperparamters" | derived by the source (formula printed on the card) | no ablation reported |
| Llama-3.1-Tulu-3.1-8B | 8B | RL | mini-batches N_mb; PPO-style update iterations K | 2; 1 | HF model card, "Hyperparamters"; `half-m` branch and `--num_epochs 1` | verified 2026-09-18 | no ablation reported |
| Llama-3.1-Tulu-3.1-8B | 8B | RL | generation temperature | 1.0 | HF model card, "Hyperparamters"; `--temperature 1.0` | verified 2026-09-18 | no ablation reported |
| Llama-3.1-Tulu-3.1-8B | 8B | RL | max prompt length; response length; max token length | 2,048; 2,048; 2,048 | HF model card, "Hyperparamters"; `--max_prompt_token_length --response_length --max_token_length` | verified 2026-09-18 | no ablation reported |
| Llama-3.1-Tulu-3.1-8B | 8B | RL | penalty for responses with no EOS token | 0.0, with `--non_stop_penalty` and `--stop_token eos` | HF model card, "Hyperparamters" and Reproduction command | verified 2026-09-18 | no ablation reported |
| Llama-3.1-Tulu-3.1-8B | 8B | RL | gradient norm threshold; warmup ratio ω | 1.0; 0.0 | HF model card, "Hyperparamters" | verified 2026-09-18 | no ablation reported |
| Llama-3.1-Tulu-3.1-8B | 8B | RL | scheduled episodes vs released checkpoint | `--total_episodes 10,000,000` scheduled; released checkpoint is step 1920, episode 1,474,560 | HF model card, "Hyperparamters" and "Learning curves" | verified 2026-09-18 | card states step 1920 was taken as the final checkpoint; selection rule not given |
| Llama-3.1-Tulu-3.1-8B | 8B | RL | prompt set | `allenai/RLVR-GSM-MATH-IF-Mixed-Constraints`, 29,946 rows, mixing weight 1.0 | HF model card Reproduction command; RLVR dataset card row count | verified 2026-09-18 | no ablation reported |
| Llama-3.1-Tulu-3.1-8B | 8B | RL | hardware | 2 nodes, 16 GPUs; `--actor_num_gpus_per_node 4 8` (12 training GPUs), `local_rollout_batch_size=4` | open-instruct `docs/tulu3.md`, "(NEW) Llama-3.1-Tulu-3.1-8B Reproduction" | conflict | the model card prints the single-node rewrite instead (`--actor_num_gpus_per_node 6`, `local_rollout_batch_size=8`); both give 48 prompts per iteration |
| Llama-3.1-Tulu-3.1-8B | 8B | RL | code version | model card: `git checkout 3f37c29ddc97d2c108a7658692d2d2c3708ef182`; docs: experiments run at `745bf58d321c`, script `grpo_vllm_thread_ray_gtrl.py` since removed | HF model card Reproduction command; open-instruct `docs/tulu3.md` note | conflict | prefer the pinned commit in `docs/tulu3.md`, which is the locus that names the produced checkpoint |

## Findings relevant to generality and negative feedback
- The reported gains are spread across knowledge, reasoning, math, coding, and instruction following, not confined to the RLVR training skills (model card, Performance table). The RL prompt set covers GSM8k, MATH, and instruction-following constraints only, so the MMLU, PopQA, DROP, and HumanEval movements are outside the trained skills.
- The one reported decrease is the 6-task safety average, 85.5 → 81.2 (model card, Performance table). The card does not discuss this change or attribute a cause.
- Negatives enter as gradient, not as content: GRPO with a group of 16 samples per prompt assigns negative advantage to below-average samples in the group. The card publishes no split of chosen versus rejected log-probabilities, no entropy trace, and no pass@k, so the effect of the negative term is not measurable from this source.
- No reward model is used, so reward-model over-optimization is not the failure mode here; the verifier's false-negative rate is not reported.

## Connections
- [[tulu-3]] — the report for the SFT and DPO stages this checkpoint inherits unchanged.
- [[tulu-3-sft-mix]] — the SFT mixture underneath the parent DPO checkpoint.
- [[rlvr-tulu3]] — the RLVR formulation and the PPO-based final stage this release replaces.
- [[grpo]] — the algorithm introduced for the final stage.
- [[open-instruct-allenai-recipes]] — the framework card holding the run scripts.
- [[tulu-3-1]] — an older library card on the same release; this card is the maintained one.
- [[olmo-3]], [[deepseek-r1]] — later open releases that also use verifier-based RL.

## Verification
- Checked on 2026-09-18 against: https://huggingface.co/allenai/Llama-3.1-Tulu-3.1-8B (card as served 2026-09-18); https://github.com/allenai/open-instruct `docs/tulu3.md`; https://huggingface.co/datasets/allenai/RLVR-GSM-MATH-IF-Mixed-Constraints
- Corrections to the previous card version:
  - Title "Tulu 3.1" → "Llama-3.1-Tulu-3.1-8B", the name the artifact is published under. The file name is unchanged because other pages link `[[tulu-3.1]]`.
  - "reported broad performance gains" and "move an 8B instruct model noticeably" → the reported 11-benchmark average moves from 64.8 to 66.3, and the safety average falls from 85.5 to 81.2 (model card, Performance table).
  - The previous card had no recipe ledger and no numbers. The full GRPO configuration, the episode index of the released checkpoint, the prompt-set size, and the hardware layout are now recorded with loci.
  - "There is no standalone Tulu 3.1 paper in the sources I used" → rewritten without first person; the arXiv link on the card is the Tülu 3 paper, which predates this checkpoint.
  - Missing **Source type** field added.
- Removed as unsupported by the source:
  - "It also shows that the open ecosystem quickly incorporated the GRPO / verifier-style RL trend after DeepSeek-R1 and DeepSeekMath made it prominent." Not stated by the model card.
  - "Tulu 3.1 is useful because it is a controlled public ablation rather than a broad marketing release", "a rare public example", "a clean public artifact". Framing claims with no source and no measurement.
- Duplicate note: [[tulu-3-1]] describes the same release and is not maintained. It states that Tülu 3.1 is a multi-base refresh across Llama 3.1 and OLMo 2 at 8B, 70B, and 405B; the model card names a single 8B checkpoint fine-tuned from `allenai/Llama-3.1-Tulu-3-8B-DPO`, and the released Model Family table lists `allenai/Llama-3.1-Tulu-3-70B` and `allenai/llama-3.1-Tulu-3-405B` unchanged at version 3. Cite this card, not [[tulu-3-1]].
- Not reported by the source: the checkpoint-selection rule behind step 1920; the reward-model-based PPO settings of the Tülu 3 8B baseline in comparable form; wall-clock or GPU-hour cost; any explanation of the safety decrease.
