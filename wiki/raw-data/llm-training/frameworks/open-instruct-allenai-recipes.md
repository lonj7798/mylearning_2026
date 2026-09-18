<!-- scope: allenai/open-instruct reproduction docs and launch files for Tülu 3 (SFT, DPO, RM, RLVR-PPO), Tülu 3.1 (GRPO), OLMo 2 post-training, and the OLMo 3 script index, pinned at commit 098424c
     deps: [[tulu-3]], [[olmo-2]]
     see-also: [[open-instruct-allenai-recipes-recipe]], [[allenai-olmo3-open-instruct-scripts]], [[olmo-2-official-configs]], [[tulu-3.1]], [[rlvr-tulu3]]
-->

# allenai/open-instruct "Tulu3 Reproduction" (`docs/tulu3.md`), with `docs/olmo2.md`, `docs/olmo3.md` and the Tülu 3 / OLMo 2 launch files (commit 098424c)
- **Core Insight:** `docs/tulu3.md` gives what Ai2 calls "(almost) the exact command" for each Tülu 3 checkpoint: SFT at LR 5e-6 (8B) and 2e-6 (70B) for 2 epochs at effective batch 64 × 1 × 2 = 128, length-normalized DPO (`dpo_norm`, β = 5) at LR 5e-7 (8B) and 2e-7 (70B, 405B) for 1 epoch, and 8B RLVR-PPO at LR 3e-7 and β = 0.05 (L39–43, L65, L144–159, L198, L249, L328, L338).
- **Guideline:** When re-running these SFT commands at the pinned commit, check the loss reduction first, because the June 2025 version of the same commands passed `--reduce_loss sum` (tulu3.md@8781471 L49, L114), PR #1024 removed the option and made mean loss the default (commit bb98dbc message), and the Tülu 3 paper reports that a sum loss with LR 5e-6 worked best in its batch-aggregation experiments (arXiv:2411.15124v5 §4.3.2).
- **Authors:** Allen Institute for AI (Ai2), open-instruct maintainers (organization)
- **Year:** 2024 (`docs/tulu3.md` added 2024-11-20; Tülu 3.1 section added 2025-02-12, commit c8b66a0; pinned commit 098424c, 2026-09-14)
- **URL:** https://github.com/allenai/open-instruct/blob/098424c479f44f153f93e9c5c0f8c4c97fc854ac/docs/tulu3.md
- **Source type:** released config/code
- **Relevant topics:** Tülu 3 SFT, length-normalized DPO, reward model, RLVR with PPO, GRPO, KL estimators, effective batch size, OLMo 2 post-training, OLMo 3 chat templates, reproduction drift

## Summary
The artifact is a documentation file (`docs/tulu3.md`, 625 lines) that lists launch commands for Llama-3.1-Tulu-3 SFT (8B, 70B), DPO (8B, 70B, 405B), the 8B reward model, RLVR (8B, 70B, 405B) and Llama-3.1-Tulu-3.1-8B (GRPO). Each block links the released Hugging Face checkpoint and an internal Beaker experiment. Companion files in the same repository were read: `docs/olmo2.md` (OLMo 2 1B SFT, DPO, two RLVR stages and stage timings), `docs/olmo3.md` (OLMo 3 chat-template settings), `configs/train_configs/{tulu3,olmo2}/*.yaml`, `scripts/train/olmo2/*.sh`, `scripts/train/tulu3/grpo_fast_8b.sh`, and `scripts/train/olmo3/README.md`. The repository reports no experiments of its own; ablations for these values are in the Tülu 3 and OLMo 2 papers ([[tulu-3]], [[olmo-2]]). The recipe ledger, with cross-checks against those papers and the Tülu 3.1 model card, is in [[open-instruct-allenai-recipes-recipe]].

## Key Contributions
- One command per released Tülu 3 checkpoint, with seeds, data mixture names, DeepSpeed stage and GPU layout (tulu3.md L8–465).
- Written rule for keeping effective batch size when GPU count changes: processes × per-device batch × gradient accumulation, 64 × 1 × 2 = 128 or 8 × 1 × 16 = 128 (L57–68).
- The Tülu 3.1 GRPO command with its batch arithmetic and a single-node variant (L468–625).
- OLMo 2 1B post-training commands and stage wall-clock times (olmo2.md L10–13, L15–426).
- An index of OLMo 3 launch scripts with the commit and Beaker run of each released stage (scripts/train/olmo3/README.md L7–25).

## Key Figures/Tables to Study (loci at 098424c)
- `docs/tulu3.md` L57–68 (effective batch rule), L270–276 (IF verifier note), L306–348 (8B RLVR), L468–547 (Tülu 3.1 GRPO), L549 (single-node batch scaling).
- `docs/olmo2.md` L10–13 (stage times), L232–281 (1B RLVR "modern reproduction" with `grpo_fast.py`).
- `configs/train_configs/olmo2/olmo2_1124_7b_sft.yaml` L9–15; `scripts/train/olmo2/finetune_7b.sh` L24–32; `dpo_7b.sh` L24–35.
- `docs/olmo3.md` L20–45 (think/instruct chat template and identity system prompt settings).

## Technical Details
- **SFT (Tülu 3).** 8B: base `meta-llama/Llama-3.1-8B`, `allenai/tulu-3-sft-mixture`, max_seq_length 4096, LR 5e-06 linear, warmup ratio 0.03, weight decay 0.0, 2 epochs, bf16, DeepSpeed ZeRO-3, seed 123 (L32–53); 8 nodes × 8 H100 = 64 GPUs (L10). 70B: same training settings except LR 2e-06, gradient checkpointing and seed 456 (L102–118). The Tülu 3 paper states the final 8B SFT ran on 32 GPUs for 6 hours (arXiv:2411.15124v5 §4.3), not 64.
- **Loss reduction drift.** At commit 8781471 (2025-06-09) both SFT commands contained `--reduce_loss sum` and `--use_flash_attn` (tulu3.md@8781471 L35, L49, L100, L114). At 098424c these flags are absent. `open_instruct/finetune.py` at 098424c backpropagates `outputs.loss` (L835–856). The same flag was removed from `scripts/train/olmo2/finetune_{7b,13b,32b}.sh` (present at 686a521, L33) and from `olmo2_1124_{7b,13b}_sft.yaml` (present at 5bcd3b2, L22).
- **DPO.** Loss `dpo_norm`, β = 5, 1 epoch, linear schedule, warmup 0.1, max_seq_length 2048 for 8B, 70B and 405B (L140–159, L194–213, L245–264). 8B: 8 GPUs × 1 × 16 (L134–143); 70B: 64 × 1 × 2 with ZeRO-3 offloading and `dpo_tune_cache.py` (L180–191); 405B: 256 × 1 × 1, tokenizer `Llama-3.1-Tulu-3-70B-SFT` (L235–248). The Tülu 3 paper names this variant "DPO-norm", the length-normalized objective in which each log-ratio is multiplied by β/|y| (arXiv:2411.15124v5 §5.1.2 Eq. 6; §5.4.1 Table 18 caption).
- **Reward model (8B).** `reward_modeling.py`, `allenai/llama-3.1-tulu-3-8b-preference-mixture`, LR 3e-6, per-device batch 1, gradient accumulation 32, max 2048 tokens, 1 epoch; evaluated on `ultrafeedback_binarized_cleaned` test_prefs (L285–298). The doc does not give the GPU count; the Tülu 3 paper states the final 8B RM trained in 9 hours on 8 H100 (§6.3).
- **RLVR, 8B (PPO).** Legacy `ppo_vllm_thread_ray_gtrl.py`, removed in PR #1132 (L310). Starts from Tülu-3-8B-DPO; `RLVR-GSM-MATH-IF-Mixed-Constraints`; prompt, response and total token length 2048; temperature 1.0; LR 3e-7; β 0.05; `total_episodes` 10,000,000; responses without EOS receive −10.0; 7 actor GPUs × `local_mini_batch_size` 32 = 224 (derived; paper Table 21 gives 224); `reward_model_path` Tülu-3-8B-RM with `reward_model_multiplier 0.0`; seed 3; save every 100 steps (L313–346).
- **RLVR, 70B and 405B.** 70B: β 0.07, warmup 0.1, LR 1e-7, 400,000 episodes, 5 × 8 actor GPUs × mini-batch 16 = 640 (L357–400). 405B: `RLVR-MATH` only, response length 1024, β 0.05, LR 1e-7, 400,000 episodes, `num_epochs 4`, 29 × 8 actor GPUs × mini-batch 8, vLLM tensor parallel 16 (L425–457). The 405B note at L411 repeats the 70B arithmetic "40 * 16 = 640", which does not match the 405B command (29 × 8 × 8 = 1,856, derived).
- **Tülu 3.1 (GRPO).** Legacy `grpo_vllm_thread_ray_gtrl.py` run at commit 745bf58d321c (L470); 2 nodes, 16 GPUs (L472). LR 5e-7 constant, β 0.01, 16 samples per prompt, `kl_estimator kl3`, `local_rollout_batch_size` 4, mini-batch = 4 × 16 / 2 = 32 ("half-m"), actor GPUs 4 + 8, ZeRO-2, vLLM TP 4, no-EOS penalty 0.0, `total_episodes` 10,000,000 (L476–535). Rollout batch 12 × 4 = 48 prompts; one node uses `local_rollout_batch_size` 8 with 6 actor GPUs (L549, L557–609).
- **Later launch file for Tülu 3.1.** `scripts/train/tulu3/grpo_fast_8b.sh` uses `grpo_fast.py` with 48 unique prompts × 16 samples, β 0.01, `kl_estimator 2`, LR 5e-7 constant, 2,000,000 episodes, `num_mini_batches 2`, `pack_length 4096` (L11–40). PR #1242 (2025-12-01) changed `kl_estimator` from `"kl1"…"kl4"` (default `"kl3"`) to integers 0–3 (default 2) and rewrote `kl3` as `2` in scripts (commit c4c1b1c diff). At 098424c index 2 is `expm1(−d) + d` and index 3 is `ratio · d` (model_utils.py L803–818).
- **OLMo 2 (docs/olmo2.md, 1B).** Flags `--add_bos` and `--use_slow_tokenizer False` are required for OLMo 1 and 2 (L6). Stage times: SFT 9 h on 8 H100; DPO 2 h; RLVR 43 h per stage for 2 million episodes on 16 GPUs, about 86 h total (L10–13). 1B SFT: `tulu-3-sft-olmo-2-mixture-0225`, LR 3e-5, 8 GPUs × 2 × 8, 2 epochs (L45–53). 1B DPO: LR 2.5e-6, 8 × 8 × 2, `dpo_norm` β 5 (L105–131). 1B RLVR: two sequential GRPO stages (mixed GSM/MATH/IF, then MATH), β 0.01, LR 5e-7, 48 prompts × 16 samples, 2,000,000 episodes each (L236–377). The original commands' experiment names say `lr_7e-7` and `lr_9e-7` while their `--learning_rate` flag is 5e-7 (L286–292, L381–387). The "modern reproduction" passes `--kl_estimator 3` (L253) where the original used `kl3` (L291); with the post-#1242 indexing, 3 is not the former `kl3`.
- **OLMo 2 7B/13B/32B launch files.** The YAML configs and the shell scripts disagree on SFT and DPO learning rates (7B SFT 1.0e-05 vs 2e-5; 13B SFT 6e-06 vs 5e-06; 7B DPO 5.0e-7 vs 1e-6; 13B DPO 5.0e-7 vs 8e-7). The scripts match the values the OLMo 2 paper marks as final (arXiv:2501.00656v3 §5). `olmo2_1124_32b_dpo.yaml` loads `OLMo-2-1124-13B-SFT` (L1), and `olmo2_1124_7b_dpo.yaml` uses the 13B preference mix (L4). Row-level values are in [[open-instruct-allenai-recipes-recipe]].
- **OLMo 3.** SFT used OLMo-core; DPO and RL used open-instruct (scripts/train/olmo3/README.md L5). The README gives the run commit for each of 17 runs (L9–25); `scripts/train/olmo3/` holds 21 `.sh` files at 098424c. PR #1547 (2026-03-20) changed DPO/GRPO defaults "to match olmo3 experiments" and removed flags from these scripts, e.g. `--warmup_ratio 0.1`, `--num_epochs 1`, `--loss_type dpo_norm` and `--beta 5` from `7b_think_dpo.sh` (diff 0047e94 → 7917d41). Values are recorded in [[allenai-olmo3-open-instruct-scripts]].

## Recipe ledger
See [[open-instruct-allenai-recipes-recipe]] (Tülu 3 8B/70B/405B, Tülu 3.1 8B, OLMo 2 1B/7B/13B/32B post-training rows, with paper and model-card conflicts).

## Findings relevant to negative feedback (launch settings; no experiments in the source)
- **No-EOS penalty.** Tülu 3 PPO assigns reward −10.0 to responses that do not stop (`--non_stop_penalty`, `--penalty_reward_value -10.0`, L324, L330, L379, L385). Tülu 3.1 GRPO and OLMo 2 1B GRPO set the value to 0.0 (L523; olmo2.md L260–261).
- **Negative as gradient in DPO.** `dpo_norm` lowers the length-normalized log-ratio of the rejected response (meaning 4 in the course standard); the doc gives no rejected-response logging or anchor term.
- **Verifier drift for IF prompts.** The doc states that the RLVR IF verifier and judge were updated and that reproducing Tülu 3 requires `IFEvalVerifierOld`; the new verifier is not compatible with the old data format (L272–276).

## Findings relevant to generality and chat templates (as reported in docs/olmo3.md)
- OLMo 3 7B Think DPO and RL used a chat template slightly different from its SFT template because of "a minor miscommunication" (L22). The 32B Think training template omits the identity prompt; the stated reason is that an identity-free training prompt is easier to fix with a system prompt at demo time, and the 7B Think model could not be retrained (L23–24).
- For OLMo 3.2+ models, Think SFT data is tokenized with the instruct template as a workaround for a bug that masks the first `<think>` token as prompt (L26–28, L39).

## Connections
- [[tulu-3]], [[rlvr-tulu3]] — paper tables that record the ablations and the final values these commands implement.
- [[tulu-3.1]], [[tulu-3-1]] — model-card view of the Tülu 3.1 GRPO run.
- [[olmo-2]], [[olmo-2-official-configs]] — OLMo 2 base-model training; the post-training rows here start from those checkpoints.
- [[allenai-olmo3-open-instruct-scripts]], [[olmo-core-olmo3-configs]], [[olmo-3]], [[ai2-dolci-think-sft-card]] — OLMo 3 stage scripts, configs and data.
- [[dpo]], [[simpo]] — standard DPO and the length-normalized variant family; [[grpo]], [[ppo]], [[john-schulman-kl-tricks]] — RL objective and KL estimators.
- [[trl-grpo]], [[openrlhf-dpo]] — other frameworks' defaults for the same stages; [[tulu-3-sft-mix]], [[allenai-tulu-sft-recipe]] — SFT data.

## Verification
- Created on 2026-09-14 from https://github.com/allenai/open-instruct/blob/098424c479f44f153f93e9c5c0f8c4c97fc854ac/docs/tulu3.md (commit 098424c; file last changed in f73ede5, 2026-05-14). Also read at 098424c: `docs/olmo2.md`, `docs/olmo3.md`, `configs/train_configs/{tulu3,olmo2}/*.yaml`, `scripts/train/olmo2/*.sh`, `scripts/train/tulu3/grpo_fast_8b.sh`, `scripts/train/olmo3/{README.md,7b_think_sft.sh,7b_think_dpo.sh,7b_think_rl.sh,7b_rlzero_math.sh}`, `open_instruct/finetune.py`, `open_instruct/model_utils.py`; earlier versions at 8781471, 686a521, 5bcd3b2, 5e5c93d, 7917d41 (via diff); cross-checks against arXiv:2411.15124v5, arXiv:2501.00656v3 and the Llama-3.1-Tulu-3.1-8B model card.
- Audit claims not found in the source: "22 scripts" in `scripts/train/olmo3/` → 21 `.sh` files plus `README.md`. The lead values "olmo2_1124_{7b,13b}_sft.yaml LR 1e-5 / 6e-6" and "_dpo.yaml LR 5e-7" are present in the YAMLs but conflict with the shell scripts and the OLMo 2 paper (see ledger).
- Not reported by the source: ablations for any value; GPU count for the reward model; which checkpoint step of the 8B, 70B and 405B RLVR runs was released.
