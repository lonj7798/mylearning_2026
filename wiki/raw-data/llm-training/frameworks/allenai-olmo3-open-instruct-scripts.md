<!-- scope: allenai/open-instruct scripts/train/olmo3 at commit d8a7f1c: README run table and the 17 launch scripts it lists for Olmo 3 SFT (OLMo-core), DPO (dpo_tune_cache.py) and RL (grpo_fast.py), plus the framework defaults those scripts depend on
     deps: [[olmo-3]], [[dpo]], [[grpo]]
     see-also: [[allenai-olmo3-open-instruct-scripts-recipe]], [[olmo-core-olmo3-configs]], [[open-instruct-allenai-recipes]], [[dapo]], [[trl-grpo]]
-->

# allenai/open-instruct `scripts/train/olmo3/` — Olmo 3 training scripts (README and launch scripts, commit d8a7f1c)
- **Core Insight:** The README maps 17 Olmo 3 post-training runs (7B/32B Think and Instruct SFT, DPO and RL; a no-pipeline variant of 7B Think RL; four 7B RL-Zero runs) to a launch script, Beaker run, W&B run and commit (README L7–25); every DPO run uses length-normalized DPO with β = 5 and learning rates from 7e-8 to 1e-6, and every RL run uses `grpo_fast.py` with β = 0.0 (no KL term), 8 samples per prompt and temperature 1.0.
- **Guideline:** When reproducing an Olmo 3 DPO or RL run from these scripts, check framework defaults at the commit named in the README row, because PR #1547 (commit 7917d41, 2026-03-20) deleted flags such as `--beta 5`, `--loss_type dpo_norm` and `--clip_higher 0.272` from the scripts and moved those values into defaults, and several script values differ from paper Table 49 (for example 32B Think RL: 64 unique prompts per step in the script, 128 in Table 49).
- **Authors:** Allen Institute for AI (allenai/open-instruct maintainers); last commit touching the directory by Finbarr Timbers.
- **Year:** 2025 (scripts added 2025-11; README run table added 2026-01-24, ff4a925; directory last changed 2026-05-10, d8a7f1c)
- **URL:** https://github.com/allenai/open-instruct/tree/d8a7f1ce97087795c19a170691b4de15a7c228a6/scripts/train/olmo3
- **Source type:** released config/code
- **Relevant topics:** Olmo 3 post-training, DPO launch settings, length-normalized DPO, GRPO/RLVR launch settings, active sampling, zero-variance filtering, truncated completions, LLM-judge rewards, config drift between scripts, defaults and paper

## Summary
The README states that Olmo 3 SFT used the OLMo-core SFT implementation "for better GPU efficiency" while DPO and RL used open-instruct (README L5). Its table lists, per run, the script, a Beaker link described as "the exact run that produced the model", a W&B URL and a commit (README L7–25, L34). SFT scripts call OLMo-core `src/scripts/train/sft/OLMo-sft.py`; DPO scripts call `open_instruct/dpo_tune_cache.py`; RL scripts call `open_instruct/grpo_fast.py`. External users are told to remove the `mason.py` wrapper and set up their own Ray cluster (README L36–37). The directory also holds scripts that the README table does not list, for example `7b_instruct_dpo_olmocore.sh` and `7b_rlzero_mix.sh`; this card does not cover them. All settings by run are in [[allenai-olmo3-open-instruct-scripts-recipe]].

## Key Contributions
- A run table linking each released Olmo 3 stage to code, logs and commit (README L9–25), plus five W&B stage reports (README L45–49).
- Initialization chain readable from checkpoint paths: 7B Think SFT starts from a checkpoint under `checkpoints/tylerr/long-context/` at step 11921 (`7b_think_sft.sh` L7); Instruct SFT starts from a checkpoint named `olmo3-7b-reasoning-sft-final` (`7b_instruct_sft.sh` L6); 32B Think DPO and RL start from `olmo3-merge-32b-1e-4-5e-5` (`32b_think_dpo.sh` L3, L10; `32b_think_rl.sh` L7).
- DPO data mixes with explicit per-dataset counts, including multi-turn preference sets (`7b_instruct_dpo.sh` L32–41).
- RL data mixes with counts and per-domain verifiers: math, code (unit-test API), instruction following, and general prompts scored by a Qwen3-32B judge (`32b_think_rl.sh` L6, L65–69).
- A no-pipeline 7B Think RL script whose comment states the released model "actually doesnt use pipelinerl" (`7b_think_rl_no_pipeline.sh` L4).

## Key Figures/Tables to Study (loci at d8a7f1c unless noted)
- `README.md` L7–25 (run table), L27–37 (reproduction instructions).
- `7b_instruct_dpo.sh` L27–51; `32b_think_dpo.sh` L5–41; pre-#1547 versions at 00d42c4 (`7b_instruct_dpo.sh` L48–54).
- `7b_instruct_rl.sh` L40–85; `32b_think_rl.sh` L25–78; `7b_rlzero_math.sh` L35–77.
- Framework code the scripts rely on: `open_instruct/dpo_utils.py` L76–105, L523–561; `open_instruct/grpo_utils.py` L96–139, L400–415, L470–481; `open_instruct/data_loader.py` L405–480, L800–815, L1070–1095, L1126–1132, L1410–1422.

## Technical Details
- **Mixer syntax.** In `dataset_mixer_list`/`mixer_list`, a value containing "." is a fraction of the dataset and an integer is a sample count; a count above the dataset size upsamples by repetition (`dataset_transformation.py` L2066–2091).
- **DPO loss.** `dpo_norm` averages token log-probabilities per response (`is_average_loss`, `dpo_utils.py` L83–84; `dpo_tune_cache.py` L525; `dpo_utils.py` L709–710) and then applies the DPO loss (L556–559):
  `L = −log σ(β · [(log π_θ(y_c|x) − log π_θ(y_r|x)) − (log π_ref(y_c|x) − log π_ref(y_r|x))])`,
  where each log π is the mean token log-probability of the response, y_c is the chosen response, y_r the rejected response, π_ref the frozen initial policy, and β the scale (default 5.0, L99). Label smoothing defaults to 0.0 (L105). Reference log-probabilities are read from a cache (L654–657).
- **DPO values.** 7B Think: 150,000 pairs (`--max_train_samples`, L40), LR 8e-8, linear schedule, max length 16,384 (L26–30). 7B Instruct: mixer counts sum to 260,000 (derived from L32–41), LR 1e-6, 16,384 tokens (L4, L42). 32B Think: 200,000 pairs, LR loop over 7e-8, 8e-8 and 9e-8, 8,192 tokens (L5, L36, L39). 32B Instruct: 260,000 (derived from L11–21), LR 1e-6, 8,192 tokens (L9, L45).
- **RL loss (code at d8a7f1c).** Default `loss_fn` is `dapo`: per-token loss max(−A·r, −A·clip(r, 1 − 0.2, 1 + 0.272)) with r the policy/old-policy ratio (`grpo_utils.py` L102, L139, L472–474). A train/inference ratio ρ is clamped at 2.0 and multiplies the loss (`use_rho_correction` True, L104–113, L400–415).
- **Advantages.** A = r_i − mean(r) over the 8 samples of a prompt when `advantage_normalization_type="centered"` (default at d8a7f1c), or divided by (std + 1e-8) when `"standard"` (`data_loader.py` L421, L1417–1420). Example: verifiable reward 10.0 (L452) for 1 of 8 samples gives A = 8.75 and −1.25 with `centered`; `standard` gives 2.646 and −0.378 (std 3.307).
- **Filtering.** Groups with zero reward std are dropped (`filter_zero_std_samples` True, L419, L814). With `--active_sampling`, dropped groups do not count toward the batch, so more prompts are drawn until it is full (L1076–1079). A prompt whose mean score is at least `no_resampling_pass_rate` of the maximum is excluded from later sampling (L809–812); 0.875 corresponds to 7 of 8 correct samples for a binary verifier (derived).
- **Truncation and code reward.** `mask_truncated_completions True` removes rollouts whose finish reason is not "stop" (L1126–1132). The code verifier returns the unit-test pass rate, or 0.0 if the pass rate is below `code_pass_rate_reward_threshold` (`ground_truth_utils.py` L966–967): 0.99 in Think/Instruct scripts, 0.0 in `7b_rlzero_code.sh` L59.
- **Judge.** General prompts are scored by `hosted_vllm/Qwen/Qwen3-32B`, judge max tokens 2,048, context 32,768, timeout 600 s (`7b_instruct_rl.sh` L75–78).
- **Script defects observed.** `7b_instruct_rl.sh` defines `nonreasoner_integration_mix_decon` (L5) but passes `${nonreasoner_math_mix_decon}` (L50), uses `${gs_model_name}` in `exp_name` before assigning it (L20–21), and ends with a line continuation after `--no_resampling_pass_rate 0.875` (L85). Only the four SFT scripts, `32b_instruct_dpo.sh` and `32b_instruct_rl.sh` carry Beaker/W&B/commit header comments; for the other runs the README row is the only link.
- **PR #1547.** Commit 7917d41 changed defaults to DPO β 5.0, `dpo_norm`, 1 epoch, warmup ratio 0.1, and GRPO `clip_higher` 0.272, `truncated_importance_sampling_ratio_cap` 2.0, `async_steps` 8, `inflight_updates` True, `advantage_normalization_type` "centered", then removed those flags from the olmo3 scripts (commit diff of `dpo_utils.py`, `grpo_utils.py`, `data_loader.py` and 15 files in `scripts/train/olmo3/`). Its CHANGELOG line lists different values (`clip_higher=0.28`, `truncated_importance_sampling_ratio_cap=10.0`, `advantage_normalization_type=mean_std`).

## Recipe ledger
Per-run DPO, SFT and RL settings, with conflicts against paper Tables 47–49, are in [[allenai-olmo3-open-instruct-scripts-recipe]].

## Findings relevant to negative feedback (code behavior; no experiments in the source)
Meanings used (course §6.1): negative as gradient (DPO rejected term; A < 0) and negative marginal value (filtered prompts).
- **DPO.** The rejected response enters as a gradient term, normalized by its length under `dpo_norm` (`dpo_utils.py` L83–84, L556–559). The mix names contain "deltas" and "maxdelta_reje" (`7b_instruct_dpo.sh` L32–33); the scripts do not describe how rejected responses were produced.
- **RL.** With centered advantages, every below-group-mean sample receives A < 0 scaled by the raw reward gap, not by the group std (`data_loader.py` L1420). All-correct and all-wrong groups give no gradient and are removed before batching (L814). Prompts solved at ≥ 0.875 are excluded from later sampling in `7b_instruct_rl.sh` and `7b_rlzero_math.sh` only (L85; L38).
- **Truncated completions** stay in the batch and receive the verifier score (`mask_truncated_completions False`) in the Think, Instruct and RL-Zero math scripts, and are removed in RL-Zero code, IF and general (`7b_rlzero_code.sh` L70; `7b_rlzero_instruction_following.sh` L69; `7b_rlzero_general.sh` L77). `non_stop_penalty` is False in all RL scripts.

## Findings relevant to generality (evaluation lists in the scripts)
- Think and Instruct DPO/RL scripts launch multi-domain evaluation lists; `7b_instruct_dpo.sh` L57 lists OMEGA-500, MATH-500, LiveCodeBench, AIME 2024/2025 pass@32, ZebraLogic, BBH, PopQA, MBPP+, MMLU, GPQA, HumanEval+, AlpacaEval v3, IFEval and AGIEval; `7b_think_rl.sh` L68 adds SimpleQA, GSM8K and Minerva MATH.
- RL-Zero scripts evaluate only the trained domain: AIME for math (`7b_rlzero_math.sh` L10), HumanEval+/MBPP+/LiveCodeBench for code (`7b_rlzero_code.sh` L9), IFEval for IF (L9), AlpacaEval/AGIEval/GPQA for general (`7b_rlzero_general.sh` L11).

## Connections
- [[olmo-3]] — the report these runs produced; Tables 47–49 give the paper-side hyperparameters.
- [[olmo-core-olmo3-configs]] — base-model pretrain, midtrain and long-context scripts that produce the SFT starting checkpoints.
- [[open-instruct-allenai-recipes]] — earlier open-instruct reproduction commands for Tülu 3 and OLMo 2.
- [[tulu-3]], [[rlvr-tulu3]] — origin of the length-normalized DPO loss and RLVR setup reused here.
- [[dpo]], [[simpo]] — base DPO objective and the other length-normalized option in `DPOLossType`.
- [[grpo]], [[dapo]], [[dr-grpo]] — group advantages, asymmetric clipping (0.272 here) and std-free advantages.
- [[rollout-training-mismatch-tis]], [[async-rollout]] — the ρ clamp and `async_steps` settings.
- [[trl-grpo]], [[verl-dapo-recipe]] — other frameworks' defaults for the same knobs.

## Verification
- Created on 2026-09-14 from https://github.com/allenai/open-instruct/tree/d8a7f1ce97087795c19a170691b4de15a7c228a6/scripts/train/olmo3 (README and the 17 listed scripts read; d8a7f1c is the last commit touching the directory as of 2026-09-14, and README plus three scripts fetched from `main` were byte-identical); pre-#1547 scripts read at 00d42c4; framework files read at d8a7f1c; cross-check against arXiv:2512.13961v2 App. A.6 Tables 47–49.
- Audit claims not found in the source: "active_sampling with no_resampling_pass_rate 0.875 in the Instruct and RL-Zero scripts" (only `7b_instruct_rl.sh` and `7b_rlzero_math.sh` set 0.875; RL-Zero code/IF/general and 7B Think RL do not enable active sampling); "mask_truncated_completions False" as a general RL setting (True in RL-Zero code/IF/general); "code_pass_rate_reward_threshold 0.99" as a general setting (0.0 in RL-Zero code); "32 prompts per rollout for RL-Zero math" is correct but all four RL-Zero scripts use 32; "LR 2e-6 for 32B Think" is correct but 32B Instruct RL also uses 2e-6 with response length 16,384; "DPO beta is not set explicitly" is true at d8a7f1c only (before #1547 every DPO script set β = 5).
- Not checked: Beaker and W&B pages (the README calls Beaker the exact run); scripts at the per-run commits in the README table.
