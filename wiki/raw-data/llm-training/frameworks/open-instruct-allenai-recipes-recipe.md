<!-- scope: recipe ledger for [[open-instruct-allenai-recipes]]: Tülu 3, Tülu 3.1 and OLMo 2 post-training launch values in allenai/open-instruct at commit 098424c, with paper and model-card cross-checks
     deps: [[open-instruct-allenai-recipes]]
     see-also: [[tulu-3]], [[tulu-3.1]], [[olmo-2]], [[olmo-2-official-configs]], [[allenai-olmo3-open-instruct-scripts]]
-->

# open-instruct Tülu 3 / Tülu 3.1 / OLMo 2 post-training recipe ledger (commit 098424c)
- **Source:** https://github.com/allenai/open-instruct/tree/098424c479f44f153f93e9c5c0f8c4c97fc854ac (`docs/tulu3.md` = "t3", `docs/olmo2.md` = "o2", `configs/train_configs/olmo2/*.yaml`, `scripts/train/{olmo2,tulu3}/*.sh`)
- **Source type:** released config/code, cross-checked against arXiv:2411.15124v5 (Tülu 3), arXiv:2501.00656v3 (OLMo 2) and the Llama-3.1-Tulu-3.1-8B model card (huggingface.co/allenai/Llama-3.1-Tulu-3.1-8B README)
- **Scope and units:** batch = sequences (SFT, RM), preference pairs (DPO), or prompts × samples (RL). "Episodes" = sampled responses, as the Tülu 3.1 card uses them (effective batch 48 × 16 = 768; step 1920 = episode 1,474,560). `total_episodes` is the planned horizon passed on the command line, not the released checkpoint's episode count. Derived products use the flags named in the locus. Status date for all `verified` rows: 2026-09-14.

## Recipe ledger

| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| Llama-3.1-Tulu-3-8B-SFT | 8B | SFT | peak LR, schedule, warmup ratio, weight decay | 5e-06, linear, 0.03, 0.0 | t3 L39–42 | verified | paper Table 11 (5 × 10⁻⁶, linear, 0.03); §4.3.2: sum loss with 5e-6 best in Llama 3.0 / Tülu 2 mix sweep (Figs. 5–6) |
| Llama-3.1-Tulu-3-8B-SFT | 8B | SFT | epochs; max length | 2; 4,096 tokens | t3 L35, L43 | verified | §4.3.2: "training for longer did not yield further improvements" |
| Llama-3.1-Tulu-3-8B-SFT | 8B | SFT | effective batch (sequences) | 128 = 64 GPUs × 1 × 2 | t3 L18–21, L65 | verified | paper Table 11: 128; no ablation reported |
| Llama-3.1-Tulu-3-8B-SFT | 8B | SFT | GPUs | 64 H100 (8 × 8) | t3 L10 | conflict | paper §4.3: final 8B trained on 32 GPUs for 6 hours |
| Llama-3.1-Tulu-3-8B-SFT | 8B | SFT | loss reduction | `--reduce_loss sum` at 8781471 (L49); flag absent at 098424c, default mean since PR #1024 | t3@8781471 L49; commit bb98dbc | conflict | §4.3.2: sum loss chosen over mean; the pinned command runs mean |
| Llama-3.1-Tulu-3-8B-SFT | 8B | SFT | data; seed | `allenai/tulu-3-sft-mixture` 1.0; 123 | t3 L49, L53 | verified | no ablation reported |
| Llama-3.1-Tulu-3-70B-SFT | 70B | SFT | peak LR; epochs; batch; length; warmup | 2e-06; 2; 64 × 1 × 2 = 128; 4,096; 0.03 | t3 L81–84, L98–106 | verified | paper Table 11 (2 × 10⁻⁶, 128, 2 epochs) |
| Llama-3.1-Tulu-3-70B-SFT | 70B | SFT | loss reduction; seed | `--reduce_loss sum` at 8781471 (L114), absent at 098424c; 456 | t3@8781471 L114; t3 L118 | conflict | same as 8B row |
| Llama-3.1-Tulu-3-8B-DPO | 8B | preference | loss; β | `dpo_norm` (length-normalized DPO); 5 | t3 L158–159 | verified | paper §5.4.1 Table 18: only DPO-norm beat the base checkpoint among DPO, SimPO, DPO-norm; Table 20 β 5 |
| Llama-3.1-Tulu-3-8B-DPO | 8B | preference | peak LR, schedule, warmup; epochs | 5e-07, linear, 0.1; 1 | t3 L144–148 | verified | paper Table 20 |
| Llama-3.1-Tulu-3-8B-DPO | 8B | preference | effective batch (pairs); max length | 8 GPUs × 1 × 16 = 128; 2,048 | t3 L135, L140–143 | verified | paper Table 20 (128, 2,048) |
| Llama-3.1-Tulu-3-8B-DPO | 8B | preference | data | `allenai/llama-3.1-tulu-3-8b-preference-mixture` | t3 L155 | verified | paper §5.2.2 Table 15 (best 8B and 70B preference mixes) |
| Llama-3.1-Tulu-3-70B-DPO | 70B | preference | peak LR; batch; β; epochs; length | 2e-07; 64 × 1 × 2 = 128; 5 (`dpo_norm`); 1; 2,048 | t3 L178–213 | verified | paper Table 19: 70B LR ablation, 2.0e-7 chosen; Table 20 |
| Llama-3.1-Tulu-3-405B-DPO | 405B | preference | peak LR; batch; β; epochs; length | 2e-07; 256 × 1 × 1 = 256; 5; 1; 2,048 | t3 L235–264 | verified | paper Table 34 (2 × 10⁻⁷, 256, β 5) |
| Llama-3.1-Tulu-3-405B-DPO | 405B | preference | data | `ai2-adapt-dev/405b_preference_mix` | t3 L260 | conflict | paper §8.1 footnote 23 names `allenai/llama-3.1-tulu-3-405b-preference-mixture` |
| Llama-3.1-Tulu-3-8B-RM | 8B | reward-model | LR; epochs; max length | 3e-6; 1; 2,048 | t3 L292, L296–298 | verified | paper App. E.2 Table 36 |
| Llama-3.1-Tulu-3-8B-RM | 8B | reward-model | effective batch | per-device 1 × grad-acc 32 × GPUs (GPU count not given in the doc) | t3 L293–295 | not reported | paper Table 36: 256; §6.3: RM trained on 8 H100 (8 × 1 × 32 = 256, derived) |
| Llama-3.1-Tulu-3-8B | 8B | RL | algorithm; start; data | PPO (`ppo_vllm_thread_ray_gtrl.py`); Tülu-3-8B-DPO; `RLVR-GSM-MATH-IF-Mixed-Constraints` | t3 L313–322 | verified | paper §6 |
| Llama-3.1-Tulu-3-8B | 8B | RL | LR; KL β | 3e-7; 0.05 | t3 L328, L338 | verified | paper Table 21: LR 3 × 10⁻⁷, β values listed [0.1, 0.05, 0.03, 0.01]; caption: final 8B model used β = 0.05 |
| Llama-3.1-Tulu-3-8B | 8B | RL | batch (responses per update) | 7 actor GPUs × 32 = 224 (derived) | t3 L334, L336 | verified | paper Table 21: 224 |
| Llama-3.1-Tulu-3-8B | 8B | RL | response / prompt length; temperature; no-EOS reward | 2,048 / 2,048; 1.0; −10.0 | t3 L320–330 | verified | paper Table 21 |
| Llama-3.1-Tulu-3-8B | 8B | RL | total episodes | 10,000,000 (planned) | t3 L329 | conflict | paper Table 21: 100,000; §6.4: evaluated every 100 steps, best MATH + IFEval checkpoint released; §6.3: "an earlier than final checkpoint" taken for all RL models |
| Llama-3.1-Tulu-3-8B | 8B | RL | PPO update iterations K; warmup | not set in command | t3 L313–346 | not reported | paper Table 21: K = 4; caption ω = 0.0 |
| Llama-3.1-Tulu-3-8B | 8B | RL | value / reward model use | RM path Tülu-3-8B-RM, `reward_model_multiplier 0.0` | t3 L323, L344 | verified | paper §6.4: value model initialized from an RM trained on the 8B preference mixture from Tülu 3 SFT |
| Llama-3.1-Tulu-3-70B | 70B | RL | LR; β; warmup | 1e-7; 0.07; 0.1 | t3 L375–383 | conflict | §6.4 text: 1 × 10⁻⁷, warmup 0.1, "β = 0.7"; Table 21 caption: β = 0.07, ω = 0.07 |
| Llama-3.1-Tulu-3-70B | 70B | RL | episodes; batch | 400,000; 40 GPUs × 16 = 640 | t3 L359, L384–391 | verified | §6.4 text: 400,000 episodes, 640; checkpoints evaluated every 40 steps |
| Llama-3.1-Tulu-3-405B | 405B | RL | data; response length; β; LR | `RLVR-MATH`; 1,024; 0.05; 1e-7 | t3 L425–441 | verified | §8.1: MATH only (GSM8K saturated, IFEval "did not help much"); Table 35 |
| Llama-3.1-Tulu-3-405B | 405B | RL | batch | 29 × 8 GPUs × 8 = 1,856 (derived); doc note says 40 × 16 = 640 | t3 L411, L448–450 | conflict | Table 35: 1,856 (matches the command, not the note) |
| Llama-3.1-Tulu-3-405B | 405B | RL | episodes; `num_epochs` | 400,000; 4 | t3 L442–443 | conflict | Table 35: 300,000 episodes, K = 1; §8.1: trained 75 steps only |
| Llama-3.1-Tulu-3.1-8B | 8B | RL | algorithm; KL estimator; start | GRPO; `kl3`; Tülu-3-8B-DPO | t3 L480, L501, L517 | verified | card: "switched from PPO to GRPO (no reward model)" |
| Llama-3.1-Tulu-3.1-8B | 8B | RL | LR, schedule; β | 5e-7 constant; 0.01 | t3 L476–477, L530 | verified | card Hyperparameters: 5 × 10⁻⁷, constant, β 0.01 |
| Llama-3.1-Tulu-3.1-8B | 8B | RL | prompts × samples per step; mini-batches | 12 GPUs × 4 = 48 prompts × 16 = 768; 2 (`half-m`) | t3 L478–486, L549 | verified | card: 48 × 16 = 768; N_mb 2; K 1 |
| Llama-3.1-Tulu-3.1-8B | 8B | RL | response length; temperature; no-EOS reward | 2,048; 1.0; 0.0 | t3 L516–523 | verified | card |
| Llama-3.1-Tulu-3.1-8B | 8B | RL | episodes: planned; released checkpoint | 10,000,000; step 1920 = episode 1,474,560 | t3 L522; card "Learning curves" | verified | card: step 1920 taken as final; 1920 × 768 = 1,474,560 (derived) |
| Llama-3.1-Tulu-3.1-8B | 8B | RL | code commit of the run | 745bf58d321c | t3 L470 | conflict | card reproduction block checks out 3f37c29 |
| OLMo-2-0425-1B-SFT | 1B | SFT | LR; batch; epochs; length; warmup; data | 3e-5; 8 × 2 × 8 = 128; 2; 4,096; 0.03; `tulu-3-sft-olmo-2-mixture-0225` | o2 L45–53 | verified | no ablation reported in the doc |
| OLMo-2-0425-1B-SFT | 1B | SFT | loss reduction | `--reduce_loss sum` at 5e5c93d (L50), absent at 098424c | o2@5e5c93d L50 | conflict | PR #1024 default mean |
| OLMo-2-0425-1B-DPO | 1B | preference | LR; batch; β; epochs; length; warmup | 2.5e-6; 8 × 8 × 2 = 128; 5 (`dpo_norm`); 1; 2,048; 0.1 | o2 L101–131 | verified | no ablation reported |
| OLMo-2-0425-1B-RLVR1 → Instruct | 1B | RL | algorithm; stages; β; LR | GRPO; mixed GSM/MATH/IF then MATH; 0.01; 5e-7 constant | o2 L236–377 | verified | no ablation reported |
| OLMo-2-0425-1B-RLVR1 → Instruct | 1B | RL | prompts × samples; episodes per stage | 48 × 16; 2,000,000 | o2 L265–277, L362–373 | verified | o2 L13: 43 h per stage on 16 GPUs |
| OLMo-2-0425-1B-RLVR1 → Instruct | 1B | RL | LR in original run names vs flag | names `lr_7e-7`, `lr_9e-7`; flag 5e-7 | o2 L286, L292, L381, L387 | conflict | unresolved in the doc |
| OLMo-2-0425-1B-RLVR1 → Instruct | 1B | RL | KL estimator | original `kl3`; "modern reproduction" `3` | o2 L253, L291 | conflict | at 098424c index 2 = former `kl3` (commit c4c1b1c; model_utils.py L803–818) |
| OLMo-2-1124-7B-SFT | 7B | SFT | peak LR | YAML 1.0e-05; script 2e-5 | olmo2_1124_7b_sft.yaml L11; finetune_7b.sh L28 | conflict | OLMo 2 paper §5: sweep 1e-5, 2e-5 (♥), 3e-5; script matches paper |
| OLMo-2-1124-7B-SFT | 7B | SFT | data | YAML `tulu-3-sft-olmo-2-mixture`; script `...-mixture-0225` | yaml L5; finetune_7b.sh L24 | conflict | OLMo 2 §5: 7B/13B used `tulu-3-sft-olmo-2-mixture`; `-0225` for 1B/32B; YAML matches paper |
| OLMo-2-1124-7B-SFT | 7B | SFT | batch; epochs; warmup; loss reduction | YAML 32 GPUs × 1 × 4 = 128 (comment L10); script 8 nodes × 8 × 1 × 2 = 128 (mason.py L538–545 multiplies processes by nodes); 2; 0.03; sum removed at HEAD | yaml L9–15, @5bcd3b2 L22; finetune_7b.sh L8, L26–32 | verified | OLMo 2 Table 17 caption: effective batch 128 |
| OLMo-2-1124-7B-DPO | 7B | preference | peak LR | YAML 5.0e-7; script 1e-6 | olmo2_1124_7b_dpo.yaml L12; dpo_7b.sh L28 | conflict | OLMo 2 §5: sweep 5e-7…1e-6, 1e-6 (♥ 7B); script matches paper |
| OLMo-2-1124-7B-DPO | 7B | preference | data; β; epochs; batch | YAML 13B mix, script 7B mix; 5; 1; 32 × 1 × 4 = 128 | yaml L4, L10–16, L22; dpo_7b.sh L24–35 | conflict | OLMo 2 §5: independent 7B (366.7k prompts) and 13B datasets; script matches |
| OLMo-2-1124-13B-SFT | 13B | SFT | peak LR; data; batch | YAML 6e-06, 1124 mix, 128; script 5e-06, `-0225` mix, 8 nodes × 8 × 1 × 1 = 64 (derived via mason.py L538–545) | olmo2_1124_13b_sft.yaml L5–11; finetune_13b.sh L8, L24–28 | conflict | OLMo 2 §5: 13B sweep 1e-6…8e-6, 5e-6 (♥) |
| OLMo-2-1124-13B-DPO | 13B | preference | peak LR; seed | YAML 5.0e-7, 1234; script 8e-7, 8 | olmo2_1124_13b_dpo.yaml L12, L26; dpo_13b.sh L28, L38 | conflict | OLMo 2 §5: 8e-7 (♥ 13B); script matches |
| OLMo-2-0325-32B-SFT | 32B | SFT | peak LR; batch; data; epochs | 4e-6; 8 nodes × 8 × 1 × 4 = 256 (derived via mason.py L538–545); `-0225` mix; 2 | finetune_32b.sh L8, L24–32 | verified | OLMo 2 §5: 32B sweep 1e-6…5e-6, best 4e-6 |
| OLMo-2 32B DPO (file `olmo2_1124_32b_dpo.yaml`) | 32B | preference | start model; LR; data | `OLMo-2-1124-13B-SFT`; 5.0e-7; 9 `*-olmo32` dataset entries (8 unique; one listed twice) | yaml L1–20 | conflict | OLMo 2 §5: 32B DPO best 2e-6; file does not load a 32B model |
| OLMo-2 32B RL (`grpo_fast_32b.sh`) | 32B | RL | β; prompts × samples; LR; episodes | 0.0; 256 × 64; 5e-7; 10,000,000 | grpo_fast_32b.sh L12–37 | conflict | OLMo 2 §5: final 32B RLVR LR 5e-7, β 0.1, 16 samples per prompt |
| OLMo 2 RL-zero (`grpo_fast_{7b,13b}_zero.sh`) | 7B, 13B | RL | start; β; prompts × samples; LR; episodes | base OLMo-2-1124; 0.0; 48 × 16; 5e-7 linear; 1,000,000 | grpo_fast_7b_zero.sh L12–47; _13b_zero L12–47 | verified | no released checkpoint named; no ablation reported |

## Notes
- The OLMo 2 7B and 13B Instruct RLVR stages used PPO (arXiv:2501.00656v3 §5); open-instruct at 098424c has no launch file for them. OLMo 2 Table 18 gives 7B β = 0.05, while §5 "Hyperparameter selection" marks 0.07 (♥ - 7B); see [[olmo-2]].
- Stage-time figures in `docs/olmo2.md` L10–13 do not name a model size.

## Starting point for a small general-purpose run
For a Llama-3.1-8B base on the Tülu 3 SFT mixture, the verified 8B rows give SFT at LR 5e-6 (linear, warmup 0.03) for 2 epochs at 128 sequences of up to 4,096 tokens, followed by `dpo_norm` DPO with β = 5 at LR 5e-7 for 1 epoch at 128 pairs of up to 2,048 tokens; the doc ran SFT on 64 H100 GPUs and DPO on one 8-GPU machine. The SFT LR was selected with a sum loss (Tülu 3 §4.3.2), so a run with mean loss is not the same recipe. These values are for Llama 3.1 bases only; the OLMo 2 paper reports that OLMo 2 bases needed higher learning rates than the Llama 3.1 recipe (§5), and its launch files disagree (see the `conflict` rows).

## Verification
- Created on 2026-09-14 from https://github.com/allenai/open-instruct/tree/098424c479f44f153f93e9c5c0f8c4c97fc854ac, arXiv:2411.15124v5, arXiv:2501.00656v3, huggingface.co/allenai/Llama-3.1-Tulu-3.1-8B (README at `main`).
- Audit claims not found in the source: none for the listed values; the lead's OLMo 2 YAML learning rates are real but marked `conflict` above.
