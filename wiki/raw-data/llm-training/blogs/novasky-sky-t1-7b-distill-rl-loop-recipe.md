<!-- scope: recipe ledger for Sky-T1-7B (Steps 1–4), Sky-T1-7B-Step2-5k-distill, Sky-T1-7B-Zero, Sky-T1-mini, and the SFT data-scaling ablation from the NovaSky post of 2025-02-13
     deps: [[novasky-sky-t1-7b-distill-rl-loop]]
     see-also: [[rloo]], [[sky-t1]]
-->

# Unlocking the Potential of Reinforcement Learning in Improving Reasoning Models — Recipe ledger
- **Parent card:** [[novasky-sky-t1-7b-distill-rl-loop]] (`blogs/novasky-sky-t1-7b-distill-rl-loop.md`)
- **Sources:** (P) the post https://novasky-ai.github.io/posts/sky-t1-7B/ (page dated 2025-02-13, updated 2026-03-14), cited by section name; (R) released code at github.com/NovaSky-AI/SkyThought@0d190f11fd8e885bbe113aeccacba5ccde5b1102: `recipes/sky-t1-7b/README.md` and `skythought/skythought-rl/examples/sky-t1/run-sky-t1-7b-{step2,step4,zero}.sh`, `run-sky-t1-mini.sh`.
- **Units:** "batch size 256" in RL is prompts per step as printed; the post writes "(~30K data)" for 127 steps and "(~15K data)" for 59 steps. SFT "batch size 96" units are not stated. Released RL scripts are verl-based (`verl.trainer.main_ppo_sky`); they are a separate fact from the post, and the post does not say that these scripts produced the released checkpoints.

## Sky-T1-7B pipeline (Qwen2.5-Math-7B base)

| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| Sky-T1-7B-Step1 | 7B | distill-SFT | base; teacher | Qwen2.5-Math-7B; QwQ | P Step 1 | verified 2026-09-14 | teacher chosen because it was the only open-weights long-reasoning model then; no ablation reported |
| Sky-T1-7B-Step1 | 7B | distill-SFT | prompt selection | NuminaMath problems; GPT-4o-mini AoPS difficulty: math > Level 3, Olympiads > Level 8, all AIME/AMC | P Step 1 | verified 2026-09-14 | no ablation reported |
| Sky-T1-7B-Step1 | 7B | distill-SFT | examples; filter | 5K responses; keep only solutions matching ground truth | P Step 1 | verified 2026-09-14 | no ablation reported |
| Sky-T1-7B-Step1 | 7B | distill-SFT | epochs; LR; batch size; system prompt | 3; 1e-5; 96; Sky-T1 system prompt | P Step 1 | verified 2026-09-14 | no ablation reported |
| Sky-T1-7B-Step1 and Step3 data | 7B | distill-SFT | teacher generation max tokens; temperature | 16384; not set in the command | R README `skythought generate --sampling-params max_tokens=16384` | verified 2026-09-14 (temperature: not reported) | no ablation reported |
| Sky-T1-7B-Step2 | 7B | RL | algorithm | "PRIME's algorithms" with PRIME prompt filtering | P Step 2 | conflict | R `run-sky-t1-7b-step2.sh` L58–59 sets `algorithm.adv_estimator=rloo`, `adv_params.verifier_gamma=1.0` and contains no process-reward-model arguments; which one produced the checkpoint is not stated |
| Sky-T1-7B-Step2 | 7B | RL | data; steps; batch size | Eurus-2-RL-Data; 127; 256 (~30K data) | P Step 2 | verified 2026-09-14 | no ablation reported |
| Sky-T1-7B-Step2 | 7B | RL | rollouts per prompt; prompt filter | 4; drop problems whose 4 rollouts are all correct or all wrong | P Step 2 | verified 2026-09-14 | no ablation reported |
| Sky-T1-7B-Step2 | 7B | RL | compute | 8xH100, around 44 hours | P Step 2 | verified 2026-09-14 | not applicable |
| Sky-T1-7B-Step2 (released script) | 7B | RL | init; LR; KL coef; entropy coef; n; max prompt / response length; mini-batch; total_epochs | NovaSky-AI/Sky-T1-7B-step1; 5e-7; 0.00; 0.; 4; 1024 / 3072; 256; 2 | R `run-sky-t1-7b-step2.sh` L14, L27–58 | verified 2026-09-14 | no ablation reported |
| Sky-T1-7B-Step2-5k-distill | 7B | distill-SFT | base; data | Qwen2.5-Math-7B; 5K rejection-sampled traces from Sky-T1-7B-Step2 with the Step 1 mixture | P Step 2 | verified 2026-09-14 | Table 1: average 55.8, equal to Step2 (55.8); Step1 with 5K QwQ traces 45 |
| Sky-T1-7B-Step2-5k-distill | 7B | distill-SFT | epochs; LR; batch size | not reported (checked post and README) | — | not reported | — |
| Sky-T1-7B-step3 | 7B | distill-SFT | base; data | Qwen2.5-Math-7B; 5K QwQ (Step 1) + 5K Sky-T1-7B-Step2 traces | P Step 3 | verified 2026-09-14 | no ablation reported |
| Sky-T1-7B-step3 | 7B | distill-SFT | epochs; LR; batch size | 3; 1e-5; 96 | P Step 3 | verified 2026-09-14 | no ablation reported |
| Sky-T1-7B | 7B | RL | algorithm | RLOO, no prompt filtering, no process reward model | P Step 4 | verified 2026-09-14 | P Step 4: chosen "to speed up the RL training"; no ablation reported |
| Sky-T1-7B | 7B | RL | data; steps; batch size; rollouts per prompt | numina_amc_aime + numina_olympiads subsets of Eurus-2-RL-Data; 59; 256 (~15K data); 8 | P Step 4 | verified 2026-09-14 | Table 1: Step3 average 51 → Sky-T1-7B 57.1 |
| Sky-T1-7B (released script) | 7B | RL | init; LR; KL coef; n; max prompt / response length; data prep | NovaSky-AI/Sky-T1-7B-step3; 5e-7; 0.00; 8; 1024 / 3072; `data_prepare_mini.py --sky-sys` | R `run-sky-t1-7b-step4.sh` L11–48 | verified 2026-09-14 | no ablation reported |
| Sky-T1-7B | 7B | eval-gate | checkpoint selection | not reported (checked post, README, scripts: `save_freq=15`, `test_freq=15` only) | — | not reported | — |

## Comparison and follow-up runs

| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| Sky-T1-7B-Zero | 7B | RL | init; algorithm; data; rollouts; steps | Qwen2.5-Math-7B; RLOO; STILL3; 4; 104 | P Results | verified 2026-09-14 | Table 1: average 51.9 vs base 36 |
| Sky-T1-7B-Zero (released script) | 7B | RL | LR; KL coef; n; max response length | 5e-7; 0.00; 4; 3072 | R `run-sky-t1-7b-zero.sh` L30–48 | verified 2026-09-14 | no ablation reported |
| Sky-T1-mini | 7B | RL | init; algorithm; reward | DeepSeek-R1-Distill-Qwen-7B; RLOO; verifier reward only | P Sky-T1-mini section | verified 2026-09-14 | Figure 3: average 72.5 vs R1-Distilled-7B 70.5 |
| Sky-T1-mini | 7B | RL | data | STILL3 + numina_amc_aime + numina_olympiads subsets of Eurus-2-RL-Data | P Sky-T1-mini section | verified 2026-09-14 | no ablation reported |
| Sky-T1-mini | 7B | RL | schedule | 119 steps (~28 h), batch 256 (~30K), cutoff 8K; then 29 steps (~8.7 h), cutoff 16K | P Sky-T1-mini section | verified 2026-09-14 | no ablation reported |
| Sky-T1-mini | 7B | RL | compute | 8xH100, 36 hours, around $870 (Lambda Cloud pricing) | P introduction | verified 2026-09-14 | not applicable |
| Sky-T1-mini | 7B | RL | rollouts per prompt; LR; KL | not reported in the post | P Sky-T1-mini section | not reported | — |
| Sky-T1-mini (released script) | 7B | RL | init; LR; KL coef; n; max prompt / response length; batch; mini-batch | deepseek-ai/DeepSeek-R1-Distill-Qwen-7B; 5e-7; 0.00; 4; 1024 / 8192; 256; 256 | R `run-sky-t1-mini.sh` L14, L27–48 | verified 2026-09-14 | no ablation reported; the script covers only an 8K response length, not the 16K continuation |
| Sky-T1-mini | 7B | RL | advantage estimator | RLOO | P Sky-T1-mini section | conflict | R `run-sky-t1-mini.sh` L58 sets `algorithm.adv_estimator=grpo`; which one produced the released checkpoint is not stated |
| SFT data-scaling ablation | 7B (base not stated) | distill-SFT + RL | SFT data sizes; RL | 30K / 60K / 120K QwQ traces; RLOO on STILL3, 4 rollouts per prompt | P Other Observations | verified 2026-09-14 | Figure 4 averages: SFT 43.8 / 49.0 / 48.0; SFT+RL 51.6 / 56.5 / 58.2 |
| all Sky-T1-7B models | 7B | eval-gate | AIME24, AMC23 protocol | 8 samples, temperature 0.6, top-p 0.95, pass@1; Qwen math eval suite | P Evaluation; R README | verified 2026-09-14 | not applicable |
| all Sky-T1-7B models | 7B | eval-gate | MATH500, OlympiadBench protocol; vLLM; system prompt | greedy; 0.6.2; Sky-T1 system prompt except Sky-T1-mini (none) | P Evaluation; R README | verified 2026-09-14 | not applicable |

## Verification
- Created on 2026-09-14 from the post and SkyThought commit 0d190f11fd8e885bbe113aeccacba5ccde5b1102 listed above.
- Audit claims not found in the source: none.
