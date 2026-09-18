<!-- scope: full recipe ledger for the OLMo 2 technical report (arXiv:2501.00656v3); companion to [[olmo-2]]
     deps: [[olmo-2]]
     see-also: [[tulu-3]], [[rlvr-tulu3]]
-->

# 2 OLMo 2 Furious — recipe ledger
- **Core Insight:** Every pretraining, mid-training, and post-training setting OLMo 2 discloses, with the locus in the report for each value.
- **Guideline:** When reusing an OLMo 2 number, take the row for the exact size, stage, and round; the 7B, 13B, and 32B runs differ in learning rate, schedule horizon, mid-training sample size, and RL algorithm.
- **Authors:** Pete Walsh, Luca Soldaini, Dirk Groeneveld, Kyle Lo, Shane Arora, Akshita Bhagia, et al. (OLMo Team, Allen Institute for AI)
- **Year:** 2025 (arXiv v1 2025-01; v3 dated 2025-10-08)
- **URL:** https://arxiv.org/abs/2501.00656
- **Source type:** official technical report
- **Relevant topics:** pretraining hyperparameters, mid-training, model merging, SFT, DPO, RLVR

## Summary
This card holds the recipe table for [[olmo-2]], split out to keep that card under the length limit. Section
numbers refer to arXiv:2501.00656v3.

## Recipe ledger
Section numbers refer to arXiv:2501.00656v3.

| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| OLMo-2-1124-7B | 7B | pretrain-stable | peak LR; warmup; cosine horizon; truncation | 3.0e-4; 2000 steps; 5T tokens; stopped at 4T | arXiv:2501.00656v3 Table 3 | verified 2026-09-18 | §4.1 Figure 11: four LR values compared; after decay to zero the averages differ by under 2 points |
| OLMo-2-1124-13B | 13B | pretrain-stable | peak LR; cosine horizon | 9.0e-4; 5T tokens, no truncation | Table 3 | verified 2026-09-18 | §4.1: the higher LR was kept from the start of the run |
| OLMo-2-0325-32B | 32B | pretrain-stable | peak LR; cosine horizon; truncation | 6.0e-4; 6.5T tokens; stopped after 6T | Table 3 | verified 2026-09-18 | no ablation reported |
| OLMo 2 (all sizes) | 7B/13B/32B | pretrain-stable | sequence length; grad clip | 4096; 1.0 | Table 3 | verified 2026-09-18 | no ablation reported |
| OLMo-2-1124-7B | 7B | mid-train | Dolmino tokens; checkpoints merged | 50B; average of 3 runs on data permutations | §4, Table 9 caption | verified 2026-09-18 | Table 14: merging 3 checkpoints equals or beats the best single run on all 6 mixes tested |
| OLMo-2-1124-13B / 0325-32B | 13B/32B | mid-train | Dolmino tokens; checkpoints merged | 3 runs on 100B plus 1 run on 300B, all 4 averaged | §4, Table 9 caption | verified 2026-09-18 | §4: averaging all four was better than averaging the three 100B runs alone |
| OLMo 2-Instruct 7B/13B | 7B/13B | SFT | mix; prompts; epochs; LR; effective batch | tulu-3-sft-olmo-2-mixture; 939,104 prompts; 2 epochs; 1e-5; 128 | §5, Table 17 | verified 2026-09-18 | Table 17: 2 epochs at 1e-5 with summed loss scores 49.97 average against 48.18-49.76 for five other settings |
| OLMo 2-Instruct 1B/32B | 1B/32B | SFT | mix; prompts; LR (32B) | tulu-3-sft-olmo-2-mixture-0225; 866,138 prompts; 4e-6 | §5 | verified 2026-09-18 | §5: LR swept over 1e-6 to 5e-6, best at 4e-6 with one extra seed |
| OLMo 2-Instruct 7B/13B | 7B/13B | preference | DPO prompts; response pool | 366.7k prompts (7B), 377.7k (13B); responses from 20 models | §5, Table 27 | verified 2026-09-18 | §5: DPO LR swept over 5e-7, 6e-7, 7e-7, 8e-7 (best at 13B) and 1e-6 (best at 7B), Figure 12 |
| OLMo 2-Instruct 32B | 32B | preference | DPO LR | 2e-6 | §5 | verified 2026-09-18 | §5: swept 8e-7 to 2.5e-6 |
| OLMo-2-1124-13B-Instruct | 13B | RL | algorithm; LR; KL β; clip ε; GAE λ; PPO iterations; episodes | PPO; 3e-7; 0.1 for the first and final rounds and 0.03 for the second; 0.2; 0.95; 4; 200,000 | Table 18 | verified 2026-09-18 | §5: β swept over 0.03/0.05/0.07/0.1; three RLVR rounds run because GSM8K fell after round 1 (Figure 13) |
| OLMo-2-1124-7B-Instruct | 7B | RL | algorithm; LR; KL β; episodes; effective batch | PPO; 4e-7; 0.05; 100,000; 224 | Table 18 | verified 2026-09-18 | §5: LR swept over 3e-7 and 4e-7; β swept over 0.03/0.05/0.07 |
| OLMo-2-0325-32B-Instruct | 32B | RL | algorithm; LR; KL β; samples per prompt; reward model | GRPO; 5e-7; 0.1; 16; none used | §5 | verified 2026-09-18 | no ablation reported |
| OLMo 2 7B / 13B | 7B/13B | pretrain-stable | compute | not reported as GPU-hours; 131 MWh (7B), 257 MWh (13B) | Table 19 | not reported | checked body, §6, Table 19, and the appendix |

## Connections
- [[olmo-2]] — the source card these rows belong to.
- [[tulu-3]] — the post-training recipe these SFT, DPO, and RLVR rows adapt.
- [[rlvr-tulu3]] — the RLVR method behind the Table 18 rows.

## Verification
- Checked on 2026-09-18 against: https://arxiv.org/abs/2501.00656 (arXiv v3, 8 Oct 2025).
- Corrections to the previous card version: none (new card, split out of [[olmo-2]] on 2026-09-18).
- Removed as unsupported by the source: GPU-hour figures for pretraining, which the report does not give.
- Not reported by the source: optimizer betas and weight decay for post-training; post-training compute.
