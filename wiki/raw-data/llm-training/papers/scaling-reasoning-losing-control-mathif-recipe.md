<!-- scope: recipe ledger for the MathIF paper's controlled training runs (distill-SFT, SFT+RL, cold-RL on Qwen2.5 bases; RL rollout-length runs on DeepSeek-R1-Distill-Qwen-1.5B) and its benchmark decoding settings
     deps: [[scaling-reasoning-losing-control-mathif]]
     see-also: [[grpo]], [[verl-grpo]], [[deepscaler]], [[qwen-qwq-traces]]
-->

# Scaling Reasoning, Losing Control — recipe ledger
- **Core Insight:** The paper's controlled runs use one hyperparameter set for all four Qwen2.5 bases (Table 7), which the authors say mostly follows VeRL defaults (App. D); only the RL maximum rollout length (Table 5) and the presence of a format reward (Table 4) are varied experimentally.
- **Guideline:** When these values are reused, keep them scoped to 1.5B and 7B Qwen2.5 bases trained on DeepScaleR with 16 H100 GPUs, because the paper reports no ablation for any other value.
- **Authors:** Tingchen Fu, Jiawei Gu, Yafu Li, Xiaoye Qu, Yu Cheng
- **Year:** 2025 (arXiv v1 2025-05)
- **URL:** https://arxiv.org/abs/2505.14810
- **Source type:** paper (training settings of its analysis experiments)
- **Relevant topics:** distill-SFT hyperparameters, GRPO hyperparameters, rollout length, format reward

## Summary
The runs are analysis experiments, not a released model. §5.2 trains Qwen2.5-1.5B, Qwen2.5-7B, Qwen2.5-Math-1.5B, and Qwen2.5-Math-7B in three ways: SFT on distilled QwQ-32B traces, the same SFT followed by GRPO, and GRPO from the base (cold-RL), plus cold-RL with a format-aware reward (Table 4). §5.3 continues RL on DeepSeek-R1-Distill-Qwen-1.5B with different maximum rollout lengths (Table 5). Table 7 gives SFT and RL hyperparameters for §5.2; the paper does not state whether §5.3 uses the same values apart from rollout length and epochs.

## Recipe ledger
Status for every `verified` row: read at the locus on 2026-09-14 in arXiv:2505.14810v2. "§5.2 bases" = Qwen2.5-1.5B, Qwen2.5-7B, Qwen2.5-Math-1.5B, Qwen2.5-Math-7B.

| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| §5.2 bases | 1.5B, 7B | distill-SFT | prompt source | DeepScaleR dataset, approximately 40k math reasoning samples | arXiv:2505.14810v2 §5.2 | verified | no ablation reported |
| §5.2 bases | 1.5B, 7B | distill-SFT | teacher; filter; examples | QwQ-32B long CoT; drop samples with a wrong teacher answer or CoT over 8192 tokens; 18k examples | §5.2 | verified | no ablation reported |
| §5.2 bases | 1.5B, 7B | distill-SFT | teacher sampling (temperature, samples per prompt) | not reported | checked §5.2, App. D, Table 7 | not reported | — |
| §5.2 bases | 1.5B, 7B | distill-SFT | max_length; truncation; sliding_window | 8192; right; none | App. D Table 7 (left) | verified | no ablation reported |
| §5.2 bases | 1.5B, 7B | distill-SFT | rope_theta | 20000 (raised from 10,000) | §5.2; Table 7 | verified | stated reason: some bases are limited to 4096 position embeddings; follows prior work [21]; no ablation reported |
| §5.2 bases | 1.5B, 7B | distill-SFT | batch_size; micro_batch_size | 256; 1 (units not stated) | Table 7 (left) | verified | no ablation reported |
| §5.2 bases | 1.5B, 7B | distill-SFT | lr; schedule; warmup_ratio | 1e-6; cosine; 0.1 | Table 7 (left) | verified | no ablation reported |
| §5.2 bases | 1.5B, 7B | distill-SFT | betas; weight_decay; clip_grad | (0.9, 0.95); 0.01; 1 | Table 7 (left) | verified | no ablation reported |
| §5.2 bases | 1.5B, 7B | distill-SFT | epochs | 3 | Table 7 (left) | verified | no ablation reported |
| §5.2 bases | 1.5B, 7B | distill-SFT | loss masking; packing | not reported | checked §5.2, App. D, Table 7 | not reported | — |
| §5.2 bases (SFT+RL and cold-RL) | 1.5B, 7B | RL | algorithm; reward | GRPO; verifiable outcome reward (correctness) | §5.2 | verified | no ablation reported |
| §5.2 bases (cold-RL w/ format reward) | 1.5B, 7B | RL | format-aware reward | 0.1 for including special reasoning tokens; 1.0 for a correct solution | §5.2 | verified | Table 4 "w/ format reward" rows: HAcc 9.52 → 10.95 on Qwen2.5-1.5B and 10.48 → 14.52 on Qwen2.5-7B vs cold-RL; negligible change on Qwen2.5-Math bases |
| §5.2 bases | 1.5B, 7B | RL | max_prompt_length; max_response_length | 1024; 3072 | Table 7 (right) | verified | no ablation on these bases; related length ablation on a different model in Table 5 |
| §5.2 bases | 1.5B, 7B | RL | rollout_n; rollout_temperature | 8; 1 | Table 7 (right) | verified | no ablation reported |
| §5.2 bases | 1.5B, 7B | RL | batch_size; mini_batch_size; rl_epoch | 128; 64; 1 (units not stated) | Table 7 (right) | verified | no ablation reported |
| §5.2 bases | 1.5B, 7B | RL | lr; schedule; warmup_ratio | 1e-6; constant; 0 | Table 7 (right) | verified | no ablation reported |
| §5.2 bases | 1.5B, 7B | RL | clip_ratio; grad_clip | 0.2; 1 | Table 7 (right) | verified | no ablation reported |
| §5.2 bases | 1.5B, 7B | RL | entropy_coeff; kl_loss_coef | 0.001; 0.001 (KL applied as a loss term per the parameter name; reference model not stated) | Table 7 (right) | verified (values) | no ablation reported |
| §5.2 bases | 1.5B, 7B | RL | total RL steps or epochs over data | not reported | checked §5.2, App. D, Table 7 | not reported | — |
| §5.2 bases | 1.5B, 7B | SFT and RL | framework; hardware | Hugging Face libraries, hyperparameters mostly following VeRL defaults; 16 NVIDIA H100 80GiB | §5.2; App. D | verified | — |
| DeepSeek-R1-Distill-Qwen-1.5B (continued RL) | 1.5B | RL | data; reward; overlong handling | DeepScaleR; pure outcome reward; overlong responses truncated and given no outcome reward | §5.3 | verified | — |
| DeepSeek-R1-Distill-Qwen-1.5B (continued RL) | 1.5B | RL | max rollout length; epochs | 1k, 2k, 4k, 8k tokens; three epochs | §5.3; Table 5 | verified | Table 5: HAcc 19.05 / 16.43 / 16.91 / 14.29 and average math accuracy 28.73 / 36.32 / 40.03 / 39.82 for 1k / 2k / 4k / 8k (original 17.14 HAcc, 36.13 accuracy) |
| DeepSeek-R1-Distill-Qwen-1.5B (continued RL) | 1.5B | RL | other hyperparameters | not reported (Table 7 is captioned for §5.2) | checked §5.3, App. D, App. F | not reported | — |
| 23 evaluated reasoning models | 0.6B–70B | eval-gate | decoding | nucleus sampling T = 1.0, p = 0.95; max generation 16,384 tokens; vLLM | §4 | verified | no ablation reported |

## Connections
- [[scaling-reasoning-losing-control-mathif]] — main card with results and findings.
- [[grpo]], [[verl-grpo]] — algorithm and framework whose defaults Table 7 mostly follows.
- [[deepscaler]] — training prompt source; [[qwen-qwq-traces]] — QwQ-series reasoning traces as a distillation source.

## Verification
- Created on 2026-09-14 from https://arxiv.org/abs/2505.14810 (arXiv v2, 2025-05-25), §4, §5.2, §5.3, Appendices D and F, Table 7.
- Audit claims not found in the source: none for this ledger.
