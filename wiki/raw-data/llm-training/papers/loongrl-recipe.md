<!-- scope: recipe ledger for the LoongRL GRPO runs on Qwen2.5-7B-Instruct and Qwen2.5-14B-Instruct (data filtering, data mix, RL hyperparameters, curriculum, compute, evaluation decoding)
     deps: [[loongrl]]
     see-also: [[grpo]], [[qwenlong-l1]]
-->

# LoongRL: Reinforcement Learning for Advanced Reasoning over Long Contexts — Recipe ledger
- **Parent card:** [[loongrl]]
- **URL:** https://arxiv.org/abs/2510.19363 (arXiv v2, 2025-10-27)
- **Source type:** paper
- **Scope:** RL from instruct checkpoints (Qwen2.5-7B-Instruct, Qwen2.5-14B-Instruct, both with a 128K context window, §3.2.2). No SFT or pretraining stage is part of the method. The paper states that RL training code and KeyChain synthesis code are in the supplementary materials (Reproducibility Statement); no config file was checked for this ledger.

## Recipe ledger

| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| LoongRL-7B and LoongRL-14B | 7B, 14B | RL | seed filtering | 277K HotpotQA/MuSiQue/2WikiMultiHopQA items; 8 answers each from Qwen2.5-32B-Instruct; discard pass rate 0 or 1; 72K remain | arXiv:2510.19363v2 §3.1 | verified 2026-09-14 | no ablation reported |
| LoongRL-7B and LoongRL-14B | 7B, 14B | RL | KeyChain context length | "approximately (<) 16,384 tokens" before chain insertion; Table 1 KeyChain lengths 14,911–20,670 | arXiv:2510.19363v2 §3.1; Table 1 | verified 2026-09-14 | no ablation reported |
| LoongRL-7B and LoongRL-14B | 7B, 14B | RL | data mix (examples) | KeyChain 7,500 (2,500 × HotpotQA, MuSiQue, 2WikiMultiHopQA); plain multi-hop QA 7,500 (2,500 × same three); RULER-style retrieval on PG19 1,024 (512 multi-key + 512 multi-value); math 5,000 (2,500 DAPO + 2,500 MATH multiple-choice) | arXiv:2510.19363v2 Table 1; §3.2.2 | verified 2026-09-14 | Table 4 (7B): replacing KeyChain with an equal amount of regular multi-hop QA gives 66.2 vs 72.4 LongBench v1 avg; other components not ablated |
| LoongRL-7B and LoongRL-14B | 7B, 14B | RL | algorithm, group size | GRPO, G = 8 | arXiv:2510.19363v2 §3.2.1; §4.1 | verified 2026-09-14 | no ablation reported |
| LoongRL-7B and LoongRL-14B | 7B, 14B | RL | KL coefficient | β = 0.001, KL(πθ ‖ πref) term in the objective (Eq. 1) | arXiv:2510.19363v2 §3.2.1 | verified 2026-09-14 | no ablation reported |
| LoongRL-7B and LoongRL-14B | 7B, 14B | RL | entropy loss | removed | arXiv:2510.19363v2 §3.2.1 | verified 2026-09-14 | follows prior work (Shang et al., 2025a); no ablation reported |
| LoongRL-7B and LoongRL-14B | 7B, 14B | RL | clip ε | not reported (symbol in Eq. 1 only) | checked §3.2.1, §4.1, App. A.1–A.9 | not reported | — |
| LoongRL-7B and LoongRL-14B | 7B, 14B | RL | reward | binary; 1 if ground truth ⊆ boxed answer or boxed answer ⊆ ground truth | arXiv:2510.19363v2 §3.2.1 Eq. 3 | verified 2026-09-14 | Table 5 (7B): 72.4 vs exact match 69.2, LLM-as-a-judge 65.2, F1 65.1 |
| LoongRL-7B and LoongRL-14B | 7B, 14B | RL | learning rate | 1 × 10⁻⁶, cosine decay; warmup not reported | arXiv:2510.19363v2 §4.1 | verified 2026-09-14 | no ablation reported |
| LoongRL-7B and LoongRL-14B | 7B, 14B | RL | gradient clipping | 1.0 | arXiv:2510.19363v2 §4.1 | verified 2026-09-14 | no ablation reported |
| LoongRL-7B | 7B | RL | batch size | 512 (unit, prompts or rollouts, not stated) | arXiv:2510.19363v2 §4.1 | verified 2026-09-14 | no ablation reported |
| LoongRL-14B | 14B | RL | batch size | 256 (unit not stated) | arXiv:2510.19363v2 §4.1 | verified 2026-09-14 | no ablation reported |
| LoongRL-7B and LoongRL-14B | 7B, 14B | RL | rollout sampling | temperature 0.6, top-p 0.95 | arXiv:2510.19363v2 §4.1 | verified 2026-09-14 | no ablation reported |
| LoongRL-7B and LoongRL-14B | 7B, 14B | RL | lengths | max output 4,096 tokens; inputs ∼16K | arXiv:2510.19363v2 §4.1 | verified 2026-09-14 | 16K chosen to avoid the cost of 128K rollouts (§3.2.2); no length ablation reported |
| LoongRL-7B | 7B | RL | curriculum steps (§4.1) | warm-up 42 steps (1 epoch, no KeyChain); Stage I 168 steps; Stage II 118 steps | arXiv:2510.19363v2 §4.1 | conflict | Fig. 4a shows LongBench v1 avg rising in each stage; no ablation of stage lengths |
| LoongRL-7B | 7B | RL | curriculum steps (App. A.6) | warm-up 1 epoch 42 steps; Stage I 4 epochs 168 steps; Stage II 9 epochs 117 steps | arXiv:2510.19363v2 App. A.6 Fig. 7 caption | conflict | differs from §4.1 by one Stage II step; the paper does not say which produced the released result |
| LoongRL-14B | 14B | RL | curriculum steps | no warm-up; Stage I 168 steps (2 epochs); Stage II 150 steps (3 epochs) | arXiv:2510.19363v2 §4.1; App. A.6 Fig. 8 caption | verified 2026-09-14 | warm-up skipped because the 14B model "can immediately handle KeyChain data" (§4.1); no ablation reported |
| LoongRL-7B and LoongRL-14B | 7B, 14B | RL | Stage II hard mining | 8 rollouts per example from the best Stage I checkpoint; drop examples correct in all 8; 30–40% of data remains | arXiv:2510.19363v2 §3.2.2 | verified 2026-09-14 | Fig. 7–8 show reward and "consistent predictions" rising within each stage; no ablation without Stage II |
| LoongRL-7B | 7B | RL | compute | 16 × A100; GPU-hours not reported | arXiv:2510.19363v2 §4.1 | verified 2026-09-14 | — |
| LoongRL-14B | 14B | RL | compute | 8 × MI300X; GPU-hours not reported | arXiv:2510.19363v2 §4.1 | verified 2026-09-14 | — |
| LoongRL-7B and LoongRL-14B | 7B, 14B | eval-gate | evaluation decoding | temperature 0.6; up to 128K input, 10K output; 8 samples; average pass@1 (non-reasoning baselines: temperature 0) | arXiv:2510.19363v2 §4.1 | verified 2026-09-14 | — |
| LoongRL-7B and LoongRL-14B | 7B, 14B | eval-gate | context extension at evaluation | YaRN to 128K for RULER 64K/128K and LongBench v2 | arXiv:2510.19363v2 Table 7 caption; App. A.7 | verified 2026-09-14 | — |
| LoongRL-7B and LoongRL-14B | 7B, 14B | eval-gate | checkpoint selection | "best checkpoint" after Stage I used for hard mining; selection metric not stated; final checkpoint rule not reported | arXiv:2510.19363v2 §3.2.2 | not reported | — |

## Notes on units and scope
- Table 1 sizes are examples (prompts). The number of rollouts per step equals prompts per step × G = 8 only if batch size counts prompts; the paper does not state this.
- The 7B warm-up uses the dataset without KeyChain data (13,524 examples by Table 1: 7,500 + 1,024 + 5,000); this count is derived from Table 1, not printed in the paper.
- Values above belong to these two runs only. They are not tested on other base models or on pretraining-stage long-context extension.
