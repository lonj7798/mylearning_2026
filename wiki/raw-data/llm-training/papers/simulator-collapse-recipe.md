<!-- scope: Recipe ledger for One Frozen Simulator Is Not Enough (arXiv:2608.12253v2) — shared multi-turn RL settings, simulator configurations, co-training pool and reward, evaluation and compute
     deps: [[simulator-collapse]]
     see-also: [[verbalized-sampling]], [[tau2-bench]], [[grpo]]
-->

# One Frozen Simulator Is Not Enough: Simulator Collapse in Multi-Agent RL — Recipe ledger
- **Core Insight:** All methods in Tables 1-2 share one RL configuration (250 steps, 16 prompts × G = 8 samples, Adam at a constant LR of 1×10⁻⁶, KL loss coefficient 0.005) and differ only in how the simulator is generated and, for Co-Training, in updating the simulator too (App. C.4, Table 6).
- **Guideline:** When reproducing these comparisons, keep the step count matched and report compute separately, because the paper compares at matched optimizer steps and Co-Training uses 2.0× per-step training compute (App. C.4, Table 8).
- **Authors:** Simon Yu, Nicholas Tomlin, Marwa Abdulhai, Ximing Lu, Derek Chong, Abe Hou, et al.
- **Year:** 2026 (arXiv v1 2026-08; v2 2026-08-17)
- **URL:** https://arxiv.org/abs/2608.12253
- **Source type:** paper
- **Relevant topics:** multi-turn RL hyperparameters, user-simulator configuration, co-training, checkpoint selection, compute

## Recipe ledger
All rows: source arXiv:2608.12253v2; status verified on 2026-09-14 unless marked otherwise.

| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| Qwen3-4B-Instruct-2507, Qwen3-8B (P4G, τ²-bench); Qwen3.5-9B, Qwen3.5-27B (CooperBench) | 4B, 8B, 9B, 27B | RL | trainable agents | as listed; CooperBench uses Qwen3.5 "since smaller models cannot complete the tasks" | §4.1; App. C.4 | verified | — |
| all methods | all | RL | policy update | REINFORCE with group z-scored terminal-reward advantage on all agent tokens; GRPO clipped importance-ratio surrogate; simulator tokens masked | §2 Eq. 2; App. C.1 Eq. 22; App. C.2 | verified | no ablation reported |
| all methods | all | RL | optimizer; betas; weight decay; grad clip; LR schedule | Adam; (0.9, 0.98); 0.1; 1.0; constant | App. C.4 Table 6 | verified | no ablation reported |
| all methods | all | RL | learning rate | 1×10⁻⁶ across all tasks | App. C.4 | verified | tuned on the first 50 steps of τ²-bench Ensemble training (stable reward growth, entropy above 1.5 nats) |
| all methods | all | RL | total training steps | 250 | Table 6; App. C.4 | verified | no ablation reported |
| all methods | all | RL | prompts per rollout batch; samples per prompt G; global batch | 16; 8; 128 | Table 6 | verified | no ablation reported |
| all methods | all | RL | rollout temperature (training); evaluator rollouts | 0.7; T = 0 | Table 6; App. C.4 | verified | no ablation reported |
| all methods | all | RL | clip ε_low / ε_high | 0.2 / 0.28 | Table 6 | verified | no ablation reported |
| all methods | all | RL | KL loss coefficient β; entropy coefficient | 0.005; 0.0 | Table 6 | verified | no ablation reported |
| all methods | all | RL | max agent response; max context; max turns (P4G / τ² / CooperBench) | 32,768 tokens; 64,000 tokens; 10 / 30 / 50 | Table 6 | verified | no ablation reported |
| all methods except Qwen3.5-27B | 4B-9B | RL | training stack | Megatron-LM TP = 4, PP = 1, BF16; SGLang rollouts; 8×H100 (colocated dual-model layout for Co-Training) | App. C.2; Table 6 | verified | — |
| Qwen3.5-27B CooperBench | 27B | RL | training stack | Tinker API with LoRA adapters | Table 2 footnote; App. C.4 | verified | — |
| RL (Single), Persona-Guided, Verbalized Sampling | 4B, 8B | RL | training simulator | GPT-5-mini (openai/gpt-5-mini, 2025-08-07), frozen | App. C.3 Table 5; C.4 Table 7 | verified | §3.3: GPT-5-mini is the least modal of three tested simulators |
| Verbalized Sampling | 4B, 8B | RL | candidates per simulator turn | K_VS = 5, one sampled by verbalized probability | App. C.4; App. D Fig. 7 | verified | no ablation of K_VS reported |
| Ensemble Models | 4B, 8B | RL | simulator pool; sampling | {Haiku 4.5, GPT-5-mini, Gemini 3 Flash}, K = 3, cyclic rotation | Table 5 | verified | App. F.1 sweeps K for the population, not for the ensemble |
| Population Co-Training | 4B, 8B | RL | simulator pool | FIFO buffer of K = 5 recent checkpoints, one every four steps, uniform sampling | Table 5; App. F.1 | verified | Fig. 17: K=5 > K=10 > K=3 > K=1 on P4G and τ²-Retail |
| Co-Training variants, τ²-bench | 4B, 8B | RL | simulator reward | exp(−(σ_π² − 0.25)²/0.02) | §4.1; App. F.8 Table 9 | verified | App. F.8: eval 0.40 vs 0.07 (adversarial) and 0.17 (cooperative) |
| Co-Training variants, P4G; CooperBench | all | RL | simulator or partner reward | donation-based adversarial reward; shared binary task success | §4.1; App. F.2 | verified | App. F.2 describes a reward-quadrant ablation; no numbers in the text |
| all methods | all | eval-gate | evaluation cadence; checkpoint rule | every 16 steps; best mean score on the 6-simulator panel (P4G, τ²) | §4.1 | conflict | App. C.4 states no evaluation-only model is used for selection |
| all methods | all | eval-gate | held-out panel | 3 training simulators + GLM-5, MiniMax-M2.7, DeepSeek-V3.1; CooperBench partners Haiku 4.5, GPT-5, Gemini-3-Flash | §4.1; Table 7; Table 2 | verified | — |
| RL (Single) vs Co-Training, τ²-Retail, Qwen3-4B | 4B | RL | wall-clock per step; total GPU-hours per 250 steps | 250 s vs 500 s; 280 vs 560 (printed as approximate) | App. C.4 Table 8 and text | verified | — |
| all methods | all | RL | seeds | 3 per method | Fig. 3, Fig. 4 captions | verified | — |
| all methods | all | RL | warmup, reference-model update, response-length penalties | not reported | checked §2, §4.1, App. C.1-C.4 | not reported | — |

## Starting point for a small general-purpose run
For a 4B-8B instruct agent trained with multi-turn RL against LLM user simulators (dialogue horizon up to 30 turns, 8×H100), the paper's shared setting is 250 steps, 16 prompts × 8 samples, constant LR 1×10⁻⁶, clip 0.2 / 0.28, KL coefficient 0.005, rollout temperature 0.7, with a K = 5 checkpoint pool and a variance-targeting simulator reward for co-training (rows above). These values were used for τ²-bench, P4G, and CooperBench only; the LR was tuned on 50 steps of one method.

## Connections
- [[simulator-collapse]] — the main card with results and findings.
- [[verbalized-sampling]] — the prompting method used for the VS simulator.
- [[grpo]] — the clipped surrogate retained in App. C.1.

## Verification
- Created on 2026-09-14 from https://arxiv.org/abs/2608.12253 (v2 PDF, 2026-08-17).
- Audit claims not found in the source: none (this ledger was built from the paper; the audit details contained no recipe values).
- Not reported by the source: warmup, reference-model refresh, response-length penalties, and matched-compute comparisons (left to future work, App. C.4).
