<!-- scope: recipe ledger for Agent Learning via Early Experience (Zhang et al., ICML 2026): per-environment expert data, rollout data, SFT, RL, and evaluation settings
     deps: [[agent-early-experience]]
     see-also: [[gigpo-verl-agent]], [[search-r1]]
-->

# Recipe ledger — Agent Learning via Early Experience
- **Parent card:** [[agent-early-experience]]
- **Source:** arXiv:2510.08558v3 (2026-05-24), App. B.2 and App. C.1-C.8. No released configuration files were checked.
- **Models:** unless a row says otherwise, values apply to all three instruction-tuned models (Llama-3.2-3B-Instruct, Qwen-2.5-7B-Instruct, Llama-3.1-8B-Instruct); the paper does not give per-model values. IL = imitation learning, IWM = implicit world modeling, SR = self-reflection.

## Recipe ledger
| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| All three models, all environments | 3B, 7B, 8B | SFT | step budget rule | IL steps selected per environment by lowest training loss and validation performance; same budget for IWM and SR; IWM = 1 epoch of the world-model objective, then supervised updates until the IL budget; SR = same epochs as IL | App. B.2 | verified 2026-09-14 | no ablation reported |
| All three models, all environments | 3B, 7B, 8B | SFT, eval-gate | hardware; metric | at most 8 H100 GPUs; each benchmark's native metric with official validators | App. B.2 | verified 2026-09-14 | not applicable |
| ALFWorld | 3B, 7B, 8B | SFT | D_expert; IWM data | 21,031 state-action pairs; 8 non-expert admissible actions sampled uniformly without replacement + expert action → 189,279 triplets | App. C.1 | verified 2026-09-14 | Fig. 4b varies K (1, 2, 4, 8) on ALFWorld with Llama-3.1-8B |
| ALFWorld | 3B, 7B, 8B | SFT | SR alternatives | up to 3 proposed by the same policy at temperature 1.0; invalid proposals replaced by uniformly sampled admissible actions | App. C.1 | verified 2026-09-14 | no ablation reported |
| ALFWorld | 3B, 7B, 8B | SFT | batch; LR; epochs; framework | 16; 1e-5; 2; LlamaFactory | App. C.1 | verified 2026-09-14 | no ablation reported |
| ALFWorld | 3B, 7B, 8B | RL, eval-gate | RL hyperparameters; eval decoding | Verl-Agent defaults (values not printed); max prompt 4096, max response 1024, temperature 0.4 | App. C.1 | verified 2026-09-14 (defaults not reported) | no ablation reported |
| WebShop | 3B, 7B, 8B | SFT | D_expert | 1,571 human trajectories → 15,464 state-action pairs | App. C.2 | verified 2026-09-14 | Fig. 4a varies the expert fraction (1/8 to 1) |
| WebShop | 3B, 7B, 8B | SFT | IWM data | policy proposals at temperatures {0.5, 0.8, 0.9} plus up to 5 uniformly sampled admissible actions per state; next state = offline textual summary (avg. 345 characters); 122,954 triplets | App. C.2 | verified 2026-09-14 | no ablation reported |
| WebShop | 3B, 7B, 8B | SFT | SR data | expert action + 3 alternatives; only trajectories finished in fewer than 15 steps; 6,235 examples | App. C.2 | verified 2026-09-14 | no ablation reported |
| WebShop | 3B, 7B, 8B | SFT, RL | batch; LR; epochs; RL | 4; 1e-5; epochs not reported; Verl-Agent defaults | App. C.2 | verified 2026-09-14 / epochs not reported | no ablation reported |
| BFCLv3 (multi-turn) | 3B, 7B, 8B | SFT | D_expert | 125 Base trajectories (75% of Base, random); Base only because no training split exists | App. C.3 | verified 2026-09-14 | not applicable |
| BFCLv3 | 3B, 7B, 8B | SFT | IWM data; SR data | 1,264 expert-derived + 11,904 augmented (10 alternatives per state); 1,200 SR examples after removing those whose conclusion differs from the expert action | App. C.3 | verified 2026-09-14 | no ablation reported |
| BFCLv3 | 3B, 7B, 8B | SFT | batch; LR; epochs | 16; 1e-5; epochs not reported | App. C.3 | verified 2026-09-14 / epochs not reported | no ablation reported |
| Tau-Bench retail | 3B, 7B, 8B | SFT | tasks; expert trajectories | 495 train / 115 eval tasks; an instruction-tuned LLaMA-family model (not named), temperature 1, 4 trajectories per task, keep one with reward 1 → 452 tasks, 5,239 pairs | App. C.4 | verified 2026-09-14 | not applicable |
| Tau-Bench retail | 3B, 7B, 8B | SFT | IWM data; SR data | 5 action candidates per observation with the expert action's tool removed from the tool set; SR uses 3 of the 5; 5,233 SR instances after filtering | App. C.4 | verified 2026-09-14 | no ablation reported |
| Tau-Bench retail | 3B, 7B, 8B | SFT | epochs and LR per method; batch | IL 6 epochs, 1e-5; IWM 1 epoch, 5e-6; SR 6 epochs, 1e-5; batch 16 for all | App. C.4 | verified 2026-09-14 | no ablation reported |
| SearchQA (MuSiQue) | 3B, 7B, 8B | SFT | tasks; expert data | 7,000 tasks (all 3-hop and 4-hop + 1,438 sampled 2-hop); Search-R1 model at temperature 1.0, 5 trajectories per task, correct answers only, at most 2 per task → 2,082 trajectories, 7,691 pairs | App. C.5 | verified 2026-09-14 | not applicable |
| SearchQA | 3B, 7B, 8B | SFT | IWM data; SR data | 30 alternatives per state at temperature 1.0; targets are summaries of retrieved documents; expert-derived WM data at 1:1 ratio with the IL set; SR with 2 alternatives, 7,691 examples | App. C.5 | verified 2026-09-14 | the authors report full-document prediction was suboptimal (no table) |
| SearchQA | 3B, 7B, 8B | SFT | tuning; GPUs; epochs; LR; context; batch | full-parameter, ZeRO-3, 4 H100; 3 epochs; 1e-5; 8,192 tokens; 2 per GPU × 16 gradient accumulation | App. C.5 | verified 2026-09-14 | no ablation reported |
| SearchQA | 3B, 7B, 8B | RL | codebase; reward; limits | Search-R1 settings on 8 H100; F1 reward; max 6 retrieval interactions; context 12,280 tokens; max output 2,048 tokens; all MuSiQue training tasks | App. C.5 | verified 2026-09-14 | no ablation reported |
| ScienceWorld | 3B, 7B, 8B | SFT | D_expert; IWM; SR | 14,506 pairs; 3 non-expert admissible actions + expert; up to 3 SR alternatives at temperature 1.0 (2 for Llama-3.1-8B) | App. C.6 | verified 2026-09-14 | no ablation reported |
| ScienceWorld | 3B, 7B, 8B | SFT, eval-gate | batch; LR; epochs; eval | 32; 5e-6; 1; one-shot prompt; max prompt 4096, max response 1024, temperature 0.4 | App. C.6 | verified 2026-09-14 | no ablation reported |
| TravelPlanner | 3B, 7B, 8B | SFT | D_expert; IWM; SR | 45 training trajectories → 1,395 pairs; all valid actions executed at each state → over 70,000 transitions; up to 30 alternatives per pair, reasoning generated by Llama-3.1-8B-Instruct at temperature 0.9, no extra filter | App. C.7 | verified 2026-09-14 | no ablation reported |
| TravelPlanner | 3B, 7B, 8B | SFT, eval-gate | epochs; LR; context; batch; eval | IL and IWM 5 epochs, 1e-5, cosine; SR max generation 8K tokens; 32K context; 16 per GPU on 8 H100 (ZeRO-3); 180 validation queries, vLLM greedy decoding | App. C.7 | verified 2026-09-14 | no ablation reported |
| WebArena-Lite | 3B, 7B, 8B | SFT | D_expert | 647 training tasks (812 − 165 eval); successful leaderboard trajectories (IBM CUGA, ScribeAgent, Learn-by-Interact, AgentOccam) → 554 trajectories, 7,044 pairs | App. C.8 | verified 2026-09-14 | not applicable |
| WebArena-Lite | 3B, 7B, 8B | SFT | IWM; SR; epochs; LR | 5 free-form alternatives + expert, model-summarized next state → 42,264 triplets; 3,190 SR examples after removing explanations that favor a non-expert action; 2 epochs, 1e-5, cosine | App. C.8 | verified 2026-09-14 | no ablation reported |
| Llama-3.3-70B-Instruct, Qwen-2.5-72B-Instruct, WebArena-Lite | 70B, 72B | SFT | adapter | LoRA for all methods with the same rank and update steps; IWM continues the same adapters in stage 2; rank not reported | §6.4; App. Table 9 | verified 2026-09-14 / rank not reported | no ablation reported |
| All GRPO runs | 3B, 7B, 8B | RL | KL coefficient, clip, samples per prompt, steps | not reported (checked §5.4, App. B-C) | §5.4 | not reported | none |

## Verification
- Created on 2026-09-14 from https://arxiv.org/abs/2510.08558 (v3, 2026-05-24).
- Audit claims not found in the source: none.
