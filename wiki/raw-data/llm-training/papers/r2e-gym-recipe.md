<!-- scope: Recipe ledger for R2E-Gym (arXiv:2504.07164v1): data collection limits and SFT hyperparameters for the editing agents (Qwen-2.5-Coder 7B/14B/32B), the testing agent (32B), and the execution-free verifier (14B), plus test-time sampling settings; companion to [[r2e-gym]]
     deps: [[r2e-gym]]
     see-also: [[deepswe-recipe]], [[swe-gym]]
-->

# R2E-Gym — Recipe ledger
Companion to [[r2e-gym]]. All rows cite arXiv:2504.07164v1 (read 2026-09-14). The paper trains three models, each with its own row group: the code-editing agent ("R2E-Gym-7B/14B/32B"), the testing agent, and the execution-free (EF) verifier. The paper does not train with RL.

Units: "trajectories" are full multi-turn agent episodes; "steps" are agent actions within one trajectory; "environments" are R2E-Gym tasks. "Batch size 8" is printed without stating whether it is per device or global. Token limits during data collection ("32K max tokens per-trajectory") and SFT max context length ("20K") are separate settings.

| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| R2E-Gym editing agent | 7B, 14B, 32B | distill-SFT | Base model | Qwen-2.5-Coder 7B, 14B, 32B | §3; App. B | verified 2026-09-14 | Table 3: 19.0 / 26.8 / 34.4 Pass@1 on SWE-Bench-Verified |
| R2E-Gym editing agent | 7B, 14B, 32B | distill-SFT | Task pool | R2E-Gym-Subset: 4,578 environments, 10 repositories, no repository overlap with SWE-Bench ("4538" in App. B) | §2; §3; Table 1; App. B | conflict (4,578 in §2, §3, Table 1; 4538 in App. B) | n/a |
| R2E-Gym editing agent | 7B, 14B, 32B | distill-SFT | Teacher and sampling | Sonnet-3.5-v2 with the paper's scaffold; temperature 0.2 | §3; App. B | verified 2026-09-14 | no ablation reported |
| R2E-Gym editing agent | 7B, 14B, 32B | distill-SFT | Quality control | rejection sampling with R2E-Gym unit tests (synthetic and existing); successful trajectories only | §3; App. B | verified 2026-09-14 | §3.1: synthetic 27.8% vs real issues 28.0% Pass@1 at 400 trajectories |
| R2E-Gym editing agent | 7B, 14B, 32B | distill-SFT | Examples | 3,321 trajectories from 2,048 environments | §3; App. B | verified 2026-09-14 | Figure 2: 100 to 3,200 trajectories; 14B begins to saturate at approximately 800 |
| R2E-Gym editing agent | 7B, 14B, 32B | distill-SFT | Collection limits | max 40 steps; 32K tokens per trajectory; 10-min trajectory timeout; 90 s per action | App. B | verified 2026-09-14 | no ablation reported |
| R2E-Gym editing agent | 7B, 14B, 32B | distill-SFT | Training targets | agent thoughts and actions | §3; §3.1 | verified 2026-09-14 | Figure 3: 34.4% with thoughts vs 30.4% without (text: 34.2%) |
| R2E-Gym editing agent | 7B, 14B, 32B | distill-SFT | Method and framework | full SFT, LLaMA-Factory | App. B | verified 2026-09-14 | n/a |
| R2E-Gym editing agent | 7B, 14B, 32B | distill-SFT | Epochs / batch / LR / warmup | 2 epochs; batch size 8; LR 1e-5; warmup ratio 0.1 | App. B | verified 2026-09-14 | no ablation reported |
| R2E-Gym editing agent | 7B, 14B, 32B | distill-SFT | Max context length | 20K ("due to computational constraints") | App. B | verified 2026-09-14 | no ablation reported |
| R2E-Gym editing agent | 7B, 14B, 32B | distill-SFT | LR decay shape, optimizer, weight decay, packing, loss masking of observations | not reported | checked §3, App. B | not reported | n/a |
| Testing agent | 32B | distill-SFT | Base model | Qwen-Coder-32B | §4.1; App. C.1 | verified 2026-09-14 | Figure 8: agent tests 51.0% vs Agentless tests 48.8% in the hybrid |
| Testing agent | 32B | distill-SFT | Examples | 2,203 Sonnet-3.5 test-generation trajectories, positive and negative, "minimal rejection sampling" | App. C.1 | verified 2026-09-14 | no ablation reported |
| Testing agent | 32B | distill-SFT | Collection limits | max 40 steps; 20K tokens per trajectory; 5-min trajectory timeout; 60 s per action | App. C.1 | verified 2026-09-14 | no ablation reported |
| Testing agent | 32B | distill-SFT | Training | full fine-tuning, LLaMA-Factory; 2 epochs; batch size 8; LR 1e-5; max context 20K; warmup ratio 0.1 | App. C.1 | verified 2026-09-14 | no ablation reported |
| Testing agent | 32B | eval-gate | Output | one test script with M = 10 tests | §4.1; App. C.1 | verified 2026-09-14 | no ablation reported |
| EF verifier | 14B | reward-model | Base model and output | Qwen2.5-Coder-14B; YES/NO tokens; score P(YES)/(P(YES)+P(NO)) | §4.1; App. C.2 | verified 2026-09-14 | Figure 4: plateau at 42.8% Best@K |
| EF verifier | 14B | reward-model | Data | 5,700 trajectories, equal positives and negatives; editing-agent SFT trajectories plus on-policy trajectories from the trained 32B model | App. C.2 | verified 2026-09-14 | Figure 7a: patch-only input 37.6% vs 42.8% Best@26 |
| EF verifier | 14B | reward-model | Training | LoRA rank 64, LLaMA-Factory; 2 epochs; batch size 8; LR 1e-5; max context 32K; warmup ratio 0.1 | App. C.2 | verified 2026-09-14 | no ablation reported |
| R2E-Gym-32B | 32B | eval-gate | Editing rollouts for TTS | 1 at T = 0 plus 25 at T = 0.8 and 0.9 (Pass@26 = 64.4%) | §4.2; Figure 14 | verified 2026-09-14 | Figure 4: Best@K against rollouts; 16 → 21 rollouts gives 47.6% → 48.4% |
| R2E-Gym-32B | 32B | eval-gate | Testing-agent sampling | "7 tests" at T = 0.8; fixed Django in-context example | §4.2 | verified 2026-09-14 | §4.2: the in-context example improves test generation for ∼2% of problems; §4.4: 5 more test rollouts give 49.3% |
| R2E-Gym-32B | 32B | eval-gate | Selection rule | hybrid s^H = Top_n(s^EF) + s^EB; regression filtering after top-n; n not reported | §4.3, Eq. 2 | verified 2026-09-14 (n: not reported) | Figure 8: Top-n 49.8% → 51.0%; regression-only 47.4%; final 51.0% Best@26 |

Starting point for a small general-purpose run: the verified SFT rows give 2 epochs, LR 1e-5, warmup ratio 0.1, and batch size 8 for full SFT of Qwen-2.5-Coder 7B-32B on 3,321 SWE agent trajectories with a 20K max context. These values were not ablated in the paper and apply to agent-trajectory SFT from a single teacher on 10 Python repositories.
