<!-- scope: Recipe ledger for DeepSWE-Preview (Qwen3-32B, RL only): values from the Together AI / Agentica blog (2025-07-02), the HF model card, and the released rLLM training script at commit 709dec43b740; companion to [[deepswe]]
     deps: [[deepswe]]
     see-also: [[r2e-gym]], [[r2e-gym-recipe]], [[dapo]], [[dr-grpo]]
-->

# DeepSWE-Preview — Recipe ledger
Companion to [[deepswe]]. Three sources are recorded separately (§5.3 rule 4 of the course standard):
- **Blog**: https://www.together.ai/blog/deepswe, sections numbered as in the post (read 2026-09-14).
- **Model card**: https://huggingface.co/agentica-org/DeepSWE-Preview (read 2026-09-14).
- **Script**: github.com/agentica-project/rllm@709dec43b74056b5ed61970d2ec4eff0de7e17af `scripts/agent/swe/deepswe_32b.sh` (identical to `examples/swe/train_deepswe_32b.sh` at the same commit, tagged "v0.1" in the commit message, 2025-07-02). Line numbers refer to that file. The only later change to `examples/swe/train_deepswe_32b.sh` (commit dc85be21ab31, 2025-07-12) sets `trainer.nnodes=8` instead of 2.

The script and the blog disagree on batch size, GPU count, step limit, and trajectory timeout. The blog describes the final training run; the repository does not state that the script reproduces that run with these values, although its README says it provides "the exact scripts to replicate our training curves". Rows with `conflict` keep both values. The rLLM docs page (docs/examples/swe.md at the same commit) labels the script "Train with 16K context", which matches neither the blog nor the script's 32768-token response length.

Units: "problems" are R2E-Gym tasks (prompts with a Docker environment). "Samples per problem" are full multi-turn trajectories. The blog's "BS=64, 8 passes" gives 512 trajectories per iteration.

| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| DeepSWE-Preview | 32B | RL | Base model and mode | Qwen/Qwen3-32B, thinking mode enabled; no SFT stage | Blog intro; model card "DeepSWE Overview"; Script L22 | verified 2026-09-14 | Blog §6: non-thinking mode gave "limited performance improvement"; RL after SFT on Claude-Sonnet 3.7/4 trajectories did not improve after 100 iterations (no numbers) |
| DeepSWE-Preview | 32B | RL | Training problems | 4.5K problems from R2E-Gym subset; repositories shared with SWE-Bench-Verified (e.g. sympy) removed; HF dataset R2E-Gym/R2E-Gym-Subset | Blog §2.1; model card dataset link; Script L14 | verified 2026-09-14 | Blog §6: SWE-Smith and SWE-Gym gave limited improvement with high solve-none rate (no numbers) |
| DeepSWE-Preview | 32B | RL | Reward | 1 if patch passes a selected sample of Pass2Pass and Fail2Pass tests within 5 minutes; 0 if any test fails or times out | Blog §2.2 "Reward" | verified 2026-09-14 | no ablation reported |
| DeepSWE-Preview | 32B | RL | Algorithm | GRPO++: clip high, no KL loss, no reward std, length normalization by max context length, leave-one-out advantage, compact filtering, no entropy loss | Blog §2.3 | verified 2026-09-14 | Blog Figure 5 (FrozenLake, GRPO++ vs GRPO); Figure 6 (Qwen3-14B, compact filtering on/off); curves only |
| DeepSWE-Preview | 32B | RL | Problems per iteration × trajectories per problem | 64 × 8 (512 containers per iteration, final run) | Blog §2.2 "Kubernetes" | conflict | no ablation reported |
| DeepSWE-Preview | 32B | RL | Problems per iteration × trajectories per problem | data.train_batch_size=8; actor_rollout_ref.rollout.n=8 | Script L16, L49 | conflict | no ablation reported |
| DeepSWE-Preview | 32B | RL | PPO mini-batch | ppo_mini_batch_size=8 (equal to train batch, one update per iteration) | Script L27 | verified 2026-09-14 | no ablation reported |
| DeepSWE-Preview | 32B | RL | Compute | 64 H100 GPUs, six days | Blog intro | conflict | n/a |
| DeepSWE-Preview | 32B | RL | Compute | trainer.nnodes=2 × n_gpus_per_node=8 (16 GPUs); nnodes=8 (64 GPUs) after commit dc85be21ab31 | Script L62-L63 | conflict | n/a |
| DeepSWE-Preview | 32B | RL | Training steps | about 200 ("With just 200 steps of RL training") | Blog Figure 2 caption; model card | verified 2026-09-14 | Blog Figure 2: SWE-Bench-Verified Pass@1 23% → 42% |
| DeepSWE-Preview | 32B | RL | Planned schedule | trainer.total_epochs=1000; save and test every 10 steps | Script L64-L65, L73 | verified 2026-09-14 | n/a; stop point is not stated in the script |
| DeepSWE-Preview | 32B | RL | Learning rate | 1e-6 | Script L24 | verified 2026-09-14 (script only; blog does not print it) | no ablation reported |
| DeepSWE-Preview | 32B | RL | Clip range | clip_ratio_high=0.28; lower clip not set in the script | Script L34 | verified 2026-09-14 (script only) | no ablation reported |
| DeepSWE-Preview | 32B | RL | KL | use_kl_loss=False; algorithm.kl_ctrl.kl_coef=0.001 is set, but the in-reward KL call is commented out in `rllm/trainer/verl/agent_ppo_trainer.py` L327-L330 at this commit | Script L33, L35, L54; trainer file | verified 2026-09-14 | Blog §2.3 gives the reason (not constraining to the SFT model's trust region); no ablation reported |
| DeepSWE-Preview | 32B | RL | Entropy coefficient | 0.0 | Script L53; Blog §2.3 "No Entropy Loss" | verified 2026-09-14 | Blog §2.3: entropy loss led to rising entropy and collapse; no figure or number |
| DeepSWE-Preview | 32B | RL | Advantage estimator | loop (leave-one-out); clip_advantages=False | Script L13, L56 | verified 2026-09-14 | no ablation reported |
| DeepSWE-Preview | 32B | RL | Loss aggregation | loss_agg_mode=seq-mean-token-sum (blog: divide by max context length); mapping between the two not checked, since the verl fork code is not in the repository tree at this commit | Script L26; Blog §2.3 | verified 2026-09-14 (as printed) | no ablation reported |
| DeepSWE-Preview | 32B | RL | Masked trajectories | blog: max context, max environment steps, 20-minute generation timeout; script: agent.overlong_filter=True, algorithm.mask_truncated_samples=False | Blog §2.3; Script L55, L70 | verified 2026-09-14 | Blog Figure 6 (Qwen3-14B): compact filtering prevents or delays reward collapse |
| DeepSWE-Preview | 32B | RL | Trajectory timeout | 20 minutes (generation) | Blog §2.3 | conflict | no ablation reported |
| DeepSWE-Preview | 32B | RL | Trajectory timeout | agent.trajectory_timeout=5400 | Script L71 | conflict | no ablation reported |
| DeepSWE-Preview | 32B | RL | Max environment steps (training) | not reported in blog; agent.max_steps=50 | Script L69 | verified 2026-09-14 (script only) | no ablation reported |
| DeepSWE-Preview | 32B | RL | Max prompt / response length | not reported in blog for training; max_prompt_length=4096, max_response_length=32768 | Script L18-L19 | verified 2026-09-14 (script only) | no ablation reported |
| DeepSWE-Preview | 32B | RL | Rollout sampling | temperature 1.0; validation n=1 at temperature 0 | Script L47, L50-L51 | verified 2026-09-14 | no ablation reported |
| DeepSWE-Preview | 32B | RL | Parallelism | FSDP with parameter and optimizer offload; Ulysses sequence parallel 8; vLLM tensor parallel 8 (async) | Script L37, L40-L44 | verified 2026-09-14 | n/a |
| DeepSWE-Verifier | not reported | reward-model | Training | execution-free verifier trained 2 epochs on correct and incorrect patches; base model and data size not printed | Blog §3 | verified 2026-09-14 | Blog Figure 10: hybrid TTS 59.0% at K=16 |
| DeepSWE-Preview | 32B | eval-gate | Evaluation settings | R2E-Gym codebase, 64k max context, 100 max environment steps; patches scored in the official SWE-bench repository; Pass@1 averaged over 16 runs | Blog §4 | verified 2026-09-14 | n/a |
| DeepSWE-Preview | 32B | eval-gate | Recommended inference | temperature 1; max tokens at least 32-64K; R2E-Gym system prompt and tools | Model card "Usage Recommendations" | verified 2026-09-14 | Blog Figure 9: gain beyond 32K max output tokens ≤2% |

Not reported in any of the three sources: optimizer betas, weight decay, warmup, gradient clipping, lower clip bound, the step count of the released checkpoint beyond "200 steps", verifier base model.

Starting point for a small general-purpose run: no row supports a small-scale default. The verified values (LR 1e-6, clip_ratio_high 0.28, 8 trajectories per problem, temperature 1.0, no KL loss, entropy coefficient 0) come from the released script for Qwen3-32B on R2E-Gym SWE tasks with 16 or 64 GPUs; the batch size, step limit, and timeout in that script conflict with the blog's final run.
