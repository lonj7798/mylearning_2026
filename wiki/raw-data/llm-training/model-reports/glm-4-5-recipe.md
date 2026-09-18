<!-- scope: Recipe ledger for the GLM-4.5 technical report (arXiv:2508.06471v1); companion to [[glm-4-5]]
     deps: [[glm-4-5]]
     see-also: [[deepseek-v3-recipe]], [[grpo]]
-->

# GLM-4.5: Agentic, Reasoning, and Coding (ARC) Foundation Models — Recipe ledger
Companion to [[glm-4-5]]. Status dates refer to reading the arXiv v1 PDF (the only version) on 2026-09-14. The report has no appendix; "not reported" means the body (§1-§5), figure labels, and table captions were checked.

Scope rules for this table:
- Pre-training and mid-training rows are labeled "GLM-4.5" because §2.2-§2.3 and Figure 3 describe GLM-4.5. The report does not state whether GLM-4.5-Air used the same token counts or hyperparameters; do not transfer them to GLM-4.5-Air.
- The report uses a cosine learning-rate schedule with no stable phase (§2.4), so pre-training rows use the stage label `pretrain` instead of `pretrain-stable`.
- Rows labeled "smaller experimental model" come from the §3.2 ablation curves, which the report states are "based on our smaller experimental model, not on GLM-4.5". Its size is not reported.
- Token counts in Figure 3 are figure labels; the report prints no unique-vs-seen distinction for them.

| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| GLM-4.5 | 355B total / 32B act. | pretrain | Architecture | 3 dense + 89 MoE layers + 1 MTP layer; hidden 5120; 96 attention heads (head dim 128), 8 KV heads; 160 routed experts, 8 active, 1 shared; MoE intermediate 1536; QK-Norm | arXiv:2508.06471v1 Table 1, §2.1 | verified 2026-09-14 | §2.1: deeper, narrower models "exhibited better reasoning capacity"; 96 heads did not lower training loss but improved MMLU and BBH (no numbers printed) |
| GLM-4.5-Air | 106B total / 12B act. | pretrain | Architecture | 1 dense + 45 MoE layers + 1 MTP layer; hidden 4096; 96 heads, 8 KV heads; 128 routed experts, 8 active, 1 shared; no QK-Norm | Table 1 | verified 2026-09-14 | no ablation reported |
| GLM-4.5 | 355B / 32B act. | pretrain | Tokens, stage 1 (general documents, mainly web) | 15T at 4K sequence length | Figure 3 label; §2.2 | verified 2026-09-14 | no ablation reported |
| GLM-4.5 | 355B / 32B act. | pretrain | Tokens, stage 2 (code, math, science up-sampled) | 7T at 4K | Figure 3 label; §2.2 | verified 2026-09-14 | no ablation reported |
| GLM-4.5 | 355B / 32B act. | pretrain | Total training tokens | 23T | Abstract | verified 2026-09-14 | no ablation reported |
| GLM-4.5 | 355B / 32B act. | pretrain | Web quality up-sampling | top quality bucket contributes over 3.2 epochs; lowest bucket discarded | §2.2 | verified 2026-09-14 | no ablation reported |
| GLM-4.5 | 355B / 32B act. | pretrain | Fill-In-the-Middle | applied to all source code data; rate not printed | §2.2 | verified 2026-09-14 (rate not reported) | no ablation reported |
| GLM-4.5 | 355B / 32B act. | pretrain | Mixture percentages | not printed | §2.2 | not reported (body and figures checked) | — |
| GLM-4.5 | 355B / 32B act. | mid-train | Repo-level code (files, issues, PRs, commits in diff-like format) | 500B at 32K | Figure 3 label; §2.3 | verified 2026-09-14 | no ablation reported |
| GLM-4.5 | 355B / 32B act. | mid-train | Synthetic reasoning data (math, science, coding competitions) | 500B at 32K | Figure 3 label; §2.3 | verified 2026-09-14 | no ablation reported |
| GLM-4.5 | 355B / 32B act. | long-context | Long-context and agent data (long documents up-sampled, synthetic agent trajectories) | 100B at 128K | Figure 3 label; §2.3 | verified 2026-09-14 | no ablation reported |
| GLM-4.5 | 355B / 32B act. | pretrain / mid-train | Max sequence length | 4,096 in pre-training; 32,768 then 131,072 in mid-training | §2.3, §2.4 | verified 2026-09-14 | no ablation reported |
| GLM-4.5 | 355B / 32B act. | pretrain / mid-train | Packing | no best-fit packing in pre-training (random truncation used as data augmentation); best-fit packing in mid-training | §2.3 | verified 2026-09-14 | rationale stated, no ablation reported |
| GLM-4.5 | 355B / 32B act. | pretrain | Optimizer | Muon for all parameters except word embedding, bias, RMSNorm weights; Newton-Schulz steps N = 5; momentum μ = 0.95; update RMS scaled to 0.2 | §2.4 | verified 2026-09-14 | §2.4: Muon "can accelerate convergence and tolerate larger batch sizes" (no numbers) |
| GLM-4.5 | 355B / 32B act. | pretrain | Optimizer for excluded parameters; gradient clipping | not printed | §2.4 | not reported | — |
| GLM-4.5 | 355B / 32B act. | pretrain → mid-train | LR schedule | cosine decay; warm-up 0 → 2.5e-4; decay to 2.5e-5 at the end of mid-training; warm-up length not printed | §2.4 | verified 2026-09-14 | §2.4: early WSD runs scored worse on SimpleQA and MMLU (no numbers) |
| GLM-4.5 | 355B / 32B act. | pretrain | Batch size | 16M → 64M tokens over the first 500B tokens, then constant | §2.4 | verified 2026-09-14 | no ablation reported |
| GLM-4.5 | 355B / 32B act. | pretrain | Regularization | weight decay 0.1; no dropout | §2.4 | verified 2026-09-14 | no ablation reported |
| GLM-4.5 | 355B / 32B act. | mid-train | RoPE base frequency | 10,000 → 1,000,000 when extending to 32K | §2.4 | verified 2026-09-14 | no ablation reported |
| GLM-4.5 | 355B / 32B act. | pretrain | Loss-free balance routing bias update rate | 0.001 for the first 15T tokens; 0.0 afterwards | §2.4 | verified 2026-09-14 | no ablation reported |
| GLM-4.5 | 355B / 32B act. | pretrain | Auxiliary sequence-level balance loss weight | 0.0001 | §2.4 | verified 2026-09-14 | no ablation reported |
| GLM-4.5 | 355B / 32B act. | pretrain | MTP loss weight λ | 0.3 for the first 15T tokens; 0.1 afterwards | §2.4 | verified 2026-09-14 | no ablation reported |
| GLM-4.5 | 355B / 32B act. | SFT | Cold-start SFT data (per expert) | "a small set" with extended CoT responses; size not printed | §3.1 | not reported (size) | — |
| GLM-4.5 | 355B / 32B act. | distill-SFT | Overall SFT data | "millions of samples" from the Reasoning, Agent, and General-chat experts, plus long-context tasks | §3.1 | verified 2026-09-14 | no ablation reported |
| GLM-4.5 | 355B / 32B act. | distill-SFT | Max context length | 128K tokens | §3.1 | verified 2026-09-14 | no ablation reported |
| GLM-4.5 (model not named for ablation) | not reported | distill-SFT | Prompt selection | drop bottom 50% of prompts by response length | §3.1 | verified 2026-09-14 | §3.1: +2%-4% on math and science tasks with half the data (benchmarks not named) |
| GLM-4.5 (model not named for ablation) | not reported | distill-SFT | Responses per hard prompt | 4 | §3.1 | verified 2026-09-14 | §3.1: additional +1%-2% (benchmarks not named) |
| GLM-4.5 | 355B / 32B act. | SFT | Learning rate, epochs, batch size, loss masking | not printed | §3.1 | not reported | — |
| GLM-4.5 | 355B / 32B act. | RL | Reasoning RL algorithm | GRPO framework, KL loss term excluded | §3.2 | verified 2026-09-14 | no ablation reported |
| smaller experimental model | not reported | RL | Max output length | single stage at 64K | §3.2, Figure 6 | verified 2026-09-14 | Figure 6: 83.4% (single-stage 64K) vs 80.6% (16K→32K→48K→64K) AIME 24 Avg@32 |
| smaller experimental model | not reported | RL | Difficulty curriculum; samples per prompt | stage 1 moderate difficulty, samples_per_prompt = 16; stage 2 extremely difficult (pass@8 = 0, pass@512 > 0) from a verified-answer pool, samples_per_prompt = 512 | Figure 5 legend and caption; §3.2 | verified 2026-09-14 | Figure 5: 83.4% (curriculum) vs 81.8% (moderate data only) AIME 24 Avg@32 |
| GLM-4.5 | 355B / 32B act. | RL | Sampling temperature rule | raise temperature when average rollout reward stabilizes; next temperature = maximum value whose held-out validation drop is ≤ 1% from the current optimum; initial value not printed | §3.2 | verified 2026-09-14 (initial value not reported) | no ablation reported |
| smaller experimental model | not reported | RL | Code RL loss aggregation | token-weighted mean (vs sequence-mean) | §3.2, Figure 7 left | verified 2026-09-14 | Figure 7 left: faster convergence on LiveCodeBench (data labels 46.5% and 46.3%) |
| smaller experimental model | not reported | RL | Science RL data | expert-verified multiple-choice questions only | §3.2, Figure 7 right | verified 2026-09-14 | Figure 7 right: GPQA-Diamond data labels 65.8% vs 62.9%; caption states expert-verified data is higher than mixed-quality data |
| GLM-4.5 | 355B / 32B act. | RL | Agentic RL objective | group-wise policy optimization over K traces from π_old; reward minus group-mean reward; loss on model-generated tokens only | §3.3.2 | verified 2026-09-14 (K not reported) | no ablation reported |
| GLM-4.5 | 355B / 32B act. | RL | Agentic reward | web search: final-answer accuracy; SWE: verifiable test cases; incorrect tool-call format halts the trace with reward 0 | §3.3.2 | verified 2026-09-14 | no ablation reported |
| GLM-4.5 | 355B / 32B act. | RL | Holistic RL prompts | roughly 5,000 prompts; 7 primary, 33 secondary, 139 tertiary categories | §3.4 | verified 2026-09-14 | no ablation reported |
| GLM-4.5 | 355B / 32B act. | RL | Instruction-following RL taxonomy and feedback | 7 major and 151 minor constraint types; rules + reward model + critique model; GRPO | §3.4, Figure 9 | verified 2026-09-14 | Figure 9 (instruction-following RL run alone): no clear reward hacking up to about 1,000 steps |
| GLM-4.5 | 355B / 32B act. | RL | RL learning rate, clip ε, KL coefficient for non-reasoning RL, prompts per step, number of steps | not printed | §3.2-§3.4 | not reported | — |
| GLM-4.5 | 355B / 32B act. | RL | Rollout precision | BF16 training; FP8 inference with online block-wise FP8 quantization per policy update | §3.5 | verified 2026-09-14 | no ablation reported |
| GLM-4.5 | 355B / 32B act. | eval-gate | AIME 24 / GPQA sampling | Avg@32 / Avg@8; LLM-based answer validation; HLE text-only questions judged by GPT-4o | §4.2.2 | verified 2026-09-14 | — |
| GLM-4.5 | 355B / 32B act. | eval-gate | SWE-bench Verified harness | OpenHands v0.34.0; 100 iterations; history truncation at 128K; temperature 0.6; top_p 1.0 | §4.2.3 | verified 2026-09-14 | — |
| GLM-4.5 | 355B / 32B act. | — | Compute (GPU type, GPU hours) | not printed | whole report | not reported | — |

Starting point for a small general-purpose run: this report gives no small-model recipe. The only small-model values are the §3.2 ablation settings of the "smaller experimental model" (single-stage RL at the target output length, two-stage difficulty curriculum, token-weighted mean loss), and the size and data of that model are not reported.
