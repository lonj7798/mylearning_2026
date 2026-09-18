<!-- scope: INTELLECT-3 technical report — a 100B+ MoE post-trained from GLM-4.5-Air-Base with two SFT stages and large-scale asynchronous RL on prime-rl, plus the verifiers / Environments Hub / Prime Sandboxes infrastructure the run used
     deps: [[glm-4-5]], [[grpo]]
     see-also: [[areal-async-rl]], [[async-rollout]], [[rollout-training-mismatch-tis]], [[verl-rollout]], [[skyrl-agent]], [[dapo]]
-->

# INTELLECT-3: Technical Report
- **Core Insight:** A fully asynchronous, disaggregated RL stack (trainer, CPU orchestrator, OpenAI-compatible vLLM pool) with continuous batching, in-flight weight updates and a bounded off-policy horizon trained a 100B+ MoE from GLM-4.5-Air-Base to AIME24 90.8, AIME25 88.0, LiveCodeBench v6 69.3, GPQA 74.4, HLE 14.6 and MMLU-Pro 81.9, above GLM-4.5-Air on every tested benchmark (Table 2), with in-loop benchmark scores still rising when the run stopped (§3.3, Fig. 9).
- **Guideline:** When rollouts are tens of thousands of tokens long and the inference pool would otherwise wait for the slowest one, use continuous batching with in-flight weight updates and discard rollouts spanning more than `max_off_policy_steps` policies, because the report measures a step time of about 1500 s at 65,536 sequence length with in-flight updating and more than 2× that without it (§3.3). When the trainer and the inference engine assign different probabilities to the same tokens, mask importance ratios outside [α, β] = [0.5, 5] instead of clipping them, and mask a whole rollout if any token ratio falls below 1e-5, because the report states this was needed to prevent runs crashing multiple days in (§3.3).
- **Authors:** Prime Intellect (organization; correspondence johannes@primeintellect.ai)
- **Year:** 2025 (arXiv v1 2025-12; companion blog post 2025-11-26)
- **URL:** https://arxiv.org/abs/2512.16144 — blog: https://www.primeintellect.ai/blog/intellect-3
- **Source type:** official technical report
- **Relevant topics:** asynchronous RL, off-policy masking, MoE training, environment interfaces, online difficulty filtering, sandboxed code execution, in-loop evaluation

## Summary
The report describes the post-training of INTELLECT-3 from the GLM-4.5-Air base model and the open infrastructure used: prime-rl (asynchronous RL framework), the verifiers environment library with the Environments Hub, and Prime Sandboxes for code execution. Both SFT stages and the RL stage, including ablations, ran on a 512 H200 cluster over two months (§3). Training data is a mixture of open environments covering math, code, science, logic, deep research and software engineering. The RL algorithm is a masked token-level importance-sampling objective with group-mean-centred advantages. The model is compared against GLM-4.5-Air, GLM-4.5, GLM-4.6, DeepSeek R1-0528, DeepSeek v3.2 and GPT-OSS 120B on six reasoning benchmarks.

## Key Contributions
- prime-rl architecture: a CPU orchestrator between an FSDP2 trainer and a vLLM inference pool with `/update_weights` and `/reload_weights` endpoints; the orchestrator uses verifiers environments so that any Environments Hub entry can be plugged into the loop (§2.1.1).
- Asynchronous off-policy training with continuous batching and in-flight weight updates, credited to AReaL and PipelineRL; a trajectory may span several policies, bounded by `max_off_policy_steps` (§2.1.2–2.1.3).
- Multi-client orchestration: one independent server per inference node with round-robin dispatch, after vLLM's multi-node data parallelism plateaued (§2.1.4).
- Online data filtering: easy/normal/hard difficulty pools by observed solve rate, plus discarding rollouts the model always solves or always fails (§2.1.5).
- Sequence-length scaling by CPU activation offloading (48k → 72k at the same hardware, MFU cost about 0.1%); context parallelism reached 256k with N_cp = 2 but halved data parallelism and degraded accuracy, so it was not used in production (§2.1.6).
- Distributed Muon over all-to-all collectives rather than overlapping gathers, using the Dion implementation (§2.1.7).
- Prime Sandboxes: a Rust gateway that bypasses the Kubernetes API for the execution path, push-based readiness webhooks, lazy image pulling and warm pools (§2.3).

## Key Figures/Tables to Study
- Table 1 (§3.2) — SFT data sources with example and token counts per stage.
- Table 2 (§4) — final benchmark comparison against six other models.
- Fig. 9 (§3.3) — in-loop evaluation every 15 steps on AIME25, AIME24, LiveCodeBench, HLE, GPQA.
- Fig. 10 (§3.3) — GSPO versus CISPO reward curves under an async-8 testbed.
- Fig. 8 (§3.2) — SFT loss curves for both stages.
- Eqs. 1–2 (§3.3) — the masked token-level importance-sampling objective.

## Technical Details
1. **Objective (§3.3, Eqs. 1–2).** J = E over rollouts of the mean over tokens of M(π_train/π_infer; α, β) · Â, where M(k) = k when k ∈ [α, β] and 0 otherwise; π_infer is the policy that generated the rollout and π_train the current trainer policy. The advantage is Â_i,t = S_i − mean({S_i}_G) with S_i the reward of rollout i and G the group size. Defaults α = 0.5, β = 5. A whole rollout is masked if any token ratio falls below 1e-5.
2. **Why masking rather than clipping (§3.3).** The report states that π_infer and π_train can produce significantly different token probabilities even with identical parameters, and that runs crashed multiple days in when this was not addressed; masking is chosen over CISPO-style clipping to avoid noisy updates from excessive importance ratios.
3. **Early algorithm ablation (§3.3, Fig. 10).** GSPO was compared against CISPO on an async-8 testbed; the report observes reward and all other metrics collapsing with GSPO. **Result (single study)**, no table of scores.
4. **In-loop evaluation (§3.3).** Scores plotted every 15 steps for AIME25 (avg@32), AIME24 (avg@32), HLE (avg@1), LiveCodeBench (avg@2) and GPQA (avg@4) over roughly 600 steps; they trend up without plateauing. Online evaluation is interleaved asynchronously on the same inference pool (§2.2.4).
5. **Environment mixture (§3.1).** Math 21.2K problems (Skywork-OR1, AceReason-Math, DAPO, ORZ-Hard) verified by math-verify plus CompassVerifier-7B as an LLM judge for answers marked wrong; code 8.6K examples with up to 15 test cases per problem and over 4000 concurrent sandboxes, completions masked on sandbox failure; science 29.3K from MegaScience; logic 11.6K across 29 tasks from SynLogic; deep research 1K SFT and 2.2K RL samples from DeepDive with search/click/open/finish tools; software engineering with R2E-Gym and mini-swe-agent-plus scaffolds, at most 200 turns, over 20,000 prebuilt repository images.
6. **Difficulty annotation (§3.1).** Solve rates are computed with Qwen3-4B (Thinking-2507 for math over 8 generations; Instruct-2507 for code over 8, and for science and logic over 16).
7. **Environment validation run (§3.1.5).** Qwen3-4B-Instruct-2507 trained on public DeepDive traces: SFT 26 steps at batch size 34 (884 samples), then 122 RL steps at group size 16 and total batch size 512; Fig. 7 shows mean reward rising.
8. **MoE handling (§2.1.8).** torchtitan grouped-GEMM MoE layers; expert parallelism was tested and not enabled because it lowered throughput at the run's sequence length and hidden dimension; expert load imbalance is logged as MaxViolation.
9. **Sandbox latency (§2.3.1, §2.3.3).** Naive Kubernetes `kubectl exec` orchestration measured 2.5 s per command at thousands of concurrent sandboxes; Prime Sandboxes reports a consistent end-to-end cold start under 10 seconds with arbitrary user images.
10. **Final scores (Table 2).** INTELLECT-3 / GLM-4.5-Air / GLM-4.5 / GLM-4.6 / DeepSeek R1-0528 / DeepSeek v3.2 / GPT-OSS 120B: AIME24 90.8 / 84.6 / 85.8 / 92.0 / 83.2 / 88.1 / 75.8; AIME25 88.0 / 82.0 / 83.3 / 90.3 / 73.4 / 84.7 / 77.7; LCB v6 69.3 / 61.5 / 64.5 / 73.0 / 62.5 / 71.6 / 69.9; GPQA 74.4 / 73.3 / 77.0 / 78.8 / 77.5 / 81.4 / 77.3; HLE 14.6 / 13.3 / 14.8 / 13.3 / 15.9 / 17.9 / 10.6; MMLU-Pro 81.9 / 73.9 / 83.5 / 83.1 / 75.3 / 84.6 / 67.1. All evaluations run through the public Environments Hub implementations, with provider APIs used for comparison models.

## Recipe ledger
| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| INTELLECT-3 (GLM-4.5-Air-Base) | 100B+ MoE | SFT | data sources; epochs; tokens per step; context length | Nemotron-Post-Training-Dataset-v1 math/code/science/tool + AM-DeepSeek-R1-0528-Distilled chat/IF at natural ratios; 1 epoch; ~33M tokens/step; 65K | arXiv:2512.16144 §3.2, Table 1 | verified 2026-09-18 | no ablation reported |
| INTELLECT-3 | 100B+ MoE | SFT | optimizer; LR; warmup; weight decay; sharding | Muon; 5e-5 warmed linearly from 1e-8 over 300 steps; 0.01; FSDP world size 64, DP replicate 8, 512 GPUs | §3.2 | verified 2026-09-18 | no ablation reported |
| INTELLECT-3 | 100B+ MoE | distill-SFT (agentic, stage 2) | data; epochs; steps; context; optimizer and LR | SWE-Swiss, Toucan Tool, synthetic traces from Environments Hub via DeepSeek-R1-0528; 2 epochs; 800 steps; 98K via context parallelism; Muon at 5e-8 decayed linearly | §3.2 | verified 2026-09-18 | no ablation reported; the 5e-8 value is printed as such in §3.2 |
| INTELLECT-3 | 100B+ MoE | SFT | stage-1 / stage-2 dataset sizes | OpenReasoning-Math 2M ex / 78.1B tok; -Code 1.9M / 94.3B; -Science 310K / 32B; -Tool 800K / 3.8B; AM General Chat 952K / 8.4B; AM Instruction Following 54K / 400M; stage 2 adds SWE Swiss 10.3K / 700M, Toucan Tool 116K / 700M, Environments Mix 38.4K / 1.9B | §3.2 Table 1 | verified 2026-09-18 | no ablation reported |
| INTELLECT-3 | 100B+ MoE | RL | prompts per step × rollouts per prompt; max context | 256 × 16; 65,536 | §3.3 | verified 2026-09-18 | no ablation reported |
| INTELLECT-3 | 100B+ MoE | RL | optimizer; LR; `max_off_policy_steps` | Muon; 1e-6; 8 | §3.3 | verified 2026-09-18 | stated as removing excessively off-policy rollouts; no sweep reported |
| INTELLECT-3 | 100B+ MoE | RL | IS mask band [α, β]; whole-rollout mask threshold | [0.5, 5]; 1e-5 | §3.3 Eqs. 1–2 | verified 2026-09-18 | report states double-sided masking was needed against trainer–inference mismatch; no ablation table |
| INTELLECT-3 | 100B+ MoE | RL | nodes; train:inference split; step time | 60 nodes × 8 H200; about 1:3 (16 training, 44 inference); ~1500 s/step at 65,536 length with in-flight weight updating | §3.3 | verified 2026-09-18 | §3.3: without in-flight weight updating, step time rises more than 2× |
| INTELLECT-3 | 100B+ MoE | RL | curriculum control | easy/normal/hard pools by observed solve rate; prompts with pass rate 1 never sampled again; always-fail/always-solve rollouts discarded | §2.1.5, §3.3 | verified 2026-09-18 | no ablation reported |
| INTELLECT-3 | 100B+ MoE | eval-gate | in-loop evaluation interval and suite | every 15 steps on AIME24, AIME25, LiveCodeBench, HLE, GPQA | §3.3, Fig. 9 | verified 2026-09-18 | §3.3, §4: scores still rising at the end of the run |
| INTELLECT-3 (whole program) | 100B+ MoE | all | cluster; duration | 512 H200; about two months including ablations | §3 | verified 2026-09-18 | no ablation reported |

## Findings relevant to generality, negative feedback and agentic training
- Breadth of the RL mixture: six domains (math, code, science, logic, deep research, software engineering) are routed through one EnvGroup with a task-ID column, so one run trains across all of them without orchestrator-side changes (§2.2.2, §3.1). **Result (single study).**
- Held-out measurement during training: HLE and GPQA are evaluated in-loop every 15 steps although the RL mixture does not target them directly, and both rise over the run (§3.3, Fig. 9).
- Negative signals: rollouts whose sandbox fails are masked out of the loss rather than scored as failures (§3.1.2, §3.1.6); prompts with a pass rate of 1 and rollouts the model always fails are removed for carrying no learning signal (§2.1.5, §3.3); tokens with importance ratios outside [0.5, 5] are masked to 0 rather than clipped (§3.3). Below-mean rewards inside a group still give negative advantages through Â_i,t (§3.3).
- Not reported: any safety, multilingual, instruction-following or chat-quality evaluation of the released checkpoint; per-domain reward curves; ablations of the data mixture; seeds.

## Connections
- [[areal-async-rl]] — credited in §2.1.3 as the origin of continuous batching with in-flight weight updates; bounds staleness with η where prime-rl uses `max_off_policy_steps`.
- [[rollout-training-mismatch-tis]] — the same sampler-versus-trainer probability gap that Eqs. 1–2 mask.
- [[glm-4-5]] — the base model's own report, which is where the parameter counts of GLM-4.5-Air come from.
- [[async-rollout]], [[verl-rollout]] — other framework implementations of asynchronous rollout with different bounds.
- [[dapo]] — one of the four math data sources in §3.1.1.

## Verification
- Created on 2026-09-18 from https://arxiv.org/abs/2512.16144 (arXiv v1, 2025-12-18), read alongside https://www.primeintellect.ai/blog/intellect-3 (2025-11-26).
- Corrections to the previous card version: none (new card).
- Removed as unsupported by the source: none. The blog's "500+ environments on the Environments Hub" figure is not stated in the report and is not used in this card.
- Chapter claims not found in the source: two labelling points in ch-58. (a) Its recipe rows call the model "106B MoE"; the report says only "100B+ parameters" on top of GLM-4.5-Air-Base, so the 106B/12B-active figures come from the GLM-4.5 report, not from this one. (b) Its "Starting point" paragraph describes the in-loop evaluation as running "at 106B MoE on 512 H200s"; §3.3 gives 60 nodes × 8 H200 (480 GPUs) for the RL run, with 512 H200 being the cluster used across all stages (§3). All other ch-58 claims — §2.1/§2.1.1 orchestrator and verifiers abstraction, `max_off_policy_steps` = 8, the [0.5, 5] and 1e-5 masking thresholds, the 16:44 node split, the ~1500 s step time and its > 2× increase, and the 15-step five-benchmark evaluation — were read at the stated loci.
- Not reported by the source: total RL steps (Fig. 9 axes run to about 600); KL coefficient or entropy control; clip ε (masking replaces clipping); reward weights per environment; dollar cost.
