<!-- scope: one row per rl-pipeline card written in the infra pass: slug, one-line core insight, outline chapters served (wiki/courses/rl-pipeline/outline.json)
     see-also: [[mai-rl-infrastructure]], [[mai-thinking-1]]
-->

# rl-pipeline insights

Rows for cards written by the second agent (search-r1, asearcher, deepresearcher, websailor-tongyi-deepresearch, minimax-m1, big-math, recipe-crosswalk) are appended by that agent.

| Slug | Core insight (one line) | Chapters |
|---|---|---|
| [[mai-rl-infrastructure]] | RL climbed from base for weeks must bound staleness (sampler lag + age / step time), match sampler and learner bitwise, and recover at the smallest scope; Table 1 estimates 17.37% of 2,490 h saved | ch-01, ch-02, ch-03, ch-04, ch-05, ch-06, ch-07, ch-08, ch-11, ch-15, ch-17, ch-20, ch-21, ch-22 |
| [[mai-thinking-1]] (infra) | Rocket v1: 4,864 GB300 job split 4,096 inference / 768 learner; BF16 sampler and learner with routing and top-p mask replay; planned resharding weight transfer over DP-group subsets | ch-01, ch-02, ch-05, ch-07 |
| [[mai-thinking-1]] (objective, sampling) | From-scratch GRPO stays stable with an entropy-targeted upper clip, an outer ratio clip (r_max = 50), two pass-rate filters (G_early = 16, G = 128), top-p = 0.97 mask replay, 8k→128k length doubling, and a 40-step staleness cutoff | ch-03, ch-10, ch-18 |
| [[mai-thinking-1]] (tasks, graders) | STEM (q, a) pairs from PDFs with MCQ/proof rewritten open-ended; SWE funnel 4.87M PRs → 265,617 verified envs with F2P/P2P tests and anti-hacking scrubs; >150 synthetic tool envs / 130,000 tasks | ch-08, ch-09, ch-11, ch-16 |
| [[mai-thinking-1]] (reward composition, IF, style) | Subjective rewards combine RM + rubric judges + verifiable checks by lexicographic priority or safety gating, not by weighted sums; IF uses rule checks + per-rubric judges with IF primary; style judge on a coarse 0/1/2 scale | ch-12, ch-14, ch-19 |
| [[mai-thinking-1]] (stage order, mixture) | Three specialist climbs (STEM, agentic + STEM, helpfulness + safety) → consolidation SFT (sample weight 56/11/33%, token weight 89/9/2%) → short final RL at 128k keeping some STEM/code; self-distillation restarts climbs after collapse; no web-search climb | ch-17 |
| [[thinking-machines-nondeterminism]] | Inference nondeterminism comes from kernels that are not batch-invariant under varying load; batch-invariant RMSNorm/matmul/fixed-split attention give 1,000/1,000 identical completions and 0 sampler–trainer KL in RL | ch-07 |
| [[flash-attention-stability]] | Flash Attention's per-tile rescaling gives ~10× more BF16 forward deviation than standard attention, growing with sequence length and shrinking with tile size; weight-level effect is 2–5× below FP16-vs-FP32 training | ch-07 |
| [[nemo-rl-tp-accuracy]] | TP accuracy loss in NeMo RL came from unequal train / log-prob micro-batch sizes (batch-variant kernels) and row-wise TP cross-GPU sums; fix by equal batch sizes and a colwise plan for o_proj/down_proj | ch-07 |
| [[deepseek-train-infer-consistency]] | DeepSeek-V4 builds end-to-end batch-invariant, deterministic kernels (dual-kernel decode attention, DeepGEMM without split-K, deterministic backward), rolls out with native FP4 after QAT, and resumes preempted rollouts from a token WAL to avoid short-length bias | ch-07, ch-06, ch-20 |
| [[truncated-importance-sampling]] | vLLM-sampler / FSDP-learner gaps make on-policy RL off-policy; multiplying by min(π_learner/π_sampler, C) at θ_old, outside the PPO clip, restores INT8-rollout runs; gap grows with parallelism differences and response length | ch-07, ch-03, ch-06, ch-18 |
| [[fp16-train-infer-mismatch]] | BF16 rounding is the main source of train–inference mismatch; FP16 in both engines cuts the sequence-level mismatch ~24× and lets plain IS policy gradient beat BF16 TIS/MIS/GSPO | ch-07, ch-06, ch-18 |
