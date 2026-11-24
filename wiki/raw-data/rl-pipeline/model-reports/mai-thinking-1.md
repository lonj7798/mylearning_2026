<!-- scope: MAI-Thinking-1 technical report (Microsoft AI, 2026), RL parts only: Sec. 3.1 recipe (GRPO objective, adaptive entropy control, outer clip, rewards, sampling, self-distillation, hyperparameters), Sec. 3.2–3.5 domain climbs, graders and consolidation, Sec. 3.6 Rocket v1 infrastructure (controller, workers, inference, numerics, weight transfer)
     deps: [[grpo]], [[async-rollout]]
     see-also: [[mai-rl-infrastructure]], [[truncated-importance-sampling]], [[fp16-train-infer-mismatch]], [[slime]], [[verl-rollout]], [[areal-async-rl]], [[rstar2-agent]], [[kimi-k2-agentic-data]]
-->

# MAI-Thinking-1: Building a Hill-Climbing Machine (RL recipe and Rocket v1)
- **Core Insight:** A from-scratch RL climb (no reasoning traces, no distillation) stays stable over thousands of steps by three mechanisms: two GRPO changes (entropy-targeted upper clip, hard outer ratio clip), self-distillation to restart climbs after collapses or base-model updates, and removal of train–inference numerical mismatch (Sec. 3, p. 30).
- **Guideline:** Train separate specialist climbs (STEM, agentic, helpfulness+safety) with one shared recipe, then consolidate by SFT on their traces plus a short final RL. Filter prompts by pass rate twice (early-exit, then full group), replay the sampler's top-p mask in training, extend max length in powers of two, and keep sampler and learner in BF16 with MoE routing replay.
- **Authors:** The Microsoft AI Team (Appendix A citation, p. 82).
- **Year:** 2026 (bibtex `urldate={2026-06-02}`; PDF created 2026-06-06).
- **URL:** https://microsoft.ai/pdf/mai-thinking-1.pdf (109 pages)
- **Source type:** model technical report
- **Relevant topics:** GRPO variants, entropy control, reward composition, pass-rate filtering, length curriculum, self-distillation, SWE environment building, tool-use environments, reward models, lexicographic/gated rewards, async RL infrastructure, weight transfer

## Summary
MAI-Thinking-1 is a 35B-active / 1T-total MoE, pre-trained on 30T tokens and mid-trained on 3.55T, 256K max context (Abstract; Sec. 1–2, p. 2, 5). The RL climb starts from a checkpoint "with no prior exposure to reasoning traces" (p. 30). Three specialists are trained with the same recipe but different prompts and rewards: STEM + competitive code (Sec. 3.2), agentic coding + tool use (Sec. 3.3), helpfulness + safety (Sec. 3.4). They are consolidated by SFT, then a final lightweight RL climb yields MAI-Thinking-1 (Fig. 12, p. 31; Sec. 3.5). Rocket, the RL framework, uses YOLO as learner and SGLang for inference; MAI built it because the open frameworks it cites (von Werra et al. 2020; Hu et al. 2024; Sheng et al. 2024, i.e. TRL, OpenRLHF, verl) did not serve "asynchronous RL across several thousand GPUs" (Sec. 3.6, p. 49). Reported results include 97.0% AIME 2025, 87.7% LiveCodeBench v6, 52.8% SWE-Bench Pro (Abstract; Table 11, p. 53). No web-search RL climb is described; the agentic climb covers SWE and general tool use only.

## Key Contributions
- Adaptive entropy control: integral controller on the upper clip bound toward target entropy, replacing an entropy bonus (Sec. 3.1.1, Eq. 7–8, Fig. 13).
- Outer ratio clip on all branches to stop gradient-norm spikes from the unclipped GRPO cases (Eq. 9).
- Self-distillation as a numerics reset and base-model hand-off, with ablation findings (Sec. 3.1.4, Fig. 15).
- Reward-combination rules for subjective domains: lexicographic shaping and gated application (Sec. 3.4.1).
- Rocket v1: 4,864 GB300 job with 4,096 inference / 768 learner chips; planned resharding weight transfer (Sec. 3.6.4–3.6.5).

## Key Figures/Tables to Study
Fig. 12 (p. 31, climb DAG); Fig. 13 (p. 32, entropy and k over 800 steps, H* = 0.3); Fig. 15 (p. 35, AIME 25 / LCB curves with self-distillation markers); Fig. 16 (p. 37, STEM Mix: open-ended 56.1%, proof 33.3%, MCQ 10.6%; mathematics 58.5%, physics 13.2%, chemistry 10.9%); Fig. 17 (p. 38, (q, a) extraction pipeline); Fig. 19 (p. 50, Rocket); Table 10 (p. 49, consolidation mixture).

## Technical Details
**Objective (Sec. 3.1.1, p. 31–33).** GRPO with token-level normalization over all tokens in the global batch (across DP ranks); advantage A_i = (R_i − mean)/std shared across tokens. Upper clip: clip(r, 1−ε, (1−ε)^−1 + k); k starts at 0 and updates k ← clip(k + δ·sign(H* − Ĥ), 0, k_max), with Ĥ an importance-weighted per-token entropy estimate. Outer clip: clip(r, r_min, r_max) applied to all branches, "similar in spirit to the dual-clip PPO".

**Reward (Sec. 3.1.2, p. 33).** R = R_task + w_lang·R_lang − w_len·R_len. R_lang = max(1 − α·n_non-English, 0), added because mixed-language CoTs "correlate with spikes in log-probability divergence between the training and inference policies". R_len = ρ_q·|y|/ℓ_max (ρ_q = pass rate), so easy problems are penalized more for length.

**Sampling (Sec. 3.1.3, p. 34).** Early exit: sample G_early responses, continue to G only if the early pass rate is in range; then a second pass-rate filter on the full group removes low-variance groups. Top-p mask from rollout is replayed in training (excluded logits set to −∞); without it, "catastrophic off-policy mismatch, causing divergence within a few training steps". Length curriculum: 8k start, doubled up to 128k.

**Hyperparameters (Sec. 3.1.5, p. 36–37).**

| Item | Value |
|---|---|
| Optimizer | AdamW β1 = β2 = 0.95, ε = 1e−15, no weight decay |
| LR | constant 1e−6; 9e−7 at higher lengths to reduce off-policiness |
| Batch | 7,040 after packing; ≤12,000 unpacked sequences |
| Clip | ε = 0.6, k_max = 2.5, δ = 0.25, H* = 0.3; r_max = 50, r_min = 0 |
| Rewards | w_lang = 0.5, α = 0.005; w_len = 0.25 up to 64k, 0 at 128k |
| Groups | G = 128, G_early = 16, early range [0.05, 0.8], full range [0.1, 0.8] |
| Sampling | top-p = 0.97 (larger p raises mask-transfer overhead) |
| Async | 5 gradient steps per inference update; drop rollouts >8 updates stale (40 steps) |
| MoE | dropless; load-balance coef 1e−5 in RL, 1e−2 in self-distillation |
| Self-distill SFT | GBS 2,048, seq 128k, wd 0.001, cosine LR 1.7e−5 → 5.2e−6, warmup 2%, dropout 0.15 |

**Self-distillation findings (Sec. 3.1.4, p. 34–36).** O(1M) traces suffice to match the teacher; more risks over-constraining exploration. Training on incorrect traces performs similarly to successful-only; later-climb traces from several checkpoints beat traces from the final checkpoint only; prompt diversity beats traces per prompt; random sampling beats shortest-trace and heuristic selection; mix mid-training data in to keep long-context behavior. Resuming from a pre-collapse checkpoint "is often not viable" because instabilities are embedded "many steps before the actual collapse".

**Domain recipes (Sec. 3.2–3.4).**

| Climb / domain | Prompt sources | Grader / reward |
|---|---|---|
| STEM (longest climb) | STEM Mix >5M samples, hardest part >550k (q, a) pairs, from textbooks, academic PDFs, forums, competition archives, vendors (p. 37–38) | SymPy-style verifier, AI judge, or tests; MCQ/proof rewritten to open-ended (3× + consensus); 4 solver tiers rate difficulty; blind grading drops suspect ground truths |
| Competitive code | 160k vendor/targeted problems, 17 languages, verified reference solutions (Sec. 3.2.2) | test cases with runtime and memory limits |
| SWE | 102M public PRs → 4.87M with linked issues → 2.08M built (42.8%) → 745,452 with F2P signal (15.3%) → 265,617 verified (5.5%) across 94,044 repos; plus synthetic tasks (BugPilot / SWE-Smith / SWE-Mirror style) (p. 42–43) | F2P + P2P hidden tests in the container; anti-hacking: network control, git history scrubbed past base commit, test files reset, LLM monitor |
| General tool use | human-curated + >150 synthetic environments, 130,000 tasks, often >50 tools per env (p. 43–44) | final-state, tool-pattern and answer graders; LLM judge per sub-task; cross-env graders for parallel calls, no duplicates, valid arguments |
| Instruction following | expert-written + synthetic from a constraint taxonomy, >40 domains, conflicting system/developer/user cases (Sec. 3.4.2) | rule checks + per-rubric binary LLM judge (averaged) + RM, combined lexicographically with IF rewards primary |
| Safety | vendor prompts, red-team, PyRIT/PAP/TAP attacks; borderline do-not-refuse slice (Table 8) | judges for policy compliance, engagement, style; safety-gated aggregator (non-compliant → minimum reward) |
| Honesty / style | vendor data, PII-filtered Copilot logs, synthetic; Arena conversations for style | 5-category factuality×confidence judge; style judge on 0/1/2 scale, applied only after verifiable and safety gates |

Mixing: the agentic climb is run jointly with STEM tasks, which "helps to stabilize the RL climb" and transfers positively; agentic tasks transfer "neither positively nor negatively" to STEM (p. 40). The RM is a post-trained MAI-Base-1 predicting k-way 1–5 scores as text, applied cyclically over k permutations; reward = P(score = 5) (p. 44–45). Consolidation SFT mixture by sample / token: STEM + coding 56% / 89%, agentic 11% / 9%, helpfulness + safety 33% / 2%; 4 epochs, LR max 1e−5 decayed 2×. Final RL uses 128k max length and keeps "a small proportion of STEM and coding data" because reasoning otherwise degrades (Sec. 3.5, p. 49).

**Rocket v1 (Sec. 3.6, p. 49–53).** Controller, problem workers and rollout workers are each a Python process / Ray actor; the controller supports on- and off-policy implementations, and "we primarily use off-policy RL for large runs". Problem worker: 16 early requests → 128 more → advantage postprocessing; it *retries* failed rollouts (contrast: the Base 2 post drops them, [[mai-rl-infrastructure]]). Grading runs on the rollout worker when isolated, on the problem worker when group-level (e.g. pairwise LLM comparison). Inference: up to 5:1 inference:learner GPUs; largest job 4,864 GB300 (4,096 inference, 768 learner); prefix caching off for single-turn (SWA eviction), 97–98% prefix-cache hit rate on multi-turn. Stability: replica self-watchdog, router circuit breaker, per-class liveness thresholds, step-progress watchdog, mid-flight retry on another replica. Numerics: "bf16 for both the learner and the inference engine", plus MoE routing replay and top-p mask replay. Weight transfer: a plan computed once at job start intersects learner and inference sharding layouts per parameter (source rank, destination rank, byte extent, dtype/layout ops); it targets one learner + one inference server and expands to live replicas; restricted to a subset of DP groups — "a 36-server fleet split four ways performs four 9-server transfers in parallel".

## Verification
Read from the 109-page PDF via `pdftotext -layout` on 2026-11-24: Sec. 1 (p. 1–2), Sec. 3–3.6 (p. 30–53), Sec. 6.2 (p. 59–60). All numbers quoted from those pages. Figure curves (Fig. 1, 13, 15) were not read beyond captions. Sections 2, 4–5 and appendices were not read except Appendix A.
