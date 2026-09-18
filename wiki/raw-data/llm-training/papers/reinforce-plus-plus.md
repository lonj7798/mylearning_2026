<!-- scope: REINFORCE++ (arXiv:2501.03262) — critic-free PPO-clip objective with global (batch-level) advantage normalization; k = 1 variant for general RLHF and a group-mean "w/ Baseline" variant with k2 KL loss for reasoning and tool use
     deps: [[vanilla-pg]], [[ppo]], [[grpo]], [[rloo]]
     see-also: [[dr-grpo]], [[rloo-vs-grpo]], [[kl-control-rlhf]], [[openrlhf-ppo]], [[prorl]]
-->

# REINFORCE++: Stabilizing Critic-Free Policy Optimization with Global Normalization
- **Core Insight:** Replacing GRPO's prompt-level advantage normalization with normalization over the whole training batch gave, in the authors' runs, a Chat-Arena-Hard score tied with GRPO at k = 1 instead of k = 4 (46.7 vs 46.8, Table 1) and a higher tool-use average than GRPO and PPO for the group-sampling variant (24.10 vs 22.58 and 21.85, Table 4).
- **Guideline:** When only one response per prompt can be sampled or scored (general RLHF, PRM-based rewards), the authors recommend REINFORCE++ with k = 1 and symmetric −1/1 rewards; when k > 1 samples and 0/1 rewards are used, they recommend the w/ Baseline variant (group-mean subtraction, then global std) (§5.1). Each comparison is a single run without reported seeds.
- **Authors:** Jian Hu, Jason Klein Liu, Haotian Xu, Wei Shen (v9); v1 was by Jian Hu alone
- **Year:** 2025 (arXiv v1 2025-01-04, titled "REINFORCE++: A Simple and Efficient Approach for Aligning Large Language Models"; v9 2025-11-10; no venue listed)
- **URL:** https://arxiv.org/abs/2501.03262 (implementation: https://github.com/OpenRLHF/OpenRLHF, per v1 abstract)
- **Source type:** paper
- **Relevant topics:** critic-free RLHF, advantage normalization, KL estimators (k1, k2, k3), PPO-clip, RLVR, tool-use RL, overfitting to small prompt sets

## Abstract
PPO uses a critic network to estimate advantages, which adds compute and memory cost. Critic-free methods such as GRPO and RLOO remove the critic but normalize advantages within the small group of responses for one prompt. The authors argue that this prompt-level normalization gives inaccurate advantages, encourages overfitting, and is a biased estimator. REINFORCE++ normalizes advantages across the global batch, which they describe as effectively unbiased as batch size grows. Two variants are proposed: REINFORCE++ (k ≥ 1) for general-domain RLHF, and REINFORCE++ w/ Baseline (k > 1) for complex reasoning. The authors report better stability and performance than existing methods in each domain, and higher scores than PPO in agentic settings.

## Key Contributions
- A proof that the GRPO advantage (r_i − mean)/std is biased for any finite group size N ≥ 2, because the local std depends on r_i (App. A.1, Theorem 1).
- REINFORCE++: PPO-clip objective with a k1 KL penalty inside the reward and advantages normalized over the global batch (§3.1, Eqs. 4–5, Algorithm 1).
- REINFORCE++ w/ Baseline: subtract the group-mean reward, normalize by global batch statistics, and add a separate k2 KL loss (§3.2, Eqs. 6–8).
- An argument that k2 is the correct separate-loss KL estimator for reverse KL and that k3 (used in GRPO) estimates forward KL (App. B.1).
- Experiments on general RLHF, small-dataset overfitting, Knights and Knaves puzzles, RL from a math base model, and multi-turn tool use (§4).

## Key Figures/Tables to Study
- Fig. 1: PPO, ReMax, GRPO, RLOO, and REINFORCE++ pipelines side by side.
- Fig. 2: training reward and KL divergence, REINFORCE++ (k = 1) vs GRPO.
- Table 2 and Fig. 3: GRPO vs REINFORCE++ trained on 30 AIME-24 questions.
- Table 4: tool-use benchmarks (average@32) for GRPO, PPO, and REINFORCE++ w/ Baseline.
- App. A (bias proof) and App. B.1 (k1/k2/k3 analysis).

## Technical Details
- **Objective.** L_PPO(θ) = E[(1/|o|) Σ_t min(s_t(θ)A_t, clip(s_t(θ), 1 − ε, 1 + ε)A_t)], s_t(θ) = π_θ(o_t | q, o_<t) / π_θold(o_t | q, o_<t) (Eq. 1). q: prompt; o: response; ε: clip threshold.
- **REINFORCE++ advantage.** A_{q,o_t} = r(o_1:T, q) − β Σ_{i=t}^{T} KL(i), KL(t) = log[π_θold(o_t | q, o_<t) / π_ref(o_t | q, o_<t)] (Eq. 4). r: sequence reward; β: KL coefficient; π_ref: reference policy.
- **Global normalization.** A^norm = (A − mean(A over D_batch)) / (std(A over D_batch) + ε) (Eq. 5). The batch is "typically large (e.g., 1024 or more)" (§3.1); this is an example, not a reported run setting.
- **w/ Baseline.** A′ = R − mean_group(R) (Eq. 6), then A^norm = (A′ − mean_batch(A′)) / (std_batch(A′) + ε) (Eq. 7); loss L = L_PPO(A^norm) − λ·J_k2, J_k2 = E[½ (log π_θ/π_ref)²] (Eq. 8). The coefficient λ is not reported.
- **Relation to PPO.** w/ Baseline equals PPO with the critic removed, GAE λ = 1 and γ = 1, and two-step global normalization as the baseline (§3.3).
- **Batch construction.** k = 1: a global batch of N samples has N prompts. k = 4: "a global batch of size N = 1024 might consist of 1024/4 = 256 unique prompts"; mean and std are over all N samples (App. B.2, example).
- **Local-normalization failure modes argued in §2.2.** (1) bias (App. A); (2) with k = 4 or 8, near-equal rewards drive the local std toward zero and the advantage "explodes"; (3) the policy is rewarded for beating other samples of the same prompt, which the authors link to overfitting on easy prompts.

## Recipe ledger
Loci are arXiv:2501.03262 v1 (2025-01-04) or v9 (2025-11-10) as marked. v9 reports no optimizer, LR, batch, clip, or KL values for its runs.

| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| v1 runs: Llama3.1-8B-SFT and Qwen2.5-7B-Instruct (per-model split not stated) | 8B / 7B | RL | KL coefficient β (in reward) | 0.01 (general), 0.001 (mathematics) | v1 §4.2 Table 1 | verified 2026-09-14 | no ablation reported |
| v1 runs (as above) | 8B / 7B | RL | clip ε | 0.2 | v1 §4.2 Table 1 | verified 2026-09-14 | no ablation reported |
| v1 runs (as above) | 8B / 7B | RL | samples per prompt | 4 | v1 §4.2 Table 1 | verified 2026-09-14 | no ablation reported |
| v1 runs (as above) | 8B / 7B | RL | rollout batch; training batch (units not stated) | 256; 128 | v1 §4.2 Table 1 | verified 2026-09-14 | no ablation reported |
| v1 runs (as above) | 8B / 7B | RL | maximum samples | 25,000 | v1 §4.2 Table 1 | verified 2026-09-14 | no ablation reported |
| v1 runs (as above) | 8B / 7B | RL | actor LR; critic LR (critic used by the PPO baseline; not stated) | 5 × 10⁻⁷; 9 × 10⁻⁶ | v1 §4.2 Table 1 | verified 2026-09-14 | no ablation reported |
| v1 runs (as above) | 8B / 7B | RL | discount γ | 1.0 | v1 §4.2 Table 1 | verified 2026-09-14 | no ablation reported |
| v1 LLaMA3 8B run | 8B | RL | training time, 70k samples, H100 (GPU count not stated) | PPO 60 h; REINFORCE++ 42 h | v1 §5.2 Table 2 | verified 2026-09-14 | n/a |
| v9 general-RLHF policy (Llama-3-8B-SFT) | 8B | RL | reward model; prompts | Bradley-Terry RM trained on ~700K preference pairs; 20,000 prompts | v9 §4.1 | verified 2026-09-14 | no ablation reported |
| v9 general-RLHF policy | 8B | RL | samples per prompt | REINFORCE++ k = 1; GRPO k = 4; RLOO k = 4; ReMax k = 1 + 1 | v9 §4.1 | verified 2026-09-14 | Table 1: 46.7 vs 46.8 (GRPO) on Chat-Arena-Hard |
| v9 ZeroTIR run (Qwen 2.5 Base 7B) | 7B | RL | framework; data; metric | OpenRLHF; ORZ and DAPO datasets; average@32 | v9 §4.3 | verified 2026-09-14 | Table 4 |
| v9 w/ Baseline runs | not stated | RL | k2 KL loss coefficient λ | not reported | v9 §3.2, App. B | not reported (body, App. A–C checked) | n/a |

## Findings relevant to generality, negative feedback, and agentic training
- **General RLHF (Result, single study).** Chat-Arena-Hard score / mean length: REINFORCE++ k = 1 46.7 / 832; GRPO k = 4 46.8 / 860; RLOO 44.6 / 866; ReMax 45.1 / 805 (v9 Table 1). GRPO's KL rose faster than REINFORCE++'s in Fig. 2, which the authors interpret as reward hacking (Interpretation, §4.1).
- **Overfitting on a small prompt set (Result, single study; model not stated).** Trained on 30 AIME-24 questions: GRPO 95.0 train pass@1, AIME-25 pass@1 0.0 and pass@16 0.4; REINFORCE++ (k > 1) 71.0, 2.5, 40.0 (v9 Table 2). §4.2 does not state whether group-mean subtraction was used in this run.
- **Difficulty OOD (Result, single study; model not stated).** Knights and Knaves: GRPO competitive at 2–3 people; REINFORCE++ higher at 4+ people; average 62.1 vs 55.7 (v9 §4.2, Fig. 4).
- **RL from a math base (Result, single study).** Qwen2.5-Math-Base (size not stated) on MATH splits: AIME-24 pass@8 21.04 vs GRPO 18.96; AMC-23 pass@8 60.47 vs 59.22; in-distribution MATH-500 pass@1 72.00 vs 73.00 (v9 Table 3).
- **Agentic tool use (Result, single study).** average@32: w/ Baseline 24.10, GRPO 22.58, PPO 21.85; GRPO is highest on AIME 24 (31.66 vs 30.83) (v9 Table 4).
- **Negative rewards.** Plain REINFORCE++ "performs best with symmetric rewards, such as −1/1"; group-mean subtraction lets w/ Baseline use 0/1 or −1/1 (§5.1). For multi-turn tool calling, the authors cite open-source reports that group-mean subtraction stabilizes training with many "void" samples (§5.1, anecdotal as presented).
- **Third-party reports cited by the authors (not verified here).** ScaleRL (16,000 GPU-hours) found batch-level normalization "slightly superior"; LitePPO and DLER also report global std as more stable (§5.2).

## Connections
- [[grpo]] — the prompt-level normalization this paper argues is biased (Eq. 3, App. A).
- [[rloo]] — leave-one-out baseline compared in §2.1 and Table 1.
- [[ppo]] — objective reused in Eq. 1 and baseline in Table 4.
- [[dr-grpo]] — separate analysis of GRPO's std-normalization bias.
- [[kl-control-rlhf]] — background for the k1-in-reward vs k2/k3-as-loss choice in App. B.1.
- [[openrlhf-ppo]] — the framework used for the experiments (§4).
- [[prorl]] — co-authored by Jian Hu; a separate prolonged-RL study.

## Verification
- Checked on 2026-09-14 against: https://arxiv.org/abs/2501.03262 v9 (2025-11-10; full text incl. App. A–C) and v1 (2025-01-04).
- Corrections: title and authors → v9 title and four authors (v1 title and sole author recorded above); "Figure 3: variance of advantages, group vs global" → v9 Fig. 3 shows training curves on the 30-question set and no variance plot exists in v1 or v9; "Table 2: RLHF results vs PPO, GRPO, RLOO" → v9 Table 1 compares GRPO, RLOO, ReMax (no PPO) on Chat-Arena-Hard, and Table 2 is the small-dataset test; "KL coef β 0.01–0.05" → 0.01 general, 0.001 math (v1 Table 1); "LR 5e-7 – 1e-6" → actor LR 5 × 10⁻⁷ (v1 Table 1); "k = 1–4" → v1 used 4, v9 used k = 1 for REINFORCE++ in §4.1; "Use REINFORCE++ when you can only afford k = 1–2" → the §5.1 recommendation above; the old card omitted the w/ Baseline variant and its k2 KL loss.
- Removed as unsupported: "Global batch size 512–2048 sequences", "Epochs per rollout 1", "Sampling T 1.0" (in neither v1 nor v9); the family comparison table rows on RLOO clip and RLOO KL placement (not stated in the paper); "trains stably on reasoning benchmarks" for v1 (v1 reports only Figures 1–3 and a time table).
- Internal inconsistencies in the source: Eq. 4 makes the advantage depend on t through Σ_{i=t}^{T} KL(i), while App. B.2 sets the advantage to 0 for t < T and to the normalized reward at t = T. v1 §4.1.1 names Llama3.1-8B-SFT but its footnote links OpenRLHF/Llama-3-8b-sft-mixture.
- Not reported by the source: seeds or variance across runs, LR/batch/clip/β for any v9 run, models for the Table 2 and Knights-and-Knaves runs, λ for the k2 loss.
