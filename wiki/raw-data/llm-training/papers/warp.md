<!-- scope: WARP (Google DeepMind, arXiv:2406.16768): RLHF with three weight-merging stages (EMA of the policy as the KL anchor, SLERP of task vectors from independent REINFORCE runs, linear interpolation towards the initialization) applied iteratively on Gemma "7B"; KL-reward Pareto fronts, side-by-side scores vs Mistral/Mixtral, zero-shot benchmarks, length bias and diversity loss
     deps: [[warm-weight-averaged-reward-models]], [[kl-control-rlhf]], [[task-arithmetic]]
     see-also: [[gemma-2]], [[gemma-3]], [[model-soups]], [[mitigating-alignment-tax-rlhf]], [[rlhf-generalisation-diversity]]
-->

# WARP: On the Benefits of Weight Averaged Rewarded Policies
- **Core Insight:** On Gemma "7B", merging two REINFORCE policies by spherical interpolation of their task vectors and interpolating back towards the initialization gives KL-reward Pareto fronts above those of the individual RL runs, and the third iteration of this procedure scores GSM8K 66.8 and MATH 31.0 against 55.6 and 25.6 for the Gemma "7B" 1.1 release (Fig. 4, Table 2).
- **Guideline:** When KL regularization to the SFT model limits reward in RLHF and several parallel RL runs per round are affordable, merge the independently trained policies with SLERP and select a point on the interpolation towards the initialization as the next round's start, because every such front in the paper lies above the RL training trajectories (§4.3-4.4); returns fall after the third iteration (Table 1), and the RM is reliable only at low KL, so reward gains at high KL need an external check (§4).
- **Authors:** Alexandre Ramé, Johan Ferret, Nino Vieillard, Robert Dadashi, Léonard Hussenot, Pierre-Louis Cedoz, et al. (Google DeepMind)
- **Year:** 2024 (arXiv v1 2024-06-24; the only version; no venue stated)
- **URL:** https://arxiv.org/abs/2406.16768
- **Source type:** paper
- **Relevant topics:** RLHF, KL regularization, alignment tax, model merging, SLERP, task vectors, exponential moving average, iterative post-training, length bias, output diversity

## Abstract
RLHF usually adds KL regularization towards the SFT model to prevent forgetting of pre-trained knowledge, but this limits reward optimization. WARP merges policies in weight space at three stages: (1) the exponential moving average (EMA) of the policy is the anchor of the KL term; (2) spherical interpolation merges independently fine-tuned policies; (3) the merged model is linearly interpolated towards the initialization to recover features from pre-training. The procedure is repeated, with each iteration's result as the next initialization, and gives higher reward at fixed KL. Experiments with Gemma policies show improved quality and alignment over other open-source LLMs (Abstract).

## Key Contributions
- EMA anchor: θ_ema ← (1 − μ)·θ_ema + μ·θ_policy at every step, used as θ_anchor in the KL term, which relaxes the constraint as training proceeds (§3.1, Observation 1).
- SLERP of task vectors δ_m = θ_m − θ_init, applied layer by layer; it raises reward with a slight KL increase, while linear interpolation (LERP) mainly lowers KL (§3.2, Observations 2-3, App. C.1).
- Linear Interpolation Towards Initialization (LITI): θ_η ← (1 − η)·θ_init + η·θ_slerp traces a Pareto front by sliding η (§3.3, Observation 5).
- Iterative WARP: θ_η from one round (usually η = 0.3) becomes θ_init for the next (§3.4, Algorithm 1).
- Measurement that RL task vectors from separate runs are close to orthogonal (Ω ≈ 90°) while full weights are collinear (App. C.2, Fig. 11).

## Key Figures/Tables to Study
- Fig. 1(b), Fig. 4(c): Pareto fronts for five WARP iterations against REINFORCE with SFT and EMA anchors.
- Fig. 3(a-b): EMA anchor vs SFT anchor with β ∈ {0, 1e-4, 1e-3, 0.01, 0.1}; Fig. 3(c): SLERP vs LERP reward over λ.
- Fig. 4(b): LITI fronts when merging M = 1 to 5 policies.
- Table 1: side-by-side preference scores vs Mistral 7B v1, v2 and Mixtral 8x7B; Table 2: zero-shot benchmarks.
- Fig. 18: output length vs KL and the length-penalty merge; Fig. 19: similarity across generations vs KL.

## Technical Details
**Objective.** argmax_θ E_(x∈X) E_(y∼π_θ(·|x)) [ r(x, y) − β·KL(π_θ(·|x) ‖ π_θanchor(·|x)) ] (Eq. 1). r is the RM score, X the prompt set, β the KL strength; the per-sample reward is r(x, y) − β·log[π_θ(y|x) / π_θanchor(y|x)] (§2). Standard RLHF sets θ_anchor = θ_sft; WARP sets θ_anchor = θ_ema (§3.1).
**SLERP (M = 2).** slerp(θ_init, θ1, θ2, λ) = θ_init + sin[(1−λ)Ω]/sin Ω · δ1 + sin[λΩ]/sin Ω · δ2, where Ω is the angle between δ1 and δ2 in that layer and λ the interpolation coefficient (§3.2). For M > 2 the paper merges iteratively with λ = 1/M; the operation is not associative, and standard deviations are small (App. B.3, Fig. 4(b)). SLERP applied to full weights behaves like LERP because full weights are collinear (App. C.2, Fig. 10(c)).
**Setup.** Gemma "7B" fine-tuned with REINFORCE on conversation prompts, using the largest available RM; no oracle control RM is used (§4). Diversity across runs comes only from prompt order (§3.2).
**EMA results.** With an SFT anchor, β = 0 raises reward fastest but KL grows within a few steps (runs stopped at KL 200, after 1k steps for β = 0); β = 0.1 saturates near reward −0.62 and β = 0.01 near −0.46; the EMA-anchor front is above and to the left of the SFT-anchor runs, with β = 0.01 matching it only at low KL (§4.1, Fig. 3(a-b)). For the SFT-anchor runs, the EMA weights perform similarly to or better than the run's policy in KL-reward terms (App. D.2, Fig. 14).
**Merge results.** SLERP of two runs gives higher reward than either endpoint, highest at λ = 0.5 and T = 9k (Fig. 3(c)). LITI fronts lie above the RL trajectories for η ∈ {0, 0.1, 0.3, 0.5, 0.8, 1.0}, and longer runs help at both high and low KL (§4.3, Fig. 4(a)); larger M gives better fronts (Fig. 4(b)). LITI with M = 1, or with a moving average of one run's checkpoints at {6k, 7k, 8k, 9k}, gives smaller gains than merging M = 2 independent runs (App. D.1, Fig. 13). RL task vectors are near-orthogonal, whereas the cited supervised fine-tuning study reports 40° to 80° (App. C.2).
**Iteration results.** Five iterations of M = 2 runs, each iteration's LITI front above its RL trajectories, with diminishing returns (§4.4). Table 1 scores (vs Mistral 7B v1 / v2 / Mixtral 8x7B; ratings ±1.5, ±1, ±0.5, ties 0): Gemma "7B" 1.1 0.37 / 0.16 / 0.08; REINFORCE with EMA anchor 0.37 / 0.16 / 0.07; WARP iteration 1 0.42 / 0.23 / 0.13; iteration 3 0.45 / 0.26 / 0.18; iteration 5 0.45 / 0.24 / 0.17 (§4.5).
**Benchmarks (zero-shot, Table 2).** Gemma "7B" 1.1 → WARP iteration 3: MBPP 39.0 → 45.4, MMLU 56.4 → 57.6, GSM8K 55.6 → 66.8, MATH 25.6 → 31.0, HumanEval 46.9 → 50.0, BBH 53.1 → 58.8 (§4.5). The η of the evaluated checkpoint is not stated.

## Recipe ledger
| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| WARP policy (Gemma "7B") | "7B" | RL | algorithm; prompts | REINFORCE variant on KL-regularized reward; "conversation prompts" (count not reported) | arXiv:2406.16768v1 §2, §4, Alg. 1 | verified 2026-09-14 | no ablation in this paper; §2 cites [2, 80, 126] for REINFORCE over PPO, DPO, IPO, RAFT |
| WARP policy (Gemma "7B") | "7B" | RL | sampling temperature; batch; optimizer; LR; warmup | 0.9; 128; Adam; 1e-6; 100 steps | §4 | verified 2026-09-14 | no ablation reported |
| WARP policy (Gemma "7B") | "7B" | RL | steps per run T | 9k default; iterations 1 / 2-3 / 4-5: 9k / 7k / 5k | §4; §4.4 | verified 2026-09-14 | Fig. 4(a): longer T better; later T reduced "for computational reasons" |
| WARP policy (Gemma "7B") | "7B" | RL | KL strength β (EMA anchor) | 0.1 | §4; App. D.2 | verified 2026-09-14 | Fig. 15: reducing μ to 0.005 or raising β to 0.2 behaves similarly, marginally better front, slower training |
| WARP policy (Gemma "7B") | "7B" | RL | EMA update rate μ | 0.01 | §3.1; §4; App. D.2 ("systematically used ... for all EMA-based runs") | conflict | §4.1 prose says μ = 0.1 for Fig. 3; App. D.2 states 0.01 was used for all EMA runs |
| WARP policy (Gemma "7B") | "7B" | RL | EMA update rate μ | 0.1 | §4.1 prose | conflict | contradicted by §3.1, §4, App. D.2 |
| WARP merge | "7B" | merge | method; M; λ | SLERP of task vectors per layer (28 layers); M = 2; λ = 0.5 (λ = 1/M for M > 2) | §4; App. B.3 | verified 2026-09-14 | Fig. 3(c): best reward at λ = 0.5; Fig. 4(b): M up to 5; App. C.1: SLERP vs LERP |
| WARP merge | "7B" | merge | LITI η for next init | 0.3 | §3.4; §4 | verified 2026-09-14 | App. D.3, Fig. 16: η = 0.5 better at high KL, η = 0.3 better below KL 65 |
| WARP merge | "7B" | merge | LITI target | the iteration's own initialization | §4.4; App. D.4 | verified 2026-09-14 | Fig. 17: fronts similar to interpolating towards SFT |
| WARP iterations | "7B" | merge | number of iterations I | 5 | §4.4 | verified 2026-09-14 | Table 1: scores stop improving after iteration 3 |
| Length-penalty run (App. E) | "7B" | RL | length penalty | −0.0005 × len(y) added to reward | App. E | verified 2026-09-14 | Fig. 18(b-c) |
| Reward model | not reported | reward-model | size; data | "a high capacity reward model, the largest available"; details not reported | §4 | not reported (checked §4, App. A-F) | n/a |
| all | "7B" | all | SFT recipe; compute; tokens | not reported | checked §3-§6, App. A-F | not reported | n/a |

## Findings relevant to generality, negative feedback, long context, agentic training, distillation
- **Generality and forgetting.** The paper uses KL to the SFT policy as its measure of forgetting pre-trained knowledge (§4), and LITI is motivated by recovering generalizable pre-training features (§3.3; Interpretation). Table 2 shows gains on all six zero-shot benchmarks for iteration 3 vs Gemma "7B" 1.1; the paper does not state whether Gemma "7B" 1.1 and the WARP runs share an SFT starting point.
- **Reward hacking and length.** Output length rises with KL, and iteration 3 is longer than iteration 1 at the same KL (App. E, Fig. 18(a)). Merging a length-penalized policy with an unpenalized one reduces length and improves the front (Fig. 18(b-c)).
- **Diversity.** BLEURT similarity between two samples at temperature 0.9 correlates positively with KL, measured on REINFORCE checkpoints and on LITI interpolations towards the SFT model (App. F, Fig. 19).
- **Distillation.** The authors describe the EMA anchor as KL distillation from a dynamic mean teacher (§3.1; Interpretation).
- **Limits.** Each iteration needs M RL runs, so training compute grows (§3, §6); results use one model size and one RM (§4).

## Connections
- [[warm-weight-averaged-reward-models]]: the reward-side merging paper; WARP is "conceived as a response to WARM" (§5).
- [[gemma-2]]: Gemma 2 §4 averages models from pipeline runs with different hyperparameters and cites this paper.
- [[gemma-3]]: Gemma 3 §3 lists "improved versions of BOND, WARM, and WARP" for its RL phase.
- [[task-arithmetic]]: task vectors (ref. [53]) that SLERP interpolates.
- [[model-soups]]: linear weight averaging of fine-tunings (ref. [137]); WiSE-FT (ref. [138]) is the supervised form of LITI.
- [[ties-merging]], [[dare-merging]]: cited as methods that reduce interference in multi-task merging with sparse task vectors (§5).
- [[mitigating-alignment-tax-rlhf]]: Lin et al. (ref. [81]) use merging to reduce alignment tax without EMA during training, without merging multiple rewarded policies, and not iteratively (§5).
- [[rlhf-generalisation-diversity]]: Kirk et al. (ref. [65]) on RLHF diversity loss, confirmed in App. F.
- [[rloo]]: Ahmadian et al. (ref. [2]), cited for REINFORCE outperforming PPO in RLHF.
- [[reward-model-overoptimization]]: Gao et al. (ref. [33]), cited for RM hacking far from the SFT policy.
- [[rlhf-length-correlations]]: Singhal et al. (ref. [119]); the length-penalty form follows this work.

## Verification
- Created on 2026-09-14 from https://arxiv.org/abs/2406.16768 (v1, 2024-06-24; full PDF including App. A-F). Gemma 2 citation checked against arXiv:2408.00118 §4; Gemma 3 statement checked against arXiv:2503.19786 §3.
- Audit claims not found in the source: none.
- Not reported by the source: RM size and training data, prompt count, SFT recipe, compute, the η of the Table 1-2 checkpoints.
