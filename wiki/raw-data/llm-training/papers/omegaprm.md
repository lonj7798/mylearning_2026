<!-- scope: OmegaPRM (Luo et al., 2024) — binary-search Monte Carlo estimation inside an MCTS-style tree to collect step-level labels for process reward model (PRM) training without human annotation
     deps: [[math-shepherd]], [[prm800k]]
     see-also: [[rstar-math]], [[step-dpo]], [[lets-verify]]
-->

# Improve Mathematical Reasoning in Language Models by Automated Process Supervision
- **Core Insight:** With a PRM trained on 1.5M automatically collected step labels, PRM-weighted majority voting over 64 samples reaches 69.4% on MATH500 for instruction-tuned Gemini Pro (about 51% base accuracy), compared with 67.2% for unweighted majority voting and 67.6% with a PRM800K-trained PRM (Abstract; §4.1, Table 1).
- **Guideline:** When questions have golden answers and a policy can sample full solutions, locate the first wrong step with binary search over Monte Carlo rollouts instead of estimating every step, because at equal compute this produced 15M data points versus 200K for per-step estimation (§4.4).
- **Authors:** Liangchen Luo, Yinxiao Liu, Rosanne Liu, Samrat Phatale, Meiqi Guo, Harsh Lara, et al. (Google DeepMind, Google)
- **Year:** 2024 (arXiv v1 2024-06; v2 2024-12)
- **URL:** https://arxiv.org/abs/2406.06592
- **Source type:** paper
- **Relevant topics:** process reward models, automated step labeling, Monte Carlo estimation, MCTS, weighted majority voting, math reasoning

## Abstract
Outcome reward models (ORMs) score only the final answer, which gives no reward or penalty to intermediate steps of long reasoning chains. Process supervision assigns rewards to intermediate steps, but existing ways to collect it rely on human annotation or on Monte Carlo estimation at every step, and both are expensive. The paper proposes OmegaPRM, a divide-and-conquer Monte Carlo Tree Search (MCTS) algorithm that finds the first error in a chain of thought by binary search and balances positive and negative examples. It collects over 1.5 million process-supervision annotations and trains PRMs on them. Combined with weighted self-consistency, the PRM improves instruction-tuned Gemini Pro from 51% to 69.4% on MATH500 and from 86.4% to 93.6% on GSM8K, and Gemma2 27B from 42.3% to 58.2% on MATH500 and from 74.0% to 92.2% on GSM8K. No human annotation is used.

## Key Contributions
- Binary-search Monte Carlo estimation that locates the first incorrect step with O(k log M) policy calls instead of O(kM) (§3.2).
- An MCTS adaptation that stores and reuses rollouts, selecting rollouts with a value-plus-exploration score (§3.3, Eqs. 2–3).
- 1.5M per-step annotations on MATH training questions, with no human labels (§4).
- Comparison of PRM objectives: pointwise soft label, pointwise hard label, and pairwise (§3.4, §4.3).
- Evaluation against PRMs trained on PRM800K and Math-Shepherd data with two policies (§4.1).

## Key Figures/Tables to Study
- Figure 2 (MC estimate, binary search, and the three MCTS stages), Figure 3 (accuracy vs number of samples), Table 1 (PRM comparison at k = 64), Table 2 (training objectives), Appendix A (question filtering), Appendix B (pairwise loss).

## Technical Details
**Monte Carlo estimate (§3.2, Eq. 1).** For question q and prefix x₁:ₜ, a completer policy samples k rollouts to a final answer.
`c_t = MonteCarlo(q, x₁:ₜ) = num(correct rollouts from step t) / num(total rollouts from step t)`
- c_t: estimated correctness of the prefix; a rollout is correct when its final answer equals the golden answer. The prefix is treated as correct if any rollout is correct (§3.2).

**Binary search (§3.2).** Following Lightman et al., labels up to the first incorrect step suffice.
1. Split the solution at midpoint m and roll out from x₁:ₘ.
2. If c_m > 0, the first half is taken as correct and the error is in the second half; if c_m = 0, the error is very likely in the first half.
3. Repeat on the erroneous half until the segment is short enough to be one step. Cost is O(k log M) versus O(kM), where M is the number of steps.

**Tree search (§3.3).** Each node stores N(s) (visit count), MC(s) (Eq. 1), and Q(s, r).
`Q(s, r) = α^(1 − MC(s)) · β^(len(r)/L)` (Eq. 2)
- α, β ∈ (0, 1] and L > 0 are constants; len(r) is the rollout length in tokens. The first factor is larger when MC(s) is close to 1, which prioritizes "supposed-to-be-correct wrong-answer" rollouts; the second factor penalizes long rollouts.
`U(s) = c_puct · sqrt(Σᵢ N(sᵢ)) / (1 + N(s))` (Eq. 3)
- c_puct sets the exploration level; Σᵢ N(sᵢ) sums visit counts over the states sᵢ of the rollout pool; the score first favours rarely visited rollouts and later high-Q rollouts (§3.3).
- Select: pop (s, r) = argmax [Q(s, r) + U(s)] from the pool of rollouts with 0 < MC(s) < 1. Binary search: locate the first error in that rollout; new rollouts with 0 < MC < 1 join the pool, and all split positions before the first error become new states. Maintain: N(s) += 1 for the selected pair; MC and Q are computed for new rollouts; there is no recursive backup to the root. Construction stops at the search limit or when the pool is empty.

**PRM training (§3.4).** Each single-step edge is one example; loss is the classification loss of Eq. 4 with prediction y = PRM(s, a).
- Pointwise soft label ŷ = MC(s) (as in MiPS); pointwise hard label ŷ = 1[MC(s) > 0] (as in Math-Shepherd); pairwise Bradley–Terry loss with normalized preference P(X preferred) = ½(1 + p − q) for MC values p, q (App. B, Eq. 6). Main results use the soft label (§3.4).

**Experimental settings (§4, App. A).**
- Data: MATH split of Lightman et al.: 12K training questions; test subset MATH500 of 500 problems. Search limit 100 per question → 1.5M per-step annotations.
- Hyperparameters: α = 0.5, β = 0.9, L = 500, c_puct = 0.125; k = 8 rollouts per MC estimate.
- Question filtering: 32 rollouts per training question; questions with no correct answer (too hard, false-negative risk) or no wrong answer (too easy, false-positive risk) are removed (App. A).
- Step granularity: any token span may be a step; the solution is divided into 16 pieces, and binary search stops when a step is shorter than average solution length / 16 (§4.2).
- Policies: Gemini Pro fine-tuned on math instruction data to about 51% on MATH; pretrained Gemma2 27B with a 4-shot prompt. All reward models are trained from pretrained checkpoints (§4).
- Metric: PRM-weighted majority voting, with the solution score equal to the product of step scores; 128 solutions generated per problem (§4; Fig. 3 caption).

**Results.**
- Table 1 (k = 64), MATH500 Gemini Pro / Gemma2 27B: majority vote 67.2 / 54.7; Math-Shepherd 67.2 / 57.4; Math-Shepherd (their implementation) 67.2 / 55.2; PRM800K 67.6 / 57.2; OmegaPRM 69.4 / 58.2. GSM8K: 92.7 / 90.6; 92.7 / 90.5; 91.8 / 91.4; 92.9 / 91.7; 93.6 / 92.2.
- As the number of samples grows, the other PRMs converge toward majority vote while OmegaPRM keeps a margin (§4.1, Fig. 3).
- Objectives (Table 2), step-classification accuracy on a small test set from MATH test: soft 70.1%, hard 63.3%, pairwise 64.2% (§4.3).
- Efficiency: same compute budget gives 200K data points (brute force) vs 15M (OmegaPRM), reported as 75×; 15M were randomly down-sampled to 1.5M for training (§4.4).
- MATH: 69.4% is an 18.4-point absolute and 36% relative gain over the 51% base (§6).

## Recipe ledger
| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| OmegaPRM data (Gemini Pro and Gemma2 27B policies) | — | reward-model | source questions | MATH train, 12K (Lightman et al. split) | arXiv:2406.06592v2 §4 | verified 2026-09-14 | no ablation reported |
| OmegaPRM data | — | reward-model | question filter | 32 rollouts per question; drop if 0 correct or 0 wrong | v2 App. A | verified 2026-09-14 | no ablation reported (stated purpose: reduce false positives/negatives) |
| OmegaPRM data | — | reward-model | rollouts per MC estimate k; search limit | 8; 100 per question | v2 §4 | verified 2026-09-14 | no ablation reported |
| OmegaPRM data | — | reward-model | α, β, L; c_puct | 0.5, 0.9, 500; 0.125 | v2 §4 (also v1 §4) | verified 2026-09-14 | no ablation reported |
| OmegaPRM data | — | reward-model | step split | 16 pieces per solution | v2 §4.2 | verified 2026-09-14 | step-length distribution similar to rule-based splits (Fig. 5) |
| OmegaPRM data | — | reward-model | annotations generated; used | 15M; 1.5M random down-sample | v2 §4.4 | verified 2026-09-14 | — |
| OmegaPRM PRM | not reported | reward-model | initialization; label | pretrained checkpoint; pointwise soft label MC(s) | v2 §4, §3.4 | verified 2026-09-14 | §4.3 Table 2: soft 70.1% > pairwise 64.2% > hard 63.3% step accuracy, single test set |
| Gemini Pro policy | not reported | SFT | math instruction tuning result | about 51% on MATH test | v2 §4 | verified 2026-09-14 | — |
| OmegaPRM PRM | — | reward-model | LR, batch, epochs, compute, PRM size | not reported | checked: v2 body §3–4, App. A–B; v1 | not reported | — |

## Findings relevant to generality, negative feedback
- Negative signals: a step is labeled incorrect when no rollout from the state after it reaches the golden answer (MC = 0); negatives are used as classifier labels for the PRM (Eq. 4), not as policy gradients (§3.2–3.4). Selection favours wrong-answer rollouts from states with MC close to 1 (Eq. 2), and the abstract states the algorithm balances positive and negative examples.
- Label noise: too-hard questions create false negatives and too-easy questions create false positives; filtering reduces but does not remove them, and "the precise impact of noise on PRM performance remains uncertain" (App. A; §5).
- Generality: all annotations come from MATH training questions; PRMs are evaluated on MATH500 and GSM8K (§4, Table 1). The method needs a question with a golden answer, so it does not apply to open-ended tasks without adaptation (§5).

## Connections
- [[math-shepherd]] — per-step Monte Carlo labeling with hard labels; used as a data baseline and reimplemented with the Gemini policy (§3.2, §4).
- [[prm800k]], [[lets-verify]] — human step labels and the MATH train/test split used here; source of the "supervise up to the first error" choice (§3.2, §4).
- [[step-dpo]] — another method built on locating the first wrong reasoning step; see that card for its use of the step.
- [[rstar-math]] — later tree-search approach to step-level signals; see that card for its method.

## Verification
- Checked on 2026-09-14 against: https://arxiv.org/abs/2406.06592 (v2, 11 Dec 2024; v1, 5 Jun 2024 compared).
- Corrections to the previous card version: title "OmegaPRM: Improve Mathematical Reasoning in LMs by…" → exact title above; author list lacked Meiqi Guo (added in v2); "lifts Gemini Pro's MATH score by 69.4% relative" → 51% → 69.4%, 18.4 points absolute, 36% relative (§6); "O(log L)" → O(k log M) vs O(kM) (§3.2); "if p_m ≈ p_0 recurse on second half" → c_m > 0 moves search to the second half, c_m = 0 to the first half (§3.2); "Math-Shepherd uses a single completion per step" → Math-Shepherd rolls out from every step (§3.2); "weighted best-of-N (PRM score × policy log-prob)" → PRM-weighted majority voting with product of step scores (§4); "PRM regresses onto MC via MSE" → classification loss with soft labels (Eq. 4); "wrong if MC < τ ≈ 0.2" → hard label is 1[MC(s) > 0] (§3.4); "Beats Math-Shepherd by ~5 MATH points" → 69.4 vs 67.2 (Gemini Pro), 58.2 vs 57.4 (Gemma2 27B) (Table 1); "Gemini Pro 1.0" → paper says "Gemini Pro"; "excludes trajectories whose first step fails / problems with no correct trajectory in K tries" → questions with 0 correct or 0 wrong answers in 32 rollouts are removed (App. A).
- Removed as unsupported by the source: "~80K problems, ~10 steps per trajectory"; "~100K TPU-hours"; "authors recommend K ≥ 16"; "K = 8–32 completions" guideline range; "seed trajectories 500–1500 tokens, 8–15 steps"; "labels more stable for deep trajectories"; "zero-supervision extensions are future work" wording (paper states adaptation to open-ended tasks is needed); "rStar-Math PPM is the direct successor" claim.
- Version note: v1 reports only Gemini Pro (Gemini Ultra distillation for the policy), a 128-sample comparison with MiPS as brute-force baseline, and lacks §4.4 and Appendices A–B; this card follows v2.
- Not reported by the source: PRM training hyperparameters, PRM and policy sizes for Gemini Pro, total compute, dataset release.
