<!-- scope: WARM (Google DeepMind, arXiv:2401.12187): weight-averaging M reward models fine-tuned from a shared pre-trained/SFT model on TL;DR preferences; linear mode connectivity, weight averaging vs prediction ensembling under distribution shift and 25% label corruption; best-of-N and REINFORCE RLHF experiments with PaLM-XXS RMs and PaLM-XS policies
     deps: [[bradley-terry-rm]], [[reward-model-overoptimization]]
     see-also: [[warp]], [[reward-ensembling]], [[model-soups]], [[rlaif-scaling]], [[gemma-3]]
-->

# WARM: On the Benefits of Weight Averaged Reward Models
- **Core Insight:** Averaging the weights of M reward models (RMs) fine-tuned from the same pre-trained model gives one RM that, used for RL on TL;DR summarization, produces a policy with a 79.4% oracle win rate against a policy trained with the best single RM (§5.2, Fig. 9(c)), with the inference cost of one RM (§3.1).
- **Guideline:** When the RL reward is a learned RM trained on noisy offline preferences and several RM fine-tunings from one pre-trained checkpoint are affordable, average their weights instead of their predictions, because on out-of-distribution (OOD) preference pairs weight averaging matched or exceeded prediction ensembling and memorized fewer corrupted labels (§4.1-4.2, Figs. 3-5); prediction ensembling remains the option when the RMs have different architectures or pre-trainings, or when an uncertainty estimate from disagreement is needed (§6).
- **Authors:** Alexandre Ramé, Nino Vieillard, Léonard Hussenot, Robert Dadashi, Geoffrey Cideron, Olivier Bachem, et al. (Google DeepMind)
- **Year:** 2024 (arXiv v1 2024-01-22; the WARP paper's reference [108] lists it as ICML 2024; the arXiv PDF states no venue)
- **URL:** https://arxiv.org/abs/2401.12187
- **Source type:** paper
- **Relevant topics:** reward modeling, reward hacking, weight averaging, linear mode connectivity, label noise in preferences, distribution shift, best-of-N, RLHF with KL regularization

## Abstract
RLHF can lead to reward hacking, where the policy exploits errors in the RM to obtain high reward without meeting the intended objective. The authors identify two causes: distribution shift between the offline preference data and the policy's generations during RL, and inconsistent (noisy) human preference labels. WARM fine-tunes several RMs and averages them in weight space, relying on the observation that fine-tuned weights stay linearly mode connected when they share pre-training. Weight averaging is more efficient than ensembling predictions and improves reliability under distribution shift and robustness to label inconsistency. On summarization with best-of-N and RL, a policy RL-fine-tuned with WARM has a 79.4% win rate against a policy RL-fine-tuned with a single RM (Abstract).

## Key Contributions
- WARM: initialize each RM from SFT weights plus a linear-probed classifier, run M fine-tunings with different hyperparameters and data orders, and use φ_WARM = (1/M) Σ_i φ_i as the RL reward (§3.1).
- Empirical check of linear mode connectivity (LMC) for RMs trained on binary preferences: interpolated RMs are at least as accurate on OOD pairs as the interpolation of the two accuracies (Observation 1, Eq. 2, Fig. 3).
- "Baklava" diversity: RM featurizers initialized from different checkpoints of one SFT run (§3.3, Fig. 2).
- Second-order analysis: with 25% swapped labels, weight averaging (WA) is worse than prediction ensembling (ENS) on the corrupted training pairs and better on OOD test pairs (Observation 3, Figs. 4-5), with a toy model in which WA weights a feature by p_j² and ENS by p_j (§4.3).
- Best-of-N and REINFORCE experiments in which WARM delays the collapse of a larger control RM's score and raises its peak (§5, Figs. 1(b), 6-9).

## Key Figures/Tables to Study
- Fig. 3: OOD accuracy along the WA and ENS interpolation for four diversity sources (checkpoints of one run, data order, learning rate, Baklava init).
- Figs. 4-5: WA vs ENS on corrupted-train, clean-train, ID-validation and OOD-test subsets.
- Fig. 8: control reward vs KL during RL for M ∈ {2, 6, 10}, ENS M = 2, and single RMs; Fig. 8(c) ablates the KL coefficient α.
- Fig. 9: oracle win rates along RL against SFT, WARM M = 6 at step 3500, and the best single RM at step 3000.
- Fig. 10: accuracy of averaging M of 10 RMs under different selection orders.

## Technical Details
**RM loss.** L_R(r_φ, D_train) = −E_(x, y+, y−) log σ(r_φ(x, y+) − r_φ(x, y−)) (Eq. 1). r_φ(x, y) is the scalar reward for prompt x and generation y; y+ is preferred over y−; σ is the logistic function.
**LMC condition.** For φ1, φ2 fine-tuned from a shared pre-training and λ ∈ [0, 1]: Acc(r_((1−λ)φ1+λφ2)) ≥ (1−λ)·Acc(r_φ1) + λ·Acc(r_φ2) (Eq. 2), where Acc is pairwise accuracy, the fraction of pairs with r(x, y+) ≥ r(x, y−) (§3.2). LMC does not hold for weights trained from scratch, even with a shared random initialization (Remark 1).
**Toy analysis.** In the bag-of-features model of §4.3, the M → ∞ ensemble predicts y·Σ_j p_j|z_j|² (Eq. 4) and the weight average predicts y·Σ_j p_j²|z_j|² (Eq. 5). p_j is the probability that one run learns feature j and z_j is that feature's vector. Derived arithmetic: a feature learned in 90% of runs keeps weight 0.81, a feature learned in 10% keeps 0.01, so run-specific features lose relative weight. With L layers the exponent becomes L (Remark 4).
**Data.** Reddit TL;DR from Stiennon et al.: 123k posts, about 5% held out as ID validation (App. B.1). Preference labels come from an instruction-tuned PaLM-L prompted with the "Detailed + CoT 0-shot" strategy, greedy decoding, up to 512 decoded tokens, run in both orderings to avoid position bias (App. B.2). The OOD test set D_ood has 92k pairwise comparisons of summaries from PaLM-XS policies sampled at high temperature, some pre-trained only, some SFT, some RLHF (§4; App. B.1). The corrupt setup swaps 25% of training labels (§4.2; App. B.2).
**Evaluation.** A PaLM-XS control RM with 80.1% accuracy on D_ood gives pointwise control reward; the PaLM-L labeler gives pairwise oracle win rates (§5). Best-of-N uses a PaLM SFT policy (N = 8, D = 15,000 prompts) and a T5 SFT policy (N = 1000, D = 1000) (§5.1); KL of best-of-N is approximated as log(N) − (N−1)/N (§5.1).
**Results.** Best-of-N summaries selected by WARM reach up to a 92.5% win rate against random selection from SFT, and no strategy exceeds 50% against WARM M = 6 (§5.1, Fig. 7). In RL, WARM M = 6 reaches a 99.8% win rate against SFT after 3500 steps, the highest of all policies (§5.2, Fig. 9(a)); M = 10 delays hacking but does not raise the peak; the authors speculate that this is related to RMs 7-10 having lower individual OOD accuracy (§5.2). ENS M = 2 policies still show early reward hacking (§5.2). The best KL coefficient for WARM is lower than for a single RM because optimal WARM policies sit at larger KL (§5.2, Fig. 8(c)). Averaging checkpoints of one RM run gave lower accuracy than averaging separate runs (Remark 2, Fig. 3(a)). Among the selection orders compared, adding RMs from best to worst OOD accuracy is described as "a reliable heuristic" (Fig. 10).

## Recipe ledger
| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| RM featurizer SFT (PaLM-XXS) | PaLM-XXS (parameters not reported) | SFT | steps; batch; optimizer; LR | 12k steps; 128; Adafactor; 1e-5 | arXiv:2401.12187v1 App. B.3 | verified 2026-09-14 | no ablation reported |
| WARM RMs (PaLM-XXS) | PaLM-XXS | reward-model | initialization | SFT checkpoints at {8k, 10k, 12k} steps (Baklava); one linear-probed classification layer shared by all RMs | App. B.3; §3.3 | verified 2026-09-14 | Fig. 3(d) (8k vs 12k init); Fig. 16 (BoN); too-early checkpoints reduce RM accuracy per [157] |
| WARM RMs (PaLM-XXS) | PaLM-XXS | reward-model | steps; batch; optimizer | 10k steps; 128; Adafactor | §4; App. B.3 | verified 2026-09-14 | no ablation reported |
| WARM RMs (PaLM-XXS) | PaLM-XXS | reward-model | LR; dropout | LR sampled from {1e-5, 4e-5, 1e-4}; dropout from {0.05, 0.1} | App. B.3 | verified 2026-09-14 | no ablation; chosen as a "mild range to preserve the LMC" following [47] |
| WARM RMs (PaLM-XXS) | PaLM-XXS | reward-model | number of runs | 10 (clean setup) | App. B.3 | verified 2026-09-14 | n/a |
| WARM M = 2 / 6 / 10 | PaLM-XXS | merge | method; members | uniform weight average of the M best RMs ranked by D_ood accuracy | App. B.3; §5 | verified 2026-09-14 | Fig. 10 selection-order comparison; Figs. 6-9 compare M |
| RL policy and value (PaLM-XS) | PaLM-XS | RL | algorithm; init | modified REINFORCE with a value baseline, following [58]; policy and value from the same SFT model | §5.2; App. B.4 | verified 2026-09-14 | no ablation reported |
| RL policy (PaLM-XS) | PaLM-XS | RL | temperature; batch; optimizer; LR; warmup | 0.9; 128; Adafactor; 1e-5; policy warmup 2k steps | App. B.4 | verified 2026-09-14 | no ablation reported |
| RL policy (PaLM-XS) | PaLM-XS | RL | KL coefficient α | 0.003 (clean labels); 0.01 (25% corrupted labels) | §5.2; App. B.4 | verified 2026-09-14 | Fig. 8(c), Figs. 19(b)-21: α ∈ {0.001, 0.003, 0.01} |
| RL policy (PaLM-XS) | PaLM-XS | RL | reference checkpoints | WARM M = 6 at 3500 steps; best single RM φ1 at 3000 steps | §5.2, Fig. 9 | verified 2026-09-14 | Fig. 9 win rates |
| all | all | all | parameter counts; tokens; compute | not reported | checked §4-§5, App. B-C | not reported | n/a |

## Findings relevant to generality, negative feedback, long context, agentic training, distillation
- **Generality (distribution shift).** WA and ENS have similar OOD accuracy along the interpolation path (Observation 2, Fig. 3); with corrupted labels the gain of WA over ENS grows as evaluation moves from the training distribution to D_ood (Observation 3, Fig. 5). The authors interpret WA as keeping predictive mechanisms that are shared across runs (Remark 3; Interpretation).
- **Limits.** If every RM relies on the same spurious cue, such as summary length, WARM keeps that cue (§6). All experiments use one task (TL;DR summarization) and AI-generated labels in place of human labels (§5; App. B.2).
- **Negative feedback.** Swapped preference labels are a form of wrong supervision. WA fits fewer of the swapped training pairs than ENS (Fig. 4(a)), and the authors link reduced memorization to a smoother reward and more stable policy gradients (Remark 5; Interpretation).
- **Distillation (reward labels).** In a setup where the control RM labels the data used to train the PaLM-XXS RMs, WARM still scores above ENS and single RMs in best-of-N (App. C.4, Fig. 22); the authors report that this setup changes diversity across RMs and use the RLAIF setup for main results (App. C.4).

## Connections
- [[warp]]: the companion policy-side method; WARP states it was "conceived as a response to WARM" (WARP §5).
- [[reward-ensembling]]: prediction ensembling of RMs (Coste et al., ref. [42]), the ENS baseline family WARM compares against.
- [[reward-model-overoptimization]]: Gao et al. (ref. [17]); WARM's App. C.4 reproduces its distillation setup.
- [[rlaif-scaling]]: Lee et al. (ref. [58]); source of the AI labeling prompt and the REINFORCE variant.
- [[best-of-n]]: Stiennon et al. (ref. [14]); source of the TL;DR preference data.
- [[model-soups]]: weight averaging of fine-tuned models (ref. [46]) that WARM applies to RMs.
- [[bradley-terry-rm]]: the pairwise loss in Eq. 1.
- [[kl-control-rlhf]]: the KL term whose coefficient α is ablated in Fig. 8(c).
- [[rlhf-length-correlations]]: Singhal et al. (ref. [26]), cited for verbose outputs as a symptom of reward hacking.
- [[gemma-3]]: the Gemma 3 report states its RL phase uses improved versions of BOND, WARM, and WARP (Gemma 3 §3).

## Verification
- Created on 2026-09-14 from https://arxiv.org/abs/2401.12187 (v1, 2024-01-22; the only version; full PDF including App. A-C). Gemma 3 connection checked against arXiv:2503.19786 §3.
- Audit claims not found in the source: none.
- Not reported by the source: PaLM-XXS and PaLM-XS parameter counts, total training tokens, compute, human-label results.
