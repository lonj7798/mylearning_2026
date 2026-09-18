<!-- scope: West-of-N synthetic preference pairs (best-of-N vs worst-of-N) for reward-model self-training
     deps: [[rlaif-scaling]]
     see-also: [[self-rewarding-lm]], [[spin]], [[ultrafeedback-construction]]
-->

# West-of-N: Synthetic Preferences for Self-Improving Reward Models
- **Core Insight:** Sampling N responses from the policy for an unlabeled prompt and pairing the best-scored with the worst-scored response under a base preference model yields pseudo-preference pairs that raise reward-model accuracy by up to about 2.3% on Reddit TL;DR, compared with about 1% reported for doubling the human preference data (§5.1).
- **Guideline:** When a base preference model already has reasonable accuracy on the target distribution and additional human preference labels are expensive, generate on-policy pairs by sampling N responses per unlabeled prompt at the same temperature used for RL sampling and keeping the best and worst under that model, because gains grow with base-model accuracy and shrink when the base labels are noisy (§5.2, Figure 8). When the base model is weak, expect small or no gain.
- **Authors:** Alizée Pace, Jonathan Mallinson, Eric Malmi, Sebastian Krause, Aliaksei Severyn
- **Year:** 2024 (arXiv v1 2024-01; v2 2024-10)
- **URL:** https://arxiv.org/abs/2401.12086
- **Source type:** paper
- **Relevant topics:** synthetic preference data, best-of-N sampling, reward model training, self-training, on-policy data

## Abstract
The quality of RLHF depends on the reward model, which in turn depends on the quantity, label accuracy, and response distribution of the preference data. The paper proposes generating synthetic preference pairs by sampling N responses to an unlabeled query from the policy and selecting the best and the worst according to a base preference model ("Best-of-N + Worst-of-N = West-of-N"). These pseudo-labeled pairs are added to the reward-model training mixture. Across three datasets the method improves any reward model tested, with an effect comparable to or greater than adding a similar quantity of human preference data.

## Key Contributions
- Extends Best-of-N sampling, previously used for policy training, to reward-model training data generation (§1).
- Shows the resulting self-trained reward models improve both held-out preference accuracy and downstream Best-of-N and PPO win rates on Reddit TL;DR, Anthropic Helpful, Anthropic Harmless, and UltraFeedback (§5.1, Figures 2 and 3).
- Proves a bound: if the base model is within ε of the ground-truth preference function everywhere, the West-of-N pair is correctly labeled with probability at least 1 − 2ε as N → ∞ (Theorem 4.1).
- Shows the extremum pairing matters: naive self-training (N = 2) and (best-of-N, random) pairing both reduce accuracy below the base model, while West-of-N raises it (Appendix B.3, Table 2).
- Adds two filters — pseudolabel confidence P_θ(y⁺ ≻ y⁻|x) and response log-likelihood under the policy — that give further gains (§5.2, Figure 5).

## Key Figures/Tables to Study
- **Figure 1** — the method diagram: sample N responses, score with the base preference model, emit the (best, worst) pair back into the reward-model training mixture.
- **Figure 2** — accuracy, high-confidence accuracy, Best-of-64 win rate, and RL win rate for HF50%, +RLAIF, +RLCD, +West-of-N, and HF100% on the three datasets.
- **Figure 4** — the N ablation: (a) gains grow from N = 2 to N = 64; (b) pseudolabel accuracy grows with N; (c) response likelihood under the policy falls with N but stays above the human-feedback data.
- **Figure 6** — a second West-of-N iteration reduces held-out accuracy gains but increases Best-of-N win rates.
- **Table 2 (Appendix B.3)** — best-vs-worst compared against naive self-training and best-vs-random.

## Technical Details
- **Datasets:** Reddit TL;DR (129k posts with human summaries; 64k human-rated summary pairs), Anthropic HH (170k rated pairs, roughly 70% helpfulness and 30% harmlessness), UltraFeedback (64k prompts, four responses each, GPT-4 rated; binarized by taking the top-scored response as positive and one of the remaining three at random as negative) (§5, Datasets).
- **Models:** policy and reward models are T5-XXL (11B) for TL;DR and AnthropicHH, and Gemma 2B for UltraFeedback (§5, Methods).
- **Base preference data:** 50% of the human feedback data (HF50%), so that West-of-N gains can be compared against the other 50% being added as human labels (HF100%) (§5, Methods).
- **Default N:** 64 responses per query, sampled from the SFT or latest RL policy at temperature 0.7 (§5, Methods; Appendix C).
- **Teacher/student split:** the base (teacher) model is trained pairwise, with no Bradley-Terry transitivity assumption; best and worst of N are recovered by an elimination tournament. The self-trained student is pointwise so it can be used in RL (§5.2; Appendix C).
- **Mixture:** student reward models are trained on a 1:1 mixture of base human preferences and West-of-N pairs (§5, Methods).
- **Base pairwise model accuracy:** 72.5% (TL;DR), 66.9% (Anthropic Helpful), 71.4% (Anthropic Harmless) (§5.2).
- **N ablation:** at N = 2 the method reduces to naive self-training and harms accuracy; accuracy and pseudolabel quality both rise through N = 8 and N = 64 (§5.2, Figure 4a-b). No reward hacking of the base model was observed up to N = 64 (§5.2).
- **Pairing ablation on TL;DR (Table 2, accuracy / high-confidence accuracy / Best-of-N win rate):** HF50% 70.9 / 78.0 / 85.2; + self-training 70.2 / 76.6 / 85.0; + (best-of-N, random) 69.5 / 76.1 / 87.0; + West-of-N 72.5 / 79.7 / 90.4.
- **Iteration:** a second round uses the first round's model as the base preference model, keeping the 1:1 human-to-synthetic ratio; it lowers the held-out accuracy gain but raises Best-of-N win rates (§5.3, Figure 6).
- **Evaluation:** held-out human-preference accuracy, plus win rate against the SFT response averaged over 1000 test queries, judged by an independent T5-Large Autorater trained on the held-out human preferences and by few-shot GPT-4, with response order randomized (§5, Evaluation).

## Recipe ledger

| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| T5-XXL policy (TL;DR, AnthropicHH) | 11B | SFT | optimizer; LR; steps; batch size; max input/output length | Adafactor; 1e-5; 4,000 steps; batch 32; 1024 in / 128 out tokens | arXiv:2401.12086v2 App. C | verified 2026-09-18 | no ablation reported |
| T5-XXL reward model | 11B | reward-model | steps; batch size; checkpoint selection | 20,000 steps; batch 8; highest validation accuracy | arXiv:2401.12086v2 App. C | verified 2026-09-18 | no ablation reported |
| T5-XXL policy | 11B | RL (PPO) | steps; batch size; LR warmup and peak; KL β; temperature; epochs | 10,000 steps; batch 8; linear warmup over 2,000 steps to 1e-4, policy LR delayed 2,000 steps; β = 0.05; T = 0.7; 8 epochs | arXiv:2401.12086v2 App. C | verified 2026-09-18 | no ablation reported |
| West-of-N generation | — | preference-data | N; sampling temperature; base:synthetic mixture | N = 64; T = 0.7; 1:1 | arXiv:2401.12086v2 §5 Methods, App. C | verified 2026-09-18 | §5.2 Figure 4a: gains increase from N = 2 to N = 64 on TL;DR |
| West-of-N filtering | — | preference-data | filter functions; retention quantile | confidence P_θ(y⁺ ≻ y⁻\|x) or joint log-likelihood log π(y⁺) + log π(y⁻); quantile chosen on validation | arXiv:2401.12086v2 §4.2, App. C | verified 2026-09-18 | §5.2 Figure 5: both filters improve accuracy and Best-of-N win rate over no filtering |
| Compute | — | all | hardware; wall clock | TPUs with 32 GiB HBM; about one day per reward model, two weeks for all experiments | arXiv:2401.12086v2 App. C | verified 2026-09-18 | not applicable |

## Findings relevant to negative feedback
- The negative member of each pair is the worst-scored on-policy sample (argmin under the base model), not a human-written or off-policy response. The paper's stated reason this works is label correctness, not hardness: the extremum pair maximizes P_θ(y⁺ ≻ y⁻|x), and Theorem 4.1 bounds the mislabeling probability by 2ε.
- Pairing the best sample against a random sample instead of against the worst lowers held-out accuracy from 72.5 to 69.5 on TL;DR (Table 2), a 3.0-point drop, and naive N = 2 self-training (70.2) falls below the base model (70.9).
- Increasing N pushes both members of the pair to lower likelihood under the policy (Figure 4c), which the paper frames as an on-policy-ness cost; it reports that at N = 64 the pairs remain far more likely than the human-feedback responses and that no reward hacking of the base model appeared in that range (§5.2).
- The paper does not analyze likelihood displacement, squeezing effects, or negative-gradient dynamics; West-of-N produces preference pairs and does not itself define a loss beyond the standard pairwise objective.

## Findings relevant to generality
- Gains are reported to depend on base preference-model accuracy: Figure 8 plots per-dataset gains against base accuracy, with the smallest gain on the noisiest base data (TL;DR RLCD) and the largest on Anthropic Harmless human feedback (Appendix B.2).
- West-of-N improves reward models built on human-labeled, RLAIF-labeled, and RLCD-generated base data, and across both model backbones tested (§5.1, Figure 3), which the authors present as evidence that the method is not tied to one preference source.
- Held-out human-preference accuracy and downstream on-policy quality do not move together: iterative West-of-N lowers the accuracy gain while raising Best-of-N win rates (§5.3). The paper attributes this to the off-policy nature of the held-out human data.

## Connections
- [[rlaif-scaling]] supplies the RLAIF baseline and evaluation prompt used in the comparison; §5.1 reports that RLAIF and West-of-N can be combined into a fully synthetic pipeline.
- [[self-rewarding-lm]] and [[spin]] are iterative self-training methods where the rejected response also comes from the policy; the paper's §2 cites concurrent work (Xu et al. 2023; Yuan et al. 2024; Meng et al. 2024) applying best-and-worst pairing to DPO data.
- [[ultrafeedback-construction]] is one of the three evaluation datasets, binarized by top-score-vs-random following Tunstall et al. (2023) (§5, Datasets).
- [[reward-ensembling]] addresses reward-model overoptimization by averaging models; West-of-N addresses reward-model input distribution by changing the data.

## Verification
- Checked on 2026-09-18 against: https://arxiv.org/abs/2401.12086 (arXiv v2, 25 Oct 2024)
- Corrections to the previous card version:
  - "Base generator: an SFT model (TULR, PaLM-2-XXS across ablations)" → policy and reward models are T5-XXL (11B) for TL;DR and AnthropicHH and Gemma 2B for UltraFeedback; PaLM 2 Bison appears only in the RLAIF and RLCD baselines (§5, Methods; Baselines).
  - "Base reward model trained on ≈20K seed pairs" → the base model is trained on 50% of each dataset's human preference data; no 20K figure appears (§5, Methods).
  - "Sample N=16 responses per prompt at T=1.0" → N = 64 at temperature 0.7 (§5, Methods; App. C).
  - "N=16 is near-optimal; marginal returns beyond N=32" → the ablation covers N ∈ {2, 8, 64} and gains increase through N = 64 (§5.2, Figure 4a).
  - "Table 1 (gain per iteration): 1 iter ≈ +6% RM accuracy; 2 iters saturates" → Table 1 is a qualitative comparison of preference-data types. The first iteration gives up to about +2.3% accuracy (§5.1); a second iteration reduces the accuracy gain but increases Best-of-N win rates (§5.3, Figure 6).
  - "4-point RM accuracy loss if you pair best against random" → 3.0 points on TL;DR, 72.5 → 69.5 (Table 2).
  - "as much as a doubling of the human preference dataset size" stated as the paper's result → the abstract says the effect is "comparable to the addition of a similar quantity of human preference data"; §5.1 contrasts about +2.3% for West-of-N against the roughly +1% that Stiennon et al. (2020) report for doubling the data.
  - "Figure 1 (RM accuracy on test vs training-set size)" → Figure 1 is the method diagram.
  - Card title and Core Insight rewritten to the exact published title.
- Removed as unsupported by the source: "works with both on-policy and off-policy generators"; "across both Bradley-Terry and DPO-implicit reward models" (the teacher is pairwise/non-BT and the student is pointwise; no DPO-implicit reward model is used); "prompt pool: 10K prompts from Reddit TL;DR + HH-RLHF" (queries come from the held-out 50% of each dataset); "pairs are sharper than human labels"; "currently deployed in many open reward-model training recipes (e.g., Ultrafeedback augmentation pipelines)"; "Figure 3 (N ablation)" as a locus.
- Not reported by the source: the number of unlabeled queries used for generation; hardness or difficulty statistics of the worst-of-N responses; likelihood-displacement or diversity measurements after RL; any evaluation outside the four preference datasets listed.
