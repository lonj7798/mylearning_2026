<!-- scope: Mahan et al. (SynthLabs + Stanford, arXiv:2410.12832, Oct 2024) — GenRM: an LLM judge trained on preference labels with next-token prediction; CoT-GenRM variants trained by STaR-SFT, post-rationalization, and STaR-DPO; Llama-3.1-8B-Instruct on UltraFeedback and UltraInteract, evaluated on RewardBench
     deps: [[bradley-terry-rm]], [[star]], [[dpo]]
     see-also: [[pairrm]], [[judge-llm-bias]], [[self-taught-evaluators]], [[meta-rewarding-lm]], [[direct-judgement-preference]]
-->

# Generative Reward Models
- **Core Insight:** With Llama-3.1-8B-Instruct trained on UltraFeedback, a judge trained by DPO on its own correct versus incorrect reasoning chains (STaR-DPO) matched Bradley-Terry, PairRM, and GenRM reward models in-distribution (73.9% vs about 73–74%) and scored 81.9% on RewardBench, above the best non-reasoning trained model (GenRM, 78.9%) (§5.1).
- **Guideline:** When a pairwise reward model must generalize beyond its preference training distribution, train an LLM judge with STaR-DPO on rationales sampled from the same model instead of a Bradley-Terry head or SFT on stronger-model rationales, because those alternatives lost accuracy on RewardBench in this study (§5.1–5.3); the evidence covers one 8B model and accuracy on preference benchmarks, not use as a reward in RL.
- **Authors:** Dakota Mahan, Duy Van Phung, Rafael Rafailov, Chase Blagden, Nathan Lile, Louis Castricato, et al. (Jan-Philipp Fränken, Chelsea Finn, Alon Albalak; SynthLabs, Stanford University)
- **Year:** 2024 (arXiv v1 2024-10; preprint)
- **URL:** https://arxiv.org/abs/2410.12832
- **Source type:** paper
- **Relevant topics:** generative reward model, LLM-as-a-judge, CoT judge, STaR, DPO on rationales, RLAIF, reward-model generalization

## Abstract
RLHF needs large sets of human preference labels, and RLAIF replaces them with LLM-generated labels that may not match human judgments. The authors propose GenRM, an iterative algorithm that trains an LLM on self-generated reasoning traces so that its preference labels match human labels. Zero-shot LLM judges under-perform Bradley-Terry reward models in-distribution by 9–36%. GenRM matches Bradley-Terry models in-distribution and outperforms them out-of-distribution by 10–45%. It also surpasses LLM judges in-distribution (by 9–31%) and out-of-distribution (by 2–6%) (Abstract).

## Key Contributions
- Shows that under the Bradley-Terry model, the RLHF reward relative to a reference response equals the log-odds of a preference probability, so a general preference model p(y_w ≻ y_l | x) can replace the pointwise reward (§3, Eq. 5–6).
- Defines GenRM (answer-indicator token, no reasoning) and CoT-GenRM (rationale, then indicator), with three training methods: STaR-SFT, STaR-Rationalizer, STaR-DPO (§4, Eq. 7–9).
- Compares them with Bradley-Terry RM, PairRM, and zero-shot judges on in-domain and RewardBench data (§5, Figs. 2, 4).
- Measures the effect of the rationale source (Llama 3.1 8B, Llama 3.1 70B, GPT-4) and of majority voting (§5.3–5.4, Table 1, Fig. 5).

## Key Figures/Tables to Study
- Fig. 1: Bradley-Terry vs GenRM vs CoT-GenRM. Fig. 2: UltraFeedback-trained models on UltraFeedback and RewardBench subsets.
- Fig. 4: UltraInteract-trained models on RewardBench Reasoning vs non-Reasoning. Fig. 5: accuracy vs majority-vote count.
- Table 1: rationale bootstrap source. Table 2 (App. A.2): hyperparameters. Table 3 (App. B): accuracy per STaR iteration.

## Technical Details
- **Bradley-Terry baseline:** p(y1 ≻ y2 | x) = σ(r(x, y1) − r(x, y2)); the reward model is the SFT model plus a linear predictor on the final embedding (§2.1.2, Eq. 1–2).
- **GenRM (no CoT):** loss −log π(I | x, y1, y2), where I is the answer-indicator token ("A" or "B"); the model is used as a classifier trained with next-token prediction (§4, Eq. 7; App. A.1 Fig. 8). Preference probabilities come from output likelihoods or majority-vote counts (§3).
- **STaR-SFT:** sample rationale r and verdict I from the current model; keep chains whose verdict matches the label; SFT on −log π(I | x, y1, y2, r) − log π(r | x, y1, y2) (§4, Eq. 8).
- **STaR-Rationalizer:** rationales come from a post-rationalization model given the correct answer (prompt: "Explain why response (A/B) is better than response (B/A)") and are trained with Eq. 8 (§4, §5.2, App. A.1 Fig. 7).
- **STaR-DPO:** DPO where the chosen output is (r_w, I_w), a rationale with the correct verdict, and the rejected output is (r_l, I_l), a rationale with the wrong verdict (§4, Eq. 9).
- **Prompts:** based on the MT-Bench judge prompt with ties removed; factors listed are helpfulness, relevance, accuracy, depth, creativity, and level of detail; the prompt also says not to let length or position influence the verdict (App. A.1 Fig. 6).
- **Setup:** all models start from Llama-3.1-8B-Instruct; training sets are UltraFeedback (61k pairs) and UltraInteract; evaluation uses a held-out split of the training dataset and RewardBench (Chat, Chat Hard, Reasoning, Safety) (§5). Generative scores in Figs. 2 and 4 are majority votes over 32 samples.
- **UltraFeedback-trained results (§5.1):** zero-shot judge without reasoning 52.25% → with CoT and self-consistency 67.75% on UltraFeedback, 60.60% → 75.18% on RewardBench. Bradley-Terry RM, PairRM, and GenRM are "around 73–74%" in-distribution; STaR-DPO 73.9%; STaR-SFT 67.4%. On RewardBench: STaR-DPO 81.9%, base-model prior 77.8%, GenRM 78.9%; Safety: STaR-DPO 91.0% vs PairRM 81.8%. The text gives 75.18% and 77.8% for zero-shot RewardBench accuracy without reconciling them.
- **UltraInteract-trained results (§5.2):** in-distribution STaR-DPO 90.2% vs base 68.8%; explicit RMs about 94%. RewardBench Reasoning: Bradley-Terry below random, best GenRM 70.8%, LLM-as-a-judge 76.6%, STaR-DPO 87.2%. RewardBench non-Reasoning: LLM-as-a-judge 78.0%, STaR-DPO 75.0%.
- **Majority voting at 32 (§5.4, Fig. 5):** +1.6% on UltraFeedback and +3.8% on RewardBench (UltraFeedback-trained); +4.6% on UltraInteract and +4.9% on RewardBench (UltraInteract-trained).
- **Rationale source (§5.3, Table 1, Maj@32, UltraFeedback / RewardBench):** Llama 3.1 8B iterations 1→3: 68.63/77.34 → 67.38/77.05; Llama 3.1 70B: 70.50/77.09 → 69.58/63.43; GPT-4: 62.85/69.58 → 71.73/78.29; GPT-4 (full, all iterations from GPT-4): 62.60/70.52. Only the first iteration uses the bootstrap source. Row-to-iteration mapping is derived: the Llama 3.1 8B rows equal STaR-SFT iterations 1–3 in Table 3.

## Recipe ledger
| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| All trained RMs and judges | 8B | reward-model | Initialization | Llama-3.1-8B-Instruct | arXiv:2410.12832v1 §5 | verified 2026-09-14 | no ablation reported |
| All | 8B | reward-model | Training data | UltraFeedback, 61k pairs; UltraInteract (size not given) | §5 | verified 2026-09-14 | no ablation reported |
| STaR-SFT / STaR-DPO / Bradley-Terry RM / GenRM / PairRM | 8B | reward-model | AdamW peak LR | 1.0e-6 | App. A.2.1 Table 2 | verified 2026-09-14 | no ablation reported |
| STaR-SFT Rationalizer | 8B | reward-model | AdamW peak LR | 2.0e-5 | App. A.2.1 Table 2 | verified 2026-09-14 | no ablation reported |
| STaR-DPO | 8B | preference | DPO β | 1.0 | App. A.2.1 Table 2 | verified 2026-09-14 | no ablation reported |
| STaR-IPO Rationalizer (row in Table 2; not described in the text) | 8B | preference | β; LR | 0.4; 1.0e-6 | App. A.2.1 Table 2 | verified 2026-09-14 | no ablation reported |
| Table 2 shared columns | 8B | reward-model | (β1, β2); weight decay; schedule; LR warmup | (0.9, 0.95); 0.1; cosine; 0.1 (unit not stated); printed once, row scope not shown in the extracted table | App. A.2.1 Table 2 | verified 2026-09-14 | no ablation reported |
| STaR methods | 8B | reward-model | Iterations; data per iteration | 3; dataset split into three equal portions, one fresh portion sampled by the latest model per iteration | App. A.3; App. B Table 3 | verified 2026-09-14 | Table 3: STaR-DPO UltraFeedback Maj@32 71.28 → 72.98 → 73.58 |
| STaR methods | 8B | reward-model | Epochs per iteration | UltraFeedback 3; UltraInteract 1 | App. A.3 | verified 2026-09-14 | chosen for similar step counts, no ablation |
| All generative models | 8B | eval-gate | Sampling | SGLang 0.3.0; temperature 1.0; top-p 0.95; majority vote over 32 | App. A.2.2; Fig. 2 caption | verified 2026-09-14 | Fig. 5 majority-vote curve |
| All | 8B | reward-model | Batch size, sequence length, samples per pair during STaR, compute | not reported | checked body, App. A–B | not reported | — |

## Findings relevant to generality, negative feedback, distillation
- **Generality (Result, single study):** explicit reward models trained on UltraInteract reach about 94% in-distribution, but Bradley-Terry falls below random on RewardBench Reasoning, while STaR-DPO reaches 87.2% (§5.2).
- **Negatives as gradient (Result, single study):** STaR-SFT discards chains with wrong verdicts and shows no gain over the base model (67.4% UltraFeedback); STaR-DPO uses them as the rejected term of DPO and reaches 73.9% and 81.9% on RewardBench (§4, §5.1). The paper does not run an ablation that separates the use of negatives from the change of loss.
- **Off-policy rationales (Result, single study; the explanation is the authors' Interpretation):** STaR-Rationalizer matches STaR-DPO in-distribution but falls below the base model on RewardBench; RewardBench Maj@1 drops from 71.73 to 67.62 over three iterations (Table 3). The authors hypothesize that rationales from the post-rationalization model are off-policy for the base model (§5.2). Llama 3.1 70B rationales raise UltraFeedback accuracy slightly and lower RewardBench accuracy (§5.3).
- **Not tested:** using GenRM as the reward in PPO or online preference optimization, and reward hacking of generative RMs; both are listed as future work (§7).

## Connections
- [[bradley-terry-rm]] — the pointwise baseline whose reward the paper rewrites as preference log-odds (Eq. 6).
- [[star]] — the rationale bootstrapping and rationalization procedure reused for the judge (§2.2).
- [[dpo]] — the objective applied to correct vs incorrect reasoning chains in STaR-DPO (Eq. 9).
- [[pairrm]] — PairRM is a baseline in Figs. 2 and 4.
- [[self-taught-evaluators]], [[meta-rewarding-lm]], [[direct-judgement-preference]] — concurrent judge-training methods cited in §6.
- [[judge-llm-bias]] — zero-shot judge failures; Fig. 3 shows a judge preferring a longer answer that ignores the instruction.
- [[ultrafeedback]] — the in-distribution training and evaluation data for §5.1.

## Verification
- Checked on 2026-09-14 against: https://arxiv.org/abs/2410.12832 (arXiv v1, 2024-10-02; no later version).
- Corrections to the previous card version:
  - Authors "Lifan Yuan, Ganqu Cui, … Maosong Sun" → Mahan, Van Phung, Rafailov, Blagden, Lile, Castricato, Fränken, Finn, Albalak (title page).
  - Second URL "arXiv:2408.15240 'Critique-out-Loud Reward Models', Ankner et al." → 2408.15240 is "Generative Verifiers" (Zhang et al.); Critique-out-Loud is 2408.11791 (reference list). Both are separate concurrent works (§6), not this artifact.
  - "reward = log P('A is better' | x, y_A, y_B, rubric)" and "score = log P('A') − log P('B')" → the paper trains −log π(I | x, y1, y2) and extracts preference probabilities from likelihoods or majority votes; no rubric input and no such formula is printed (§3, §4, Eq. 7).
  - "rubric of correctness, helpfulness, safety, conciseness" → MT-Bench factors: helpfulness, relevance, accuracy, depth, creativity, level of detail (App. A.1).
  - "critiques human-written or bootstrapped from GPT-4" → rationales self-sampled (STaR) or generated by a post-rationalization model; GPT-4 and Llama 3.1 70B are tested as bootstrap sources (§4, Table 1).
  - "GenRM Pareto-dominates in Fig. 2"; "Fig. 4 calibration plot" → Fig. 2 shows STaR-DPO strongest on RewardBench with comparable in-distribution accuracy; Fig. 4 compares UltraInteract-trained models (§5.1–5.2).
  - Pointwise 1–10 scoring → the paper evaluates pairwise judgments only.
- Removed as unsupported by the source: "calibrated uncertainties"; "CoT improves accuracy 3–10 pp on RewardBench"; "less vulnerable to sycophancy/length bias when the rubric names them"; "steerable via its own context"; "slower but scales with model capability"; "mitigated by using a judge from a different model family"; "makes adversarial exploitation harder"; "GenRM ensembles give calibrated uncertainty".
- Not reported by the source: batch size, sequence length, compute, UltraInteract pair count, any RL run using the trained judge.
