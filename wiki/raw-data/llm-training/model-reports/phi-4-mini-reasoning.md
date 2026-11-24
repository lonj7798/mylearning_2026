<!-- scope: Phi-4-Mini-Reasoning (Microsoft, 2025) — 4-stage small-model math reasoning recipe on Phi-4-Mini 3.8B: distillation as mid-training, distillation as SFT, rollout DPO on rejected teacher samples, RLVR with stability fixes
     deps: [[phi-4]], [[deepseek-r1]]
     see-also: [[phi-3]], [[limo]], [[s1]], [[bespoke-stratos]], [[open-thoughts]], [[openr1]], [[grpo]], [[dpo]], [[distillation-source-matters]], [[rstar2-agent]]
-->

# Phi-4-Mini-Reasoning: Exploring the Limits of Small Reasoning Language Models in Math
- **Core Insight:** For a 3.8B model, small curated long-CoT sets hurt (LIMO: MATH-500 71.8 → 57.8; S1K: → 47.0), while a staged recipe — large-scale distilled mid-training, a curated distilled SFT subset, DPO on teacher-rejected rollouts, then RLVR — raises MATH-500 to 94.6 and AIME24 to 57.5 (§2, Table 1; §5.4, Table 3).
- **Guideline:** For a small non-reasoning base, absorb a large volume of distilled reasoning traces first (packed, until validation saturates) before any small high-quality SFT set, and reuse the teacher's incorrect rollouts as DPO negatives rather than discarding them (§2, §3.1–3.3).
- **Authors:** Haoran Xu, Baolin Peng, Hany Awadalla, Dongdong Chen, Yen-Chun Chen, Mei Gao, et al. (Microsoft; 14 authors, last-listed Weizhu Chen)
- **Year:** 2025 (arXiv v1 2025-04-30; only version)
- **URL:** https://arxiv.org/abs/2504.21233
- **Source type:** model report
- **Relevant topics:** small-model reasoning, distillation as mid-training, long-CoT SFT, rollout DPO, RLVR stability, temperature annealing, GRPO length variance

## Abstract
The report presents a four-step recipe for small language models: (1) large-scale mid-training on diverse distilled long-CoT data, (2) SFT on high-quality long-CoT data, (3) Rollout DPO on a curated preference set, and (4) RL with verifiable reward. Applied to Phi-4-Mini (3.8B), the resulting Phi-4-Mini-Reasoning outperforms DeepSeek-R1-Distill-Qwen-7B by 3.2 points and DeepSeek-R1-Distill-Llama-8B by 7.7 points on MATH-500.

## Key Contributions
- Evidence that naive small-set distillation (LIMO, S1K) degrades a 3.8B model on all three benchmarks (§2, Table 1).
- Per-stage ablation of the four-stage recipe (Table 3).
- Rollout DPO: preference pairs from the same teacher rollouts that rejection sampling would discard (§3.3).
- Three RL stability fixes for a small model: prompt selection by length uniformity, reward rebalancing by oversampling and filtering, and temperature annealing (§3.4).
- DAPO comparison in which DAPO's cons@16 on AIME degrades during training (§5.5, Fig. 2b).

## Key Figures/Tables to Study
- Table 1: small-set distillation failures. Table 2: data sources. Table 3: stage-by-stage scores.
- Fig. 2a: pass@k on AIME24 for base, Rollout DPO, and RL model. Fig. 2b: DAPO vs their RL recipe (cons@16).

## Technical Details
**Stage 1 — distillation as mid-training (§3.1).** Causal LM on question + correct CoT answer; answers sampled from DeepSeek-R1 with rejection sampling; packed sequences; train "as much CoT training data as possible until model performance saturates on a validation dataset."

**Stage 2 — distillation as SFT (§3.2).** A compact subset of the mid-training data spanning math domains with difficulty above "college level"; non-packed so the model learns where to stop.

**Stage 3 — Rollout DPO (§3.3, Eq. 1).** Keep questions labeled high-school level or above by GPT-4o-mini; correct rollouts are preferred, incorrect rollouts dispreferred; standard DPO loss.

**Stage 4 — RLVR (§3.4).** Reward +1 if verified, −1 otherwise (Eq. 7). PPO (Eqs. 2–4) and GRPO (Eqs. 5–6) are described; the paper does not state which one is used for the final model. Observed problems in pilot GRPO: positive responses for one prompt ranged ~12K–20K tokens; DAPO-style 0/1-accuracy filtering still left unstable gradients from length variance; hard prompts needed a GRPO group of 128 to get one positive. Fixes:
1. Prompt optimization: sample multiple rounds; keep prompts whose responses have relatively uniform token lengths.
2. Reward rebalancing: oversample hard prompts; keep all positives and an equal number of randomly sampled negatives; drop prompts whose group accuracy exceeds a threshold (example given: 50%).
3. Temperature annealing: 1.0 → 0.6 linearly over the first 50% of steps, then fixed at 0.6.

**Data (§4, Table 2).** Non-reasoning sources (questions only, answers re-sampled from DeepSeek-R1 671B, ~8 rollouts per question): AquaRAT 98K, Ape210K 210K, MetaMathQA 395K, MathInstruct 262K, TAL-SCQ5K 5K. Reasoning sources used as given: OpenR1-Math 220K, Bespoke-Stratos-17k 17K, OpenThoughts-114K 114K. Plus unnamed in-house seed sets. Total: ~10M rollouts over 1.6M samples. Verification by a math-verification tool, then GPT-4o-mini re-verifies rollouts flagged incorrect (to reduce false negatives). Each sample is annotated with domain, difficulty (elementary to graduate), and presence of repetitive patterns. Mid-training uses the full set; later stages use subsets.

**Results (Table 3; AIME24 / MATH-500 / GPQA-Diamond, pass@1, mean of 3 runs at T=0.6, top_p 0.95, 32K).** Phi-4-Mini 10.0 / 71.8 / 36.9; + distill mid-training 30.0 / 82.9 / 42.6; + distill fine-tuning 43.3 / 89.3 / 48.3; + Rollout DPO 50.0 / 93.6 / 49.0; + RL 57.5 / 94.6 / 52.0. Baselines: DeepSeek-R1-Distill-Qwen-7B 53.3 / 91.4 / 49.5; DeepSeek-R1-Distill-Llama-8B 43.3 / 86.9 / 47.3; o1-mini (reported) 63.6 / 90.0 / 60.0.

## Recipe ledger
| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| Phi-4-Mini-Reasoning | 3.8B | mid-train | batch; LR; epochs; warm-up; seq len; packing | 128; 1e-5; 5; 0.1; 16K; packed | arXiv:2504.21233v1 §5.3 | verified 2026-11-24 | Table 3: +20.0 AIME, +11.1 MATH-500 over base |
| Phi-4-Mini-Reasoning | 3.8B | mid-train | data volume | ~10M rollouts over 1.6M samples (full set) | §4 | verified 2026-11-24 | stopping rule: until validation saturates (§3.1) |
| Phi-4-Mini-Reasoning | 3.8B | SFT | batch; LR; epochs; warm-up; seq len; packing | 128; 1e-5; 5; 0.1; 20K; not packed | §5.3 | verified 2026-11-24 | Table 3: +13.3 AIME over mid-trained model |
| Phi-4-Mini-Reasoning | 3.8B | preference | LR; epochs; seq len | 5e-7; 1; 16K | §5.3 | verified 2026-11-24 | Table 3: +6.7 AIME, +4.3 MATH-500 |
| Phi-4-Mini-Reasoning | 3.8B | RL | LR; seq len | 5e-7; 25K | §5.3 | verified 2026-11-24 | Table 3: +7.5 AIME; Fig. 2b vs DAPO |
| Phi-4-Mini-Reasoning | 3.8B | RL | temperature schedule | 1.0 → 0.6 linear over first 50% of steps, then 0.6 | §3.4 | verified 2026-11-24 | no ablation reported |
| Phi-4-Mini-Reasoning | 3.8B | RL | algorithm; group size; KL β; DPO β; number of SFT examples; number of DPO pairs | not reported | checked §3–5 | not reported | — |

## Findings relevant to generality and negative feedback
- **Negative samples.** Incorrect DeepSeek-R1 rollouts, normally discarded by rejection sampling, become DPO dispreferred responses (§3.3). In RL, negatives are subsampled to equal the positive count for hard prompts (§3.4). Rollout DPO raised pass@k on AIME24 across k, which the authors read as extending the capability boundary (§5.5, Fig. 2a).
- **Label noise.** Rule-based verification produces false negatives on complex solutions; GPT-4o-mini re-checks flagged rollouts (§4).
- **Generality.** All reported evaluations are math or GPQA-Diamond; no instruction-following, coding, or forgetting measurement is reported. The safety statement refers to an external model card (§5.6).
- **Small-model specificity.** The authors hypothesize the RL instabilities are more prominent for small models (§3.4); no larger-model comparison is run.

## Connections
- [[phi-4]], [[phi-3]]: family reports; this card covers the reasoning variant of Phi-4-Mini only.
- [[limo]], [[s1]]: the small-set distillation recipes that degrade Phi-4-Mini (Table 1).
- [[bespoke-stratos]], [[open-thoughts]], [[openr1]]: reasoning-trace sources in Table 2.
- [[grpo]], [[dpo]]: objectives used; DAPO is compared in Fig. 2b.
- [[distillation-source-matters]]: related evidence on teacher-trace choice for distillation.

## Verification
- Checked on 2026-11-24 against https://arxiv.org/abs/2504.21233 (v1 PDF, full text, 10 pages including references).
- Table 1 and Table 3 values were read from the PDF text layer, where columns are serialized; column assignment checked against the abstract (94.6 − 91.4 = 3.2; 94.6 − 86.9 = 7.7).
- Not reported: in-house seed data size; SFT subset size; DPO pair count; RL prompt count, steps, group size, KL; whether final RL used GRPO or PPO; compute.
