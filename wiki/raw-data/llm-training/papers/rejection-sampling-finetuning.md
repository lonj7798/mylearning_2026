<!-- scope: the paper that introduced Rejection sampling Fine-Tuning (RFT) and measured what drives it — distinct reasoning paths, not sample count
     deps: [[star]]
     see-also: [[best-of-n]], [[west-of-n]], [[iterative-sft-rl]], [[rest-em]], [[llama-2]]
-->

# Scaling Relationship on Learning Mathematical Reasoning with Large Language Models
- **Core Insight:** Rejection sampling Fine-Tuning (RFT) — sample k reasoning paths from an SFT model, keep the ones with the correct answer and correct intermediate calculations, deduplicate them by equation list, and fine-tune the base model on the result — raises LLaMA-7B GSM8K accuracy from 35.9 (SFT) to 49.3 when rejection samples from four models are combined, and the quantity that predicts the gain is the number of **distinct** reasoning paths per question, not the number of accepted samples (Abstract; §3.3, Tab. 1; Figure 4).
- **Guideline:** When budget allows one round of self-generated SFT data, run rejection sampling at k ≈ 100 with temperature 0.7, deduplicate by calculation path, and pool samples from several smaller SFT models rather than raising k on one large model, because distinct paths per question grow only 1.17 → 5.25 as k goes 1 → 100 for LLaMA-7B, while the pooled 400-sample U13B set reaches 12.84 (§3.3, Tab. 2).
- **Authors:** Zheng Yuan, Hongyi Yuan, Chengpeng Li, Guanting Dong, Keming Lu, Chuanqi Tan, et al.
- **Year:** 2023 (arXiv v1 2023-08; text checked against v2, 2023-09-13)
- **URL:** https://arxiv.org/abs/2308.01825
- **Source type:** paper
- **Relevant topics:** rejection sampling fine-tuning, RFT, self-generated SFT data, reasoning-path diversity, GSM8K, pre-training loss as a predictor

## Abstract
The paper studies how pre-training loss, supervised data amount, and augmented data amount govern
the math reasoning performance of a supervised LLM. Pre-training loss predicts performance better
than parameter count. SFT accuracy grows log-linearly with supervised data amount, and better
pre-trained models gain less from more data. To add data without human labeling, the authors propose
Rejection sampling Fine-Tuning (RFT): an SFT model generates candidate reasoning paths, the
incorrect ones are filtered out, and the survivors become augmented fine-tuning data. RFT improves
more when the augmented samples contain more distinct reasoning paths, and improves less-performant
models more. Combining rejection samples from several models raises LLaMA-7B on GSM8K to 49.3%,
against 35.9% for SFT. Code and the augmented data are released at
https://github.com/OFA-Sys/gsm8k-ScRel.

## Key Contributions
- Defines RFT as a named procedure and separates it from generic best-of-n filtering by adding calculation-level verification and equation-list deduplication (§3.3; App. A.3, Algorithm 1).
- Identifies distinct reasoning path count, not accepted-sample count, as the quantity that tracks RFT gain (§3.3, Figure 4; Tab. 2).
- Shows RFT gains shrink as the base model improves, and are negative for LLaMA-33B self-sampling (§3.3, Tab. 1).
- Shows pooling rejection samples across models (D'_U13B, D'_U33B) beats self-sampling at every size up to 33B (§3.3, Figure 5-6).
- Gives a FLOPs and GPU-hour comparison putting SFT at ~1e-5 and RFT at ~1e-4 of pre-training cost (§4.2, Tab. 4).

## Key Figures/Tables to Study
- **Table 1** — ICL / SFT / RFT k=100 maj1@1 and maj1@100 across LLaMA and LLaMA-2 sizes.
- **Table 2** — distinct reasoning paths per question as a function of k.
- **Table 3** — RFT-U13B against other open and proprietary baselines.
- **Table 4** — FLOPs and A100 hours for pre-training, SFT, RFT inference, and RFT training.
- **Figure 4** — accuracy against k, with and without deduplication.
- **Figure 6** — Venn diagram of which model contributed which calculation paths to D'_U33B.

## Technical Details
- Procedure: for each training question `q_i`, sample k = 100 candidate paths at temperature 0.7; discard paths whose final answer differs from the gold answer or whose in-line equations fail Python evaluation; for each distinct equation list keep one path, choosing the one with the largest summed Levenshtein distance to the already-kept paths; fine-tune the **base** model on `D ∪ {accepted paths}` (§3.3; App. A.3, Algorithm 1). Different operand order (`3+4=7` vs `4+3=7`) and different equation order count as distinct (§3.3).
- Fine-tuning: 3 epochs, batch size 128, peak LR 2e-5 with 3% warmup, evaluated at the final epoch, on NVIDIA A100 (App. A.1). maj1@1 uses greedy decoding; maj1@100 samples 100 times at temperature 0.7 and majority-votes (App. A.1; §3.3).
- GSM8K training set is 7,473 questions; RFT inference GPU hours in Tab. 4 are for those questions at 100 samples each (Tab. 4 caption).
- SFT vs RFT k=100 maj1@1 / maj1@100 (Tab. 1): LLaMA-7B 35.9/48.7 → 41.7/52.7; LLaMA2-7B 41.6/55.4 → 47.5/58.7; LLaMA-13B 43.0/55.2 → 49.1/59.9; LLaMA2-13B 50.0/61.7 → 54.8/65.4; LLaMA-33B 54.6 → 54.5 (no gain).
- RFT-U13B (104K samples pooled from LLaMA-7B, LLaMA2-7B, LLaMA-13B, LLaMA2-13B SFT models): 49.3/61.8 (7B), 50.3/65.6 (7B-2), 52.1/66.2 (13B), 55.4/69.1 (13B-2), 56.5 (33B), 59.0 (65B), 62.3 (70B-2) — the 65B and 70B rows are at or below their SFT accuracies of 59.3 and 63.2 (Tab. 5-6).
- Gains over SFT from pooling: +13.4 (LLaMA-7B), +8.7 (LLaMA2-7B), +9.1 (LLaMA-13B), +5.4 (LLaMA2-13B) (§1).
- Distinct paths per question by k (Tab. 2, 7B column): k=1 → 1.17; k=3 → 1.44; k=6 → 1.74; k=12 → 2.20; k=25 → 2.93; k=50 → 3.94; k=100 → 5.25; the pooled 400-sample U13B set → 12.84. LLaMA-33B reaches only 2.78 at k=100.
- k=3 already beats SFT by about 2 points; returns fall as k doubles (§3.3).
- Deduplication is not a loss: k=100 with and without dedup perform similarly, and dedup is better on 3 of 4 models with much less training time (§3.3).
- Cost (Tab. 4, A100 80GB): LLaMA-7B pre-train 82k GPU-hours, SFT 0.6, RFT inference 10, RFT-U33B training 9. LLaMA-33B: pre-train 530k, SFT 80, RFT inference 4.5k, RFT training 1.2k.
- Scaling claim: from LLaMA-7B to LLaMA2-7B, 4.2e22 additional pre-training FLOPs buy 2.1 points of RFT-U33B accuracy and 0.05 of pre-training loss (§4.2).

## Recipe ledger
| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| LLaMA / LLaMA-2 SFT on GSM8K | 7B-70B | SFT | Epochs; batch size | 3; 128 | arXiv:2308.01825v2 App. A.1 | verified 2026-09-18 | App. A.2: 3 epochs is poor on small data subsets, so an epoch search {3, data-fraction} is used there (Tab. 5) |
| LLaMA / LLaMA-2 SFT on GSM8K | 7B-70B | SFT | Peak LR; warmup | 2e-5; 3% | App. A.1 | verified 2026-09-18 | no ablation reported |
| LLaMA / LLaMA-2 SFT models | 7B-33B | distill-SFT (self-generated) | Sampling k; temperature | 100; 0.7 | §3.3; App. A.3 | verified 2026-09-18 | Figure 4: accuracy vs k ∈ {1,3,6,12,25,50,100}; returns diminish as k doubles |
| LLaMA-33B-SFT | 33B | distill-SFT (self-generated) | Sampling temperature (alternative) | 1.0 | §3.3 | verified 2026-09-18 | §3.3: 82.4 correct and 4.77 distinct paths per question at t=1.0 vs 88.7 correct and 2.78 distinct at t=0.7 |
| RFT-U13B | 7B-70B | distill-SFT | Augmented set size | 104K samples | Tab. 5 | verified 2026-09-18 | Figure 5: U13B beats self-sampled k=100 at every size up to 33B |
| RFT k=100 | 7B-33B | distill-SFT | Augmented set size | ~47K samples | Tab. 5 | verified 2026-09-18 | Tab. 2: 5.25 distinct paths per question at k=100 (7B) |
| RFT | 7B-33B | distill-SFT | Fine-tuning target | the base LLM, not the SFT model | §3.3 | verified 2026-09-18 | no ablation reported |
| RFT (all) | 7B-70B | distill-SFT | Compute | A100 80GB; ZeRO3 for 33B+ | Tab. 4 caption | verified 2026-09-18 | Tab. 4: RFT cost ≈ 1e-4 of pre-training |

## Findings relevant to generality
- The authors attribute the gain to reasoning-path diversity: RFT "provides multiple reasoning paths which makes LLMs have better reasoning generalization" (§1). This is the paper's **Interpretation**; the supporting measurement is that accuracy tracks distinct-path count rather than sample count (Figure 4, Tab. 2).
- After RFT on D'_U13B, the trained models produce more distinct correct calculation processes per test question than SFT or self-sampled RFT models, measured by sampling 100 paths per test question at temperature 0.7 (§4.1, Figure 7).
- The method does not help strong models: LLaMA-33B self-sampled RFT is 54.5 against SFT 54.6, and RFT-U13B at 65B/70B is 59.0/62.3 against SFT 59.3/63.2 (Tab. 1, Tab. 5). The authors attribute this to larger SFT models overfitting the training questions and generating fewer distinct paths (§3.3).
- Evaluation is GSM8K only; the paper regresses no scaling law and lists 65B/70B RFT and math-corpus pre-training as missing (§7 Limitations).

## Findings relevant to negative feedback
- Negatives here are of type (1), negative marginal value: wrong-answer and wrong-calculation paths are discarded, not trained against. The filter is two-stage — final-answer match plus Python evaluation of every in-line equation — so answer-correct-but-calculation-wrong paths are also removed (§3.3).
- The paper reports a separate attempt to use failures as content: a self-revising model trained to rewrite a sampled path into the gold path reaches 36.09% against a 35.90% fine-tuning baseline at K=1 and **degrades** as K grows, which the authors attribute to lexical distribution mismatch between train-set and test-set samples. Selecting the sampled path with the largest Levenshtein distance from the gold path restores a gain across all K; an N-fold split does not (App. D.2, Figure 8).

## Connections
- [[star]] — the earlier self-generated-rationale loop; RFT adds calculation verification and path deduplication.
- [[rest-em]] — later formalization of the same generate-filter-finetune loop as expectation-maximization.
- [[best-of-n]] and [[west-of-n]] — inference-time and preference-data uses of the same accept-the-top-scorer step.
- [[iterative-sft-rl]] — stacks where rejection-sampling SFT is one round.
- [[llama-2]] — applies rejection sampling inside an RLHF loop with a reward model rather than a verifier; a different artifact, previously mis-cited as the source of RFT.

## Verification
- Checked on 2026-09-18 against: https://arxiv.org/abs/2308.01825 (arXiv v2, 2023-09-13).
- Corrections to the previous card version:
  - Title "Rejection-Sampling Fine-Tuning", authors "Commonly surfaced in Llama 2 and later open post-training recipes", URL https://arxiv.org/abs/2307.09288 (the Llama 2 report) → the card now describes the artifact that introduced RFT, Yuan et al. 2023, arXiv:2308.01825, with exact authors. The old card mixed a method name with a different paper's URL.
  - "Rejection-sampling fine-tuning is not a standalone algorithm paper so much as a recurring post-training recipe" → RFT is defined and named in §3.3 of a single paper, which reports the ablations the old card lacked.
  - "Keep top candidates" scored by "RM, judge model, or rule-based verifier" → the paper accepts all paths passing answer and calculation checks, then deduplicates by equation list; it does not rank by a reward model (§3.3).
  - "Often delivers large quality gains before expensive online RL" (no number, no condition) → gains stated per model with loci, including the sizes where RFT gives no gain.
- Removed as unsupported by the source:
  - "Turned policy sampling into a self-improving supervised-data loop" — the paper cites prior rejection-sampling augmentation work (Huang et al. 2022; Zelikman et al. 2022; Ni et al. 2023; Zhu et al. 2023) in §2 and does not claim priority for the loop itself.
  - "Provides cleaner data than raw teacher distillation because selection is prompt-specific" — no comparison against teacher distillation is run.
  - "this is often the highest-leverage first iteration" — not measured; the paper's own conclusion is that lowering pre-training loss dominates (§4.2, §5).
- Not reported by the source: results on non-math tasks; RFT for 65B/70B self-sampling; any scaling law fit; reward-model-scored variants.
