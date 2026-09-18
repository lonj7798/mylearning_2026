<!-- scope: Chinchilla scaling law extended with lifetime inference cost (FLOPs and dollars); 47-run validation of loss and downstream accuracy up to 10,000 tokens per parameter
     deps: [[chinchilla-compute-optimal]]
     see-also: [[kaplan-scaling-laws]], [[overtraining-downstream-scaling]], [[overtrained-lms-harder-to-finetune]], [[chinchilla-replication]], [[cooldown-scaling-beyond-fixed-durations]]
-->

# Beyond Chinchilla-Optimal: Accounting for Inference in Language Model Scaling Laws
- **Core Insight:** When the Chinchilla loss law is combined with inference cost, a developer who expects about 10^9 inference requests should train a smaller model on more tokens; for a 30B-Chinchilla-quality model with 10^13 inference tokens, a 13.6B model trained on 2.84× the data uses 28% fewer total FLOPs (Abstract; §2).
- **Guideline:** When expected lifetime inference tokens approach or exceed the pretraining token count, choose a smaller model trained past the Chinchilla ratio, because total cost falls (§2 Fig. 2; §6 Fig. 6) and 47 validation runs show no loss plateau up to 10,000 tokens per parameter (§4); when inference demand is far below the pretraining token count, the Chinchilla allocation is already close to compute-optimal (§2).
- **Authors:** Nikhil Sardana, Jacob Portes, Sasha Doubov, Jonathan Frankle (Databricks MosaicML)
- **Year:** 2023 (arXiv v1 2023-12; ICML 2024, PMLR 235)
- **URL:** https://arxiv.org/abs/2401.00448
- **Source type:** paper
- **Relevant topics:** scaling laws, compute-optimal training, inference cost, over-training, tokens per parameter, scaling-law fitting

## Abstract
Scaling laws such as Chinchilla estimate loss from parameter count and training tokens but ignore inference cost. The authors modify the Chinchilla law to find the parameter count and pretraining data size that minimize the cost of training plus serving a model of a fixed quality (pretraining loss) under a given inference demand. They do this in FLOPs and in dollar cost. They find that developers expecting about 1B requests should train smaller models for longer than Chinchilla-optimal. They train 47 models to test whether quality keeps improving at extreme tokens-per-parameter ratios and find improvement up to 10,000. Refitting the Chinchilla coefficients shows that a law fit only on typical ratios overestimates the benefit of extra tokens at extreme ratios.

## Key Contributions
- A constrained objective: minimize training plus inference FLOPs subject to `L(N, D_tr) = ℓ` (§2 Eq. 2–3).
- A proof that the optimum has no general closed form once inference is included; the authors solve it numerically with Newton root-finding (§2; App. A Theorem A.1).
- A dollar-cost version with separate MFU for training, prompt tokens, and generated tokens, and separate cost per FLOP (§6 Eq. 5–6).
- 47 MPT-style models from 150M to 6B parameters at 10 to 10,000 tokens per parameter, evaluated on loss and a 5-category downstream Gauntlet (§3; App. C Table 4).
- A refit of the Chinchilla parametric law on progressively more extreme runs (§5 Table 1, Fig. 5).

## Key Figures/Tables to Study
- Fig. 1: 13B model with 2T inference tokens versus a 7B model on more data.
- Fig. 2 and Fig. 6: ratios of total cost, parameters, and tokens versus Chinchilla-style models, in FLOPs and in dollars.
- Fig. 3: loss and Gauntlet Average versus tokens per parameter; Gauntlet Average versus loss.
- Fig. 4: loss versus FLOPs grouped by tokens-per-parameter ratio.
- Table 1 and Fig. 5: fitted coefficients by data subset. App. B Tables 2–3: selected configurations.

## Technical Details
- **Loss law.** `L(N, D_tr) = E + A/N^α + B/D_tr^β` (§2 Eq. 1). `N` is parameter count, `D_tr` pretraining tokens, `E` irreducible loss, `A, B, α, β` fitted constants. The analysis uses A = 406.4, B = 410.7, E = 1.69, α = 0.336, β = 0.283; the rounded Chinchilla values 0.34 and 0.28 are replaced following De Vries (App. A, footnote 2).
- **Cost model.** Total FLOPs `6·N·D_tr + 2·N·D_inf`, with 6N FLOPs per training token and 2N per inference token (§2 Eq. 3). `D_inf` is the sum of input plus output tokens over all requests (§2). Derived example: serving 2T tokens on a 13B model costs 2 × 13e9 × 2e12 = 5.2e22 inference FLOPs.
- **Assumptions.** Inference demand is independent of model size at equal loss, and demand can be estimated before training (§2; App. A). Footnote 1 notes that smaller models may attract more demand through lower latency.
- **Compute-optimal examples.** 13B model with 2T inference tokens: a 7B model on more data saves 1.7e22 FLOPs (17%) (Fig. 1). 7B-Chinchilla quality with 10^11 inference tokens: 6B parameters on 1.18× the data (§2). 30B-Chinchilla quality with 10^13 tokens: 13.6B on 2.84× the data, 28% fewer FLOPs (§2). 70B-Chinchilla model (4.26T training tokens) with 10T inference tokens: 41.6B on 7.92T tokens, 12% fewer FLOPs (App. B.1 Table 2).
- **Cost-optimal settings.** Training and prompt MFU 50%, generation MFU 1%; 70 input and 215 output tokens per request from LMSYS-Chat averages; training on A100-80GB at $1.50/hr and inference on A100-40GB at $1.10/hr after INT8 quantization (Fig. 6 caption; App. B.2–B.3).
- **Cost-optimal examples.** 30B-Chinchilla quality with 1.5B requests: a 16B model on 3.35T tokens costs 17% less (§6). At 2T inference tokens (7.02B requests), a Chinchilla-70B model needs 1.3% more FLOPs than the FLOP-optimal model but costs 36% more than the cost-optimal model; the authors attribute the gap to the 50× lower MFU of output tokens (§6). 70B-Chinchilla with 35.1B requests: 21.5B parameters on 27T tokens, 54% savings (App. B.2 Table 3).
- **Experiments.** MPT architecture with ALiBi and GQA; data is "trillions of tokens of general web text and code", one epoch, no repeats (§3; App. C). The 150M model reaches 10,000 tokens/parameter; larger models were stopped at 1,000 or lower because of compute, and the 6B model has one run at 20 (§4; App. C Table 4).
- **Gauntlet.** World knowledge, commonsense reasoning, reading comprehension, language understanding, symbolic problem solving; each task is baseline-subtracted, normalized, and weighted equally (§3).
- **Fitting.** Huber loss with δ = 10^-3 on log-sum-exp predictions, minimized with L-BFGS from a grid of initializations, following Chinchilla App. D.2 (§5 Eq. 4).

## Recipe ledger
| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| MPT-style sweep | 150M–6B | pretrain | optimizer | Lion, β1 = 0.9, β2 = 0.95 | arXiv:2401.00448v3 App. C | verified 2026-09-14 | no ablation reported |
| MPT-style sweep | 150M–6B | pretrain | weight decay | equal to the learning rate | App. C | verified 2026-09-14 | no ablation reported |
| MPT-style sweep | 150M–6B | pretrain | warmup / schedule | "cosine warmup (αf = 0.1) with a duration equal to 3 times the number of model parameters" | App. C | verified 2026-09-14 | no ablation reported |
| MPT-style sweep | 150M–6B | pretrain | grad clip; max sequence length | norm clipping, threshold 1; 4096 tokens | App. C | verified 2026-09-14 | no ablation reported |
| MPT-style sweep | 150M | pretrain | peak LR; batch size (unit not printed) | 4.603e-4; 160 (960 for the 10,000 tok/param run) | App. C Table 4, footnote 3 | verified 2026-09-14 | smaller batch so short runs get enough steps (App. C); no ablation reported |
| MPT-style sweep | 370M | pretrain | peak LR; batch size | 3.453e-4; 320 | App. C Table 4 | verified 2026-09-14 | no ablation reported |
| MPT-style sweep | 750M | pretrain | peak LR; batch size | 2.302e-4; 480 | App. C Table 4 | verified 2026-09-14 | no ablation reported |
| MPT-style sweep | 1.3B | pretrain | peak LR; batch size | 1.726e-4; 960 | App. C Table 4 | verified 2026-09-14 | no ablation reported |
| MPT-style sweep | 2.5B | pretrain | peak LR; batch size | 1.381e-4; 960 | App. C Table 4 | verified 2026-09-14 | no ablation reported |
| MPT-style sweep | 6B | pretrain | peak LR; batch size | 8.632e-5; 960 | App. C Table 4 | verified 2026-09-14 | no ablation reported |
| MPT-style sweep | 150M–6B | eval-gate | loss reported | smoothed final training loss over the last ten batches | App. C | verified 2026-09-14 | no ablation reported |

## Findings relevant to generality
- **Downstream breadth (Result, single study).** The Gauntlet Average rises with tokens per parameter with no saturation point observed, and at lower loss a given loss decrease gives a larger accuracy gain (§4, Fig. 3b–c).
- **Loss as proxy.** Loss and Gauntlet Average are tightly correlated in these runs (§4, Fig. 3c). Per-category averages correlate less; symbolic problem solving stays near zero for all models (§4; App. D Fig. 7).
- **Efficiency by ratio.** For ratios ≥ 20, loss-versus-FLOPs lines are nearly parallel up to 500 tokens/parameter; ratios below 20 are less compute-efficient (§4, Fig. 4).
- **Fit bias.** Fitted coefficients change as more extreme runs are added (≤ 100 tokens/param: α = 0.08, β = 0.13, E = 0.17; all runs: α = 0.18, β = 0.24, E = 1.45) and the fitted curves become flatter. The authors conclude that laws fit at typical ratios overestimate the loss reduction from extra tokens at extreme ratios; no fit matches the long 150M runs well (§5 Table 1, Fig. 5).
- **Not studied.** Fine-tuning, post-training, or adaptability of over-trained models; models above 6B; ratios above 10,000 (§4; §5; §8).

## Connections
- [[chinchilla-compute-optimal]] — source of Eq. 1 and the coefficients this paper extends.
- [[kaplan-scaling-laws]] — source of the 6N and 2N FLOP approximations (§2).
- [[overtraining-downstream-scaling]] — Gadre et al., 100 models at 20–640 tokens/parameter, discussed in §7.
- [[overtrained-lms-harder-to-finetune]] — later study of fine-tuning cost of over-training, which this paper does not measure.
- [[chinchilla-replication]] — Besiroglu et al., cited for rounding bias in the Chinchilla coefficients (§5, §7).
- [[resolving-scaling-discrepancies]] — Porian et al., cited in §7.
- [[data-constrained-scaling]] — Muennighoff et al., the repeated-data regime this paper excludes (§1).
- [[lmsys-chat-1m]] — source of the 70-input / 215-output token request profile (App. B.2).
- [[cooldown-scaling-beyond-fixed-durations]] — cites this paper for training past compute-optimality to save inference.
- [[llama-2]], [[llama-3]] — 2T and 15T token models cited as over-trained examples (§1, §5).

## Verification
- Created on 2026-09-14 from https://arxiv.org/abs/2401.00448 (arXiv v3, 2025-04-14; full PDF text including App. A–D). v1 date 2023-12-31 and ICML 2024 comment checked on the arXiv abs page.
- Audit claims not found in the source: none.
- Not reported by the source: batch-size unit, total tokens or compute for the 47 runs, dataset name and mixture, downstream results after fine-tuning.
