<!-- scope: Hernandez et al. (OpenAI, Feb 2021) — effective data transferred from English-text pre-training to Python fine-tuning, fitted as a power law in parameters and fine-tuning data; data multiplier; ossification in the high-data regime
     deps: [[kaplan-scaling-laws]]
     see-also: [[scaling-laws-unreliable-downstream]], [[data-constrained-scaling]], [[scaling-laws-forgetting]], [[atlas-multilingual-scaling-laws]], [[data-mixing-laws]]
-->

# Scaling Laws for Transfer
- **Core Insight:** In the low-data regime, the Python data that text pre-training is worth (effective data transferred, D_T) fits D_T = k·D_F^α·N^β with α = 0.18 and β = 0.38 for text → Python, so a 10× larger model is worth about a 100× larger fine-tuning set in this setting (§1.1, Eq. 1.1, Table 1).
- **Guideline:** When fine-tuning data is expensive and the target task is in the low-data regime, fine-tune a pre-trained model at 1% and 10% of the dataset and at several model sizes to estimate α and β before deciding whether to collect more data, because the fitted power law held over more than 4 orders of magnitude of model size and 3 of fine-tuning data (Fig. 2, §6.3); when the fine-tuning set approaches the size needed to saturate a from-scratch model, do not assume pre-training helps, because small pre-trained models ended worse than from-scratch models there (§3.1, Fig. 5).
- **Authors:** Danny Hernandez, Jared Kaplan, Tom Henighan, Sam McCandlish (OpenAI; Kaplan also Johns Hopkins University)
- **Year:** 2021 (arXiv v1 2021-02; single version; preprint)
- **URL:** https://arxiv.org/abs/2102.01293
- **Source type:** paper
- **Relevant topics:** transfer learning, scaling laws, fine-tuning data efficiency, distribution proximity, generality measurement, ossification

## Abstract
The paper studies scaling laws for transfer between distributions in unsupervised fine-tuning. Transformers trained from scratch on a fixed-size dataset stop improving in cross-entropy loss once they are data-limited; models pre-trained on a large language dataset show a reduced but non-zero slope. The authors define effective data transferred as the additional target data a same-size from-scratch model would need to reach the fine-tuned model's loss. In the low-data regime this quantity is a power law of parameter count and fine-tuning dataset size. The authors believe the exponents correspond to the generality of the model and the directed proximity of the two distributions. Pre-training in effect multiplies the fine-tuning dataset size, and transfer scales predictably in parameters, data, and compute.

## Key Contributions
- A data-denominated measure of transfer, D_T, defined relative to from-scratch training of the same model size (§1.1, Fig. 1, §1.2).
- The transfer power law D_T = k(D_F)^α(N)^β and fitted coefficients for two pre-training distributions (Eq. 1.1, Table 1).
- The effective data multiplier (D_F + D_T)/D_F ≈ k·N^β / D_F^(1−α), which decreases as D_F grows (Eq. 1.2).
- Evidence of "ossification": pre-training reduces effective data for small models in the high-data regime (§3.1, §6.6, Figs. 5–6).
- A proposed low-cost experiment for comparing the value of additional fine-tuning data with the value of a larger model (§6.3).

## Key Figures/Tables to Study
- Figure 1 (definition of D_T; 40M model on 3e5 characters, D_T ≈ 1000× D_F); Figure 2 (global fits); Table 1 (k, α, β).
- Figure 3 (from-scratch curves go flat with more parameters; fine-tuned curves only change slope); Figures 5–6 (ossification); Figure 8 (converged compute); Appendix F (contamination control).

## Technical Details
**Definitions (§1.2, App. B)**
- D_F: fine-tuning dataset size in characters. D_T: additional Python characters a from-scratch model of the same size needs to match the pre-trained model's Python loss. D_E = D_F + D_T is total effective data (Eq. B.1).
- N: parameters excluding vocabulary and positional embeddings. L: cross-entropy in nats per token (§1.2).
- D(N): data needed to reach 99% of the infinite-data performance for size N, defined through L(D → ∞) = 0.99·L(D(N)) (§3.1, footnote 7). Low-data regime: D_F ≤ 10% of D(N) (§1.1 footnote 2, App. A).

**Fitted law (§1.1, Table 1)**
- D_T = k·(D_F)^α·(N)^β (Eq. 1.1). Text → Python: k = 1.9e4, α = 0.18, β = 0.38. 50% text + 50% non-Python code → Python: k = 2.1e5, α = 0.096, β = 0.38 (Table 1).
- β is identical for both pre-training mixes; the authors hypothesize that β depends only on architecture and target distribution and measures how the architecture generalizes on the target (§1.1).
- α measures directed proximity of two distributions; smaller α means closer (§1.1). The larger k for the mixture means more transfer in the low-data regime; the smaller α means that benefit shrinks toward the high-data regime (Table 1 caption).
- For text → Python, β ≈ 2α, so a 10× increase in N is worth about a 100× increase in D_F (§1.1).
- Worked check (derived from Table 1): with α = 0.18 and β = 0.38, 10× more D_F multiplies D_T by 10^0.18 ≈ 1.51; 10× larger N multiplies D_T by 10^0.38 ≈ 2.40; 100× more D_F gives 10^0.36 ≈ 2.29.
- Fit procedure: individual logit fits of D_T/(D_F + D_T) per D_F had about the same exponent, so the authors fitted the fits (Eq. C.1, App. C, Fig. 10). The fit breaks down outside the low-data regime (App. D, Fig. 13).

**Regimes and compute (§3)**
- Pre-trained models keep improving with N at fixed D_F where from-scratch models are flat (Fig. 3).
- In the high-data regime, 1M-parameter models trained from scratch on datasets larger than 1e8 characters beat the fine-tuned ones (Fig. 5 caption); small fine-tuned models did not reach from-scratch performance even at 10× or 100× D(N) (§3.1, §6.6).
- Ignoring pre-training compute, fine-tuning is more compute efficient in the low-data regime (Fig. 4, 3e8 Python characters) and stays near the compute-efficient frontier for more of training (§3.2, Fig. 7).
- Fine-tuning on large datasets needs 1–10 epochs for code, similar to from-scratch; the smallest dataset (100,000 tokens) needed 2–5× fewer epochs than from-scratch (App. G, Fig. 15).

**Speculative extrapolations (§6.2)**
- Extrapolated to about 1 character and a GPT-3-size model, text pre-training is estimated to equal 3.7e8 Python characters from scratch; 300 characters multiply D_E by 2.8. For text + code pre-training, D_E = 4.1e9 characters and 300 characters give 1.7×. The authors label this speculative: two orders of magnitude up in model size and five down in data.

## Recipe ledger
| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| Hernandez et al. transformers (all runs) | range spans 4 orders of magnitude; sizes not listed | pre-train, fine-tune, from-scratch | stopping rule | trained to convergence at the optimal early-stopping point | arXiv:2102.01293v1 §2 | verified 2026-09-14 | no ablation reported |
| same | same | all | optimizer; LR | Adam; LR and optimization parameters "similar to" Kaplan et al. 2020, peak LR not printed | §2 | verified (optimizer); not reported (LR value; checked §2, §5, App. A–G) | a handful of LR scans for larger models on small data showed no improvement (§5 item 1) |
| same | same | all | batch size; sequence length | 256 (unit not stated); 2048 tokens | §2 | verified 2026-09-14 | no ablation reported |
| same | same | all | warmup; vocabulary | 3000 steps; 50257 | §2 | verified 2026-09-14 | no ablation reported; for small datasets training ended before warmup finished (§5 item 2) |
| text pre-training set | — | pre-train | data | WebText2, Common Crawl, English Wikipedia, public Internet Books; 24 billion characters | §2 | verified 2026-09-14 | no ablation reported |
| non-Python code half of mixture | — | pre-train | data | 240 billion characters / 340 GB; 21% .c, 18% .java, 17% .js, 12% .cpp, 7.6% php, 6.5% .cs, 4.4% .md, 3.2% .cc, 3.2% .ts, 2.6% .go, 1.8% .m, 1.7% .rb, .55% .sh | §1.1 footnote 3 | verified 2026-09-14 | Table 1 compares against text-only pre-training |
| Python set | — | fine-tune, from-scratch | data; held-out | 22 billion characters (31 GB) from public GitHub; 3% held out for evaluation | §2 | verified 2026-09-14 | no ablation reported |

The paper prints 24 billion characters for the text dataset and 240 billion characters for the code half of the "equal mix"; it does not state how the two were balanced. Compute totals are not reported.

## Findings relevant to generality
- The paper proposes α and β as measures of generality and distribution proximity (abstract, §1.1). This is an interpretation from one target distribution (Python) and two source mixes (Interpretation).
- Adding other programming languages to pre-training raised zero-shot transfer by about 10×, which the authors call modest; they conclude that training on the distribution of interest is preferable when possible (§1.1).
- Ossification: pre-training can reduce effective data when fine-tuning data is plentiful, observed for small models; larger models were not tested in that regime (§3.1, §6.6).
- Contamination control: adding a known 0.3% Python to pre-training raised effective Python data 4.5× instead of the 2× expected if trace Python explained zero-shot transfer (App. F).
- Limits stated by the authors: only Python as target; only unsupervised fine-tuning, so supervised and RL settings are untested; transformers only; hyperparameters tuned for from-scratch language modeling (§5).

## Connections
- [[kaplan-scaling-laws]] — source of the from-scratch loss law (Eq. 6.1 substitutes D_T for D) and of the optimization settings.
- [[data-constrained-scaling]] — later study of the data-limited regime for from-scratch pre-training with repeated data.
- [[scaling-laws-forgetting]] — scaling view of what fine-tuning removes, the counterpart to what pre-training transfers.
- [[atlas-multilingual-scaling-laws]] — transfer scaling laws between languages.
- [[data-mixing-laws]] — fitted mixture-ratio prediction, a direction listed in §6.7 items 6–7.
- [[scaling-laws-unreliable-downstream]] — evidence that downstream task scaling is often not predictable, in contrast to the loss-based transfer fits here.

## Verification
- Created on 2026-09-14 from https://arxiv.org/abs/2102.01293 (arXiv v1, 2021-02-02; the only version), full PDF text including Appendices A–G.
- Audit claims not found in the source: none. The audit wording that the exponents "reflect" generality is stated in the paper as the authors' belief (abstract) and hypothesis (§1.1), and the power-law fit is restricted to the low-data regime (§1.1, App. D).
- Not reported by the source: exact list of model sizes, peak learning rate, batch-size unit, total compute.
