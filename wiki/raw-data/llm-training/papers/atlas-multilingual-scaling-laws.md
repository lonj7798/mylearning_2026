<!-- scope: ATLAS (Longpre et al., ICLR 2026): 774 multilingual pretraining/finetuning runs on MADLAD-400; repetition- and transfer-aware scaling law, 38x38 bilingual transfer matrix, curse-of-multilinguality law for adding languages, pretrain-vs-finetune crossover
     deps: [[chinchilla-compute-optimal]], [[data-constrained-scaling]]
     see-also: [[multilinguality-curse-250-languages]], [[uberweb-multilingual-curation]], [[fineweb-2]], [[vocabulary-scaling-laws]]
-->

# ATLAS: Adaptive Transfer Scaling Laws for Multilingual Pretraining, Finetuning, and Decoding the Curse of Multilinguality
- **Core Insight:** Across 774 training runs (10M-8B parameters, MADLAD-400), the Adaptive Transfer Scaling Law reaches R² = 0.98 overall and R²(M) = 0.82 on held-out multilingual mixtures, versus 0.64 and 0.61 for the Chinchilla law (Table 1), and its curse-of-multilinguality fit (ϕ = 0.11, ψ = −0.04) implies that serving 4× as many languages at unchanged loss needs 1.4× parameters and 2.74× total tokens (§5).
- **Guideline:** When the number of evenly sampled languages grows from K to rK and per-language loss must not degrade, scale parameters by r^0.243 and total tokens by r^0.728 (compute by r^0.970), because these are the compute-optimal iso-loss multipliers of the paper's all-language fit (§5, Fig. 5); the fit uses uniform language sampling, no repetition term and models up to 8B, so skewed mixtures need their own fit.
- **Authors:** Shayne Longpre, Sneha Kudugunta, Niklas Muennighoff, I-Hung Hsu, Isaac Caswell, Alex Pentland, et al. (MIT, University of Washington, Stanford University, Google Cloud AI, Google DeepMind)
- **Year:** 2025 (arXiv v1 2025-10; ICLR 2026)
- **URL:** https://arxiv.org/abs/2510.22037
- **Source type:** paper
- **Relevant topics:** multilingual scaling laws, data repetition, cross-lingual transfer, curse of multilinguality, data mixtures, continued pretraining from multilingual checkpoints

## Abstract
Scaling-law research has focused on English although major models serve users in many languages. The authors run what they describe as the largest multilingual scaling-law study to date: 774 training experiments, 10M-8B parameters, 400+ training languages and 48 evaluation languages. They introduce ATLAS for monolingual and multilingual pretraining, which improves out-of-sample fit over existing laws, often by more than 0.3 R². They derive a cross-lingual transfer matrix over 38 × 38 = 1,444 language pairs, a language-agnostic law for scaling model size and data when adding languages without losing performance, and compute crossover points between pretraining from scratch and finetuning a multilingual checkpoint.

## Key Contributions
- ATLAS, a one-stage, repetition-aware law with separate terms for target, transfer, and other-language tokens (§3, Eqs. 1-3, Table 1).
- A 38 × 38 bilingual transfer matrix and its relation to script, family and symmetry (§4, Figs. 2-3, C.2).
- A scaling law in the number of training languages K, with closed-form compute-optimal multipliers for K → rK (§5, Eq. 7, App. B.7).
- A pretrain-from-scratch vs finetune-from-Unimax crossover in tokens and compute (§6, Figs. 6-7).

## Key Figures/Tables to Study
- Fig. 1: optimal N-D trajectories for six languages under three vocabulary/training regimes. Table C.1: fitted per-language laws.
- Table 1: R² on held-out N, D, C and mixture M for each law, with ATLAS term ablations.
- Fig. 2 / C.2: transfer matrix (30 × 30 / 38 × 38). Fig. 3: symmetry and similarity. Fig. C.1: transfer vs N and D.
- Fig. 4: relative loss vs K, N, D. Fig. 5: iso-loss frontiers for K → rK. Figs. 6-7: pretrain vs finetune crossover.
- Tables B.1-B.5: experiment inventory, hyperparameters, model shapes, Unimax sampling rates, capacity mixtures.

## Technical Details
- **Data and metric.** MADLAD-400 (CommonCrawl-based, over 400 languages); vocabulary-insensitive loss on per-language test sets of min(20M tokens, 20%) plus Flores-101 (§2, App. B.2). The paper gives 48 evaluation languages (abstract, L_eval in Table B.1) and 50 test-set languages (§2, App. B.1).
- **Runs.** Table B.1 totals 774: 140 monolingual, 20 Unimax, 70 multilingual-vocabulary monolingual, 290 language-pair, 120 capacity, 134 finetuning runs. Model sizes range from 9,044,352 (scale 0) to 8,452,190,208 parameters (scale 46) (Table B.3). §2 prose gives "over 750" runs and sizes "10M − 2B".
- **ATLAS** (§3): `L(N, D_eff) = E + A/N^α + B/D_eff^β`, `D_eff = S_λ(D_t; U_t) + Σ_{i∈K_t} τ_i S_λ(D_i; U_i) + τ_other S_λ(D_other; U_other)`, with `S_λ(D; U) = D` if D ≤ U and `U[1 + (1 − exp(−λ(D/U − 1)))/λ]` if D > U.
  N: parameters; E: irreducible loss; A, B: coefficients; α, β: exponents; D_t: target-language tokens; U: unique tokens of a source; λ: repetition parameter shared across sources; K_t: the 3 languages most co-sampled with t; τ_i, τ_other: transfer weights (τ_i initialized from §4 transfer scores); D_other = D_tot − D_t − Σ_{K_t} D_i.
- **Held-out R² splits** (App. B.3): R² random 20%; R²(D) top 20% of D; R²(N) N = 660M and 8B; R²(C) largest C = 6ND; R²(M) all mixtures that are not monolingual, bilingual or Unimax.
- **Table 1, monolingual (R² / R²(N) / R²(D) / R²(C)).** Chinchilla 0.94 / 0.68 / 0.94 / 0.90; data-constrained law 0.93 / 0.78 / 0.93 / 0.88; ATLAS (D_t only) 0.92 / 0.88 / 0.91 / 0.88. Averaged over EN, FR, RU, ZH, HI, SW.
- **Table 1, multilingual (… / R²(M)).** Chinchilla 0.64 / −0.99 / 0.72 / 0.66 / 0.61; MSL (He et al. 2024) 0.67 / −0.65 / 0.73 / 0.67 / 0.70; ATLAS D_t only 0.70 / −0.75 / 0.80 / 0.72 / 0.64; + D_other 0.98 / 0.89 / 0.97 / 0.97 / 0.66; + transfer languages 0.98 / 0.89 / 0.96 / 0.98 / 0.82. Adds ES, DE to the average. The §3 text prints MSL R²(M) as 0.69.
- **Compute-efficiency tax** (§3, Fig. 1). Multilingual vocabulary and Unimax training shift the optimal frontier upward relative to monolingual vocabulary and training, most for English; Hindi and Swahili curves bend upward from repeated epochs. Unique tokens: EN 2.8T, RU 705B, FR 363B, ZH 125B, HI 7.9B, SW 770M (Table C.1).
- **Bilingual Transfer Score** (§4, App. B.5): `BTS_{s→t} = −(σ_bi(L_t(d_mono)) − 2·d_mono) / d_mono`.
  d_mono: reference token count of the monolingual model (42B tokens, about 10,000 steps at 2B); L_t(d): monolingual loss on t after d tokens; σ_bi(ℓ): tokens a 50/50 (s, t) bilingual model needs to reach loss ℓ. BTS = 0: no transfer; > 0: positive transfer; < 0: interference.
- **Transfer matrix.** 2B models. Direct BTS for 80 pairs (§4 footnote; App. B.6 says 90 experiments); the rest is predicted by a 300-tree random forest with 5-fold CV R² = 0.85 and Spearman ρ = 0.88 (App. B.6).
- **Transfer results** (§4). English is a top-5 source for 19 of 30 targets, French 16, Spanish 13, Hebrew 11. Urdu and Pashto show negative transfer with all other languages. Same script: mean score −0.23 vs −0.39 for different scripts; family and script effects p < .001. Overall A→B vs B→A Pearson r = −0.11; same family and script pairs are closer to symmetric.
- **Transfer vs scale** (App. C.2). The gap between synergistic (es→pt) and interfering (en→zh) pairs appears early and stays; larger models move interfering pairs toward zero.
- **Curse of multilinguality** (§5, Eq. 7): `L(K, N, D_t) = L∞ + A·K^ϕ/N^α + B·K^ψ/D_t^β`, D_tot = K·D_t under even sampling. K: number of training languages; ϕ: capacity exponent in K; ψ: data exponent in K (ψ < 0 means positive transfer). Fit R² ≥ 0.87; > 0.8 for each of 8 languages; all-language fit ϕ = 0.11, ψ = −0.04 (§5). Mixtures of 4 to 50 languages, 11 model sizes, 120 runs (Tables B.1, B.5).
- **Iso-loss multipliers** (§5, App. B.7): N′/N = r^{ϕ/α}, D_t′/D_t = r^{ψ/β}, D_tot′/D_tot = r^{1+ψ/β}, C′/C = r^{1+ϕ/α+ψ/β}. Fig. 5: (tokens, params) = (×1.66, ×1.18) for 2K, (×2.74, ×1.4) for 4K, (×4.54, ×1.66) for 8K, (×7.52, ×1.96) for 16K. Worked example (§5): at 4K, 2.74/4 means 1 − 0.685 = 32% fewer tokens per language. N has larger marginal effect than D_tot: |∂S/∂log N| > |∂S/∂log D| (§5, Fig. 4).
- **Pretrain vs finetune** (§6, Fig. 6). 2B models; finetuning from the Unimax checkpoint leads early, and pretraining from scratch overtakes it after EN 144B, JA 145B, PT 148B, DE 160B, RU 189B, VI 234B, ES 239B, ZH 283B tokens. Unimax samples English at 5% and each other language at 1.4% (§6; 1.42% in Table B.4).
- **Crossover compute** (§6, Fig. 7). The caption prints C = 10283128 × N^1.65, the §6 text prints log(C) = 1113708 × N^1.65, and the figure legend prints exponent 1.54.

## Recipe ledger
| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| ATLAS study models (mono, bilingual, Unimax, capacity) | 10M-8B | pretrain | LR schedule; decay period | WSD; 10% | arXiv:2510.22037v2 App. B.2 Table B.2 | verified 2026-09-14 | cites Hu et al. 2024, Hägele et al. 2024; no ablation reported |
| same | 10M-8B | pretrain | Optimizer; base LR; warmup | AdamW; 2e-4; 1000 steps | Table B.2 | verified 2026-09-14 | "initial experiments" on LR, batch, optimizer, dropout mentioned (B.2); no table |
| same | 10M-8B | pretrain | Sequence length; training steps | 2048; 30k+ (varied per experiment) | Table B.2 | verified 2026-09-14 | no ablation reported |
| same | 10M-8B | pretrain | Dropout | 0.1 | Table B.2 | verified 2026-09-14 | little difference vs 0.0 except low-resource languages with repeated epochs (B.2); no table |
| same | <150M / <1B / <2B / 2B+ | pretrain | Batch size (unit not stated) | 256 / 512 / 1024 / 2048 | Table B.2 | verified 2026-09-14 | "following initial empirical results"; no table |
| same | 10M-8B | pretrain | Vocabulary | 64k SentencePiece (+512 special tokens), T = 100, 99.9995% character coverage; monolingual variant trained per language | §2, Table B.2, Table B.3 | verified 2026-09-14 | Fig. 1 compares monolingual vs multilingual vocabulary |
| Unimax base checkpoints | 10M-8B | pretrain | Language sampling | EN 5.00%, 1.42% for most high-resource languages, 420 languages | App. B.4, Table B.4 | verified 2026-09-14 | adopted from Chung et al. 2023; no ablation reported |
| Unimax base checkpoints | 10M-8B | pretrain | Training tokens | 1T | App. B.4, B.5.2 | conflict | §6 states "trained for 1B tokens"; B.4 and B.5.2 state 1T |
| Finetuned models (continued monolingual pretraining) | 10M-8B | mid-train | Hyperparameters | same as pretraining, LR schedule reset, at least 30k+ steps | App. B.4 | verified 2026-09-14 | no ablation reported |
| Capacity models | 11 sizes | pretrain | Mixture sampling | uniform over K ∈ {4, 6, 8, 12, 16, 24, 32, 50} languages | §5, Table B.5 | verified 2026-09-14 | design choice for models serving all K languages (§5) |

Not reported (checked body and App. A-C): Adam betas, weight decay, gradient clipping, batch unit (sequences or tokens), precision, hardware, total compute.

## Findings relevant to generality
- **Adding languages costs per-language loss.** K affects relative loss more than N or D, and the penalty grows monotonically with K; larger N and D_tot reduce it, with N more effective (§5, Fig. 4). The ϕ > 0, ψ < 0 fit means a capacity-driven penalty offset in part by positive transfer (§5).
- **Transfer is structured and asymmetric.** Shared script and family predict positive transfer; benefit from A to B cannot be assumed to imply benefit from B to A (§4). English benefits less from other languages than they benefit from English (Interpretation, §3, Fig. 1).
- **Scale reduces interference.** Larger models move negative transfer scores toward zero (App. C.2).
- **Starting point.** Below 144B-283B target tokens, depending on the language, continued pretraining from a multilingual checkpoint reaches lower loss than pretraining from scratch at equal compute (§6); the crossover depends on the base mixture and its training length (§6 limitation).
- **Measurement limits.** All results are loss on held-out text; no downstream task accuracy is reported.

## Connections
- [[chinchilla-compute-optimal]] — the Chinchilla law ATLAS extends and compares against (Table 1).
- [[data-constrained-scaling]] — Muennighoff et al. data-constrained law; ATLAS is a one-stage variant of its repetition term (§3).
- [[kaplan-scaling-laws]] — earlier English scaling laws cited as the prior focus (§1).
- [[multilinguality-curse-250-languages]] — Chang et al. 2024, cited for the curse of multilinguality in smaller models (§1, §5).
- [[uberweb-multilingual-curation]] — argues many multilingual regressions come from data quality rather than capacity; a counterpoint to the capacity framing.
- [[fineweb-2]] — multilingual pretraining data pipeline; ATLAS uses MADLAD-400 instead.
- [[vocabulary-scaling-laws]], [[tokenizer-language-unfairness]] — vocabulary effects behind the compute-efficiency tax of multilingual vocabularies.
- [[data-mixing-laws]], [[regmix]], [[doremi]] — mixture-optimization methods; ATLAS supplies per-language transfer terms for such mixtures.
- [[llama-3]] — cited as training on 8% non-English tokens with brief multilingual scaling-law discussion (§1).

## Verification
- Created on 2026-09-14 from https://arxiv.org/abs/2510.22037 (arXiv v2, 25 Feb 2026; full PDF including App. A-C).
- Audit claims not found in the source: none.
- Inconsistencies inside the source: model-size range 10M-8B (abstract, App. B) vs 10M-2B (§1 prose); 774 runs (Table B.1) vs "over 750" (§2); 80 vs 90 directly measured transfer pairs (§4 vs App. B.6); MSL R²(M) 0.69 (text) vs 0.70 (Table 1); Unimax checkpoint 1B (§6) vs 1T tokens (App. B.4); crossover-compute constants and exponents differ between §6 text, Fig. 7 caption and legend.
