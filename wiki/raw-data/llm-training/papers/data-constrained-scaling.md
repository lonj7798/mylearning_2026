<!-- scope: Muennighoff et al. 2023: scaling laws when unique pre-training data is limited; value of repeated epochs, compute allocation, and code-mixing and filtering as alternatives (GPT-2-architecture models up to 8.7B parameters, 900B tokens)
     deps: [[c4]]
     see-also: [[deduplicating-training-data]], [[d4]], [[rephrasing-the-web]], [[synthetic-data-scaling-laws]]
-->

# Scaling Data-Constrained Language Models
- **Core Insight:** At a fixed compute budget, training on data repeated for up to 4 epochs gives test loss almost equal to training on unique data (an 8.7B model at 4 epochs ends with 0.5% higher validation loss than at 1 epoch), while further repetition has diminishing value, with a fitted decay constant R*_D ≈ 15.4 repetitions (Abstract, §6, App. A).
- **Guideline:** When unique text is the binding constraint, spend extra compute on more epochs faster than on more parameters, repeat for up to about 4 epochs, and add Python code for up to 50% of tokens, because in the paper's runs these choices changed held-out loss negligibly or improved it and did not reduce the 19-task natural-language average (§5, §6, §7, Fig. 6); apply perplexity or deduplication filtering mainly to noisy datasets (§7, App. O).
- **Authors:** Niklas Muennighoff, Alexander M. Rush, Boaz Barak, Teven Le Scao, Aleksandra Piktus, Nouamane Tazi, et al. (Hugging Face, Harvard University, University of Turku)
- **Year:** 2023 (arXiv v1 2023-05; NeurIPS 2023)
- **URL:** https://arxiv.org/abs/2305.16264
- **Source type:** paper
- **Relevant topics:** data-constrained scaling laws, multi-epoch training, compute allocation, code in pre-training mixtures, perplexity filtering, deduplication

## Abstract
Extrapolating the trend of scaling both parameters and training tokens suggests that training data may soon be limited by the amount of text on the internet. The paper studies language-model scaling when data is constrained, with experiments that vary data repetition and compute budget up to 900 billion training tokens and 9 billion parameters. With constrained data and fixed compute, up to 4 epochs of repeated data changes loss negligibly compared with unique data. With more repetition, the value of added compute eventually decays to zero. The authors propose and validate a compute-optimal scaling law that accounts for the decreasing value of repeated tokens and of excess parameters. They also test code augmentation and the removal of common filters as ways to mitigate data scarcity. Models and datasets from 400 training runs are released at github.com/huggingface/datablations.

## Key Contributions
- More than 400 models from 10 million to 9 billion parameters, trained for up to 1500 epochs, with final test loss recorded (§1).
- A data-constrained generalization of the Chinchilla loss formula using effective data D′ and effective parameters N′ (§3.1, Eq. 5–6), fit on 182 runs (§5, App. A).
- Allocation result: with fixed unique data, compute should go to epochs faster than to parameters (§5).
- Return result: up to about 4 epochs the loss penalty is negligible, and meaningful gains stop at around 16 epochs (§6).
- Tests of code filling, perplexity filtering, and deduplication filtering on 19 downstream tasks (§7, App. M–O).

## Key Figures/Tables to Study
- **Figure 1** — return on compute when repeating (4.2B model) and the data-constrained efficient frontier.
- **Figure 3** — empirical and predicted IsoLoss contours for 100M unique tokens.
- **Figure 4** — the three IsoFLOP budgets and eight data budgets each.
- **Figure 6** — downstream average for repetition, code filling, and filtering (4.2B, 84B tokens).
- **App. A Table 1** — fit quality of alternative decay formulas.

## Technical Details
- **Definitions (§3):** D_C is the unique-data budget. U_D = min{D_C, D} is the number of unique tokens used, and R_D = D/U_D − 1 is the number of repetitions (epochs − 1). U_N is the compute-optimal parameter count for U_D (or N if smaller), and R_N = N/U_N − 1.
- **Loss formula (§3.1):** L(N, D) = A / N′^α + B / D′^β + E, where N′ is effective parameters, D′ is effective tokens, and A, B, α, β, E are fitted constants as in Chinchilla.
- **Effective data (Eq. 5):** D′ = U_D + U_D · R*_D · (1 − e^(−R_D / R*_D)). R*_D is a fitted constant: at R_D = R*_D, repeated tokens are worth on average 1 − 1/e of fresh tokens (§3.1). N′ uses the same form with R_N and R*_N (Eq. 6).
- **Limits of the formula (§3.1):** for R_D ≪ R*_D, D′ ≈ D. No amount of repetition gives a better loss than one epoch over U_D + U_D · R*_D fresh tokens.
- **Fitted constants (App. A):** R*_N = 5.309743 and R*_D = 15.387756. Because R*_N < R*_D, excess parameters lose value faster than repeated tokens. The full fitted law is L = 521 / (U_N + 5.3 · U_N(1 − e^(−R_N/5.3)))^0.35 + 1488 / (U_D + 15.4 · U_D(1 − e^(−R_D/15.4)))^0.35 + 1.87 (App. A, Eq. 17).
- **Worked example (derived from Eq. 5 with R*_D = 15.387756):** 4 epochs means R_D = 3, so D′ = U_D(1 + 15.39 × (1 − e^(−3/15.39))) = 3.73 U_D against D = 4 U_D, or 93% of the value of unique tokens. At 16 epochs (R_D = 15), D′ = 10.58 U_D, or 66%.
- **Fit quality (App. A Table 1, v5):** R² = 0.7722 for Eq. 14 (the form used), against 0.4452 when no decay is assumed.
- **Setup (§4):** GPT-2 architecture and tokenizer; subsets of C4 nested so runs with less data use a subset of runs with more data (Fig. 2); the whole available data is repeated and shuffled after each epoch; no early stopping; held-out test loss. Training loss is not used because models overfit repeated data (App. H, Fig. 14).
- **Allocation, 100M unique tokens (§5, Fig. 3):** the one-epoch compute-optimal model has about 7M parameters. The lowest loss is at about 20–60× more parameters and epochs, about 7000× more FLOPs, with more than 50% loss reduction.
- **Allocation at scale (§5):** at 9.3 × 10^21 FLOPs with 25B unique tokens, the model suggested by the data-constrained frontier has 27% fewer parameters than the Chinchilla-suggested model and achieves better loss and downstream performance (Fig. 1, Table 8).
- **Return (§6, Fig. 4–5):** IsoFLOP runs at 2.8B/55B, 4.2B/84B, and 8.7B/178B tokens. The fit underestimates loss for runs whose loss rises midway through training, such as 44 epochs (§6).
- **Deduplicated C4 (App. E, Fig. 10):** 146M-parameter models on 100M unique tokens reach their best loss at 59 epochs with or without extra deduplication.
- **Filtering and code setup (§7, App. N):** maximum data budget D_C = 84B tokens for 4.2B models. Perplexity filtering keeps the 25% of samples with lowest perplexity under a KenLM model trained on Wikipedia, giving 44B tokens repeated about 2 epochs. Deduplication removes documents sharing a 100-character span, giving 21B tokens repeated 4 epochs. Both filters start from about 178B tokens. Evaluation uses 19 tasks with 0–5 shots, 114 rescaled scores per model (§7).

## Recipe ledger
| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| datablations GPT-2-architecture study models | all | pretrain-stable | LR schedule | linear warmup over 1% of tokens to 2e-4, cosine decay to 2e-5 | arXiv:2305.16264v5 App. S; §4 | verified 2026-09-14 | based on prior work and test runs (App. S); no ablation reported |
| same | < 2B | pretrain-stable | batch size | 256 (unit not stated) | App. S | verified 2026-09-14 | no ablation reported |
| same | 2–5B | pretrain-stable | batch size | 512 (unit not stated) | App. S | verified 2026-09-14 | no ablation reported |
| same | > 5B | pretrain-stable | batch size | 1024 (unit not stated) | App. S | verified 2026-09-14 | no ablation reported |
| same | all | pretrain-stable | optimizer | Adam, β1 = 0.9, eps = 1e-8; β2 = 0.999, except β2 = 0.95 for FLOP budgets 9.3 × 10^20 and 2.1 × 10^21 | App. S | verified 2026-09-14 | β2 = 0.95 gave slightly lower final loss and fewer loss spikes than 0.999 (App. S, no table) |
| same | all | pretrain-stable | dropout; weight decay; grad clip | 0.1; 0.1; 1.0 | App. S | verified 2026-09-14 | no ablation reported |
| same | all | pretrain-stable | precision; sequence length; vocabulary | bfloat16; 2048; 50257 | App. S, Eq. 23 | verified 2026-09-14 | no ablation reported |
| same | all | pretrain-stable | compute | up to 256 AMD Instinct MI250X GPUs on up to 64 nodes (LUMI); about 3 million GPU hours in total | App. S | verified 2026-09-14 | not applicable |
| same | 4.2B | pretrain-stable | total tokens; Python share | 84B; 0–90% in steps of 10% (The Stack) | §7, App. M Table 10 | verified 2026-09-14 | up to 50% code: no drop in 19-task average (Fig. 6, 5 seeds) |

## Findings relevant to generality
- **Repetition and downstream tasks:** models trained for 1 epoch and for up to 4 epochs show no significant downstream differences, and performance starts dropping after about 4 epochs (App. L, §7, Fig. 6).
- **Code:** filling up to 50% of the data with Python (42B tokens) shows no deterioration on natural-language tasks; beyond 50% performance decreases (§7). WebNLG and bAbI improve as soon as code is added (§7). At 4.2B, the bAbI score rises from 0.0 with no code to 23.2 with 50% code (App. M Table 10). The authors hypothesize that code teaches long-range state tracking (§7, Interpretation).
- **Effective tokens:** mixing in code gives a 2× increase in effective tokens on natural-language tasks (§1). Doubling data with code and then repeating 4 epochs gives 8× the tokens, which the authors expect to match 8× more unique data (§7).
- **Filtering:** perplexity filtering helps on C4, while deduplication does not improve downstream performance on C4 (§7). Both are more effective on the noisier OSCAR corpus (App. O). The authors note deduplication may have value their benchmark does not measure, such as reduced memorization (§7).
- **Limits (App. Q):** only whole-dataset repetition is studied, not repetition of a fraction; returns from epochs may depend on learning rate, dropout, and optimizer; only C4 and OSCAR are used; fine-tuning data constraints are not studied.

## Connections
- [[c4]] — the main training corpus (§4).
- [[deduplicating-training-data]] — its suffix-array method is used for the 100-character deduplication filter (App. N).
- [[d4]] — cites this paper for loss degradation beyond four epochs of random repetition and tests repetition of a selected subset (D4 §4.2).
- [[rephrasing-the-web]], [[synthetic-data-scaling-laws]] — other responses to limited unique data, not tested in this paper.

## Verification
- Checked on 2026-09-14 against: https://arxiv.org/abs/2305.16264 (arXiv v5, 2025-06-28; full text including App. A–X; version history v1 2023-05-25 to v5).
- Corrections to the previous card version:
  - Title "Data-Constrained Scaling" → "Scaling Data-Constrained Language Models".
  - Authors "Thomas Muennighoff and collaborators" → first author Niklas Muennighoff; Thomas Wolf is the eighth author.
  - "the value of repeating existing data versus adding lower-quality new data depends on where the model sits in the compute-data regime" → the stated results: up to 4 epochs negligible, value decays with R*_D ≈ 15.4, code and filtering tested as alternatives (Abstract, §6, §7).
- Removed as unsupported by the source: "Helped motivate later work on data quality, synthetic augmentation, and dedup"; connections to [[the-pile]], [[dolma]], [[fineweb]], [[phi-textbooks]], [[hf-cosmopedia]] (not discussed by the source).
- Not reported by the source: "R_T ≈ 4" as a fitted constant, and the formula "D′ = U(1 − e^(−R/R_T))". The fitted constant is R*_D = 15.387756, and 4 epochs is the range with negligible loss change (§6, App. A). ch-13 and ch-14 read.md attribute the "R_T ≈ 4" form to this card.
