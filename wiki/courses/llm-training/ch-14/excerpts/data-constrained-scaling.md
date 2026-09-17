---
chapter: ch-14
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/data-constrained-scaling.md
source_url: https://arxiv.org/abs/2305.16264
primary_version: arXiv:2305.16264v5 (2025-06-28; v1 2023-05)
created_at: "2026-09-15"
revised: 2026-09 (generality revision; replaces the 2026-04 excerpt, which used an incorrect formula)
---

# Excerpt: Scaling Data-Constrained Language Models (Muennighoff et al.)

Verbatim quotations and equations used by ch-14 `read.md`, with loci. Authors: Niklas Muennighoff, Alexander M. Rush, Boaz Barak, Teven Le Scao, Aleksandra Piktus, Nouamane Tazi, et al. Checked against the v5 PDF on 2026-09-15. Source type: paper.

## Abstract
> "We find that with constrained data for a fixed compute budget, training with up to 4 epochs of repeated data yields negligible changes to loss compared to having unique data. However, with more repetition, the value of adding compute eventually decays to zero."

## Definitions (§3)
> "we split the Chinchilla total data term D into two parts: the number of unique tokens used, U_D, and the number of repetitions, R_D (i.e. epochs - 1). Given total training tokens D and data budget D_C these terms are simply computed as U_D = min{D_C, D} and R_D = (D/U_D) − 1."

## Motivation for a new law (§5)
> "their parametric fit explicitly relies on the assumption that models are trained for a single epoch only. Thus, there is no guarantee that their scaling predictions hold for repeated data."

## Evaluation on held-out data (§4, App. H)
> "As repeating data can result in extreme overfitting (see Appendix H), we report loss on a held-out test set unless otherwise specified" (§4)
> "when repeating data for multiple epochs, training loss is a bad metric as models will overfit to the limited data available" (App. H)

## Loss form and effective data (§3.1, Eq. 5-6)
- Loss: L(N, D) = A / N′^α + B / D′^β + E, where N′ is effective parameters and D′ is effective data.
- Eq. 5: D′ = U_D + U_D · R*_D · (1 − e^(−R_D / R*_D)).
- Eq. 6: N′ = U_N + U_N · R*_N · (1 − e^(−R_N / R*_N)).
> "Note that for R_D = 0 (no repetitions), D′ = U_D = D." ... "The formula implies that no matter how many times we repeat the data, we will not get a better loss than could be obtained with a single epoch on U_D + U_D R*_D fresh tokens."
> "at R_D = R*_D, the number of effective tokens D′ is U_D + U_D R_D (1 − e^−1) which means that the U_D R_D repeated tokens are worth on average 1 − 1/e fraction of fresh ones."

## Derivation assumption (App. A)
> "Assume that each time a model trains on a token, it learns a 1 − δ fraction of the information in it for some constant 0 ≤ δ ≤ 1." R*_D is defined as (1 − δ)/δ, and "D′ 'plateaus' at U + R*_D U as R_D goes to infinity."

## Fitted constants (App. A, Eq. 17)
> "we are able to get a fairly stable fit resulting in R*_N = 5.309743 and R*_D = 15.387756. Since R*_D > R*_N, excess parameters decay faster."
- Eq. 17: L = 521 / (U_N + 5.3 · U_N (1 − e^(−R_N/5.3)))^0.35 + 1488 / (U_D + 15.4 · U_D (1 − e^(−R_D/15.4)))^0.35 + 1.87, "where U_N = U_D · 0.051".
- The fit uses 182 runs "with parameters varying from 7 million up to 9 billion and epochs ranging from 1 to 500" (App. A).

## Return (§6)
- Figure 4 (IsoFLOP): 2.8B parameters trained for 55B tokens, 4.2B for 84B tokens, 8.7B for 178B tokens, each with several data budgets (epochs).
> "the N = 8.7 billion parameter model trained for four epochs (D_C = 44 billion unique tokens) finishes training with only 0.5% higher validation loss than the single-epoch model (D_C = 178 billion unique tokens)."
> "it significantly underestimates the final test loss of failing models where loss increases midway through training, such as models trained for 44 epochs (not depicted)."
> "Meaningful gains from repeating data can be made up to around 16 epochs (R*_D) beyond which returns diminish extremely fast."

## Allocation (§5)
> "We confirm this at scale by training the data-constrained compute-optimal model for 9.3 × 10^21 FLOPs and 25 billion unique tokens as suggested by our efficient frontier. Despite having 27% fewer parameters, this model achieves better loss and downstream performance than the model suggested by the Chinchilla scaling laws."

## Complementary strategies (§7, 4.2B parameters, 84B total tokens)
> "For repeating data, differences in downstream performance are insignificant for up to around 4 epochs (25% budget) and then start dropping ... Filling up to 50% of data with code (42 billion tokens) also shows no deterioration. Beyond that, performance decreases quickly on natural language tasks."
> "Of the filtering approaches, we find perplexity-filtering to be effective, while deduplication does not help." ... "Deduplication may have value not captured in our benchmark, such as reducing memorization."
> "we recommend reserving filtering for noisy datasets and using both code augmentation and repeating to increase data tokens."

## Training hyperparameters (App. S)
> "For all training runs we use 1% of tokens for linear warm-up of the learning rate to a maximum learning rate of 2e-4 that is decayed to 2e-5 following a cosine schedule." ... "We use a dropout rate of 0.1, a weight decay rate of 0.1 and clip gradients at 1.0."

## Filtering setup (§7, App. N-O)
> "For perplexity filtering, we select the top 25% samples with the lowest perplexity according to a language model trained on Wikipedia. This results in 44 billion tokens that are repeated for close to two epochs to reach the full data budget. For deduplication filtering, all samples with a 100-char overlap are removed resulting in 21 billion tokens that are repeated for four epochs during training."
> "We also investigate filtering on a different noisier dataset in Appendix O, where we find it to be more effective."

## Limitations (App. Q)
> "In this work we focus on repeating the entire unique dataset for several epochs. Alternatively, one can repeat only a fraction of the dataset."
> "The returns from additional epochs may heavily depend on hyperparameters such as learning rate, dropout, or the optimizer choice."
> "More investigations of resolving data-constraints when fine-tuning LLMs may be of interest for future work."
