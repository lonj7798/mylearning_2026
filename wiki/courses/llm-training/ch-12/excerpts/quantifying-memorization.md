---
chapter: ch-12
course: llm-training
phase: read
excerpt_of: primary source arXiv:2202.07646v3 (no library card as of 2026-09-15)
source_url: https://arxiv.org/abs/2202.07646
primary_version: arXiv:2202.07646v3 (2023-03-06; v1 2022-02; ICLR 2023), full text read 2026-09-15
created_at: "2026-09-15"
---

# Excerpt: Quantifying Memorization Across Neural Language Models

**Paper:** Nicholas Carlini, Daphne Ippolito, Matthew Jagielski, Katherine Lee, Florian Tramèr, Chiyuan Zhang (Google Research; University of Pennsylvania; Cornell). arXiv v1 2022-02; ICLR 2023. Source type: paper. This chapter uses the duplication results; the full memorization treatment is in ch-12a.

## Definition (§3.1, §3.2)
- Definition 3.1: a string s is extractable with k tokens of context from model f if some length-k string p exists such that [p || s] is in f's training data and f produces s from p under greedy decoding.
- Evaluation: for sequence lengths 50 to 500 tokens, the model is prompted with the first length − 50 tokens and a sequence counts as extractable if the next 50 tokens are emitted exactly. 50 tokens average 127 characters or 25 words in the GPT-Neo training set (§3.2).
- Sampling: a uniform sample of 50,000 sequences, and a sample normalized by sequence length and duplication count, about 500,000 sequences in total; duplicate counts come from the suffix-array construction of Lee et al. (§3.2).

## Models (§4)
GPT-Neo 125M, 1.3B, 2.7B, and 6B (GPT-J), trained on The Pile (825GB). GPT-2 models, trained on WebText, serve as a baseline for text that is predictable without having been trained on.

## Results
- **Model size (§4.1, Fig. 1a):** on the duplication-normalized sample, a log-linear fit with R² 99.8%: a tenfold increase in model size corresponds to 19 percentage points more memorization. GPT-2 completes about 6% of the evaluation set, against 40% for GPT-Neo 1.3B.
- **Duplication (§4.2, Fig. 1b):** for buckets of sequences duplicated between 2 and 900 times (1,000 sequences per bucket), memorization follows a log-linear trend in the number of duplicates. Memorization still happens with a few duplicates, so deduplication "will not perfectly prevent leakage".
- **Context (§4.3, Fig. 1c):** 33% of evaluated sequences are extractable from the 6B model with 50 tokens of context, 65% with 450 tokens.
- **Uniform sample (§4.4):** at least 1% of The Pile is extractable by GPT-J 6B but not by GPT-2 XL.
- **Search definition (§4.4):** at 100 repetitions, 32.6% of outputs occur somewhere in the dataset, but 15.8% match the true continuation.

## Replication on deduplicated training data (§5.2, Fig. 4c)
- Models: the 1.5B causal models of Lee et al. trained on C4 Original, C4 NearDup, and C4 ExactSubstr.
- For sequences repeated below 35 times in original C4, the ExactSubstr model memorizes 1.2% of sequences on average against 3.6% without deduplication (a factor of 3×, p < 10⁻¹⁵).
- Deduplication helps for sequences repeated up to about 100 times, not for sequences repeated more often; extractability of examples repeated at least 408 times is significantly higher than for any lower repeat count. The authors hypothesize that any scalable deduplication is imperfect and that different valid definitions of a duplicate make deduplication non-exhaustive (Interpretation).

## Other replications (§5.1, §5.3)
- T5 (masked LM) on C4: the model-size trend holds, but absolute memorization is about an order of magnitude lower; the duplication trend is non-monotonic with large variance, partly because buckets full of whitespace-heavy sequences are easier to predict (§5.1).
- OPT (trained on a deduplicated dataset overlapping The Pile): same scaling trends with an effect size orders of magnitude smaller; the authors cannot separate careful curation from distribution shift as the cause (§5.3).
