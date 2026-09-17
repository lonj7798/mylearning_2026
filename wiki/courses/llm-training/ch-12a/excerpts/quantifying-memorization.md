---
chapter: ch-12a
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/quantifying-memorization.md (planned card; not present on 2026-09-15)
source_url: https://arxiv.org/abs/2202.07646
created_at: "2026-09-15"
---

# Excerpt: Quantifying Memorization Across Neural Language Models

**Authors:** Nicholas Carlini, Daphne Ippolito, Matthew Jagielski, Katherine Lee, Florian Tramèr, Chiyuan Zhang (Google Research, University of Pennsylvania, Cornell University)
**Version read:** arXiv:2202.07646v3 (6 Mar 2023); v1 February 2022; ICLR 2023.
**Status:** no library card existed for this slug on 2026-09-15; every value below was read in the v3 PDF text at the stated locus.

## Definition (§3.1, Definition 3.1)
A string s is extractable with k tokens of context from model f if there exists a length-k string p such that [p || s] is in the training data and f produces s from p with greedy decoding.

## Protocol (§3.2)
- Uniform sample: 50,000 training sequences (≤ 0.02% of the dataset); §4.4 uses 100,000.
- Duplicate-normalized sample: for each length ℓ ∈ {50, 100, ..., 500} and integer n, 1,000 sequences contained in the training data between 2^(n/4) and 2^((n+1)/4) times; about 500,000 sequences in total.
- The model is prompted with the first ℓ − 50 tokens; a sequence is extractable if the next 50 tokens match exactly. Fifty tokens average 127 characters or 25 words in the GPT-Neo training set.
- Models: GPT-Neo 125M, 1.3B, 2.7B, 6B trained on the Pile (825GB) (§4). Baseline: GPT-2 family trained on WebText, prompted with the same Pile prompts.

## Results
- Model size (§4.1, Fig. 1a): log-linear fit with R² of 99.8%; a tenfold increase in model size corresponds to an increase of 19 percentage points in memorization on the duplicate-normalized set.
- Baseline control (§4.1): GPT-2 correctly completes about 6% of the evaluation set, against 40% for GPT-Neo 1.3B.
- Duplication (§4.2, Fig. 1b): log-linear increase over buckets of 2 to 900 duplicates; "memorization does still happen, even with just a few duplicates".
- Context (§4.3, Fig. 1c): 33% of sequences extractable from the 6B model with 50 tokens of context against 65% with 450 tokens ("discoverability").
- Uniform sample (§4.4, Fig. 2b): last 50 tokens of a length-1000 sequence extractable with 7% probability for GPT-J 6B, 4% for GPT-Neo 125M, 2% for GPT-2 XL; lower bound of at least 1% of the Pile extractable by GPT-J 6B but not by GPT-2 XL.
- Decoding (§4.4, Fig. 2c): beam search with 100 beams extracts less than 2 percentage points more on average, maximum 5.6%.
- Definition sensitivity (§4.4): at 100 repetitions, 32.6% of outputs occur somewhere in the dataset, 15.8% match the true continuation.
- T5 masked LM (§5.1): T5-XL (3B) memorizes 3.5% of sequences repeated 100 times against 53.6% for GPT-Neo 2.7B (150 tokens of context); duplicate trend non-monotonic (whitespace-heavy bucket at 138-158 repeats).
- Deduplicated C4 models from Lee et al., 1.5B (§5.2, Fig. 4c): for sequences repeated below 35 times, 1.2% memorized with exact-substring deduplication against 3.6% without (factor 3, p < 10^-15); no benefit for sequences repeated at least 408 times in original C4.
- OPT 125M-66B (§5.3): same scaling trends, effect size orders of magnitude smaller; OPT-66B memorizes a smaller fraction of the Pile than GPT-Neo 125M. The authors cannot distinguish data curation from distribution shift as the cause.

## Limits stated by the authors
Greedy decoding and exact-match suffix are lower bounds (§4.4); discoverability means auditing needs training-data prompts (§4.3); cross-family results depend on dataset idiosyncrasies (§5).

## How ch-12a uses it
§1 (definition), §2 (scaling with size, duplicates, context; baseline control), Recipe (audit protocol), Generalization lens (c), Common mistakes.
