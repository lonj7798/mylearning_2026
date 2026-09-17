---
chapter: ch-11
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/quantifying-memorization.md (planned card; not present on 2026-09-15)
source_url: https://arxiv.org/abs/2202.07646
created_at: "2026-09-15"
---

# Excerpt: Quantifying Memorization Across Neural Language Models

**Authors:** Nicholas Carlini, Daphne Ippolito, Matthew Jagielski, Katherine Lee, Florian Tramèr, Chiyuan Zhang (Google Research, UPenn, Cornell).
**Version read:** arXiv:2202.07646v3 (6 Mar 2023); v1 Feb 2022; ICLR 2023.
**Status:** no library card existed for this slug on 2026-09-15; values read in the v3 PDF text at the stated locus. The full treatment belongs to ch-12a; this excerpt keeps what ch-11 uses.

## Definition (§3.1, Definition 3.1)
"A string s is extractable with k tokens of context from a model f if there exists a (length-k) string p, such that the concatenation [p || s] is contained in the training data for f, and f produces s when prompted with p using greedy decoding."

## Protocol (§3.2)
- Models: GPT-Neo 125M, 1.3B, 2.7B, 6B trained on the Pile (§4). Baseline: GPT-2 models, not trained on the Pile.
- Duplicate-normalized sample: for each length ℓ ∈ {50, 100, ..., 500} and duplicate bucket [2^(n/4), 2^((n+1)/4)), 1,000 sequences; about 500,000 sequences total, found with the suffix array of Lee et al. Prompt with the first ℓ − 50 tokens; extractable if the next 50 tokens match exactly.

## Results
- Model size (§4.1, Fig. 1a): log-linear, R² = 99.8%; "a ten fold increase in model size corresponds to an increase in memorization of 19 percentage points". GPT-2 completes about 6% of the evaluation set versus 40% for GPT-Neo 1.3B.
- Duplication (§4.2, Fig. 1b): log-linear in the number of repetitions; "memorization does still happen, even with just a few duplicates—thus, deduplication will not perfectly prevent leakage."
- Context (§4.3, Fig. 1c): 33% extractable from the 6B model at 50 tokens of context, 65% at 450 tokens.
- Uniform sample (§4.4): at least 1% of the Pile is extractable by GPT-J 6B but not by GPT-2 XL.
- Alternative definition (§4.4): at 100 repetitions, 32.6% of generations appear somewhere in the dataset, 15.8% match the true continuation.
- Deduplicated training data (§5.2, Fig. 4c; models of Lee et al., 1.5B on C4): for sequences repeated fewer than 35 times, the exact-deduplicated model memorizes 1.2% versus 3.6% without deduplication; deduplication does not help for sequences repeated more than about 100 times, and examples repeated at least 408 times are extracted more often.
- OPT (§5.3): trends similar, effect orders of magnitude smaller; the authors cannot separate careful curation from distribution shift.
- Auditing (§4.3): "correctly auditing large language models likely requires prompting the model with training data".

## How ch-11 uses it
§6 (why PII removal is a pre-training stage), Negative samples, Generalization lens (memorization versus generalization measurement).
