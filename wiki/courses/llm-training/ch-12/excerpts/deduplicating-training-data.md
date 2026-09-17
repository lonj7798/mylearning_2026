---
chapter: ch-12
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/deduplicating-training-data.md (card verified 2026-09-14)
source_url: https://arxiv.org/abs/2107.06499
primary_version: arXiv:2107.06499v2 (2022-03-24; v1 2021-07; ACL 2022); §4.1, §4.2 and App. A re-read 2026-09-15
created_at: "2026-04-23"
revised: 2026-09 (generality revision) — rewritten to match the verified card; the earlier version swapped the MinHash bucket parameters, gave a wrong train-validation overlap table, called 3.04% a token share, stated O(N log N) construction, and attributed a linear duplication-memorization relation and downstream gains to the paper
---

# Excerpt: Deduplicating Training Data Makes Language Models Better

Katherine Lee, Daphne Ippolito, Andrew Nystrom, Chiyuan Zhang, Douglas Eck, Chris Callison-Burch, Nicholas Carlini (Google Research; University of Pennsylvania). Library card: [[deduplicating-training-data]].

## Datasets (§3)
Wiki-40B English 2.9M pages (mean 768 BPE tokens); LM1B 30M sentences (32); C4 360M documents (486); RealNews 31M documents (793).

## ExactSubstr (§4.1, App. B)
- "When two examples x_i and x_j share a sufficiently long substring ... that substring is removed from one of them" (§4.1). Minimum length k = 50 tokens.
- All examples are concatenated into one sequence S built from the bytes of the BPE tokenization; a suffix array A(S) = arg sort all_suffixes(S) is built. Example: the suffixes of "banana" sort to the suffix array (6 4 2 1 5 3) (§4.1.1).
- A repeated sequence appears at adjacent suffix-array positions; a linear scan finds adjacent entries sharing a prefix of at least 50 tokens (§4.1.2).
- Suffix arrays need 8 bytes per input token and can be built in linear time (§4.1.1). App. B: SA-IS on parallel splits, merge O(N·m·log K); the 350GB C4 suffix array takes under 12 hours on one 96-core, 768GB machine and occupies 1.5TB.
- Threshold choice (App. B, Fig. 5): matches under 10 tokens are common; manual inspection found no false positives at 25 tokens; the authors doubled 25 to 50.

## NearDup (§4.2, App. A)
- Documents are space-tokenized; each 5-gram is hashed; k = 9,000 minimum hashes are "partitioned into r buckets, with b hashes per bucket" (App. A).
- Candidate probability: Pr(d_i, d_j | Jaccard(d_i, d_j) = s_ij) = 1 − (1 − s_ij^b)^r, "where b = 20 and r = 450" (§4.2). So there are 450 buckets, each of 20 hashes.
- Candidates are verified: actual Jaccard index above 0.8, then edit similarity above 0.8, where EditSim(x_i, x_j) = 1 − EditDistance(x_i, x_j) / max(|x_i|, |x_j|). Duplicate pairs form a graph; connected components are clusters.
- Stated reason for b and r: "to make sure a collision at the desired Jaccard index threshold of 0.8 had a high probability of occurring" (App. A).
- Alternative configuration: Jaccard ≥ 0.9 and edit similarity ≥ 0.9 with b = 20, r = 40, k = 800; the similarity histograms of both configurations vanish at the same point (App. A, Fig. 4).

## Amount of duplication
| Dataset | Train examples with near-duplicates (Table 2) | Validation examples with a near-duplicate in train (Table 2) | Train tokens in 50-token repeats (Table 3) | Validation tokens in 50-token matches with train (Table 3) |
|---|---|---|---|---|
| C4 | 3.04% | 4.60% | 7.18% | 1.38% |
| RealNews | 13.63% | 14.35% | 19.4% | 3.37% |
| LM1B | 4.86% | 4.92% | 0.76% | 0.019% |
| Wiki-40B | 0.39% | 0.72% | 2.76% | 0.67% |
- C4 clusters (§5.1, Fig. 1): 1.8M clusters are pairs; 280 clusters have more than 5,000 examples; the largest has 250,933. One 61-word sequence occurs 61,036 times in training and 61 times in validation (§1). 77% of C4 examples removed by NearDup also contain an ExactSubstr match (§5.1).
- Cross-split rule (§5): when text occurs in more than one split, the validation or test copy is kept and the training copy removed.

## Models and results (§6, App. C)
- Models: 110M base (3 seeds) and 1.5B XL decoder-only models on C4 Original, C4-NearDup, C4-ExactSubstr; maximum length 512 tokens; XL about 2 epochs.
- Unprompted memorization (Table 4): 100,000 samples of up to 512 tokens, top-k = 50; a token is memorized if it is part of a 50-token substring found in training. 1 epoch: Original 1.926%, NearDup 0.189%, ExactSubstr 0.138%. 2 epochs: 1.571%, 0.264%, 0.168%.
- Prompted memorization (Fig. 3): with 32-token prompts from duplicated training examples, the Original model reproduces the true continuation (edit similarity above 0.8) over 40% of the time.
- Perplexity (§6.1, Fig. 2): similar on the full C4 validation set and its unique subset; deduplicated models have higher perplexity on validation examples that have training duplicates; ExactSubstr lowers XL perplexity on Wiki-40B by almost 3 points. Dataset size falls by up to 19% (§1).
- Released models (Table 5): Transformer-XL on LM1B validation, perplexity 21.77 overall, 10.11 on examples with training near-duplicates, 23.58 on unique examples.

## Limits stated in the paper
- Downstream tasks were not evaluated (§2).
- Negative consequences of deduplication were not investigated, including tasks that need memorization such as closed-book question answering (§7).
- Deduplication is not sufficient to remove privacy-sensitive data (§7, §8).
