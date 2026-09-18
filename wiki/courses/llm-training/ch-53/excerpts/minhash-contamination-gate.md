---
chapter: ch-53
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/deduplicating-training-data.md
source_url: https://arxiv.org/abs/2107.06499
primary_text_checked: card verified 2026-09-14; Tülu 3 loci read 2026-09-17
revised: 2026-09 (generality revision)
---

# Excerpt: the contamination gate — decision rule and index parameters

Used by ch-53 §7. The gate has two parts that are often conflated: a **decision rule** that says when a
score may not be reported, and an **index** that makes the search affordable. They come from different
sources and have different thresholds.

## Part 1 — the decision rule (Tülu 3, arXiv:2411.15124 §3.2)

Matching is computed on prompts only, because completions in training sets are frequently regenerated
by a model. The authors compared full-string, n-gram, and embedding matching and kept n-gram matching:
embedding methods could not separate distributional similarity from paraphrase, while n-gram matching
caught instances that differ trivially, such as a math problem in which only the numbers changed.

1. Tokenize the evaluation prompt and the training prompt.
2. A token of the evaluation instance counts as matched when both instances share an **8-gram**
   containing that token.
3. The evaluation instance **overlaps** a training instance when more than **50%** of its tokens are
   matched against that same training instance.
4. A training set is **contaminated** with respect to an evaluation when its instances overlap more
   than **2%** of that evaluation's instances.

The 2% figure is Tülu 3's own dataset-level rule. It is not a threshold from the deduplication
literature, and Lee et al. 2021 propose no contamination threshold of any kind.

Base rates the rule produced (Table 37, share of evaluation instances overlapping the dataset):
Evol CodeAlpaca / HumanEval 70.7%; LMSys Chat 1M / AlpacaEval 46.5%; DaringAnteater / MATH 30.7%;
NuminaMath-TIR / MATH 18.2%; LMSys Chat 1M / MMLU 10.3%, GSM8K 8.9%.

## Part 2 — the index (Lee et al. 2021, arXiv:2107.06499 §4.1–4.2)

**ExactSubstr.** All examples are concatenated and a suffix array is built; adjacent suffix-array
entries sharing a prefix of at least **50 tokens** mark a repeated substring. The threshold came from
inspection: matches shorter than 10 tokens are common, manual inspection of 25-token matches found no
false positives, and the authors doubled 25 to 50 for margin (App. B). On one 96-core machine the
350 GB C4 suffix array takes under 12 hours to build.

**NearDup.** Documents are space-tokenized into **5-grams**. The MinHash signature has **k = 9,000**
hash values split into **r = 450 buckets of b = 20 hashes each**. A pair becomes a candidate with
probability

```
P(candidate | Jaccard s) = 1 − (1 − s^b)^r ,   b = 20 hashes per bucket, r = 450 buckets
```

Candidates are confirmed when Jaccard index > 0.8 **and** edit similarity > 0.8, where
`EditSim(x_i, x_j) = 1 − EditDistance(x_i, x_j) / max(|x_i|, |x_j|)`. Connected components of the
duplicate graph form clusters. An alternative setting is also reported: 0.9/0.9 with b = 20, r = 40,
k = 800 (App. A, Fig. 4).

### The parameter order has to be checked against the curve

With b = 20 and r = 450 the candidate probability is 0.0004 at s = 0.5, 0.016 at s = 0.6, 0.302 at
s = 0.7 and 0.995 at s = 0.8; the midpoint of the S-curve is `(1/r)^(1/b) = (1/450)^(1/20) = 0.737`.

If the two numbers are exchanged — 20 buckets of 450 hashes — the midpoint becomes
`(1/20)^(1/450) = 0.993` and a pair at Jaccard 0.9 is proposed with probability below 1e−15, so the
index returns almost nothing. Library APIs usually take the pair as `(bands, rows_per_band)`, which is
the reverse of the paper's "buckets of hashes" wording. Evaluate the S-curve at s = 0.8 before running
the gate; the value must exceed 0.9. `figures/lsh-band-curve.html` plots both settings.

## Base rates from the deduplication paper (for calibration, not as a gate)

Validation examples with a near-duplicate in training (Table 2): C4 4.60%, RealNews 14.35%,
LM1B 4.92%, Wiki-40B 0.72%. Validation tokens inside 50-token matches with training (Table 3):
C4 1.38%, RealNews 3.37%, LM1B 0.019%, Wiki-40B 0.67%. Training examples flagged as near-duplicates:
C4 3.04%, RealNews 13.63%, LM1B 4.86%, Wiki-40B 0.39%. Memorization after one epoch on C4 fell from
1.926% of generated tokens to 0.189% (NearDup) and 0.138% (ExactSubstr) (Table 4).

## When no training data is available

String matching requires the training set. For closed checkpoints, ch-53 §7.3 substitutes a
memorization probe ([[swe-bench-illusion]], [[gsm1k]]). A probe result never reports as `clean`; the
gate records `unknown-training-data`.

Related: [[deduplicating-training-data]], [[minhash-lsh]], [[tulu-3]], [[read]].
