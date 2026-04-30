<!-- scope: foundational paper showing dedup halves memorization and improves LM quality
     deps: [[c4]]
     see-also: [[minhash-lsh]], [[d4]], [[dolma]], [[fineweb]]
-->

# Deduplicating Training Data Makes Language Models Better
- **Core Insight:** Standard pretraining corpora contain massive near-duplicate content; removing it 10× reduces memorized output, improves perplexity, and shortens training — with almost no downside.
- **Guideline:** Before any other quality filter, run exact-substring and MinHash-based near-duplicate removal; treat dedup as mandatory infrastructure, not an optional ablation.
- **Authors:** Katherine Lee, Daphne Ippolito, Andrew Nystrom, Chiyuan Zhang, Douglas Eck, Chris Callison-Burch, Nicholas Carlini
- **Year:** 2021 (ACL 2022)
- **URL:** https://arxiv.org/abs/2107.06499
- **Relevant topics:** deduplication, memorization, train-test contamination, data quality

## Abstract
We find that existing language modeling datasets contain many near-duplicate examples and long repetitive substrings. As a result, over 1% of the unprompted output of language models trained on these datasets is copied verbatim from the training data. We develop two tools that allow us to deduplicate training datasets — for example removing from C4 a single 61 word English sentence that is repeated over 60,000 times. Deduplication allows us to train models that emit memorized text ten times less frequently and require fewer train steps to achieve the same or better accuracy. We can also reduce train-test overlap, which affects over 4% of the validation set of standard datasets, thus allowing for more accurate evaluation.

## Key Contributions
- Quantified that **>1% of unprompted LM output is verbatim training-set copy**.
- Showed **>4% train-test overlap** in standard LM validation sets — silently inflating reported metrics.
- Released **two dedup tools**: exact-substring (ExactSubstr via suffix array) and near-duplicate (NearDup via MinHash+LSH).
- Established dedup as the single highest-ROI pretraining data operation — adopted in every subsequent open corpus.

## Key Figures/Tables to Study
- **Figure** showing distribution of duplicate count per document in C4 and Wiki-40B — long-tail of extreme duplicates.
- **Table** of memorization rate before vs after dedup (10× reduction).
- **Train-test overlap table** across LM1B, C4, RealNews, Wiki-40B.
- **Training-curve figure** showing dedup reaches target perplexity in fewer steps than baseline.

## Technical Details
**Two complementary methods:**

1. **ExactSubstr** — suffix-array-based exact substring matching.
   - Find all duplicate substrings of length ≥ **50 tokens** (the threshold chosen empirically — long enough to avoid common phrases).
   - Remove one copy of each duplicate span.
   - Runs in O(N log N) via suffix array construction on the concatenated corpus.

2. **NearDup** — MinHash + LSH for fuzzy document-level dedup.
   - Compute 5-gram shingles per document.
   - Build 9000 MinHash signatures (aggressive signature count for high recall).
   - LSH with b = 20 bands of r = 450 rows, threshold ≈ (1/b)^(1/r) ≈ 0.8 Jaccard similarity.
   - Any document exceeding threshold against another is dropped.

**Findings on C4:**
- 3.04% of training tokens are in near-duplicate clusters.
- A single 61-word English sentence repeats **>60,000 times**.
- Removing near-dups reduces training set by ~5% while improving all downstream metrics.

**Findings on memorization:**
- Without dedup, ~1% of unprompted 256-token completions are verbatim training copies.
- With dedup, this drops ~10×.
- Privacy implication: dedup alone substantially reduces the surface for membership-inference and training-data-extraction attacks.

**Train-test contamination:**
- 4.6% of LM1B validation overlaps training.
- 3.2% of C4 validation overlaps training.
- Reported perplexity improvements on benchmarks may be partly illusory without dedup.

## Connections
- The foundation under every modern pretraining corpus — see [[c4]], [[the-pile]], [[dolma]], [[fineweb]].
- Technical prerequisite for [[minhash-lsh]] and the semantic variant [[d4]] / SemDeDup.
- Directly motivated [[data-constrained-scaling]]'s question: if dedup costs 5% tokens, how many epochs of repetition are OK?
- Memorization findings connect to later work on training-data extraction attacks (Carlini 2021 line).
