---
chapter: ch-17
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/deduplicating-training-data.md
source_url: https://arxiv.org/abs/2107.06499
created_at: "2026-04-23"
---

# Excerpt: Lee et al. 2021 — the exact/near-dedup numbers ch-17 §3 and §4 implement

**Source library:** `wiki/raw-data/llm-training/papers/deduplicating-training-data.md`
**Paper:** Lee, Ippolito, Nystrom, Zhang, Eck, Callison-Burch, Carlini 2021, "Deduplicating Training Data Makes Language Models Better" (ACL 2022).

---

## Why this paper is the numerical backbone of the lab

Ch-17's §3 picks a minimum duplicate-span length of 50 tokens. §4 picks 9000 MinHash signatures at Jaccard ≥ 0.8 (or 128 on the constrained path). Every one of those numbers comes from this paper. They are not convention — they are empirically tuned, and the paper walks through why.

From the source (lines 32–41):

> **ExactSubstr** — suffix-array-based exact substring matching. Find all duplicate substrings of length ≥ **50 tokens** (the threshold chosen empirically — long enough to avoid common phrases). Remove one copy of each duplicate span. Runs in O(N log N) via suffix array construction on the concatenated corpus.
>
> **NearDup** — MinHash + LSH for fuzzy document-level dedup. Compute 5-gram shingles per document. Build 9000 MinHash signatures (aggressive signature count for high recall). LSH with b = 20 bands of r = 450 rows, threshold ≈ (1/b)^(1/r) ≈ 0.8 Jaccard similarity.

The 50-token threshold is the single most important hyperparameter in §3. At 20 tokens you match proverbs and boilerplate HTML. At 100 you miss real duplication. The paper reports that 50 is where "common idioms" disappear but "boilerplate repeats" stay — above 99% of matches at 50 tokens are templated text, not language.

The (b=20, r=450) choice for NearDup is the aggressive high-recall end of the LSH design space. At s = 0.8 it produces near-certain recall; at s = 0.5 it produces essentially zero false positives. The cost is 9000 hash computations per document. Ch-17's small-scale variant (128 hashes, b=16 × r=8) preserves the S-curve shape at much lower cost — good enough for 1 GB.

---

## The numbers ch-17 must measure

Three quantitative findings from the paper are *targets* for the lab to reproduce in miniature:

1. **~3% of training tokens are in near-duplicate clusters** (source line 44). On a CC slice, expect 2–6%. If your MinHash pass drops < 1% of your data, either your slice is unusually clean (check for CommonCrawl's built-in WET dedup) or your (b, r) is too permissive.
2. **A single 61-word English sentence repeats >60,000 times in C4** (source line 45). This is the most famous artifact in the paper — the giveaway boilerplate on a specific news template. On a fresh CC slice you will find similar long-tail repeats. Log the top-10 most-repeated 5-grams before and after MinHash and paste them into the memo; this is the kind of spot-check Karpathy's recipe (see the excerpt for [[karpathy-training-neural-net-recipe]]) treats as mandatory.
3. **4.6% of LM1B validation overlaps training, 3.2% of C4** (source lines 53–55). This is the decontamination FP numerator. The lab's decontamination step is a direct rerun of the paper's §5 train-test-overlap measurement on the student's own slice + eval suite.

From the source (lines 49–51):

> Without dedup, ~1% of unprompted 256-token completions are verbatim training copies. With dedup, this drops ~10×. Privacy implication: dedup alone substantially reduces the surface for membership-inference and training-data-extraction attacks.

The memo should not spend space on memorization rates — the ch-17 models are too small to verbatim-emit in any reliable way — but one paragraph citing this as "the reason dedup is mandatory even when the downstream delta is small" is correct framing.

---

## The decontamination-FP methodology the lab copies

The paper uses n-gram overlap (specifically, n = 13 was the follow-on Chinchilla default; this paper uses varying n and reports the tradeoff). The crucial move is:

> **Train-test contamination:** 4.6% of LM1B validation overlaps training. … Reported perplexity improvements on benchmarks may be partly illusory without dedup.

The failure mode is subtle: a 13-gram match between a training doc and a HellaSwag continuation does not always mean the training doc leaked the eval item. Sometimes it means "the quick brown fox jumps over the lazy dog" appears in both places by coincidence. The paper does not fully quantify the FP rate; it is the ch-17 memo's job to produce that number by hand-labelling 100 dropped docs. The lab's 5–15% FP expectation at n=13 is the empirical result students should expect to land near.
