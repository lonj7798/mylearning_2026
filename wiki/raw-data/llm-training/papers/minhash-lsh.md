<!-- scope: classical MinHash + LSH reference for approximate deduplication
     see-also: [[deduplicating-training-data]], [[ccnet]]
-->

# MinHash and LSH for Approximate Document Deduplication
- **Core Insight:** Exact dedup is not enough for web corpora; MinHash plus locality-sensitive hashing gives a cheap approximation for near-duplicate detection at web scale.
- **Guideline:** For large corpus dedup, fingerprint documents by shingles, compute compact MinHash signatures, and use LSH buckets to surface near-duplicates before full comparison.
- **Authors:** Classical reference line: Andrei Broder and later LSH work
- **Year:** 1997-1998
- **URL:** https://doi.org/10.1109/SEQUEN.1997.666900
- **Relevant topics:** MinHash, locality-sensitive hashing, near-duplicate detection, deduplication

## Abstract
The classical MinHash/LSH line introduces a scalable approximation to document resemblance and containment. In LLM pipelines it matters because web corpora contain enormous numbers of near-copies, boilerplate variants, mirrored pages, and templated spam that exact dedup misses.

## Key Contributions
- Defined resemblance-based document similarity for large text collections.
- Made near-duplicate search practical without all-pairs comparison.
- Became the default primitive behind large-scale corpus dedup pipelines.

## Technical Details
- Convert each document to shingles.
- Hash shingles under many permutations or approximations to form a MinHash signature.
- Use LSH to place similar signatures in shared candidate buckets.
- Run exact or tighter similarity checks only on candidate pairs.

## Connections
- Core algorithmic primitive behind [[deduplicating-training-data]], [[ccnet]], and later open data stacks.
- Also relevant to diversity-aware filtering because near-dedup and semantic dedup are distinct problems.

