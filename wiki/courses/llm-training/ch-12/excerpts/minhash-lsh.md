---
chapter: ch-12
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/minhash-lsh.md (card verified 2026-09-14)
source_url: https://doi.org/10.1109/SEQUEN.1997.666900
primary_version: author PDF of Broder (1997), "preliminary version"; loci are section numbers of that PDF
created_at: "2026-04-23"
revised: 2026-09 (generality revision) — rewritten to match the verified card; the earlier version presented LSH banding, the name "MinHash", multi-permutation signatures, and a Lee et al. parameter table as content of this paper, and repeated the swapped Lee et al. bucket parameters
---

# Excerpt: On the resemblance and containment of documents

Andrei Z. Broder, Proc. Compression and Complexity of SEQUENCES 1997, pp. 21–29. Library card: [[minhash-lsh]].

## Definitions (§2)
- A document is a canonical token sequence; a shingle is a contiguous subsequence of w tokens; S(D, w) is the set (or labelled bag) of w-shingles.
- Resemblance: r_w(A, B) = |S(A,w) ∩ S(B,w)| / |S(A,w) ∪ S(B,w)|. Containment: c_w(A, B) = |S(A,w) ∩ S(B,w)| / |S(A,w)|.
- Worked example (§2), set option B: A = (a, rose, is, a, rose, is, a, rose), B = (a, rose, is, a, flower, which, is, a, rose). Resemblance is 60%, 50%, 42.85% for w = 1, 2, 3. For w = 1, A's set {a, rose, is} has 3 elements, B's set has 5, intersection 3, union 5, so 3/5.
- A larger w makes resemblance more sensitive to reordering and to single-token edits, because one changed token affects w shingles. Resemblance is not transitive (§2).

## Theorem 1 (§3)
- π is a uniformly random permutation of the shingle universe. Let α be the smallest element of π(S(A,w) ∪ S(B,w)). Then Pr(α ∈ M(A) ∩ M(B)) = |S(A,w) ∩ S(B,w)| / |S(A,w) ∪ S(B,w)| = r_w(A, B).
- The fixed-size sketch M(A) = MIN_s(π(S(A,w))) gives an unbiased estimate of resemblance; the variable-size MOD_m sketch estimates resemblance and containment.

## Implementation (§4)
- Shingles are mapped to l-bit fingerprints (Rabin fingerprints recommended); the clustering experiment used l = 40 (§4.1).
- Sample size: "100 samples seems reasonable and 200 seems more than enough", giving sketches of 300 to 800 bytes (§4.2).
- Comparing two sorted sketches costs O(s); all pairs among r documents cost O(r²s); greedy clustering costs O(rs) (§4.4).

## Web clustering result (§1)
Over 30,000,000 AltaVista documents (over 150 GB), clustered at 50% resemblance: 3.6 million clusters with 12.3 million documents; 2.1 million clusters held only identical documents (5.3 million documents).

## Not in this paper
The name "MinHash", signatures from many independent hash functions, and LSH banding do not appear. The banding candidate probability 1 − (1 − s^r)^b for b bands of r signature rows is derived in Leskovec, Rajaraman, Ullman, *Mining of Massive Datasets*, Ch. 3 §3.4.2; its Example 3.12 gives 0.99965 at s = 0.8 for b = 20 bands and r = 5 rows (pointer recorded in the card). Note that this textbook uses r for rows per band, while Lee et al. use b for hashes per bucket and r for the number of buckets.
