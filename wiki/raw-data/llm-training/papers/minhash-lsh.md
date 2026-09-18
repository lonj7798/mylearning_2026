<!-- scope: Broder (1997) — resemblance and containment of documents, estimated from min-wise sampling sketches (the basis of MinHash near-duplicate detection)
     see-also: [[deduplicating-training-data]], [[ccnet]]
-->

# On the resemblance and containment of documents
- **Core Insight:** The resemblance of two documents (the Jaccard ratio of their shingle sets) can be estimated from small per-document sketches, because under one uniformly random permutation the smallest element of the union lies in both documents' sketches with probability equal to the resemblance (§3, Theorem 1). The author judges 100 samples per document reasonable and 200 more than enough (§4.2).
- **Guideline:** When a collection is too large to compare documents pairwise, use a fixed-size min-wise sketch per document to estimate resemblance, because Theorem 1 makes the estimate unbiased and §4.2 bounds its error with the sample size. Otherwise, when containment of one document in another is needed, use the variable-size MOD_m sketch, and expect error-prone containment estimates for very short documents inside much longer ones, because the short document contributes few samples (§3).
- **Authors:** Andrei Z. Broder
- **Year:** 1997 (Proc. Compression and Complexity of SEQUENCES 1997, IEEE, pp. 21–29)
- **URL:** https://doi.org/10.1109/SEQUEN.1997.666900
- **Source type:** paper
- **Relevant topics:** resemblance (Jaccard similarity), containment, shingling, min-wise sampling sketches, Rabin fingerprints, near-duplicate clustering

## Abstract
For two documents A and B, the paper defines resemblance r(A, B), intended to capture "roughly the same", and containment c(A, B), intended to capture "roughly contained". Both measures are reduced to set-intersection problems. The relative size of the intersection is estimated by random sampling that is done independently for each document. Resemblance can be estimated from a fixed-size sample per document. The paper discusses the mathematical properties of both measures and an efficient implementation of the sampling with Rabin fingerprints.

## Key Contributions
- Defines w-shingling and two set constructions: labelled shingles (option A) and the plain set of shingles (option B) (§2).
- Defines resemblance and containment as ratios of set sizes, and states that 1 − r_w is a metric; the proof is omitted in this version (§2).
- Theorem 1: sketches built from one random permutation give unbiased estimates of resemblance (MIN_s and MOD_m sketches) and of containment (MOD_m sketch) (§3).
- Implementation analysis: fingerprint length and collision error, sketch maintenance cost, sample size, and greedy clustering (§4).
- Reports a clustering of over 30,000,000 web documents at 50% resemblance (§1).

## Key Figures/Tables to Study
- The paper has no figures or tables. Study Theorem 1 and its proof (§3) and the sample-size probability p(s, r, ε) (§4.2).

## Technical Details
**Definitions (§2).**
- A document is a canonical token sequence. Tokens may be letters, words, or lines. A parser removes differences the user chooses to ignore, such as punctuation, formatting, and capitalization (§2).
- A shingle is a contiguous subsequence of tokens. The w-shingling of D is the bag of all shingles of size w in D (§2).
- `r_w(A, B) = |S(A,w) ∩ S(B,w)| / |S(A,w) ∪ S(B,w)|` and `c_w(A, B) = |S(A,w) ∩ S(B,w)| / |S(A,w)|` (§2).
  S(D, w) is the shingle set of document D under option A or option B; w is the shingle size.
- Worked example (§2): A = (a,rose,is,a,rose,is,a,rose), B = (a,rose,is,a,flower,which,is,a,rose). Under option A, resemblance is 70%, 50%, 30% for w = 1, 2, 3. Under option B it is 60%, 50%, 42.85%. Check option B at w = 1: A's set {a, rose, is} has 3 elements, B's set has 5, the intersection has 3 and the union 5, so 3/5 = 60%.
- A larger w makes resemblance more sensitive to reordering of text. A change in one token affects w shingles, so a large w can be over-sensitive to small edits (§2). Resemblance is not transitive (§2).

**Estimator (§3, Theorem 1).**
- Symbols: Ω is the set of all shingles of size w, totally ordered. π is a permutation of Ω chosen uniformly at random. MIN_s(W) is the set of the s smallest elements of W, or W if |W| < s. MOD_m(I) is the set of elements divisible by m. g: Ω → N is an injection.
- With M(A) = MIN_s(π(S(A,w))), the value `|MIN_s(M(A) ∪ M(B)) ∩ M(A) ∩ M(B)| / |MIN_s(M(A) ∪ M(B))|` is an unbiased estimate of resemblance.
- With L(A) = MOD_m(g(π(S(A,w)))), `|L(A) ∩ L(B)| / |L(A) ∪ L(B)|` estimates resemblance and `|L(A) ∩ L(B)| / |L(A)|` estimates containment, both unbiased.
- Proof step: let α be the smallest element of π(S(A,w) ∪ S(B,w)). Then `Pr(α ∈ M(A) ∩ M(B)) = |S(A,w) ∩ S(B,w)| / |S(A,w) ∪ S(B,w)| = r_w(A, B)`.
- M(D) has a fixed size but estimates only resemblance. L(D) grows with D and estimates both measures (§3).
- To bound L(D), documents with size between 100·2^i and 100·2^(i+1) store MOD_{2^i}; the expected sketch size is between 50 and 100 (§3).

**Implementation (§4).**
- A shingle of 7 English words is about 40–50 bytes, so each shingle is first mapped to an l-bit id (§4.1).
- With probability better than 99.9% over the choice of f, `|r_{w,f} − r_w| < |S(A,w) ∪ S(B,w)| / 2^(l−11)` (§4.1).
- Rabin fingerprints are recommended because their collision probability is well understood; it is bounded by max(|s1|, |s2|)/2^(l−1) in an adversarial model (§4.1).
- In the clustering experiment, a typical shingle set had about 1000 elements, a shingle was about 400 bits, and fingerprints used l = 40 (§4.1).
- Under option A, MIN_s is kept in a heap at O(s log s log(n/m)) expected cost, where n is the number of tokens and m the modulus. Option B needs a balanced search tree at the same asymptotic cost (§4.1).
- Sample size: the number of common shingles in the sample is approximated as binomial. "100 samples seems reasonable and 200 seems more than enough", which gives sketches of 300 to 800 bytes (§4.2).
- For the entire web, the probability that any pair with resemblance below 50% is estimated above 90% is less than 0.1% (§4.2).
- Comparing two sorted sketches costs O(s). All pairs among r documents cost O(r²s). Greedy clustering costs O(rs), assuming each fingerprint belongs to few clusters (§4.4).

**Web clustering result (§1).** Input: over 30,000,000 documents from an AltaVista crawl, over 150 GB. Threshold: 50% resemblance. Output: 3.6 million clusters containing 12.3 million documents. Of these, 2.1 million clusters contained only identical documents (5.3 million documents), and 1.5 million clusters contained 7 million documents that mixed exact and near duplicates. Details are in Broder, Glassman, Manasse, Zweig (WWW 1997), ref. [5].

## Connections
- [[deduplicating-training-data]] — applies MinHash near-duplicate detection to language-model training data.
- [[ccnet]] — web-corpus pipeline card that links here for its deduplication stage.
- Not in this paper: the name "MinHash", signatures built from many independent permutations, and LSH banding. The banding candidate probability `1 − (1 − s^r)^b` (b bands of r signature rows, Jaccard similarity s) is derived in Leskovec, Rajaraman, Ullman, *Mining of Massive Datasets*, Ch. 3 §3.4.2 (http://infolab.stanford.edu/~ullman/mmds/ch3.pdf). Its Example 3.12 gives 0.99965 at s = 0.8 for b = 20, r = 5. Its §3.11 attributes minhashing to Broder, Charikar, Frieze, Mitzenmacher (STOC 1998) and the original LSH works to Indyk and Motwani (STOC 1998) and Gionis, Indyk, Motwani (VLDB 1999).

## Verification
- Checked on 2026-09-14 against: author PDF (marked "preliminary version", §2) at https://www.cs.princeton.edu/courses/archive/spring13/cos598C/broder97resemblance.pdf, and Crossref metadata for DOI 10.1109/SEQUEN.1997.666900 (title, venue, pp. 21–29). The IEEE version was not accessed; loci are section numbers of the author PDF. MMDS Ch. 3 was checked for the pointers under Connections.
- Corrections to the previous card version:
  - Title "MinHash and LSH for Approximate Document Deduplication" and author "Classical reference line: Andrei Broder and later LSH work" → the URL identifies one paper: Andrei Z. Broder, "On the resemblance and containment of documents" (1997). The card now describes only that paper.
  - "Year: 1997-1998" → 1997.
  - "Hash shingles under many permutations or approximations to form a MinHash signature" → the paper uses one random permutation and keeps the s smallest elements (MIN_s) or the elements divisible by m (MOD_m) (§3, Theorem 1).
  - "Use LSH to place similar signatures in shared candidate buckets" → LSH does not appear in the paper. For many documents it proposes greedy clustering in O(rs) (§4.4).
  - "Made near-duplicate search practical without all-pairs comparison" → sketches remove the need to compare full documents. Comparing all pairs of sketches still costs O(r²s) (§4.4).
  - "Exact dedup is not enough for web corpora" → the paper's stated problem is that standard string distances (Hamming, Levenshtein) do not capture "roughly the same" and require comparing entire documents (§1).
- Removed as unsupported by the source: "Became the default primitive behind large-scale corpus dedup pipelines"; "web corpora contain enormous numbers of near-copies, boilerplate variants, mirrored pages, and templated spam that exact dedup misses"; "Run exact or tighter similarity checks only on candidate pairs"; "near-dedup and semantic dedup are distinct problems".
- Not reported by the source: effects on language-model training; precision or recall of near-duplicate detection against labels; LSH banding parameters.
