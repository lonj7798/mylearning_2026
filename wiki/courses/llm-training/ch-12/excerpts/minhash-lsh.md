---
chapter: ch-12
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/minhash-lsh.md
source_url: https://doi.org/10.1109/SEQUEN.1997.666900
created_at: "2026-04-23"
---

# Excerpt: Broder's MinHash and the LSH Banding Construction

**Source library:** `wiki/raw-data/llm-training/papers/minhash-lsh.md`
**Heritage:** Broder 1997 (resemblance + MinHash) -> Indyk-Motwani 1998 (LSH) -> Leskovec-Rajaraman-Ullman *Mining of Massive Datasets* Ch. 3 (modern pedagogy)

---

## Why this source anchors ch-12

Lee 2021 ([[excerpts/deduplicating-training-data]]) does not re-derive MinHash. It cites Broder and runs. Ch-12 §3-§4 is the missing derivation, because picking `(r, b)` for LSH is one of the small number of places in LLM data pipelines where a researcher *should* compute a formula rather than copy a default. This excerpt reconstructs the primitives end-to-end.

---

## The central theorem, stated precisely

Let `U` be the universe of shingles (5-grams, byte-level n-grams, etc.). Let `S(A), S(B) subset U` be the shingle sets of two documents. Let `J(A, B) = |S(A) ∩ S(B)| / |S(A) ∪ S(B)|` be their Jaccard index.

Let `pi : U -> {1, ..., |U|}` be a uniformly random permutation. Define the **MinHash** signature:

    h_pi(A) = min_{s in S(A)} pi(s)

**Theorem (Broder 1997).**

    Pr[ h_pi(A) = h_pi(B) ] = J(A, B).

**Proof.** Consider `V = S(A) ∪ S(B)`. Under a uniformly random `pi`, every element of `V` is equally likely to be the argmin. Let `s* = argmin_{s in V} pi(s)`. Then `h_pi(A) = h_pi(B)` iff `s*` lies in both `S(A)` and `S(B)`, i.e. `s* in S(A) ∩ S(B)`. The probability of that event is exactly `|S(A) ∩ S(B)| / |V| = J(A, B)`. ∎

This is the single theorem the whole pipeline rests on. Everything else — signature length, banding, S-curve — is engineering around it.

---

## Why one permutation is not enough

A single `h_pi` is a Bernoulli trial with success probability `J`. To estimate `J` to additive error `eps` with confidence `1 - delta`, Hoeffding gives

    m >= (1 / (2 eps^2)) * ln(2 / delta)

For `eps = 0.05, delta = 0.01`, `m ~ 1060`. Real systems approximate this by using `m` independent hash functions rather than true permutations:

    h_i(s) = (a_i * s + b_i) mod p       for i = 1..m

with `p` a large prime and `(a_i, b_i)` drawn per `i`. The signature of `A` becomes

    sig(A) = ( min_{s in S(A)} h_1(s), ..., min_{s in S(A)} h_m(s) )

Each coordinate is Bernoulli(`J`). The fraction of matching coordinates estimates `J`.

Ch-12 §3 notes Lee 2021 uses `m = 9000` — 9x more than Hoeffding demands. The reason is that Lee et al. care about the *right tail* (high-`J` pairs), not the full `J` distribution, and want to maximize recall of true near-duplicates even at the cost of compute.

---

## LSH banding — the construction, derivation, and S-curve

Signature comparison is still O(N^2) without LSH. Partition the signature into `b` bands of `r` rows, so `m = r * b`. Hash each band's `r`-tuple into a bucket; documents sharing *any* band's bucket are LSH candidates.

**Per-band agreement.** Two documents agree on a single row with probability `J`. Assuming independence across coordinates (true in the random-permutation idealization, approximately true under universal hashing), they agree on all `r` rows of one band with probability

    P(one band matches) = J^r

**Per-band disagreement.** The probability they disagree in a band (differ in at least one row) is `1 - J^r`. Over `b` independent bands:

    P(disagree in every band) = (1 - J^r)^b

**Candidate probability** (the event of interest):

    P_LSH(J; r, b) = 1 - (1 - J^r)^b

This is the S-curve ch-12 §4 plots. Its shape:

- For `J << t`: `J^r` is tiny, `(1 - J^r)^b ~ 1`, candidate probability ~ 0.
- For `J >> t`: `J^r` approaches 1, candidate probability ~ 1.
- Transition width scales as `1/(r * b)`.

**The 0.5-threshold.** Set `P_LSH(t) = 0.5`:

    1 - (1 - t^r)^b = 0.5
    (1 - t^r)^b = 0.5
    1 - t^r = 0.5^(1/b)
    t^r = 1 - 0.5^(1/b)
    t = (1 - 0.5^(1/b))^(1/r)

For `b >= 8`, the first-order approximation `0.5^(1/b) ~ 1 - (ln 2)/b` gives the textbook form:

    t ~ (ln 2 / b)^(1/r) ~ (1/b)^(1/r)

**Worked examples.**

| r | b | m = r*b | t (where P_LSH(J=t) = 0.5) |
|---|---|---|---|
| 5 | 20 | 100 | 0.509 |
| 9 | 20 | 180 | 0.687 |
| 10 | 50 | 500 | 0.652 |
| 20 | 50 | 1000 | 0.807 |
| 450 | 20 | 9000 | 0.993 |

The last row is Lee 2021's configuration. The threshold `0.993` means "LSH alone collides almost every pair" — it is a **recall-max** setting paired with an explicit Jaccard verifier, not a standalone threshold. Skipping the verifier with these parameters would produce an enormous false-positive rate (see the left panel of [[ch-12]]'s companion figure).

---

## What can go wrong — the practitioner's list

From [[minhash-lsh]] and 25 years of production experience:

**Shingle choice.** 5-grams over whitespace tokens is the Lee 2021 default. For code, byte-level n-grams are better (whitespace is semantically meaningful; tokenization differs). For short documents (<20 shingles), Jaccard is noisy; fall back to exact hashing.

**Hash independence.** The `(a, b)` parameters must be drawn from a pairwise-independent family. Using `h_i(s) = h(s + i)` for a fixed `h` is a common bug: the resulting signatures are correlated, the `J^r` independence assumption breaks, and the S-curve is wrong.

**Band collision false positives.** Two unrelated documents have a `P_LSH(0, r, b) = 0` probability of sharing a bucket under the ideal model, but under a finite-range hash there is a small residual collision rate `~ 1/bucket_count`. Make the bucket space large (64-bit) so this is negligible.

**Memory pressure at scale.** A 1T-token corpus with 256 signatures per 2KB document is ~32 GB just for signatures, manageable. At 9000 signatures per document (Lee 2021) it is ~1.1 TB, which is why production pipelines (FineWeb's datatrove, Dolma) do not use 9000 signatures outside of research.

---

## When MinHash alone is not enough

Two cases the document-level primitive misses.

**Case 1: small verbatim block in a large document.** Two 5000-shingle documents that share a 50-shingle verbatim block have `J ~ 50/9950 ~ 0.005` — below any reasonable threshold. ExactSubstr (suffix array) is the right tool.

**Case 2: paraphrase.** Two documents that are paraphrases with minimal surface overlap have `J ~ 0`, invisible to shingle-based MinHash regardless of threshold. Semantic dedup (SemDeDup / D4, see [[excerpts/d4]]) is the right tool.

Ch-12 §1's tool-vs-duplicate-type table lays this out: MinHash is the document-level near-duplicate tool and nothing else. Using it for span-level or semantic cases is a category error.

---

## Connections

- [[excerpts/deduplicating-training-data]] — Lee 2021 uses this primitive as NearDup.
- [[excerpts/d4]] — the semantic extension for cases MinHash cannot see.
- [[excerpts/fineweb]] — per-snapshot vs global MinHash ablation, one of the few places the default got overturned.
- [[excerpts/dolma]] — the production cascade context; Dolma uses MinHash for The Stack code sub-corpus.
- [[ch-12]] §3 (derivation), §4 (S-curve), §7 (production cascade).
