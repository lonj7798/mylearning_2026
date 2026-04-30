---
chapter: ch-29
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/minhash-lsh.md + wiki/raw-data/llm-training/papers/deduplicating-training-data.md
source_url: https://arxiv.org/abs/2107.06499
created_at: "2026-04-23"
---

# Excerpt: MinHash-LSH — the dedup primitive ch-29's Stage 3 implements

**Source library:**
- `wiki/raw-data/llm-training/papers/minhash-lsh.md` (classical definition)
- `wiki/raw-data/llm-training/papers/deduplicating-training-data.md` (operational parameters)

**Artifact:** shingle → MinHash signature → LSH band-bucket → exact check

---

## Why this source anchors ch-29

Evol-Instruct is a *paraphrase machine*. Run the `deepening` operator twice on the same seed and you get two near-duplicates that share no token boundary but have Jaccard similarity > 0.85. Exact dedup catches zero of them. [[self-instruct]]'s default is ROUGE-L > 0.7, which is O(N²) and blows up at ch-29's 10K+ candidate scale. MinHash-LSH is the standard answer: sub-linear, approximate, and attested by the single highest-ROI dedup paper in LLM pretraining ([[deduplicating-training-data]]).

---

## The primitive, stated exactly

From [[minhash-lsh]] (lines 21–25):

1. Convert each document to shingles (k-grams).
2. Hash shingles under many permutations to form a MinHash signature.
3. Use LSH to place similar signatures in shared candidate buckets.
4. Run exact or tighter similarity checks only on candidate pairs.

The key property: if two documents have Jaccard similarity ≥ `J`, LSH places them in the same bucket with high probability (`1 - (1 - J^r)^b`) where `b` is the number of bands and `r` is rows per band. Tighter bands → higher recall at higher `J`.

---

## The attested parameters from [[deduplicating-training-data]]

From the source (lines 38–42):

- 5-gram shingles per document.
- **9000 MinHash signatures** (aggressive signature count for high recall on web-scale C4).
- LSH with **b = 20 bands of r = 450 rows**, threshold ≈ `(1/b)^(1/r)` ≈ **0.8 Jaccard similarity**.
- Any document exceeding threshold against another is dropped.

These are pretraining-scale numbers. Ch-29's lab-scale numbers are:
- 5-gram shingles (same — attested).
- **128 MinHash permutations** (`datasketch` default; sufficient at 10K pool, not 100M).
- LSH threshold **0.8** by default; drop to **0.7** if your Evol-Instruct paraphrases are escaping (tune and log).
- Single-pass insert-then-query loop (see below).

---

## The ch-29 implementation — what `datasketch` does and does not do

```python
from datasketch import MinHash, MinHashLSH
lsh = MinHashLSH(threshold=0.8, num_perm=128)
for i, s in enumerate(samples):
    mh = fingerprint(s.instruction + " " + s.output)
    if not lsh.query(mh):        # no collision → keep
        lsh.insert(str(i), mh)
        kept.append(s)
```

- `lsh.query(mh)` does the band-bucket lookup and returns candidate IDs.
- `lsh.insert` adds the signature to all its band buckets.
- **No exact Jaccard check** — the library treats bucket collision as "near-duplicate." At 128 perms this has a false-positive rate ~5% at true J=0.75; acceptable for the lab, not for a production corpus.

For tighter precision (ch-29 resource-constrained path does not need this), add the exact `MinHash.jaccard(other)` check on every collision before rejecting.

---

## Why 3% is the expected loss, and why >30% is a red flag

[[deduplicating-training-data]] (line 45): "Removing near-dups reduces training set by ~5% while improving all downstream metrics."

On a well-bounded synthetic pool post-IFD, expected dedup loss is 5–20%. If your dedup kills >30% of the post-IFD pool, one of:

1. Your Evol-Instruct rounds are producing trivial paraphrases (too-similar operator applied twice in a row to the same seed) — reduce rounds or increase seed diversity.
2. Your IFD filter is keeping a near-identical top-K across seeds — lower `keep_frac` or add diversity weighting.
3. Your seed pool itself is collapsed — re-hand-write the seeds with more topical spread.

The `dedup_check.py` acceptance gate (ch-29 gate 3) exists to make this failure mode visible.

---

## The memorization angle — not used by ch-29, but worth knowing

From [[deduplicating-training-data]] (lines 48–51):

> Without dedup, ~1% of unprompted 256-token completions are verbatim training copies. With dedup, this drops ~10×. Privacy implication: dedup alone substantially reduces the surface for membership-inference and training-data-extraction attacks.

Synthetic pools inherit this property transitively: if your teacher generates a verbatim common Stack Overflow answer in multiple paraphrases and you do not dedup, the SFT model will memorize that one answer. This argument is why dedup is mandatory even for teacher-originating data, not only web scrapes.

---

## What ch-29 keeps, changes, drops

| [[deduplicating-training-data]] default | Ch-29 choice | Reason |
|-----------------------------------------|--------------|--------|
| 5-gram shingles | Same | attested |
| 9000 MinHash signatures | 128 | lab scale; 9000 is overkill < 100K docs |
| b=20, r=450 (≈0.8) | `datasketch` default (b=16, r=8) at threshold=0.8 | library handles band/row split |
| Exact substring (50-token) via suffix array | Not used | MinHash alone is sufficient for 1K–10K pool |
| Document-level only | Same (inst + output concat) | chapter's granularity |

---

## Connections

- **ch-25** — the full-read chapter on [[minhash-lsh]] + [[deduplicating-training-data]].
- **ch-26** — semantic dedup ([[d4]] / SemDeDup) is the stricter follow-up; out of scope for the lab.
- **ch-29 §4** — the `no-dedup` ablation should surface at least one duplicate cluster; gate 3 enforces this.
