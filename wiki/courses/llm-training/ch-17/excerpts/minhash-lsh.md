---
chapter: ch-17
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/minhash-lsh.md
source_url: https://doi.org/10.1109/SEQUEN.1997.666900
created_at: "2026-04-23"
---

# Excerpt: MinHash + LSH — the banding math that picks (b, r) in ch-17 §4

**Source library:** `wiki/raw-data/llm-training/papers/minhash-lsh.md`
**Reference:** Broder 1997 and the subsequent LSH literature.

---

## Why the banding math is non-negotiable for the lab

Ch-17 §4 asks students to pick `BANDS` and `ROWS` for MinHashLSH. The §4 code sketch uses (16, 8) on the 1 GB path and cites Lee et al.'s (20, 450) for the full-budget path. Those numbers are not arbitrary — they come from the LSH collision-probability curve. A student who picks (b, r) without understanding this curve will either over-merge (drop ~60% of a clean corpus) or under-merge (drop ~1% and fail to hit the dedup target).

From the source (lines 22–25):

> - Convert each document to shingles.
> - Hash shingles under many permutations or approximations to form a MinHash signature.
> - Use LSH to place similar signatures in shared candidate buckets.
> - Run exact or tighter similarity checks only on candidate pairs.

The four-step pipeline is straightforward. The subtlety is in step 3 — the LSH bucketing. Two documents with Jaccard similarity `s` collide in at least one LSH band with probability `P(collision) = 1 − (1 − s^r)^b`. This function is the S-curve at the centre of every near-dedup pipeline.

---

## The S-curve and why it is sharp around the target

For (b=16, r=8):
- `s = 0.9` → P(collision) = 1 − (1 − 0.43)^16 ≈ 0.9999
- `s = 0.8` → P = 1 − (1 − 0.168)^16 ≈ 0.95
- `s = 0.7` → P = 1 − (1 − 0.058)^16 ≈ 0.62
- `s = 0.5` → P = 1 − (1 − 0.0039)^16 ≈ 0.06
- `s = 0.3` → P = 1 − (1 − 0.00007)^16 ≈ 0.001

The "threshold" is conventionally defined as the Jaccard where P(collision) = 0.5, which for (b=16, r=8) is about `s ≈ 0.69` — not 0.8. The (0.5)^(1/r) = (0.5)^(0.125) ≈ 0.917 and then solving (1 − (1 − 0.917)^b = 0.5) — the exact algebra gives `s_50 = (1 − 0.5^(1/b))^(1/r)`. The point: if you want the 50%-probability threshold to match your Jaccard target, pick (b, r) so that `(1/b)^(1/r) ≈ s_target`. For s_target = 0.8 and r = 8, `b = (1/0.8)^8 ≈ 5.96` — so (b=6, r=8) centres the curve on 0.8. The (b=16, r=8) choice skews sharper: it produces P ≈ 0.95 at s=0.8 and therefore a much more aggressive dedup.

Lee et al.'s (20, 450) is the other extreme: extremely sharp, extremely high recall, at 9000-hash cost. At s = 0.8 it produces P ≈ 0.9999; at s = 0.7 it collapses to P ≈ 0. That is why the [[deduplicating-training-data]] paper treats 0.8 as a hard Jaccard threshold — the banding choice makes the threshold effectively discrete.

---

## What the memo should say about the (b, r) choice

The ch-17 memo's §4 paragraph must justify its (b, r). A correct justification looks like:

> "I used (b=16, r=8) on the 1 GB slice. The P(collision | s) curve gives 0.95 at s = 0.8 and 0.06 at s = 0.5, which means documents at my target Jaccard (0.8) collide in 19 out of 20 runs and near-random document pairs collide in 1 out of 17 — a ~280× signal-to-noise ratio at the LSH stage before any similarity check. This matches the behaviour of Lee et al. 2021's (20, 450) at ~70× lower hash cost, which is the right tradeoff at 1 GB where exact verification of each candidate pair is cheap."

If the memo says "I used (16, 8) because the example code did," that is a failing paragraph. The ch-17 lab is exactly the place to stop treating these hyperparameters as magic.

---

## Why "convert to shingles" is also a choice with consequences

From the source (line 22):

> Convert each document to shingles.

The shingle length is a load-bearing choice. Lee et al. use 5-gram word shingles. Shorter shingles (2-gram) collapse too many unrelated documents into the same signature space; longer shingles (10-gram) make small-edit near-duplicates look independent. Ch-17 inherits the 5-gram default.

At the extreme, character-level shingles (k = 9 characters) are used in some spam-detection pipelines. They are more robust to typos but produce very noisy MinHash signatures. The lab does not explore this axis, but the memo should name the word-5-gram choice as deliberate, not default — especially if the slice contains code, which word-shingles handle badly. (This is one of the [[dolma]] source-specific quirks: `The Stack` code uses MinHash on code tokens, not word 5-grams. Ch-17's slice is web text; 5-grams work.)
