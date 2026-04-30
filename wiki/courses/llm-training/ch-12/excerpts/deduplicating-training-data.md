---
chapter: ch-12
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/deduplicating-training-data.md
source_url: https://arxiv.org/abs/2107.06499
created_at: "2026-04-23"
---

# Excerpt: Lee 2021 — Deduplicating Training Data Makes Language Models Better

**Source library:** `wiki/raw-data/llm-training/papers/deduplicating-training-data.md`
**Authors:** Katherine Lee, Daphne Ippolito, Andrew Nystrom, Chiyuan Zhang, Douglas Eck, Chris Callison-Burch, Nicholas Carlini (ACL 2022)

---

## Why this source anchors ch-12

Every pretraining corpus built after 2022 — C4-refresh, The Pile, RedPajama, Dolma, FineWeb, Nemotron — cites this paper as the reason it runs dedup. Before Lee 2021, dedup was sometimes done and sometimes not; papers cited "light deduplication" without specifying method or threshold. After Lee 2021, dedup is infrastructure. Ch-12 §2 is this paper compressed into the minimum reading the rest of the chapter requires.

---

## The three load-bearing findings

**1. Memorization is a linear function of duplication count.**

> Without dedup, ~1% of unprompted 256-token completions are verbatim training copies. With dedup, this drops ~10x.

The implication is not just "less parroting." At frontier scale the memorization rate *is* the training-data-extraction attack surface. A document duplicated `k` times is memorized with probability roughly proportional to `k` (Lee 2021 Figure 4, and the follow-up Carlini 2022 "Quantifying Memorization" line). Dedup is therefore the first-line privacy control — no differential-privacy machinery needed, just remove the duplicates.

**2. Train-test contamination is pervasive and silent.**

The paper's contamination table (quoted verbatim in ch-12 §2):

| Corpus | % of validation that overlaps training (>= 50 tokens) |
|---|---|
| LM1B | 4.6% |
| C4 | 3.2% |
| RealNews | 1.6% |
| Wiki-40B | 0.6% |

Ch-12's §2 reads this table as "reported perplexity improvements across a decade of LM papers are partly memorization, not generalization." LM1B in particular had been the standard billion-word LM benchmark; a 4.6% overlap means any model trained on anything derived from Common Crawl had been tested on data it had seen. The paper's fix — dedup train against val — is now the default integrity check for any new benchmark release.

**3. Dedup shortens training.**

> Removing near-dups reduces training set by ~5% while improving all downstream metrics.

The naive prediction is that fewer tokens = worse perplexity. The actual observation is that fewer *unique* tokens is the binding constraint, and removing duplicates leaves unique-token count untouched while removing gradient steps that do nothing. This is why ch-12 treats dedup as an *efficiency* intervention, not just a quality one.

---

## The two tools, quoted

From the paper's Technical Details (source file lines 29-41):

> **ExactSubstr** — suffix-array-based exact substring matching.
> - Find all duplicate substrings of length >= **50 tokens** (the threshold chosen empirically — long enough to avoid common phrases).
> - Remove one copy of each duplicate span.
> - Runs in O(N log N) via suffix array construction on the concatenated corpus.

> **NearDup** — MinHash + LSH for fuzzy document-level dedup.
> - Compute 5-gram shingles per document.
> - Build 9000 MinHash signatures (aggressive signature count for high recall).
> - LSH with b = 20 bands of r = 450 rows, threshold ~ (1/b)^(1/r) ~ 0.8 Jaccard similarity.
> - Any document exceeding threshold against another is dropped.

Two things to flag for ch-12 §4 readers.

First, the quoted formula `t ~ (1/b)^(1/r)` with `b=20, r=450` evaluates numerically to `t ~ 0.9934`, not 0.8. The paper's `~0.8` number is the Jaccard of the *verifier* step that runs after LSH candidate generation, not the S-curve's 0.5-threshold. This is a common pedagogical trap: the LSH parameters are tuned for recall, not for being the final threshold. The verifier is what makes the threshold 0.8. Ch-12 §4 derives this explicitly so the reader does not memorize "Lee used 0.8 therefore the formula says 0.8."

Second, **9000 signatures is aggressive.** Hoeffding-style concentration on `J` requires ~1000 signatures for `eps=0.05`; Lee et al. spent 9x more because they cared about the right tail (finding candidate duplicates), not the uniform estimate. Production MinHash pipelines in 2024-2025 typically use 128-256 signatures with careful `(r, b)` tuning — see datasketch's defaults and the FineWeb implementation.

---

## The 61-word sentence

From the abstract:

> Removing from C4 a single 61 word English sentence that is repeated over 60,000 times.

This one line did more for corpus-dedup adoption than the contamination table. It is concrete, absurd, and easy to reproduce: grep C4 for any of several legal-boilerplate phrases and you find >10^4 hits. A 60,000-copy sentence at 61 tokens is 3.66 M training tokens of identical gradient signal. Even if the sentence is legitimate (cookie notice, license text), the model does not need 60,000 gradient steps to learn it; by the 50th it has memorized it.

Ch-12 §1's "duplicate types" table lists this as "long verbatim substrings across docs" — the ExactSubstr workload. It is not a document-level duplicate (the hosting documents are genuinely distinct) and MinHash with document granularity cannot catch it. This is the paper's main argument for shipping *both* tools: they cover disjoint failure modes.

---

## Why the paper shipped both methods

From the paper's framing (implicit in "Two complementary methods"):

- **ExactSubstr** finds *substrings*: span-level duplicates regardless of host document. Cost: O(N log N) and high memory (suffix array is ~8 bytes/token).
- **NearDup** finds *documents*: documents that are near-copies as a whole. Cost: O(N) via LSH buckets, low memory.

Neither subsumes the other. A document mostly-rewritten but containing a 100-token verbatim block is caught by ExactSubstr, missed by NearDup. A document that is 95% paraphrased from another (different surface words, same structure) is caught by NearDup if shingles are short enough and the paraphrase preserves them, missed by ExactSubstr. The paper's quiet point: the decision is not "which tool" but "cascade them."

Ch-12 §7 builds the production cascade exactly this way: URL-hash first, MinHash second, ExactSubstr (or paragraph-hash Bloom as a cheaper proxy) third, paragraph-level last after quality filtering.

---

## Operational checklist the paper leaves implicit

Things ch-12 highlights that the paper assumes the reader infers:

- Dedup runs **before** quality filtering when the goal is token-efficiency (don't score documents you'll throw away) but **after** quality filtering when the goal is diversity (don't delete rare-but-genuine content that a later quality filter would have kept).
- The `(r, b)` knob is the **cost vs recall** lever; the **threshold** is enforced by the post-filter.
- Per-source dedup is safer than cross-source dedup — see Dolma's source-specific passes.
- Global vs per-snapshot is an open question even in 2024 — FineWeb flipped the default to per-snapshot.

---

## Connections

- [[excerpts/minhash-lsh]] — the algorithmic primitive Lee 2021 plugs in as NearDup.
- [[excerpts/d4]] — the semantic-dedup extension that catches what 5-gram-shingle MinHash misses.
- [[excerpts/fineweb]] — the per-snapshot vs global-MinHash ablation that updates Lee 2021's defaults.
- [[excerpts/dolma]] — the production filter cascade that embeds Lee's methods.
- [[ch-12]] §2 (the evidence), §5 (ExactSubstr mechanics), §7 (the production cascade).
