---
chapter: ch-48
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/deduplicating-training-data.md
source_url: https://arxiv.org/abs/2107.06499
created_at: "2026-04-23"
---

# Excerpt: Deduplicating Training Data — the canonical contamination methodology

**Source library:** `wiki/raw-data/llm-training/papers/deduplicating-training-data.md`
**Artifact:** ExactSubstr (suffix-array, 50-token threshold), NearDup (MinHash+LSH, 5-gram shingles, 9000 sigs, 20×450 bands, Jaccard ~0.8), empirical train-test overlap numbers on LM1B / C4.

---

## Why this source is the foundation of ch-48

Every modern decontamination workflow is a repurposing of the two primitives this paper introduced. ExactSubstr + NearDup were originally designed to remove training-set near-duplicates; ch-48 uses the *same primitives* against *eval-set hashes pinned before training* to measure train-eval overlap. The FP/FN table in §2 of the chapter is anchored to the empirical behaviour characterised here.

---

## The two primitives the chapter inherits

Source §Technical Details / ExactSubstr:

> Find all duplicate substrings of length ≥ **50 tokens** (the threshold chosen empirically — long enough to avoid common phrases). Runs in O(N log N) via suffix array construction on the concatenated corpus.

Source §Technical Details / NearDup:

> Compute 5-gram shingles per document. Build 9000 MinHash signatures. LSH with b = 20 bands of r = 450 rows, threshold ≈ (1/b)^(1/r) ≈ 0.8 Jaccard similarity. Any document exceeding threshold against another is dropped.

These are the `MINHASH_SIGS = 9000`, `LSH_BANDS = 20`, `LSH_ROWS = 450` constants in ch-48 §3's decontamination pseudocode. Do not pick different values unless the chapter's memo §7 explains why.

---

## The empirical contamination numbers the chapter cites

Source §Technical Details / Train-test contamination:

> - 4.6% of LM1B validation overlaps training.
> - 3.2% of C4 validation overlaps training.
> - Reported perplexity improvements on benchmarks may be partly illusory without dedup.

Ch-48 §1 opens with these numbers because they establish the base-rate: even in 2021 research corpora constructed with explicit care, multi-percent eval contamination is the default state, not the exceptional case.

---

## Why 50 tokens (not 8, not 13)

Source §Technical Details / ExactSubstr:

> the threshold chosen empirically — long enough to avoid common phrases

50 tokens of English text has vanishingly small collision probability under any reasonable null distribution (common-phrase entropy at 50 tokens is overwhelmingly dominated by rare-word bigram combinations). This is the "essentially zero FP" row of ch-48's detection table — but it pays for that precision with zero recall on any paraphrase, translation, or even whitespace-normalised edit.

The 8-token and 13-token rows in the detection table fill in the tradeoff curve: n=8 for recall-favoured detection (used by open-eval harnesses that want high recall and accept manual triage), n=13 for the GPT-3 / Llama convention, n=50 for a precision floor.

---

## Memorization as proxy evidence for contamination

Source §Technical Details / Findings on memorization:

> - Without dedup, ~1% of unprompted 256-token completions are verbatim training copies.
> - With dedup, this drops ~10×.

For ch-48's memo §7 "what the memo does NOT claim" section: memorization rate is an *indirect* contamination signal. A model with 1% verbatim-emit rate is structurally at risk of leaking eval answers even if the decontamination report says "zero n-gram hits" — because the eval may have been paraphrased in the corpus, then memorized through statistical repetition of its semantic structure. This is why the defensible memo enumerates the recall gap, not just the detector output.

---

## What ch-48 keeps, changes, adds

| Source primitive | Ch-48 usage | Reason |
|---|---|---|
| ExactSubstr ≥50 tokens, internal dedup | Adapted as precision-floor eval-set detector | Same primitive, different pinned set |
| NearDup 5-gram / 9000 sigs / 20×450 | Kept verbatim as MinHash paraphrase detector | Attested threshold Jaccard ~0.8 |
| One-shot corpus dedup | Per-stage (pretrain / SFT / RM / rollouts) | Downstream contamination pathway per [[llama-3]] |
| Eval overlap reported at corpus level | Per-eval-instance overlap fraction | Corpus-level hides instance-level concentration |

---

## Connections to the rest of the chapter

- **[[dolma]], [[fineweb]]** — apply these primitives at 3T / 15T scale respectively; ch-48 inherits their streaming-pass architecture.
- **[[llama-3]]** — the iterative-rounds pipeline that turns a single contamination event into a downstream amplifier.
- **[[bespoke-stratos]]** — concrete case where NearDup on seed prompts is insufficient because memorization lives in the *trace*, not the prompt.
- **[[faithful-synth-eval]]** — the external-verifier analog for detecting contamination in synthesis pipelines.
