---
chapter: ch-09
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/the-pile.md
source_url: https://arxiv.org/abs/2101.00027
created_at: "2026-04-23"
---

# Excerpt: The Pile — the 22-subset hand-curation and what each subset taught the field

**Source library:** `wiki/raw-data/llm-training/papers/the-pile.md`
**Paper:** Gao et al. 2020, "The Pile: An 800GB Dataset of Diverse Text for Language Modeling" (EleutherAI).

---

## Why this source anchors ch-09 §2 and §3

Ch-09's central argument — *"a corpus is a composition, not a pile of text"* — is the thesis of this paper named literally "The Pile." EleutherAI's 2020 release is the first public, explicit, hand-curated pretraining mixture; every later open corpus either imitates its explicit-mixture discipline ([[dolma]], RedPajama, SlimPajama) or deliberately departs from it to a single-classifier regime ([[fineweb]]).

The 22-subset table is the literal figure on the chapter's §3 spine. This excerpt walks through *why each subset was chosen*, *what happened to it by 2026*, and what the Pile's one-shot-release format got right and wrong about maintainability.

---

## The mixture-as-scaling-variable argument

From the source (lines 7-8):

> - **Core Insight:** Diversity of source domains is itself a scaling variable; broad, high-quality mixtures outperform monolithic web corpora on cross-domain generalization.
> - **Guideline:** When building a pretraining mix, do not rely only on generic web text; add curated academic, code, forum, and book-like sources with explicit mixture control.

This is the argument that justifies ch-09's four-axis framework. If token count were the only relevant scaling variable, you could just throw more CommonCrawl at the model and be done. [[the-pile]] is the experimental refutation: on cross-domain evals, their 22-subset mixture outperforms pure CC at matched token counts. Gao et al. formalize this as the "diversity bonus" and give explicit weights:

- Upsampled slices (weight > 1): Wikipedia, Gutenberg, arXiv, PubMed Central, StackExchange, USPTO — the high-quality domain-specialist texts.
- Native-weight slices: Pile-CC, OpenWebText2, GitHub — already abundant.
- Downsampled slices (weight < 1): none explicitly; a few subsets have weights approaching 1.

The Pile's upsampling weights are the 2020 expression of what [[scaling-laws-data-quality]] later formalized as effective-sample-size scaling: structured domain text is *worth more per token* than raw web, and the mixture weights encode that per-source multiplier empirically.

---

## The 22 subsets — what each contributed

From the source (lines 22-25):

> ## Technical Details
> - 22 component datasets with manually chosen mixture weights.
> - Includes sources such as PubMed, arXiv, GitHub, books, StackExchange, and web text.
> - Emphasizes domain coverage rather than only crawl cleanup.
> - Also documents risks and problematic sources, which helped push later data documentation standards.

Table of the 22, with the 2026 post-mortem for each:

| # | Subset | ~% (token-weighted) | 2026 status |
|---|---|---|---|
| 1 | Pile-CC | 18.1% | Superseded by FineWeb (15T) and DCLM (~3T); web filtering moved from heuristic to classifier. |
| 2 | Books3 | 8.1% | **Removed from every derivative** after 2023 Rhode Island litigation. |
| 3 | PubMed Central | 7.6% | Aged well; present in peS2o ([[dolma]]). |
| 4 | GitHub | 7.6% | Superseded by The Stack (licence-filtered) and Starcoder. |
| 5 | OpenWebText2 | 5.0% | Obsoleted by FineWeb's classifier-scored web. |
| 6 | arXiv | 4.5% | Aged well; present in peS2o and Proof Pile II. |
| 7 | FreeLaw | 4.1% | Aged well; legal-domain text still rare and valuable. |
| 8 | StackExchange | 2.6% | Aged well; Q&A structure is a high-value format. |
| 9 | USPTO | 1.7% | Aged well; patents are structured domain text. |
| 10 | PubMed Abstracts | 1.5% | Aged well; absorbed into academic slices. |
| 11 | OpenSubtitles | 1.0% | Dropped by most derivatives (quality issues). |
| 12 | Gutenberg (PG-19) | 0.9% | Aged well; public-domain books are the licence-safe replacement for Books3. |
| 13 | DM Mathematics | 0.7% | **Aged poorly** — synthetic problems without chain-of-thought; superseded by Qwen2.5-Math synthetic CoT. |
| 14 | Wikipedia EN | 0.5% | Aged well; still the canonical anchor. |
| 15 | BookCorpus2 | 0.5% | Mostly dropped; licence concerns and quality. |
| 16 | Ubuntu IRC | 0.4% | Dropped; narrow-domain chat log. |
| 17 | EuroParl | 0.3% | Dropped; machine-translation noise. |
| 18 | YouTube Subtitles | 0.3% | Dropped; auto-caption errors. |
| 19 | HackerNews | 0.3% | Kept by some, dropped by others. |
| 20 | PhilPapers | 0.2% | Kept in peS2o and academic slices. |
| 21 | NIH ExPorter | 0.1% | Kept; grant-abstract structure is useful. |
| 22 | Enron Emails | 0.1% | Dropped; PII concerns. |

Aggregated: ~40% of the Pile's tokens (by weight) have survived essentially intact into 2024-2026 open corpora. ~60% have been replaced or removed. The largest single change is Books3 (8.1%) — dropped outright — and the next-largest is Pile-CC (18.1%) — superseded by classifier-filtered CC.

---

## Books3 — the canonical licence mistake

Books3 was the Pile's 2.5% books slice, sourced from Bibliotik (a shadow library / BitTorrent tracker). In 2020 the legal risk was speculative; by 2023, it had crystallised as the central exhibit in multiple lawsuits. From the source (lines 25-26):

> Also documents risks and problematic sources, which helped push later data documentation standards.

EleutherAI did document Books3's provenance — this is one of the reasons the paper is credited with pushing data-documentation norms. But *documenting* is not the same as *avoiding*. By 2024:

- **RedPajama v1** dropped Books3, replacing with Project Gutenberg.
- **SlimPajama** inherited RedPajama's swap.
- **Dolma** explicitly cites the Books3 lesson and uses Gutenberg.
- **OLMo-Mix-1124** and **Dolma 3 Mix** use Gutenberg + other permissively-licenced books.
- **The Pile's own download page** at `the-eye.eu` was taken down; the Pile-without-Books3 is a separate distribution.

For ch-09 §5's licence discussion, Books3 is the single most important case study. It is also the reason "every opt-out register" conversation takes place — post-Books3, the ambient expectation is that training-data use of copyrighted text is auditable.

---

## Why Pile-CC underperforms FineWeb

From the source (line 15):

> The Pile is an 825 GiB English text corpus built from 22 diverse high-quality subsets spanning academic text, code, books, web text, and forums.

"High-quality" is doing a lot of work in that sentence for Pile-CC. The Pile-CC recipe is roughly:

1. Start from Common Crawl WARC files.
2. jusText extraction.
3. Language filter (English).
4. Heuristic quality filters (similar to C4's).
5. Deduplication.

This is the 2020 state of the art. [[fineweb]] (2024) is the next-generation recipe:

1. Start from 96 CC WARC snapshots.
2. Trafilatura extraction (higher-quality than jusText).
3. fastText language ID.
4. Stronger heuristics (Gopher + C4 stack).
5. Per-dump MinHash (not global — a counterintuitive finding).
6. PII redaction.
7. (For FineWeb-Edu) Llama-3-70B-annotated classifier filter at score ≥ 3.

The headline result: FineWeb produces better per-token training signal than Pile-CC on MMLU, ARC, and reasoning benchmarks. For ch-09 §4's "the shift from raw-web maximalism to classifier-filtered web" narrative, Pile-CC vs FineWeb is *the* concrete comparison. Same substrate (Common Crawl), same language (English), 100× scale difference, fundamentally different filter philosophy.

---

## Why the Pile was a one-shot release, not a pipeline

From the source (lines 17-21):

> ## Key Contributions
> - Made explicit mixture design a first-class pretraining decision.
> - Released a broad-source open corpus that became a baseline for open LMs.
> - Showed benefits of curated-domain coverage beyond raw crawl scale.

What EleutherAI delivered in 2020 was a *corpus*: 825 GiB on disk, static, released, done. The code for regenerating it was published but never updated. No Pile-v2 ever shipped.

By contrast, [[dolma]] and [[fineweb]] were released with their *pipelines* as first-class artefacts — `dolma` CLI and `datatrove` codebase, both maintained. Re-running on a 2026 Common Crawl snapshot produces a current corpus. This is the single biggest structural improvement of open-data work between 2020 and 2024.

For ch-09's reader: the Pile's static nature is why its subset-level lessons transferred into later pipelines rather than being rebuilt from scratch. The 22-subset *template* is what survived; the specific corpus file is a 2020 snapshot.

---

## What to take from The Pile for ch-09

1. **Mixture is a design decision, not an accident.** The Pile made this explicit first; every later corpus either inherits that discipline or is measured against it.
2. **40% of hand-curated slices aged well; 60% did not.** Structured domain text (arXiv, PubMed, GitHub-via-The-Stack, StackExchange, FreeLaw, USPTO, Wikipedia, Gutenberg) survives. Shadow-library books, narrow chat logs, auto-captioned media, synthetic math-without-reasoning do not.
3. **Documentation is not the same as licence safety.** The Books3 lesson: "we documented it" does not shield against the 2023+ litigation environment.
4. **One-shot releases lose to pipelines.** Static corpus files are artefacts of their release year; maintainable pipelines remain current.
5. **Pile-CC → FineWeb is the canonical web-filter recipe transition.** Heuristics → classifier, hand-tuned → LLM-annotated.

---

## Connections

- [[excerpts/dolma]] — the six-stage-cascade successor to the Pile's hand-curation philosophy.
- [[excerpts/fineweb]] — the single-classifier successor that challenges the "mixture is necessary" claim for web text.
- [[excerpts/llama-3]] — the closed-corpus counterpart that does not publish its mix at all.
- [[excerpts/olmo-3]] — the maximum-transparency 2025 open-corpus release, Dolma 3 stages.
- [[excerpts/qwen-3]] — the 36T multilingual corpus that includes synthetic data at large scale.
- [[ch-09]] — §2 (comparison table row: The Pile), §3 (22-subset aging), §5 (Books3 as the licence case study).
