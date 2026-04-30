---
chapter: ch-09
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/dolma.md
source_url: https://arxiv.org/abs/2402.00159
created_at: "2026-04-23"
---

# Excerpt: Dolma — the six-stage cascade and the "reproducibility as scientific contribution" thesis

**Source library:** `wiki/raw-data/llm-training/papers/dolma.md`
**Paper:** Soldaini et al. 2024 (ACL), "Dolma: an Open Corpus of Three Trillion Tokens for Language Model Pretraining Research" (Allen AI).

---

## Why this source anchors ch-09 §2, §5, and §6

Dolma is the transparency benchmark for ch-09's disclosure axis. Three trillion tokens, six-stage filter cascade, every threshold published, every ablation released, per-source sub-pipeline documented, and the `dolma` CLI shipped as open source. For ch-09 §6's four-axis framework, Dolma is the "every axis maximally disclosed" row — the data-discipline ceiling against which [[llama-3]] (token-total-only) and [[qwen-3-5]] (no report at all) are measured.

This excerpt walks through the six-stage cascade, the per-source quirks, and the framing thesis that the Dolma paper makes explicit: *a reproducible corpus is itself a scientific contribution*.

---

## The framing thesis — reproducibility as a first-class contribution

From the source (line 15):

> Information about pretraining corpora used to train the current best-performing language models is seldom discussed: commercial models rarely detail their data, and even open models are often released without accompanying training data or recipes to reproduce them. As a result, it is challenging to conduct and advance scientific research on language modeling, such as understanding how training data impacts model capabilities and limitations. To facilitate scientific research on language model pretraining, we curate and release Dolma, a three-trillion-token English corpus […]

This is Dolma's political thesis, not just its methodological one. The paper explicitly frames the lack of data disclosure in commercial models as a *problem for science* — that without access to the corpus, downstream research cannot isolate data effects from algorithm effects. Ch-09 §4's "why frontier labs stopped disclosing" section is the other side of this coin: labs have strategic reasons not to disclose, Dolma's authors argue that science needs disclosure, and the field bifurcates.

For the learner: treat Dolma's paper as the reference citation whenever you need to argue "data-disclosure deltas matter for experimental comparability." The Allen AI team, through Dolma and the OLMo series, is the community that operationalized this stance most consistently.

---

## The six-stage filter cascade

From the source (lines 29-35):

> **Filter cascade (order matters):**
> 1. **URL / document-level deduplication first** using Bloom filters — remove exact-URL repeats across CC snapshots.
> 2. **Language identification** via fastText; keep English (lang score ≥ threshold).
> 3. **Quality filters** adapted from Gopher/C4 heuristics (line-length, symbol-to-word ratio, stopword ratio, fraction of lines ending in punctuation, duplicate-line fraction).
> 4. **Content filters**: fastText classifiers trained on the **Jigsaw Toxic Comments** dataset produce `hate` and `NSFW` scores; documents above threshold are dropped.
> 5. **PII filtering** targets three high-precision categories: email addresses, IP addresses, phone numbers.
> 6. **Paragraph-level deduplication last**, via Bloom filter exact-match on paragraphs. Alternative `dolma-ngram` splits paragraphs into n-grams and marks a paragraph duplicate if the fraction of duplicated n-grams exceeds threshold T (default T = 1.0).

This is the spine of ch-10 (Open Curation Pipelines), teed up here. Each stage has a *reason for its position in the order*:

- **Stage 1 (URL/doc dedup first)**: exact-URL duplicates across CC snapshots are the cheapest thing to remove. Doing it first keeps the downstream stages' compute bounded.
- **Stage 2 (language ID)**: cheap, high-value. No sense running expensive quality heuristics on Mandarin text if you're building an English corpus.
- **Stage 3 (quality heuristics)**: removes the worst ~30-50% of surviving docs. Gopher + C4 heuristics are the well-established baseline — line lengths, symbol ratios, terminal punctuation.
- **Stage 4 (content classifiers)**: toxicity / NSFW. Jigsaw Toxic Comments is the training set; fastText classifiers are the production filter. Threshold is tuned per deployment.
- **Stage 5 (PII)**: email, IP, phone. High-precision rule-based redaction. Name/address redaction is *not* attempted (too high false-positive rate).
- **Stage 6 (paragraph dedup last)**: the deliberate choice.

From the source (line 37):

> **Why this order:** doing paragraph dedup last is deliberate — earlier stages change which paragraphs survive, so dedup only matters over the surviving distribution.

This is the subtlety. If you paragraph-dedup first, you dedup paragraphs that will later be dropped anyway (low quality, wrong language). Do it last and you dedup the *surviving* corpus — the one that will actually reach training.

Contrast with [[fineweb]]'s MinHash-per-dump (not global) finding: both are examples of dedup design being non-trivial and benchmark-ablated rather than ritualistic.

---

## Per-source sub-pipelines — web vs code vs academic vs forums

From the source (lines 39-43):

> **Source-specific quirks:**
> - `peS2o` (scientific) uses different quality filters than web — it trusts publication structure.
> - `The Stack` code uses near-dedup via MinHash on code tokens.
> - Social media (Reddit) is filtered by subreddit-level quality lists.

The six-stage cascade is the *web* pipeline. Other sources have different cascades because their quality signals differ:

- **peS2o (academic)**: Semantic Scholar Open Access papers. Trust publication structure (DOI, abstract, references) as quality signal. Skip web heuristics (line-length ratios are useless — papers have tables, equations, long sentences). Use section-level detection for cleanup.
- **The Stack (code)**: MinHash on code tokens (not prose). Licence-filter — only permissively licensed repos. Remove generated code (boilerplate, vendored files).
- **Reddit (forums)**: subreddit-level quality lists (keep /r/AskScience, drop /r/4chan). Within kept subreddits, thread-level structure preserved.

This is ch-11 territory (Data Operations — tokenizers, shards, lineage) but seeded here. **Different sources need different cascades.** One filter pipeline is not enough for a full corpus; you need N pipelines that each produce a documented shard, which then get *mixed* according to a disclosed weighting.

---

## Token-count table — what 3T actually is

Dolma v1.7's per-source composition (approximate, from the paper's Table 1):

| Source | Tokens (approx.) | Stage of mix |
|---|---|---|
| CommonCrawl (filtered through Dolma 6-stage) | ~2.4 T | bulk web slice |
| The Stack (filtered to permissive licences) | ~240 B | code slice |
| peS2o (Semantic Scholar OA papers) | ~150 B | academic slice |
| Reddit (subreddit-filtered) | ~90 B | forum slice |
| Wikipedia + Wikibooks | ~60 B | encyclopedic slice |
| Project Gutenberg | ~60 B | books slice |

Totals: ~3 T tokens. For ch-09 §2's comparison table, this maps onto the ~80%/8%/5%/3%/2%/2% breakdown.

Comparison with The Pile (825 GiB, ~300B tokens): Dolma is **10× larger**, web slice is **3× larger proportionally** (80% vs 23%), code and books are proportionally smaller. The shift from Pile to Dolma is *toward web dominance*, which is the same shift [[fineweb]] makes explicit with its 100%-web-only stance.

---

## The `dolma` tooling — pipelines as artefacts

From the source (line 44):

> **Tooling:** the `dolma` CLI accepts YAML configs, runs filters as streaming passes over JSONL shards, and emits per-document `attribute` files (one score per filter) so the final keep/drop decision is a separate, cheap pass.

Two design choices worth noting:

1. **Streaming over JSONL shards.** 3 T tokens is ~15 TB of text on disk. No single pass materializes the whole corpus in memory; filters stream over shards, and shards are independent (parallelizable across nodes).
2. **Per-document attribute files.** Each filter emits a score per document. The final keep/drop decision is a downstream `dolma tag ... --keep score > T` pass. This separates filter *computation* from filter *decision*, which means thresholds can be re-tuned without re-running filters — a huge ops win.

For ch-09's "pipeline as artefact" point and ch-10's pipeline-deep-dive: the attribute-file pattern is the discipline that makes threshold ablations cheap. Allen AI publishes the attributes alongside the corpus; reproducers can compute their own keep/drop decisions from the same scores.

---

## Ablation discipline — each stage justified against a downstream eval

From the source (line 20):

> - Ablation-driven defense of the filter order: each stage is shown to improve downstream OLMo eval scores.

This is the paper's methodological contribution beyond the corpus itself. Each of the six stages is ablated: train a small OLMo on Dolma-minus-stage-X, compare against Dolma-full, measure the eval delta. Stages that don't improve downstream are not worth their compute.

Ch-09 §7 (how to read a new pipeline) operationalizes this: for any new pipeline, ask "what ablation justifies each stage?" If the answer is "author intuition," treat the stage with skepticism. If the answer is a downstream eval delta, the stage is principled.

---

## What to take from Dolma for ch-09

1. **Disclosure is a stance, not an accident.** Dolma's explicit framing is that reproducibility *is* a scientific contribution. Every threshold and ablation is published.
2. **The six-stage cascade is the modern open-pipeline template.** URL dedup → lang → quality → content → PII → paragraph dedup, in that order, with justifications.
3. **Different sources, different cascades.** Web gets six stages; code gets MinHash + licence; academic trusts publication structure; Reddit uses subreddit lists.
4. **Attribute files separate compute from decision.** Filters emit scores; keep/drop is a downstream pass; thresholds are cheap to re-tune.
5. **Each stage is ablated.** No filter survives without a measured downstream eval delta.

---

## Connections

- [[excerpts/the-pile]] — the hand-curated predecessor; Dolma systematizes what the Pile did informally.
- [[excerpts/fineweb]] — the classifier-first successor; Dolma's heuristic-first stance is the counter-philosophy.
- [[excerpts/olmo-3]] — Dolma 3 extends Dolma's approach to stage-level curriculum disclosure.
- [[excerpts/llama-3]] — the closed-disclosure counterpart; Dolma's thesis explicitly responds to this gap.
- [[ch-09]] — §2 (Dolma comparison-table row), §5 (licence-regime discipline), §6 (disclosure ceiling), §7 (ablation as pipeline-reading tool).
- [[ch-10]] — the full six-stage pipeline deep dive builds on this excerpt.
