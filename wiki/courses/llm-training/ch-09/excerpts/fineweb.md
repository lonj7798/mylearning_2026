---
chapter: ch-09
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/fineweb.md
source_url: https://arxiv.org/abs/2406.17557
created_at: "2026-04-23"
---

# Excerpt: FineWeb — the 15T classifier-filtered web corpus and the per-dump MinHash surprise

**Source library:** `wiki/raw-data/llm-training/papers/fineweb.md`
**Paper:** Penedo et al. 2024, "FineWeb: Decanting the Web for the Finest Text Data at Scale" (HuggingFace).

---

## Why this source anchors ch-09 §2 and §4

FineWeb is the 2024 open-data headline. Fifteen trillion tokens, all from Common Crawl, all published, all with a single-classifier filter producing the FineWeb-Edu 1.3T premium subset. For ch-09's 2020 → 2026 composition-shift narrative, FineWeb is the operational proof that classifier-filtered web text *alone* can match the data-quality ceiling that hand-curated Pile-style mixtures previously held.

This excerpt walks through the pipeline, the classifier training, and the two findings that surprised the field: per-dump MinHash outperforming global MinHash, and a single-score threshold beating every stacked-heuristic recipe on MMLU/ARC.

---

## The headline claim — 15T tokens, fully open, matches Llama 3

From the source (lines 7-8):

> - **Core Insight:** A single, classifier-scored quality filter (educational value, annotated by an LLM) on deduplicated Common Crawl beats every prior hand-crafted web recipe on knowledge-and-reasoning benchmarks.
> - **Guideline:** For web data at 2024+ scale, invest effort in a small LLM-labeled classifier tuned to your downstream benchmark family (e.g., educational content for MMLU), then apply a single score threshold — don't stack dozens of heuristics.

This is the 2024 counter-thesis to The Pile's 2020 mixture argument. [[the-pile]] said: "diverse domain mixture is a scaling variable." FineWeb says: "for web text specifically, a single educational-value classifier is the dominant quality signal, and you don't need to stack heuristic filters to reach Llama-3-adjacent quality on MMLU."

Both claims can be simultaneously true. The Pile's mixture argument is about *cross-domain* generalization — academic + code + books + web. FineWeb's classifier argument is about *intra-web* quality. Modern open stacks ([[olmo-2]] OLMo-Mix-1124, [[olmo-3]] Dolma 3 Mix) combine both: classifier-filtered CC as the web slice, plus hand-curated code / math / academic / books / encyclopedic slices. FineWeb does not claim web-only is sufficient for frontier performance — it claims web-only is sufficient *for the web-sized share of the mix*.

---

## The pipeline — seven stages, explicitly documented

From the source (lines 29-36):

> **Pipeline (FineWeb base):**
> 1. **URL filter** (blocklist) on Common Crawl WARC files.
> 2. **Trafilatura** for HTML-to-text extraction (higher-quality than CCNet's extractor).
> 3. **fastText language ID** → English only.
> 4. **Quality heuristics** adapted from Gopher + C4 (symbol ratios, line length, etc.).
> 5. **MinHash deduplication per snapshot** (not globally). Global dedup hurt downstream because it removed documents that only re-appear once per snapshot but are high-quality.
> 6. **PII redaction** (email, phone).

For ch-09's "read a pipeline" discipline (§6), this is a textbook example of what a disclosed recipe looks like. Every stage is named, the choice at each stage is justified against an alternative, and the order matters.

Notable design decisions:

- **Trafilatura over jusText.** The 2020 standard ([[the-pile]], [[c4]]) was jusText or custom HTML-to-text extractors. Trafilatura is a 2020-era Python library that treats extraction as a structured content-detection problem. FineWeb's ablations show it reduces boilerplate leakage and improves downstream MMLU.
- **Per-dump MinHash, not global.** See §4 below; this is the counterintuitive finding.
- **Heuristic filters are still present** (step 4) but they are not the quality ceiling — the classifier (step 7, FineWeb-Edu only) is.
- **No explicit content / toxicity classifier.** FineWeb-base does not filter on toxicity; FineWeb-Edu's classifier effectively handles this as a side effect of "educational value."

---

## The per-dump MinHash surprise

From the source (line 35):

> MinHash deduplication per snapshot (not globally). Global dedup hurt downstream because it removed documents that only re-appear once per snapshot but are high-quality.

This is the single most surprising finding of the FineWeb ablations. The intuition everyone had in 2020-2023 was: more deduplication is better, so dedupe across all 96 CC snapshots globally. FineWeb's ablations show the opposite for this specific setup:

**Global dedup hurts.** Why?

- A high-quality reference page (e.g., a canonical Wikipedia-adjacent article, a landmark blog post, a well-maintained documentation page) appears once per snapshot across multiple snapshots. Global dedup removes all-but-one copy, leaving a single lonely occurrence.
- But the "re-appearance" itself is a quality signal: a document that survives crawls month after month is more likely to be well-maintained, well-linked, and useful.
- Training on the multiple copies gives the model *repeated exposure* to the canonical document without accidentally over-representing junk that happens to appear on many URLs within one snapshot.

**Per-dump dedup** preserves this cross-snapshot repetition while removing within-snapshot near-duplicates (multiple copies of the same page under slight URL variations within a single crawl).

The mechanism is not obvious in advance, but the ablation is concrete: per-dump-MinHash-trained 350M models beat global-MinHash-trained 350M models on MMLU by ~2 pp at matched tokens.

For ch-09's message — "read pipelines critically, not ritualistically" — the per-dump finding is the specific example. A dedup ritual ("always globally dedupe") is defeated by an ablation against downstream eval.

---

## The FineWeb-Edu classifier — LLM-annotated quality

From the source (lines 37-42):

> **FineWeb-Edu classifier:**
> - Trained on **450K web samples** annotated by **Llama-3-70B-Instruct** with integer scores 0–5, where 0 = not educational, 5 = highly educational.
> - Small classifier head on top of a frozen embedding model.
> - Filter keeps documents with score ≥ 3. This threshold **removes ~92% of FineWeb**, leaving 1.3T tokens.
> - On a hold-out of 46,867 Llama-3-annotated samples, the score≥3 binary classifier reaches F1 = 82%.
> - Threshold 3 was chosen for best trade-off: higher thresholds gain MMLU but lose HellaSwag.

This is the 2024 canonical pattern for LLM-annotated data filtering. Four design decisions to internalize:

1. **LLM-as-annotator.** 450K samples is too many for human labeling, too few for an ML-only signal. Llama-3-70B-Instruct is the labeler; its "educational value" judgment is the training target.
2. **Small classifier, big labeler.** The deployed filter is a cheap head on a frozen embedding model, which can be applied at 15T-token scale. The expensive 70B labeler never touches production inference; it only produces the 450K training labels.
3. **Threshold is a tuned hyperparameter, not a binary.** Score ≥ 3 keeps 8%; score ≥ 4 keeps 2%; score ≥ 2 keeps 25%. FineWeb-Edu picked 3 because the MMLU/HellaSwag Pareto frontier bends there.
4. **Benchmark-tuned, not benchmark-agnostic.** "Educational value" is a proxy for MMLU/ARC; HellaSwag (which measures common-sense completion more than knowledge) actively disagrees with high thresholds. There is no universal "quality classifier" — only a classifier tuned for a benchmark family.

For ch-09 §4 (shift to classifier-filtered web) and §7 (how to read a new pipeline): the FineWeb-Edu classifier is the template that [[qwen-3]]'s "large-scale annotation for educational value, domain, and safety" likely mirrors (Qwen does not publish its classifier; the recipe is nearly identical to FineWeb-Edu's published pattern).

---

## FineWeb vs FineWeb-Edu — the 92% cut

The FineWeb (15T) → FineWeb-Edu (1.3T) relationship is not a different corpus; it is a filter threshold. FineWeb-Edu is literally `FineWeb[classifier_score >= 3]`. The 92% removed is ~13.7T tokens — more than the entire Llama 3 pretraining corpus.

What this means operationally:

- Train on **FineWeb (15T)** if you care about raw token diversity and can afford the compute. Example: a 70B-parameter Chinchilla-optimal run with 1.4T tokens uses ~9% of FineWeb; a 405B-parameter extended run with 15T tokens uses all of it.
- Train on **FineWeb-Edu (1.3T)** if you prioritize MMLU/ARC and can accept the smaller token budget (Chinchilla-optimal for 70B).
- Most 2024-2025 open mixes (OLMo-Mix-1124, OLMo 3's Dolma 3 Mix) effectively combine: use FineWeb-Edu as the "curriculum-end" slice and FineWeb (or equivalent wider recipe) as the bulk.

For ch-09 §7's point about stage-level budgets: OLMo 3's Stage 2 (Dolmino, 100B tokens) and Stage 3 (Longmino, 50B tokens) are budget-sized specifically because their higher-quality slices are 1-2 orders of magnitude smaller than the Stage 1 pool they filter against. FineWeb-Edu is in the same league — a 1.3T "ceiling quality" slice that frontier open mixes draw from for mid-training phases.

---

## Why this was released open at all

From the source (lines 17-21):

> ## Key Contributions
> - **FineWeb**: 15T tokens from 96 Common Crawl dumps — largest fully-open web corpus at release.
> - **FineWeb-Edu**: 1.3T-token subset filtered by an educational-quality classifier; outperforms FineWeb on MMLU/ARC/OpenBookQA.
> - Full ablation study of dedup strategies (global vs per-snapshot MinHash) and filter order.
> - Public release of classifier, codebase (`datatrove`), and all ablation checkpoints.

HuggingFace's strategic logic (inferable from its positioning, not the paper): if the open community has parity with Llama 3's *data* — a corpus that matches 15.6T undisclosed tokens at 15T fully-disclosed tokens — then the closed-vs-open gap collapses to pure algorithm / compute, where the open community can compete.

For ch-09 §4's non-disclosure discussion: FineWeb's existence is why frontier labs' non-disclosure is actionable. A closed lab training on 15T tokens of "high-quality web" can no longer hide behind "our data is better" — FineWeb is *right there*, matched, ablatable. Any remaining performance gap has to be explained by something else (code mix, synthetic data, architectural gains, post-training).

---

## What to take from FineWeb for ch-09

1. **Classifier-filtered web is the 2024+ frontier-quality substrate.** Single threshold beats stacked heuristics for benchmark-family optimization.
2. **LLM-as-annotator is the scalable labeling pattern.** 450K samples from a frontier model → small classifier → filter 15T tokens.
3. **Per-dump MinHash beats global MinHash.** Cross-snapshot repetition is a quality signal, not a redundancy to remove.
4. **FineWeb : FineWeb-Edu :: bulk : curriculum-end slice.** Threshold choice is Pareto-tuned; there's no universal quality classifier.
5. **Open-data parity at 15T changes the closed-disclosure conversation.** The gap between published and un-published mixes is now measurable, because there's a same-size reference point.

---

## Connections

- [[excerpts/the-pile]] — the 2020 counter-predecessor; hand-curated mixture vs classifier-filtered web.
- [[excerpts/dolma]] — the heuristic-first open pipeline that FineWeb's classifier approach challenges at matched web-slice size.
- [[excerpts/llama-3]] — the closed 15.6T counterpart to FineWeb's 15T; disclosure-gap reference.
- [[excerpts/qwen-3]] — the closed corpus that probably uses a similar classifier pattern without disclosing it.
- [[excerpts/olmo-3]] — combines classifier-filtered CC (FineWeb-Edu-style) with additional slices.
- [[ch-09]] — §2 (comparison-table FineWeb row), §4 (classifier-filtered-web shift), §6 (disclosure gradient).
