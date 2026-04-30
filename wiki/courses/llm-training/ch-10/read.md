<!-- chapter: ch-10
     track: data
     title: Open Curation Pipelines — CCNet, C4, Dolma, FineWeb
     kind: content
     deps: [ch-09]
     sources: [[ccnet]], [[c4]], [[dolma]], [[fineweb]], [[scaling-laws-data-quality]], [[rephrasing-the-web]], [[deduplicating-training-data]], [[minhash-lsh]]
     figures: figures/pipeline-compare.html
-->

# Chapter 10 — Open Curation Pipelines: CCNet, C4, Dolma, FineWeb

> **Core insight.** Four pipelines, five years, one conceptual skeleton. CCNet (2019) fixed the order — **language ID → dedup → quality** — and every open pipeline since has been an argument about *which* quality signal belongs at step three. C4 answered with heuristics, Dolma answered with heuristics + transparency + ablations, FineWeb answered with an LLM-labeled classifier. The right way to read a new pipeline in 2026 is the same way you read one in 2019: **what filters, in what order, with what thresholds, ablated how.**
>
> **Guideline.** When auditing or designing a web pipeline: (1) identify the *quality signal* (KenLM perplexity / heuristic stack / classifier); (2) locate dedup relative to quality (CCNet puts it before, Dolma puts it around, FineWeb puts it after extraction but before the classifier); (3) demand an ablation table that moves *one* stage against a fixed downstream eval at a fixed model size; (4) treat any pipeline without such a table as unfalsifiable.

---

## Why this chapter exists

[[ch-09]] mapped the *landscape* of pretraining datasets — what exists, what each corpus omits, which labs stopped publishing their mixes. This chapter goes one level deeper into the four pipelines that shaped open pretraining from 2019 to 2026: **[[ccnet]]** (the template), **[[c4]]** (the heuristic baseline everyone copies), **[[dolma]]** (the transparency benchmark), **[[fineweb]]** (the classifier-era replacement). By the end you should be able to open a new data-release paper, skim the pipeline diagram, and identify which filter is load-bearing, which is ornamental, and which threshold you would argue with.

Three historical currents matter. First, the classical signal was **perplexity against a trusted anchor corpus** — CCNet trains a KenLM on Wikipedia and scores the crawl by it. Second, the heuristic current — C4's terminal-punctuation and bad-word rules — chose speed and legibility over semantic targeting. Third, the classifier current — FineWeb-Edu's Llama-3-labeled educational-value score — bet that a small supervised classifier beats any hand-rolled rule stack at modern scale. [[scaling-laws-data-quality]] provides the theoretical counterpart: quality is a scaling variable, so two corpora with equal token counts can sit on different loss curves. The pipelines you study here are the empirical manifestation of that claim.

A fourth current, mostly implicit in all four papers, is **ordering**. The same six filters applied in a different sequence produce materially different corpora, because each filter operates on whatever distribution survives the previous ones. The CCNet → Dolma → FineWeb evolution is not only about *what* to filter; it is about *when* in the cascade. You cannot read these pipelines as bags of rules — you must read them as pipelines.

The companion interactive [figures/pipeline-compare.html](figures/pipeline-compare.html) lays the four pipelines side-by-side with click-to-inspect stages; use it as the running reference while reading §1–§5 below.

---

## 1. CCNet — the template (2019)

[[ccnet]] is the conceptual skeleton later pipelines mutate. Three stages, each doing one job:

1. **Paragraph-level exact dedup, per ~5 GB shard.** Hash normalized paragraphs (lowercase, digits → 0, punctuation stripped). Within a shard of ~1 M docs, drop duplicates. Done *first* because it is cheap and removes boilerplate that would bias the language classifier.
2. **fastText language ID.** Run `lid.176` on each document; keep top-predicted language with score ≥ 0.5. Route documents to per-language downstream processing. CCNet is multilingual by design — this is the step where later English-only recipes (C4, FineWeb) differ.
3. **KenLM perplexity vs Wikipedia.** For each language, train a 5-gram KenLM on Wikipedia of that language. Score each document's perplexity. Partition into **head / middle / tail** by perplexity percentile. *Label, don't drop* — CCNet emits all three partitions and lets the downstream user pick.

Why this became the template: the three stages are orthogonal (each fixes a distinct failure mode — duplicate-boilerplate noise, language confusion, off-distribution text) and each uses a *cheap* signal (hash, tiny classifier, n-gram LM). The recipe runs on a handful of CPU nodes and produces usable multilingual pretraining data. Every open pipeline since has kept the skeleton and argued about step three.

**Why Wikipedia as the anchor?** It is the only multilingual corpus available in 2019 that is (a) cleaned by humans, (b) large enough (at least millions of tokens per language) to train a 5-gram LM, and (c) stylistically aligned with the "encyclopedic explanation" register that pretraining language models mostly want to imitate. The choice is not neutral — Wikipedia has its own style biases (formal register, Western-encyclopedic framing, particular topic distribution) and CCNet's "head" partition inherits those. But the alternative in 2019 was no anchor at all, and that comparison is not close.

The dedup-before-langID ordering is deliberate and sometimes copied wrong. [[dolma]] re-orders (dedup around quality, not before langID) because Dolma's dedup is document-level and expensive enough that you want to throw away non-English documents first. [[fineweb]] reverts to a CCNet-style structural-first ordering (URL filter → extraction → langID → quality → dedup) and argues from ablation that this is correct at 15T-token scale.

**The "head partition" convention.** A subtle point new readers miss: CCNet itself does not ship a "high quality" corpus — it ships a *perplexity-partitioned* one. The head partition is the lowest-perplexity third (most Wikipedia-like), the tail is the highest-perplexity third (most colloquial / noisiest). Downstream users — RedPajama-V1, Llama-1's training mix, early open releases — picked the head partition as "CCNet-filtered Common Crawl" and threw away the middle and tail. This is a *use-site* decision, not a pipeline decision, and it is where most of CCNet's effective quality-filtering power actually sits. If you adopt CCNet, you are really adopting CCNet-head.

---

## 2. C4 — the heuristic baseline (2019)

[[c4]] was built for T5 and popularized the idea that a flat, reproducible English crawl was enough to train a competitive model. The pipeline is a sequence of ~8 rules applied to one Common Crawl snapshot (April 2019 WET):

- **Terminal-punctuation rule** — keep only lines ending in `.` `!` `?` `"`. Drops ~50 % of raw lines. The most aggressive single filter in C4, and the bluntest: it deletes code, bullets, and titles along with menu items.
- **Line length ≥ 5 words.** Targets navigation fragments and captions.
- **Bad-word blocklist** — drop any document containing any of ~400 words from the "List of Dirty, Naughty, Obscene and Otherwise Bad Words." Document-level, not line-level.
- **JavaScript-line strip** — remove lines containing the word "javascript" (case-insensitive). Targets "Please enable JavaScript" boilerplate.
- **Curly-brace strip** and **"lorem ipsum" strip** — delete pages containing template placeholders.
- **langdetect → English with P ≥ 0.99.** Strict threshold; C4 is explicitly English-only.
- **Three-sentence line dedup** — emit all contiguous 3-sentence spans, keep one copy of each unique span globally. Catches the long tail of templated boilerplate that document-level dedup misses.

C4 made web pretraining reproducible (pin the snapshot, run the rules, get the same corpus) and it ran fast. It also baked in biases that later audits (Dodge et al. 2021, cited by [[c4]]) documented: the bad-word blocklist disproportionately removes documents about LGBTQ identities, medical content, and some dialects of English; the P(English) ≥ 0.99 threshold drops African-American English and code-switched prose. These are not bugs in the blocklist or the classifier — they are the *unavoidable cost of document-level heuristic filtering without feedback*. C4 has no ablation table; no one re-ran T5 with each rule toggled individually.

**Read C4 as a reference point, not a recipe.** Its heuristics reappear almost verbatim in every subsequent pipeline (Gopher borrowed them, Dolma stacks them, FineWeb applies them pre-classifier). Its omission of ablations is the gap Dolma later filled.

**One observation about the terminal-punctuation rule.** It is the single most consequential heuristic in the open-pretraining literature because it is copied *implicitly*. Any pipeline whose Gopher-style "fraction of lines ending in terminal punctuation > 0.6" threshold is tuned from C4's rule is downstream of C4's blind spot: documents whose lines are *structurally valid* but not prose-terminated (bullet-heavy technical docs, code-heavy explainers, poetry, song lyrics) are persistently under-kept. If your downstream model is surprised by poetry or by markdown-heavy docs, the chain of blame ends here.

---

## 3. Dolma — the transparency benchmark (2024)

[[dolma]]'s contribution is not a new filter — it is a **fully-documented six-stage cascade with an ablation row per stage**. AI2 published every config, every threshold, and a toolkit (`dolma` CLI) that runs the pipeline as streaming passes over JSONL shards. Each filter emits a per-document *attribute file* with its score; the final keep/drop decision is a cheap second pass over the attribute files. This decoupling is itself a design contribution — it makes re-running the pipeline with different thresholds almost free.

The cascade (web lane):

1. **URL + Bloom dedup** across CC snapshots (FP ≤ 1e-6). Re-crawls dominate CC volume; deduping URLs first eliminates the easiest near-duplicates cheaply.
2. **Document-level near dedup** via Bloom filter on normalized text. Same content served at different URLs.
3. **fastText language ID → English.**
4. **Quality filter stack** adapted from Gopher + C4: symbol-to-word ratio < 0.1, mean line length > 5, fraction-of-lines-with-terminal-punctuation > 0.6, stopword ratio > 0.06, duplicate-line fraction < 0.3.
5. **Content filter** — fastText toxicity/NSFW classifier trained on Jigsaw Toxic Comments; threshold-drop.
6. **PII filter** — high-precision regex for three categories only: emails, IPs, phone numbers. Documents above a PII-density threshold are dropped (not redacted in place).
7. **Paragraph-level dedup last** — Bloom exact-match; alternative `dolma-ngram` marks a paragraph duplicate if the fraction of duplicated n-grams exceeds threshold T (default T = 1.0).

**Why paragraph dedup is last.** Earlier stages change which paragraphs survive; dedup should operate over the surviving distribution, not the raw one. The Dolma paper's ablation (Table 5 in the arXiv version) toggles each stage with a fixed 1B-parameter OLMo-style model trained on 150B tokens from the stage-off corpus. The full-pipeline row sits at the top of downstream task averages; removing document-dedup costs the most; removing paragraph-dedup costs less but non-zero; removing the quality stack costs roughly half of the full-pipeline delta. The ablation is what makes the rest of the pipeline *falsifiable*.

**Two subtleties in Dolma's ordering.** First, language ID *after* document dedup is the opposite of CCNet's "dedup after langID" — justified because Bloom-based doc dedup is cheap enough to run on the full multilingual crawl, and dropping non-English documents after dedup does not re-introduce duplicates. Second, content filtering (toxicity, NSFW) runs *before* the PII stage because the NSFW classifier would otherwise score documents whose PII has already been shifted out of context; keeping the content classifier close to the raw text preserves its calibration.

**Per-source lanes.** The web lane above is only one of Dolma's pipelines. `peS2o` (scientific papers) uses a different quality stack — it trusts publication structure, so terminal-punctuation rules are replaced with section-header and LaTeX-noise heuristics. `The Stack` (code) uses MinHash near-dedup on code tokens rather than paragraph hash, because code reuse patterns are long-range (long verbatim copies of the same file across repositories). Social-media lanes (Reddit) filter by subreddit-level quality lists, not by text classifier. Books (Project Gutenberg) use near-zero filtering because the source is already curated. **The right filter depends on the source distribution**, and Dolma is the first open pipeline to make that explicit.

**The `dolma` toolkit as a design artifact.** The toolkit accepts YAML configs, runs filters as streaming passes over JSONL shards, and emits per-document attribute files (one score per filter) so the final keep/drop decision is a separate cheap pass. This two-pass architecture — *score everything, decide later* — is a genuine engineering contribution. You can re-sweep thresholds without re-running the expensive filter passes. It also makes ablation cheap: the ablation table is a script over the attribute files, not a re-run of the pipeline.

See [[excerpts/dolma]] for the full ablation table with filter × model-size × eval numbers.

---

## 4. FineWeb and FineWeb-Edu — the classifier era (2024)

[[fineweb]] replaces the heuristic-stacking question with a supervised-learning question: *can a small classifier, trained on LLM-labeled data, do what no regex stack can?* The answer shipped in HF's 15T-token corpus: yes, on knowledge-and-reasoning benchmarks, by a wide margin.

**FineWeb base pipeline.**

1. **URL blocklist filter** on WARC files.
2. **Trafilatura** for HTML-to-text extraction, replacing the WET/boilerpipe extractor used by CCNet. Trafilatura preserves paragraph structure and strips chrome more aggressively. Ablation in [[fineweb]]: swapping WET for Trafilatura gives a measurable quality bump *before any semantic filter runs*. This is the most under-cited lesson of the paper — the extractor choice is not infrastructure, it's a filter.
3. **fastText language ID → English**, P ≥ 0.65 (looser than C4's 0.99; FineWeb tightens later via quality).
4. **Gopher + C4 heuristic cascade**, identical to Dolma's stage 4.
5. **MinHash dedup per snapshot**, not global. MinHash-LSH on 5-grams, Jaccard threshold 0.8, applied within each of 96 CC snapshots.
6. **PII redaction** (email, phone) — in-place replace, not document drop.

The **per-snapshot MinHash** finding is the genuinely new empirical result in [[fineweb]]. Global dedup (standard [[deduplicating-training-data]] recipe) was tested and found to hurt downstream evals. Penedo et al. argue that globally-deduplicated "evergreen" pages — documents that appear in many snapshots because they are high-quality reference material — are exactly the pages whose multiple appearances carry training signal. Dedup within a snapshot keeps that signal; dedup across snapshots destroys it.

**FineWeb-Edu: the educational-value classifier.**

- **Label source.** Llama-3-70B-Instruct annotates 450 K web samples on a 0–5 integer scale (0 = not educational, 5 = highly educational).
- **Classifier.** A small classification head on top of a frozen embedding model (public release: `HuggingFaceFW/fineweb-edu-classifier`).
- **Hold-out performance.** On a 46,867-sample Llama-3-annotated hold-out, score-≥-3 binary classification reaches F1 = 82 %.
- **Threshold.** Score ≥ 3 is the shipped default. This **discards ~92 % of FineWeb**, leaving 1.3 T tokens from 15 T.
- **Why 3 and not 4?** Threshold sweep shows threshold 4 gains MMLU but loses HellaSwag. Threshold 3 is the MMLU/ARC vs HellaSwag trade-off sweet spot.

The classifier-vs-heuristic margin on MMLU and ARC is the FineWeb-Edu headline. See [[excerpts/fineweb]] for the full threshold × token-count × benchmark table.

**Why a classifier beats heuristics at 2024+ scale.** Heuristics encode surface features — punctuation, ratios, line lengths. A classifier trained on LLM-labeled educational-value data encodes a semantic judgment that no regex can express: *does this document have the vibe of a textbook, a lecture, or a well-written explainer?* The FineWeb paper reports that stacking additional heuristics beyond the Gopher + C4 cascade plateaus on MMLU — more rules stop helping. The classifier is the first signal that breaks the plateau. This is the empirical content of [[scaling-laws-data-quality]]'s quality-as-a-scaling-variable argument: heuristics exhaust their per-token information at some scale, and you need a richer signal to keep climbing.

**What replicating FineWeb-Edu actually costs.** Labelling 450 K documents with Llama-3-70B-Instruct is not free — at ~1000 tokens per document and Llama-3's 2024 API rates, the annotation pass is roughly a low-four-figure dollar cost plus multi-day wall time on a cluster. Training the classifier head on frozen embeddings is fast (hours on a single GPU). The 15 T → 1.3 T filtering pass is one streaming read over FineWeb. *Most of the cost is in the anchor-LLM annotation pass.* This makes FineWeb-Edu easy to specialize — annotate 450 K documents with a *different* prompt ("rate on math reasoning value" / "rate on code explanation quality") and you get a task-specialized Edu-style corpus for the same marginal cost. HF's Nemotron-CC and later task-specific corpora follow exactly this recipe.

**The DCLM-baseline comparison.** DCLM (DataComp-LM, Li et al. 2024, not in this chapter's source list but worth naming) runs a head-to-head of C4-style heuristics, CCNet-style perplexity, Dolma's stack, and a FineWeb-Edu-style classifier on the same DCLM benchmark harness. The DCLM leaderboard is the clearest public ranking of these pipelines on matched evals: classifier-based pipelines dominate the top, heuristic-only pipelines cluster in the middle, and raw Common Crawl sits at the bottom. When you are in doubt about which pipeline to borrow, the DCLM leaderboard is the current empirical answer; the papers in this chapter are what let you understand *why* the ranking comes out that way.

---

## 5. The filter-by-filter comparison

This is the table to memorize. Rows are filter *families*; columns are the four pipelines. Entries are the concrete signal + threshold + (where reported) ablation delta.

| Filter family | CCNet (2019) | C4 (2019) | Dolma web (2024) | FineWeb / -Edu (2024) |
|---|---|---|---|---|
| Source snapshots | Multiple CC WET, multilingual | 1 CC snapshot (Apr 2019) WET | Multiple CC, English slice | 96 CC snapshots, English |
| HTML→text | WET (boilerpipe) | WET (boilerpipe) | WET + per-source variants | **Trafilatura** (paper: measurable quality gain over WET) |
| URL filter | — | — | URL + Bloom dedup (FP ≤ 1e-6) | URL blocklist + URL dedup |
| Language ID | fastText lid.176, score ≥ 0.5, multilingual | langdetect, P(English) ≥ 0.99 | fastText, English-only | fastText, P(English) ≥ 0.65 |
| Heuristic quality | — (uses perplexity instead) | Terminal-punct + ≥5-word lines + JS/curly/lorem strip | Gopher + C4 stack: symbol-ratio < 0.1, mean-line > 5, term-punct > 0.6, stopword > 0.06, dup-line < 0.3 | Same Gopher + C4 stack as Dolma |
| Semantic quality | **KenLM perplexity vs Wikipedia**; partition head/mid/tail | — | — (heuristic only) | **FineWeb-Edu classifier, score ≥ 3** (Llama-3-labeled, F1 = 82 %) |
| Blocklist / content | — | ~400-word bad-word list, doc-level drop | fastText toxicity/NSFW (Jigsaw) | URL blocklist at stage 1 only |
| PII | — | — | Regex: email/IP/phone, doc-drop on density | Regex: email/phone, in-place redact |
| Exact dedup | Paragraph, per shard, before langID | 3-sentence line span, global | Doc-level Bloom + paragraph Bloom last | — |
| Near dedup | — | — | Optional `dolma-ngram` paragraph variant | **MinHash per snapshot** (not global; ablation: global hurts) |
| Ablation published? | No (partitions speak for themselves) | No | Yes — per-stage × downstream eval (Table 5) | Yes — dedup strategy + classifier threshold sweeps |
| Output scale | ~2 T tokens (all languages, all partitions) | ~750 B tokens | ~3 T tokens | 15 T (base) → 1.3 T (Edu, score ≥ 3) |

Two rows are load-bearing. The **semantic-quality row** — where each pipeline's "quality signal" lives — is the axis along which the field has actually moved (perplexity → heuristics → classifier). The **ablation-published row** is the axis along which the field has become *scientific* rather than *artisanal*.

---

## 6. How to read and critique a new pipeline

When you open a new data release (GPT-5-era open corpus, Qwen-3 data card, any lab's next FineWeb clone), run this four-question checklist:

1. **What is the quality signal?** Perplexity against what anchor? Which heuristics at which thresholds? What classifier trained on what labels? If the paper does not answer this in one paragraph, the pipeline is not reproducible.
2. **In what order relative to dedup and langID?** Pre-dedup quality wastes cycles scoring duplicates; post-dedup quality may score a distribution the classifier was not trained on. Ask: why *this* order?
3. **With what thresholds, justified how?** A threshold without a sweep is a superstition. FineWeb-Edu's score ≥ 3 is defended by a 0–5 sweep against MMLU + ARC + HellaSwag. C4's P(English) ≥ 0.99 is defended by nothing; we now know it cost the corpus certain English dialects.
4. **Ablated how?** One stage off, fixed model size, fixed downstream eval. No ablation → treat the pipeline as an engineering artifact, not a scientific claim. [[dolma]] is the current bar. [[c4]] is below the bar and has been for five years despite being the most-copied recipe.

**Two orthogonal directions to keep in view.** First, [[scaling-laws-data-quality]] argues quality is a scaling variable — the same pipeline delta can be worth a different model-size jump at different scales. A classifier that helps at 1B may plateau at 70B, or vice versa; you cannot rank pipelines without pinning the model size and the downstream eval. Second, [[rephrasing-the-web]] (WRAP) argues that even classifier-filtered web text is noisy enough that *rewriting* it into cleaner styles (Wikipedia-like, Q&A, terse) produces further multiplicative gains — i.e. filtering has a ceiling, and synthetic rephrasing is the next move. Chapters on deduplication ([[ch-12]]) and synthetic data (Track 2) pick up each thread.

**Three failure modes to watch for in new pipelines.** (a) *Ablations at the wrong scale* — a filter that helps at 150 M parameters may regress at 7 B; demand the ablation model size and token count. (b) *Eval leakage* — a classifier trained on data that overlaps the benchmark test set will look miraculous; FineWeb-Edu explicitly decontaminates against MMLU/ARC/HellaSwag, many follow-ups do not. (c) *Overfitting the filter to the benchmark* — if the classifier's training labels are "rate for MMLU usefulness," the downstream MMLU win is near-tautological; the meaningful test is transfer to held-out evaluations the classifier was not tuned for.

**A worked example of pipeline critique.** Suppose a 2026 paper announces "DeepWeb-Pure," a 20 T-token web corpus from 120 CC snapshots. The paper reports +3.5 pp MMLU over FineWeb-Edu. Questions to run the claim through:

- What is the quality signal? If it is "a classifier trained on GPT-5-labeled educational value," ask which labelling prompt was used and whether the labels are public. Unreleased labels = unreproducible pipeline.
- Where is dedup? If it is global MinHash across all 120 snapshots, the per-dump finding from [[fineweb]] suggests the claim should be re-checked with per-snapshot MinHash before attributing gains to the classifier.
- What ablation? If the table is "full pipeline vs no pipeline," the +3.5 pp could live entirely in one stage (extraction, dedup, or classifier) — you cannot tell. Ask for per-stage rows.
- Eval decontamination? If the paper does not say, assume the MMLU gain is contaminated and discount it by a first-order correction.

Running this checklist on any new pipeline sharpens the intuition faster than reading one more paper. The four pipelines in this chapter give you the vocabulary; the checklist is how you use it.

**The shape of future pipelines.** Three trends are visible in 2025–2026 releases.

- *Multi-classifier cascades* — instead of one educational-value classifier, pipelines stack specialized classifiers (math-value, code-quality, reasoning-density) and blend their scores.
- *Rephrasing hybrids* — [[rephrasing-the-web]]-style pipelines that filter then rewrite, turning the curation pipeline into a generation pipeline with a quality gate in front.
- *Domain-native per-source pipelines* — Dolma's per-source lanes were the proof of concept; modern frontier training mixes run 8–12 per-source pipelines in parallel and the web pipeline is just one of them.

Each of these directions keeps the CCNet skeleton in place and extends step three with additional machinery. None of them escape the four-question checklist above.

You audit a multi-classifier cascade with the same questions you audit a single-classifier pipeline — what signal, what order, what threshold, ablated how. The checklist is load-bearing precisely because the skeleton is stable: as long as pipelines fit the CCNet shape, the same four questions remain the right ones.

---

## 7. Where this leaves you

By the end of this chapter the mental model should look like:

- **CCNet is still the skeleton.** Structural stage → dedup stage → quality stage. Every later pipeline is a specific instantiation of this shape with different components and orderings.
- **C4 is a reference, not a recipe.** Copy its rules if you must for reproducibility, but understand which of them are doing work (three-sentence dedup, langdetect) and which are legacy noise (JavaScript-line strip, lorem-ipsum strip).
- **Dolma is the scientific bar.** Per-source pipelines, published thresholds, ablation tables. If your next data release does not match this level of transparency, readers cannot evaluate it.
- **FineWeb-Edu is the 2024+ baseline for web-quality filtering.** One classifier, one threshold, one ablation-driven choice of that threshold. It is also a recipe: swap the anchor-LLM annotation prompt and you can re-run it for your own domain.

Chapters [[ch-11]] (tokenizers, shards, lineage, PII ops) and [[ch-12]] (deduplication) take the downstream half of the pipeline apart in more detail. This chapter stops at the point a pipeline emits cleaned text; the next two chapters cover what you do with that text before it touches a model.

---

## Connections and what's next

- **[[ccnet]] / §1** — three-stage template; the KenLM-vs-Wikipedia perplexity signal.
- **[[c4]] / §2** — heuristic baseline; the bias case study.
- **[[dolma]] / §3** — transparency benchmark; per-source pipelines and the ablation table.
- **[[fineweb]] / §4** — classifier era; per-dump MinHash; FineWeb-Edu's Llama-3-labeled quality model.
- **[[scaling-laws-data-quality]]** — quality as an explicit scaling variable.
- **[[rephrasing-the-web]]** — the "filters have a ceiling" critique and the rephrase-the-web complement.
- **[[deduplicating-training-data]] / [[minhash-lsh]]** — the dedup methods the pipelines in §3–§4 apply.
- **ch-11 (data ops)** — tokenizers, shards, lineage, PII as *operations*; what you build *after* the pipeline of this chapter emits its tokens.
- **ch-12 (dedup)** — the Lee 2021 foundation and the near/semantic dedup methods used by Dolma and FineWeb.

## Further reading

- [[ccnet]] — Wenzek et al. 2019, the three-stage multilingual template.
- [[c4]] — Raffel et al. 2019 (T5 paper); the heuristic baseline plus Dodge et al. 2021 bias audit in Connections.
- [[dolma]] — Soldaini et al. 2024; the ablation table (Table 5) is the thing to photograph.
- [[fineweb]] — Penedo et al. 2024; the per-dump MinHash finding and the FineWeb-Edu classifier recipe.
- [[scaling-laws-data-quality]] — Subramanyam et al. 2025; quality as a scaling variable.
- [[rephrasing-the-web]] — Maini et al. 2024; the filter-ceiling critique.

## Companion visualization

**[figures/pipeline-compare.html](figures/pipeline-compare.html)** — four vertical pipeline lanes (CCNet, C4, Dolma, FineWeb-Edu). Click any stage to see the exact filter signal, threshold, and (where the paper reports one) the ablation delta. The page is the running reference for §5's comparison table — stages colored by *kind* (structural, quality, content, dedup) so you can see at a glance where each pipeline invests its filter budget.
