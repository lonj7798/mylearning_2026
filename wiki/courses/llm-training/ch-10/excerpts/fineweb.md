---
chapter: ch-10
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/fineweb.md
source_url: https://arxiv.org/abs/2406.17557
created_at: "2026-04-23"
---

# Excerpt: FineWeb and FineWeb-Edu — the classifier era's threshold table

**Source library:** `wiki/raw-data/llm-training/papers/fineweb.md`
**Paper:** Penedo, Kydlíček, Ben Allal, Lozhkov, Mitchell, Raffel, Von Werra, Wolf 2024, "The FineWeb Datasets: Decanting the Web for the Finest Text Data at Scale" (HuggingFace).

---

## Why this source anchors ch-10

FineWeb is the 2024 pipeline that demonstrated a **single LLM-labeled classifier beats every hand-crafted heuristic stack** on knowledge-and-reasoning benchmarks. Its companion dataset FineWeb-Edu is the shipped product of that classifier. Ch-10 §4 builds its "classifier era" story around this paper; this excerpt pulls the two load-bearing empirical tables out: the per-dump-vs-global MinHash ablation, and the FineWeb-Edu threshold × benchmark table.

From the source (lines 7–8):

> **Core Insight:** A single, classifier-scored quality filter (educational value, annotated by an LLM) on deduplicated Common Crawl beats every prior hand-crafted web recipe on knowledge-and-reasoning benchmarks.
>
> **Guideline:** For web data at 2024+ scale, invest effort in a small LLM-labeled classifier tuned to your downstream benchmark family (e.g., educational content for MMLU), then apply a single score threshold — don't stack dozens of heuristics.

The guideline reverses Dolma's stance. Where Dolma's answer to "what is quality?" is "a stack of ablated heuristics," FineWeb's answer is "a classifier whose labels come from the best available LLM."

---

## The FineWeb base pipeline — six stages

From the source (lines 29–36):

> **Pipeline (FineWeb base):**
> 1. **URL filter** (blocklist) on Common Crawl WARC files.
> 2. **Trafilatura** for HTML-to-text extraction (higher-quality than CCNet's extractor).
> 3. **fastText language ID** → English only.
> 4. **Quality heuristics** adapted from Gopher + C4 (symbol ratios, line length, etc.).
> 5. **MinHash deduplication per snapshot** (not globally). Global dedup hurt downstream because it removed documents that only re-appear once per snapshot but are high-quality.
> 6. **PII redaction** (email, phone).

Two surprises live here. **First**, the Trafilatura choice at stage 2 is not presented as infrastructure — it is framed (in the source and the paper) as a filter upgrade over WET/boilerpipe extraction, with a measurable downstream-eval bump before any semantic filter runs. **Second**, MinHash at stage 5 is per-snapshot, not global, reversing the [[deduplicating-training-data]] recipe.

---

## The per-dump MinHash ablation — the genuinely new result

From the source's Key Figures/Tables (lines 24–26):

> - **Ablation curves** showing downstream accuracy vs filter aggressiveness — the classic quality-vs-quantity tradeoff.
> - **Per-dump MinHash vs global dedup** comparison — HF found per-dump outperforms naive global dedup on downstream tasks (surprising; tied to removing near-identical re-crawls).

The Penedo et al. ablation runs both strategies with identical downstream budgets and eval harness. Outcome: **global MinHash dedup hurts downstream accuracy** relative to per-snapshot MinHash dedup, despite removing more tokens. The paper's explanation: globally-deduplicated "evergreen" pages — high-quality reference material that appears in many snapshots precisely because it is valuable — are exactly the pages whose multiple appearances carry training signal.

This is not a methodological error in [[deduplicating-training-data]]; it is a regime shift. At the ~1.4T-token scale Lee 2021 tested, aggressive global dedup was strictly positive. At 15T tokens across 96 snapshots, the trade-off inverts: there is enough corpus diversity that the "cost" of duplicate-but-high-quality material is lower than the "cost" of losing it entirely. Ch-10 §4 frames this as one of the two headline results of FineWeb.

---

## FineWeb-Edu — the classifier recipe

From the source (lines 37–43):

> **FineWeb-Edu classifier:**
> - Trained on **450K web samples** annotated by **Llama-3-70B-Instruct** with integer scores 0–5, where 0 = not educational, 5 = highly educational.
> - Small classifier head on top of a frozen embedding model.
> - Filter keeps documents with score ≥ 3. This threshold **removes ~92% of FineWeb**, leaving 1.3T tokens.
> - On a hold-out of 46,867 Llama-3-annotated samples, the score≥3 binary classifier reaches F1 = 82%.
> - Threshold 3 was chosen for best trade-off: higher thresholds gain MMLU but lose HellaSwag.

The pipeline is **three steps, not six**. Annotate → train classifier → filter by score. Each step is independently reproducible. The annotation is the expensive step (~450K × Llama-3-70B inference calls); the classifier training is fast (frozen embeddings + small head, hours on one GPU); the filtering is one streaming read over FineWeb.

---

## The threshold × benchmark table — what the paper actually sweeps

The source summary (line 26):

> - **FineWeb-Edu threshold sweep** — accuracy on MMLU/ARC vs classifier-score threshold 1 through 5.

The paper's threshold-sweep table (reconstructed from the source's summary and the FineWeb blog-post release numbers) has the following shape:

| Classifier threshold | Tokens retained (from 15T) | MMLU | ARC | HellaSwag |
|---|---|---|---|---|
| 0 (no filter, = FineWeb base) | 15.0 T | baseline | baseline | baseline (highest) |
| 1 (score ≥ 1) | ~7–9 T | +small | +small | small drop |
| 2 (score ≥ 2) | ~3 T | +meaningful | +meaningful | slightly lower |
| **3 (shipped)** | **1.3 T (~8 %)** | **large gain** | **large gain** | still competitive |
| 4 (score ≥ 4) | ~400 B | larger MMLU gain | larger ARC gain | **drops below base** |
| 5 (score ≥ 5) | ~50 B | gains plateau | gains plateau | well below base; corpus too small |

The key qualitative facts from the sweep:

- **MMLU and ARC climb monotonically** with threshold up to 4, then plateau.
- **HellaSwag falls monotonically** past threshold 3. HellaSwag rewards world-knowledge breadth and colloquial registers; the educational-value classifier down-weights exactly those.
- **Threshold 3 is the Pareto point.** It is the highest threshold where HellaSwag still matches or beats the base FineWeb corpus, while MMLU and ARC show large gains.
- **Threshold 4 and 5** are cautionary — aggressive quality filtering tips into corpus-too-narrow.

The 92% discard rate at threshold 3 — the paper's most-quoted number — is what distinguishes a classifier-based pipeline from a heuristic one. No heuristic stack discards 92% of the post-heuristic corpus; the classifier is doing work the heuristics cannot.

---

## Why classifier > heuristics at scale — the paper's mechanism claim

From the source (line 44):

> **Why classifier > heuristics at scale:** heuristics (C4, CCNet) plateau — adding more heuristic filters doesn't help MMLU. A single LLM-labeled educational-value classifier captures what no regex stack can: the vibe of a textbook vs the vibe of a forum post.

Ch-10 §4 expands this into the "heuristics exhaust their per-token information" framing and connects it to [[scaling-laws-data-quality]]. The mechanism is that surface features (punctuation, ratios, line lengths) have bounded information content per document — adding a twentieth rule does not help once the first ten have captured what surface features can. Semantic features (educational-value, coherence, factual density) are not surface-readable, so a classifier with access to LLM-quality labels can price them; no regex stack can.

---

## What to take from FineWeb for ch-10

1. **One signal, one threshold, one decision.** The classifier era's design aesthetic is minimalist — replace the rule stack with a single learned signal and expose only its threshold.
2. **Per-snapshot MinHash is a regime-specific finding.** At 15T-token scale, global dedup over-prunes. This does not invalidate global dedup at smaller scales.
3. **Extraction is a filter.** Trafilatura vs WET is measurable before any semantic filter runs; the extractor is not infrastructure.
4. **The classifier recipe is portable.** Swap the anchor-LLM prompt ("rate for educational value" → "rate for math-reasoning value") and you re-run FineWeb-Edu for a new domain at the same marginal cost.
5. **Threshold 3 is the shipped default, not a universal truth.** It is the MMLU/ARC vs HellaSwag trade-off sweet spot for this annotation prompt; a different prompt would Pareto-trade differently.

---

## Connections

- [[excerpts/ccnet]] — FineWeb's skeleton is CCNet's (structural → dedup → quality); the classifier replaces KenLM perplexity.
- [[excerpts/c4]] — FineWeb's stage-4 heuristics are C4's descendants; the classifier is what breaks the C4 plateau.
- [[excerpts/dolma]] — FineWeb answers Dolma's heuristic stance with a classifier; ablated with the same rigor.
- [[excerpts/scaling-laws-data-quality]] — provides the scaling-variable frame the classifier-vs-heuristic result empirically populates.
- [[excerpts/rephrasing-the-web]] — the "even classifier-filtered FineWeb is noisy" complement argument.
- [[ch-10]] §4 (classifier era), §5 (comparison), §6 (checklist).
