<!-- scope: 15T-token Common Crawl–derived pretraining corpus with classifier-based quality filtering
     deps: [[ccnet]], [[c4]]
     see-also: [[dolma]], [[deduplicating-training-data]]
-->

# FineWeb: Decanting the Web for the Finest Text Data at Scale
- **Core Insight:** A single, classifier-scored quality filter (educational value, annotated by an LLM) on deduplicated Common Crawl beats every prior hand-crafted web recipe on knowledge-and-reasoning benchmarks.
- **Guideline:** For web data at 2024+ scale, invest effort in a small LLM-labeled classifier tuned to your downstream benchmark family (e.g., educational content for MMLU), then apply a single score threshold — don't stack dozens of heuristics.
- **Authors:** Guilherme Penedo, Hynek Kydlíček, Loubna Ben Allal, Anton Lozhkov, Margaret Mitchell, Colin Raffel, Leandro Von Werra, Thomas Wolf
- **Year:** 2024
- **URL:** https://arxiv.org/abs/2406.17557
- **Relevant topics:** pretraining data curation, classifier-based filtering, deduplication ablations, FineWeb-Edu

## Abstract
The performance of a large language model (LLM) depends heavily on the quality and size of its pretraining dataset. However, the pretraining datasets for state-of-the-art open LLMs like Llama 3 and Mixtral are not publicly available and very little is known about how they were created. In this work, we introduce FineWeb, a 15-trillion token dataset derived from 96 Common Crawl snapshots that produces better-performing LLMs than other open pretraining datasets. To advance the understanding of how best to curate high-quality pretraining datasets, we carefully document and ablate all of the design choices used in FineWeb, including in-depth investigations of deduplication and filtering strategies. In addition, we introduce FineWeb-Edu, a 1.3-trillion token collection of educational text filtered from FineWeb. LLMs pretrained on FineWeb-Edu exhibit dramatically better performance on knowledge- and reasoning-intensive benchmarks like MMLU and ARC. Along with our datasets, we publicly release our data curation codebase and all of the models trained during our ablation experiments.

## Key Contributions
- **FineWeb**: 15T tokens from 96 Common Crawl dumps — largest fully-open web corpus at release.
- **FineWeb-Edu**: 1.3T-token subset filtered by an educational-quality classifier; outperforms FineWeb on MMLU/ARC/OpenBookQA.
- Full ablation study of dedup strategies (global vs per-snapshot MinHash) and filter order.
- Public release of classifier, codebase (`datatrove`), and all ablation checkpoints.

## Key Figures/Tables to Study
- **Ablation curves** showing downstream accuracy vs filter aggressiveness — the classic quality-vs-quantity tradeoff.
- **Per-dump MinHash vs global dedup** comparison — HF found per-dump outperforms naive global dedup on downstream tasks (surprising; tied to removing near-identical re-crawls).
- **FineWeb-Edu threshold sweep** — accuracy on MMLU/ARC vs classifier-score threshold 1 through 5.

## Technical Details
**Pipeline (FineWeb base):**
1. **URL filter** (blocklist) on Common Crawl WARC files.
2. **Trafilatura** for HTML-to-text extraction (higher-quality than CCNet's extractor).
3. **fastText language ID** → English only.
4. **Quality heuristics** adapted from Gopher + C4 (symbol ratios, line length, etc.).
5. **MinHash deduplication per snapshot** (not globally). Global dedup hurt downstream because it removed documents that only re-appear once per snapshot but are high-quality.
6. **PII redaction** (email, phone).

**FineWeb-Edu classifier:**
- Trained on **450K web samples** annotated by **Llama-3-70B-Instruct** with integer scores 0–5, where 0 = not educational, 5 = highly educational.
- Small classifier head on top of a frozen embedding model.
- Filter keeps documents with score ≥ 3. This threshold **removes ~92% of FineWeb**, leaving 1.3T tokens.
- On a hold-out of 46,867 Llama-3-annotated samples, the score≥3 binary classifier reaches F1 = 82%.
- Threshold 3 was chosen for best trade-off: higher thresholds gain MMLU but lose HellaSwag.

**Why classifier > heuristics at scale:** heuristics (C4, CCNet) plateau — adding more heuristic filters doesn't help MMLU. A single LLM-labeled educational-value classifier captures what no regex stack can: the vibe of a textbook vs the vibe of a forum post.

## Connections
- Successor in spirit to [[c4]] and [[ccnet]]; challenges [[dolma]]'s heuristic-first stance.
- FineWeb-Edu recipe is mirrored by Nemotron and inspired the data mixes of later open releases.
- Classifier-based filtering motivates interest in [[doremi]]-style data reweighting as a complementary tool.
- Dedup ablations extend the findings of [[deduplicating-training-data]] — the per-dump finding is a genuinely new result.
