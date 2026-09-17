---
chapter: ch-09
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/fineweb.md (card verified 2026-09-14)
source_url: https://arxiv.org/abs/2406.17557
primary_version: arXiv:2406.17557v2 (2024-10-31; v1 2024-06)
created_at: "2026-04-23"
revised_at: "2026-09-15"
---

# Excerpt: The FineWeb Datasets: Decanting the Web for the Finest Text Data at Scale

Values and quotations used by ch-09 `read.md`, matching the verified library card. Authors: Guilherme Penedo, Hynek Kydlíček, Loubna Ben allal, Anton Lozhkov, Margaret Mitchell, Colin Raffel, et al. (Hugging Face).

## Two datasets, two selection methods
- **FineWeb:** 15T GPT-2 tokens from 96 Common Crawl snapshots (§3, §3.7). Pipeline: WARC extraction with trafilatura → base filtering (URL blocklist, fastText English score ≥ 0.65, MassiveText quality and repetition filters) → per-snapshot MinHash deduplication → C4 filters except terminal punctuation → three custom heuristic filters → PII anonymization (§3.7). No quality classifier is applied to FineWeb.
- **FineWeb-Edu:** 1.3T tokens kept from FineWeb by an educational classifier: Llama-3-70B-Instruct scores 460,000 pages on an additive 0-5 scale (§4, App. F.1); a linear regression head on Snowflake-arctic-embed-m is trained on 410,000 annotations; pages with score ≥ 3 are kept (§4). Derived: 1.3T / 15T ≈ 8.7% of FineWeb tokens kept.

## Results (§4; 1.71B models, 350B tokens)
FineWeb-Edu raises MMLU from 33% to 37% and ARC from 46% to 57% over FineWeb. Threshold 3 was the trade-off between knowledge and reasoning benchmarks and "other benchmarks like HellaSwag" (§4).

## Topic shift from the classifier (§4.1, Fig. 18)
> "The educational classifier heavily favors topics such as 'Education, Learning, Teaching' or 'History, Culture, Politics', while down-sampling 'Business, Finance, Law', 'Entertainment, Film, Theater' and 'Places, Travel, Real Estate', among others."

Card values: "Education, Learning, Teaching" +3.2%; "History, Culture, Politics" +2.2%.

## Domain fit (§4.2, Fig. 12)
FineWeb has lower Paloma perplexity on broad web sources and on Twitter AAE, Manosphere, Gab, 100 Subreddits, and 4chan; FineWeb-Edu tends to be lower on Wikipedia sources, academic text, and 100 programming languages.

## Release and limits
Datasets under ODC-By, datatrove code, Llama 3 annotations, classifier, and ablation models are released (§1, App. C). Most experiments are at 1.71B scale; the data is web-only; the datasheet notes that code is likely not prevalent (§6, App. A).

## Correction to earlier ch-09 material
The April 2026 version of ch-09 described FineWeb as "100% web after classifier" and stated that "a single LLM-labeled quality classifier on CC beats every heuristic stack". FineWeb itself uses heuristic filters only; the classifier defines FineWeb-Edu, which is applied on top of FineWeb's heuristic pipeline.
