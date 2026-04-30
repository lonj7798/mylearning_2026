---
chapter: ch-48
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/fineweb.md
source_url: https://arxiv.org/abs/2406.17557
created_at: "2026-04-23"
---

# Excerpt: FineWeb — per-snapshot vs global dedup has contamination-detection consequences

**Source library:** `wiki/raw-data/llm-training/papers/fineweb.md`
**Artifact:** 15T-token corpus, per-snapshot MinHash dedup surprise result, filter cascade details.

---

## Why FineWeb matters for ch-48

Ch-48 §2's detection table treats MinHash as a single row. FineWeb complicates that picture by showing that the *granularity* at which MinHash is applied (global vs per-snapshot) affects both downstream performance and, by extension, contamination recall. A decontamination pipeline that picks the wrong granularity misses cross-snapshot leakage.

---

## The per-snapshot dedup surprise

Source §Key Figures/Tables to Study:

> **Per-dump MinHash vs global dedup** comparison — HF found per-dump outperforms naive global dedup on downstream tasks (surprising; tied to removing near-identical re-crawls).

For pretraining quality, per-snapshot wins. For contamination detection against a *pinned eval set*, the logic is different: global matching is strictly necessary — an eval sentence appearing in snapshot A and again in snapshot B is a two-hit contamination that per-snapshot dedup would leave intact because each snapshot independently looks clean.

Ch-48 §3's pseudocode uses global matching (single LSH index across the corpus stream) for this reason. If a team follows FineWeb's per-snapshot dedup policy, they must still run a *separate* global decontamination pass — the two objectives are different.

---

## FineWeb-Edu's classifier as a contamination risk

Source §Technical Details / FineWeb-Edu classifier:

> Trained on **450K web samples** annotated by **Llama-3-70B-Instruct** with integer scores 0–5 … Filter keeps documents with score ≥ 3.

The classifier is trained on LLM-labeled data. If Llama 3's training corpus contained benchmark questions, the classifier has inherited a latent bias toward rating benchmark-adjacent content as "educational." This is a second-order contamination pathway: the classifier-driven quality filter systematically *upweights* documents that resemble benchmarks.

Ch-48 memo §7 "what the memo does NOT claim" should explicitly disclaim: "no audit of classifier-based quality filter for benchmark-proximity bias."

---

## The filter cascade ordering — where decontamination fits

Source §Technical Details / Pipeline (FineWeb base):

> 1. URL filter (blocklist)
> 2. Trafilatura for HTML-to-text extraction
> 3. fastText language ID → English only.
> 4. Quality heuristics (symbol ratios, line length, etc.).
> 5. MinHash deduplication per snapshot
> 6. PII redaction

Decontamination is not an explicit step in this cascade. In a defensible workflow it slots in as step 5b or 6b — after MinHash dedup, before downstream consumption. The ch-48 §3 pseudocode assumes this position: the stream being decontaminated has already passed quality filters and dedup, so every remaining flag is signal, not artefact.

---

## Scale implications for contamination cost

Source §Abstract:

> we introduce FineWeb, a 15-trillion token dataset derived from 96 Common Crawl snapshots

At 15T tokens, a decontamination pass costs roughly the same as a dedup pass — dominated by the streaming hash computation. This is important because it removes the "we can't afford to decontaminate" excuse. If a team is running MinHash dedup at this scale, adding eval-set hash lookup adds <10% to the pipeline cost (Bloom filter lookups are O(1)) while catching per-instance overlap that dedup cannot.

---

## What ch-48 takes from FineWeb

| FineWeb practice | Ch-48 contamination implication |
|---|---|
| Per-snapshot dedup for quality | Contamination needs *global* matching on top |
| Classifier-based quality filter | Second-order bias toward benchmark-adjacent text |
| Trafilatura HTML extraction | Text form determines n-gram hashability; normalise consistently |
| 15T-scale streaming pass | Decontamination at scale is a <10% overhead, no cost excuse |

---

## Connections

- **[[deduplicating-training-data]]** — the MinHash primitive FineWeb uses.
- **[[dolma]]** — alternative filter cascade; explicit vs FineWeb's classifier-driven.
- **[[olmo-3]]** — builds on both Dolma and FineWeb-era practice with an explicit per-stage decontamination utility.
- **[[scaling-laws-data-quality]]** — the quality-as-scaling-variable lens under which contamination inflates the apparent scaling curve.
