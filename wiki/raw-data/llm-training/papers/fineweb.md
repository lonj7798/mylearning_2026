<!-- scope: FineWeb (15T-token English Common Crawl corpus) and FineWeb-Edu (1.3T-token educational subset): extraction, filtering, dedup ablations, classifier filtering
     deps: [[ccnet]], [[c4]]
     see-also: [[dolma]], [[deduplicating-training-data]], [[phi-3]], [[llama-3]]
-->

# The FineWeb Datasets: Decanting the Web for the Finest Text Data at Scale
- **Core Insight:** For a 1.71B-parameter model trained on 350B tokens, FineWeb-Edu (1.3T tokens kept from FineWeb by an LLM-annotated educational classifier at score ≥ 3) raises MMLU from 33% to 37% and ARC from 46% to 57% over FineWeb (§4), and deduplicating each Common Crawl snapshot separately outperformed global deduplication (§3.4, Fig. 5).
- **Guideline:** When filtering web data with a quality classifier, choose the threshold with ablations that include non-knowledge benchmarks, because the FineWeb-Edu threshold of 3 was the trade-off between knowledge/reasoning benchmarks and benchmarks such as HellaSwag (§4), and the filtered set shifts topics and domain fit toward education and Wikipedia-like text (§4.1-4.2).
- **Authors:** Guilherme Penedo, Hynek Kydlíček, Loubna Ben allal, Anton Lozhkov, Margaret Mitchell, Colin Raffel, et al. (Hugging Face)
- **Year:** 2024 (arXiv v1 2024-06, v2 2024-10; NeurIPS 2024 Datasets and Benchmarks Track)
- **URL:** https://arxiv.org/abs/2406.17557
- **Source type:** paper
- **Relevant topics:** pretraining data curation, text extraction, heuristic filter selection, MinHash deduplication, classifier-based filtering, FineWeb-Edu

## Abstract
Pretraining data for leading open LLMs such as Llama 3 and Mixtral is not released and is poorly documented. The paper introduces FineWeb, a 15-trillion-token dataset from 96 Common Crawl snapshots that yields better LLMs than other open pretraining datasets. It documents and ablates every design choice, with detailed studies of deduplication and filtering. It also introduces FineWeb-Edu, a 1.3-trillion-token subset of educational text; models trained on it perform better on knowledge- and reasoning-intensive benchmarks such as MMLU and ARC. The datasets, the curation codebase (datatrove), and all ablation models are released.

## Key Contributions
- FineWeb: 15T GPT-2 tokens from 96 snapshots, built by a sequence of data ablations (§3, §3.7).
- A procedure for deriving heuristic filters from statistics that separate a high- and a low-quality version of the same crawl (§3.6).
- Deduplication study: global vs per-snapshot MinHash, and lighter global methods (§3.4, App. E).
- FineWeb-Edu and its classifier, trained on Llama-3-70B-Instruct annotations (§4); topic, domain-fit, and bias analyses (§4.1, §4.2, §5).
- Release of datasets (ODC-By), datatrove, Llama 3 annotations, classifier, and ablation models (§1, App. C).

## Key Figures/Tables to Study
- Fig. 3-5: global vs per-snapshot MinHash. Fig. 9: gain from each step. Fig. 10: comparison with 10 other open datasets at 350B tokens.
- Fig. 11: MMLU vs tokens. Fig. 17: FineWeb-Edu thresholds 2, 3, 4. Fig. 12 and Fig. 18: domain fit and topic shift.

## Technical Details
- **Ablation protocol (§3.1).** Models identical except data; two runs per data version with different data subsets and seeds, scores averaged. Filtering ablations use ~28B tokens; some dedup and cumulative runs use 350B. Benchmarks: CommonSense QA, HellaSwag, OpenBook QA, PIQA, SIQA, WinoGrande, ARC, MMLU (large benchmarks truncated to 1,000 samples), chosen for low seed variance, near-monotonic improvement, and above-random scores at this scale.
- **Extraction (§3.2, Fig. 1).** trafilatura on WARC files outperforms Common Crawl WET text in a 28B-token ablation.
- **Base filtering (§3.3).** URL blocklist; fastText English score ≥ 0.65; MassiveText quality and repetition filters with original thresholds. Result over 96 snapshots: ~36T tokens.
- **MinHash (§3.4, App. E.1).** 5-grams; 112 hash functions in 14 buckets of 8; documents match if all 8 hashes agree in any bucket; transitive clusters; one random document kept per cluster. Match probability P(s) = 1 − (1 − s⁸)¹⁴, where s is the n-gram similarity of two documents. P = 56%, 77%, 92%, 98.8% at s = 0.7, 0.75, 0.8, 0.85 (App. E.1).
- **Global dedup (§3.4, Fig. 3-4).** Iterating from the newest snapshot (2023-50) to the oldest removed up to 90% of old snapshots and left 4T tokens. A 350B-token run improved little over non-deduplicated data. In snapshot 2013-48, the ~31B kept tokens trained a worse model than 171B tokens obtained by deduplicating the ~460B removed tokens on their own.
- **Per-snapshot dedup (§3.4, Fig. 5).** Deduplicating each snapshot independently gave 20T tokens and matched RefinedWeb. Lighter global methods applied afterwards (URL dedup, 71.5% of tokens removed; line dedup, 77.8%; line dedup with min words, 85%; 3-line dedup, 80.9%) all performed worse (App. E.3, Fig. 15).
- **C4 filters (§3.5, Fig. 6).** On the 2019-18 crawl, C4's terminal-punctuation filter gave the largest single gain but removed ~30% of tokens. All other C4 filters together removed ~7% and scored higher than terminal punctuation alone; these were adopted.
- **Custom filters (§3.6).** From over 50 statistics, 16 candidate thresholds were tested at 28B tokens; three were kept: fraction of lines ending in punctuation ≤ 0.12 (10.14% of tokens removed), fraction of characters in duplicated lines ≥ 0.1 (12.47%), fraction of lines shorter than 30 characters ≥ 0.67 (3.73%). Together: ~22% removed, aggregate score about +1%.
- **Final pipeline (§3.7).** WARC extraction → base filtering → per-snapshot MinHash → C4 filters → custom filters; PII step anonymizes email and public IP addresses. The App. A datasheet lists the same steps with MinHash after the custom filters.
- **FineWeb-Edu annotation (§4, App. F.1).** Llama-3-70B-Instruct scores 460,000 pages from CC-MAIN-2024-10 on an additive 0-5 scale; the prompt focuses on grade-school and middle-school knowledge to avoid favoring highly technical pages such as arXiv abstracts.
- **Classifier (§4).** Linear regression head on Snowflake-arctic-embed-m, embedding and encoder frozen; trained on 410,000 annotations for 20 epochs at learning rate 3e-4; checkpoint with best F1 on the remaining 50,000; outputs rounded to integers 0-5. Threshold 3 gives binary F1 82%. Applying the classifier to all 15T tokens took 6,000 H100 GPU hours.
- **Results (§4, Fig. 10-11, Fig. 17).** Threshold 3 had the best aggregate among FW-Edu-2/3/4 at 28B tokens (App. F.2). At 350B tokens FineWeb-Edu outperforms all compared open datasets on the aggregate. FineWeb-Edu reaches 33.6% MMLU at 38B tokens; Matrix reaches similar accuracy at 300B (Fig. 11). Derived: 1.3T / 15T ≈ 8.7% of FineWeb tokens are kept.

## Recipe ledger
| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| FineWeb data-ablation model | 1.71B | pretrain-stable | Architecture | Llama; 24 layers; 32 heads; 32 KV heads; tied embeddings; vocab 50,257 (GPT-2 tokenizer); init std 0.02 | arXiv:2406.17557v2 App. D.1 | verified 2026-09-14 | no ablation reported |
| FineWeb data-ablation model | 1.71B | pretrain-stable | Sequence length; global batch | 2,048; ~2 million tokens | §3.1; App. D.2 | verified 2026-09-14 | no ablation reported |
| FineWeb data-ablation model | 1.71B | pretrain-stable | Global batch in sequences | 1,024 sequences = 64 (dp) × 4 (micro-batch) × 4 (accumulation); × 2,048 = 2,097,152 tokens | App. D.2 | derived | — |
| FineWeb data-ablation model | 1.71B | pretrain-stable | Optimizer | Adam β1 0.9, β2 0.95, ε 1e-8; weight decay 0.1; grad clip 1.0 | App. D.3 | verified 2026-09-14 | no ablation reported |
| FineWeb data-ablation model | 1.71B | pretrain-stable | LR schedule | peak 3e-4; 500 linear warmup steps; cosine decay to 3.0e-5 | App. D.3 | verified 2026-09-14 | no ablation reported |
| FineWeb data-ablation model | 1.71B | pretrain-stable | Tokens per run | ~28B (filtering ablations); 350B (some dedup runs, cumulative checks, dataset comparisons) | §3.1; §3.7 | verified 2026-09-14 | 28B described as roughly Chinchilla-optimal for 1.71B (§3.1) |
| FineWeb data-ablation model | 1.71B | eval-gate | Seeds per data version | 2 (different data subset and initialization), averaged | §3.1 | verified 2026-09-14 | no ablation reported |
| FineWeb ablations (all) | 1.71B | pretrain-stable | Compute | over 70 models; estimated 80,000 H100 GPU hours | §3.1 | verified 2026-09-14 | — |
| FineWeb-Edu classifier | not reported | data filter (no §5.2 stage) | Annotations; epochs; LR | 410,000 train / 50,000 validation; 20; 3e-4 | arXiv:2406.17557v2 §4 | conflict | checkpoint selected by best validation F1 (§4) |
| FineWeb-Edu classifier | not reported | data filter (no §5.2 stage) | Annotations; epochs; LR | 450,000 training samples; hold-out 46,867; 20; 3e-4 | huggingface.co/HuggingFaceFW/fineweb-edu-classifier README (read 2026-09-14) | conflict | model card; which count produced the released classifier is not stated |
| FineWeb-Edu | — | data filter (no §5.2 stage) | Score threshold | ≥ 3 | §4; App. F.2 Fig. 17 | verified 2026-09-14 | FW-Edu-2/3/4 at 28B tokens, 3 best aggregate; trade-off vs HellaSwag-type benchmarks (§4) |

Not reported (checked body, App. A-G): total steps, dropout, mixed precision, shard size or file layout of the release, a comparison with DCLM, classifier results at thresholds other than 2-4.

## Findings relevant to generality
- **Threshold trade-off.** Threshold 3 balanced knowledge/reasoning benchmarks against "other benchmarks like HellaSwag" (§4).
- **Topic shift (§4.1, Fig. 18).** Clusters over 50k + 50k embedded samples: FineWeb-Edu gains "Education, Learning, Teaching" (+3.2%) and "History, Culture, Politics" (+2.2%) and down-samples "Business, Finance, Law", "Entertainment, Film, Theater", and "Places, Travel, Real Estate".
- **Domain fit (§4.2, Fig. 12).** Paloma perplexity without decontamination: FineWeb is lower on broad web sources (C4, mC4, Falcon, Dolma V1.5, RedPajama CommonCrawl) and on Twitter AAE, Manosphere, Gab, 100 Subreddits, and 4chan. FineWeb-Edu tends to be lower on Wikipedia sources (WikiText-103, M2D2 Wikipedia), academic text (M2D2 S2ORC, RedPajama ArXiv), and 100 programming languages.
- **Deduplication.** The authors hypothesize that most of the dedup gain comes from removing very large duplicate clusters, and that further removing small clusters (under ~100 copies) can hurt (Interpretation, §3.4). Dedup effects are hard to see in small samples: in a simulation with 100 identical 200B-token snapshots, a 1B-token sample is almost all unique (App. E.2, Fig. 14).
- **Limits (§6).** Most experiments are at 1.71B scale; evaluation uses academic benchmarks without instruction tuning; the data is web-only. The datasheet notes that code is likely not prevalent (App. A).

## Connections
- [[c4]] — its heuristic filters are ablated; all except terminal punctuation are adopted (§3.5).
- [[ccnet]] — pipeline behind CC-100 and RedPajama, both compared in Fig. 10 (§2).
- [[dolma]] — Dolma 1.6 and 1.7 Common Crawl subsets are comparison baselines (Fig. 10).
- [[deduplicating-training-data]] — cited for the harm of duplicated pretraining data (§2, §3.4).
- [[llama-3]], [[phi-3]] — cited as non-public datasets that used educational-content classifiers (§4).
- [[data-constrained-scaling]] — cited for pretraining with one or a few passes over the data (§2).

## Verification
- Checked on 2026-09-14 against: https://arxiv.org/abs/2406.17557 (arXiv v2, 31 Oct 2024; v1 PDF also read for annotation counts, which match v2).
- Corrections: title "FineWeb: Decanting the Web…" → "The FineWeb Datasets: Decanting the Web for the Finest Text Data at Scale". "450K samples; hold-out 46,867" → paper: 460,000 annotated, 410,000 train, 50,000 validation (§4); 450K/46,867 are from the classifier model card. "Small classifier head" → linear regression on Snowflake-arctic-embed-m (§4). "Threshold sweep 1 through 5 on MMLU/ARC" → FW-Edu-2/3/4 on aggregate accuracy (Fig. 17). "Global dedup hurt because it removed high-quality documents that reappear once per snapshot" → global dedup gave little gain; data it kept in 2013-48 was worse than data it removed (§3.4, Fig. 3-4). "PII: email, phone" → email and public IP addresses (§3.7). "Trafilatura higher quality than CCNet's extractor" → compared with WET files (§3.2). "Pipeline: Gopher + C4 heuristics, then per-snapshot MinHash, then PII" → base filtering (URL, language, MassiveText) → per-snapshot MinHash → C4 filters → custom filters → PII (§3.7). "Single classifier beats hand-crafted recipes; don't stack heuristics" → FineWeb-Edu is a classifier applied on top of FineWeb's heuristic pipeline (§3.7, §4).
- Removed as unsupported: "removes ~92% of FineWeb" (replaced by derived 8.7% kept); "largest fully-open web corpus at release"; "heuristics plateau on MMLU"; "the vibe of a textbook vs a forum post"; "mirrored by Nemotron"; "challenges Dolma's heuristic-first stance"; "motivates DoReMi-style reweighting"; "the per-dump finding is a genuinely new result".
