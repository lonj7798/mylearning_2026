<!-- scope: WebOrganizer (Princeton / AI2, Feb 2025; ICML 2025) — 24 topic and 24 format domains for web pages via LLM annotations distilled into 140M classifiers, RegMix-predicted domain mixtures for MMLU and HellaSwag, composition with quality filters, and quality filters analyzed as implicit domain mixtures (1B DCLM 1b-1x runs)
     deps: [[regmix]], [[dclm]]
     see-also: [[fineweb]], [[dolma]], [[doremi]], [[data-mixing-laws]], [[paloma]], [[pretrainers-guide-training-data]]
-->

# Organize the Web: Constructing Domains Enhances Pre-Training Data Curation
- **Core Insight:** For 1B models trained on ~29B tokens selected from a 200B-token deduplicated Common Crawl pool, sampling by a predicted topic × format mixture raises the 9-task average from 51.6 to 54.6 (the FineWeb-Edu quality filter alone reaches 54.2), and applying the same mixture on top of FineWeb-Edu raises it to 56.2 (Table 1).
- **Guideline:** When pretraining data is selected with a document-level quality classifier, measure the topic and format mixture it implies and consider setting per-domain token quotas before selecting within each domain, because at 1B scale quotas added 2.0 points to FineWeb-Edu and 1.0 to DCLM-fasttext, and the implicit mixture with random within-domain sampling recovers 84% of FineWeb-Edu's average gain but only 35% of DCLM-fasttext's (Tables 1-2).
- **Authors:** Alexander Wettig, Kyle Lo, Sewon Min, Hannaneh Hajishirzi, Danqi Chen, Luca Soldaini (Princeton University; Allen Institute for AI; UC Berkeley; University of Washington)
- **Year:** 2025 (arXiv v1 2025-02; v3 2025-07; ICML 2025)
- **URL:** https://arxiv.org/abs/2502.10341
- **Source type:** paper
- **Relevant topics:** pretraining data curation, domain mixing, quality filtering, topic and format taxonomies, RegMix, LLM-to-classifier distillation, corpus documentation

## Abstract
Web-crawled pretraining corpora have no internal structure, which makes their contents hard to reason about. WebOrganizer builds two taxonomies of web pages, by topic and by format, and annotates a pretraining corpus by distilling annotations from a large language model into efficient classifiers. The domains are used to study how mixing data from different domains changes downstream task performance, and topic and format mixtures can be combined for further gains. Domain mixing also improves existing quality-based selection methods. The paper compares how quality filters implicitly change the domain mixture and concludes that constructing and mixing domains complements quality-based curation.

## Key Contributions
- Two taxonomies with 24 categories each: topic (subject matter) and format (style, intent, venue), drafted from curlie.org, Google Adsense, the Wikipedia ontology, and frequent URL domains, then refined by reviewing Llama-3.1-405B-Instruct annotations (§2.1, App. A).
- Topic and format classifiers: gte-base-en-v1.5 (140M parameters, 8192-token context) trained with a soft knowledge-distillation loss, first on 1M Llama-3.1-8B-Instruct annotations, then on 80K Llama-3.1-405B-Instruct annotations (§2.2, App. B).
- RegMix adapted to downstream targets: bits-per-byte of the correct answer under a 5-shot prompt for MMLU (train split) and HellaSwag (validation split) (§3, App. C).
- Topic × Format selection under an independence assumption, and domain quotas combined with FineWeb-Edu and DCLM-fasttext (§4.2-4.3).
- Quality filters as implicit domain mixers, measured by replacing within-domain quality selection with random sampling (§5, Table 2).
- Released code, classifiers, and the annotated corpus (§1).

## Key Figures/Tables to Study
- **Figure 1:** token share per topic and format in the 200B corpus. **Figure 3 / Table 9:** corpus vs predicted mixtures for MMLU, HellaSwag, and both.
- **Table 1:** main results on 9 tasks. **Table 2:** implicit mixtures of the two quality filters. **Figure 4:** implicit compositions vs corpus and RegMix.
- **Table 7:** classifier accuracy ablations. **Table 8:** RegMix held-out Spearman. **Table 10:** per-target results with DCLM Core and held-out perplexity.

## Technical Details
- Corpus (§4.1, App. E): DCLM 1b-1x pool, 1.64T raw tokens extracted with resiliparse; RefinedWeb heuristic filters without DCLM's high-quality URL filter; Bloom-filter deduplication from Dolma instead of MinHash; result 200B tokens, ~1B held out for validation. Each run selects 30B tokens, slightly above the 29B needed, because tokenization and packing drop tokens.
- URL domains are too granular for mixing: 18.5k URL domains have more than 1k documents and 14.7M have fewer in the 200B corpus (§2 Desiderata, Fig. 5).
- Redundancy (§2.3-2.4): NMI(T;F) ≈ 0.10 between topics and formats. 24 k-means clusters of gte embeddings give NMI(C;T) ≈ 0.46 vs NMI(C;F) ≈ 0.13; with 576 clusters these stay within ±0.03.
- Classifier training (App. B): soft labels from next-token probabilities normalized over category letters; category and example order randomized per annotation; 5 epochs per stage, batch 512 sequences, LR 1e-4 with 10% warmup and linear decay; input `{url}\n\n{text}`. On 10K validation pages where the 405B model is at least 75% confident (86% of topic, 79% of format annotations): topic 93.5 average / 87.1 worst-group, format 91.8 / 80.5; without 2-stage training 91.8 / 84.3 and 90.2 / 74.1; without URL 92.1 / 86.0 and 88.9 / 80.2 (Table 7). A second 405B seed agrees on 98% (topic) and 97% (format); the authors estimate the classifiers add 4.4%-5.1% error.
- RegMix adaptation (§3, App. C): 512 mixtures per domain set sampled from Dirichlet(αp) with corpus prior softened by τ = 2 and log α ~ Uniform(log 0.1, log 10); 50M models on 1B tokens each (~360 H100 hours for 512 runs); gradient-boosted tree regression; adaptive search with KL penalty γ·KL(p‖π), N = 0.5M, T = 15, γ = 0.002, η = 0.2, 2 seeds; upsampling capped at 6.5× so 30B of 200B tokens can be selected without repetition. Two-task targets average two regression models. Held-out Spearman between predictions and proxy runs is around 0.90 (Table 8); Data Mixing Laws gave worse correlation.
- Predicted mixtures (§3): MMLU upsamples Science & Tech., History, Health, Academic Writing, and Q&A Forums; HellaSwag upsamples Home & Hobbies, Fashion & Beauty, and Tutorials. Coding targets upsample the Software Engineering topic and Documentation format; Natural Questions is the only target whose mixture upsamples the Entertainment topic, which App. D describes as heavy (values in Fig. 8).
- Topic × Format (§4.2): P̃(topic, format) = P̃_T(topic) · P̃_F(format); if a pair lacks data, all its documents are taken and the remainder is upsampled.
- Quotas with quality filters (§4.3): the mixture sets tokens per domain; the top-scored documents are selected inside each domain, which varies the quality threshold per domain.
- Table 1 averages over MMLU, HellaSwag, PIQA, WinoGrande, CSQA, SIQA, ARC-e, ARC-c, OBQA (OLMES, 5-shot): baseline 51.6; + k-means clusters 53.2; + Topic 53.7; + Format 53.4; + Topic × Format 54.6 (gains on 8 of 9 tasks; SIQA −0.6); FineWeb-Edu 54.2 → 56.2 with Topic × Format; DCLM-fasttext 55.1 → 56.1. FineWeb-Edu's HellaSwag is 56.0 (1.5 below baseline) and 62.5 with quotas.
- Table 2 (implicit Topic × Format mixture, share of the filter's gain recovered): FineWeb-Edu 73% on MMLU and 84% on the task average; DCLM-fasttext 56% on MMLU and 35% on the average. Held-out perplexity on the baseline corpus: baseline 12.1, FineWeb-Edu 14.7, DCLM-fasttext 14.0, implicit mixtures 12.2-12.9.
- BoolQ is excluded because results were unreliable: random-sampling baseline 63.8% vs DCLM-fasttext 54.4% (App. E).

## Recipe ledger
| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| WebOrganizer DCLM 1b-1x runs (all curation variants) | 1,439,795,200 params | pretrain | tokens trained; batch; sequence length | 28,795,904,000 tokens; 256 sequences; 2048 tokens | arXiv:2502.10341v3 App. E | verified 2026-09-14 | DCLM 1b-1x reference setting; no ablation reported |
| same | 1.44B | pretrain | selection pool | 30B tokens per run from a 200B-token base corpus | §4.1, App. E | verified 2026-09-14 | no ablation reported |
| same | 1.44B | pretrain | mixture (main run) | RegMix Topic × Format mixture targeting MMLU + HellaSwag; per-domain weights in Table 9 (not transcribed here) | §4.2, App. D Table 9 | verified 2026-09-14 | Table 1: 54.6 vs 51.6 baseline average |
| same | 1.44B | pretrain | compute | 183 H100 hours per run (with torch.compile) | App. E | verified 2026-09-14 | — |
| same | 1.44B | pretrain | optimizer; LR schedule | not reported in this paper (uses the DCLM 1b-1x reference setting) | checked §4.1, App. E | not reported | — |
| RegMix proxy models | 50M | pretrain (proxy) | runs; tokens; compute | 512 runs per domain set; 1B tokens each; ~360 H100 hours for 512 runs | §3, App. C | verified 2026-09-14 | Table 8: Spearman ~0.90 on 50 held-out mixtures |
| RegMix proxy models | 50M | pretrain (proxy) | architecture | hidden 512, intermediate 1536, SwiGLU, 8 heads, 8 blocks, RoPE base 10000 | App. C Table 6 | verified 2026-09-14 | no ablation reported |
| RegMix proxy models | 50M | pretrain (proxy) | optimizer; LR; batch | Adam β (0.9, 0.95); peak LR 3e-3; cosine cooldown 3e-4; warmup 10%; batch size 128 | App. C Table 6 | verified 2026-09-14 | no ablation reported |
| Topic and format classifiers | 140M (gte-base-en-v1.5) | classifier distillation | annotations per stage | stage 1: 1M Llama-3.1-8B-Instruct; stage 2: 80K Llama-3.1-405B-Instruct (of 100K, 20K reserved) | §2.2, App. B | verified 2026-09-14 | Table 7: 2-stage raises worst-group accuracy 84.3 → 87.1 (topic), 74.1 → 80.5 (format) |
| Topic and format classifiers | 140M | classifier distillation | epochs; batch; LR | 5 epochs per stage; 512 sequences; 1e-4, 10% warmup, linear decay | App. B | verified 2026-09-14 | no ablation reported |

## Findings relevant to generality and distillation
- **Target choice changes the mixture (Result, single study).** MMLU and HellaSwag call for different mixtures, and optimizing both requires trade-offs; the authors conclude that "data quality" depends on the chosen downstream tasks (§3, §5).
- **Transfer and narrowing.** The two-task Format mixture improves 6 of 7 non-target tasks and the Topic mixture transfers to ARC and OBQA (§4.4). A mixture tuned only for MMLU (Topic × Format) reaches MMLU 33.2 but lowers HellaSwag from 57.5 to 54.1 and PIQA from 71.3 to 69.9 (Table 10).
- **Quality filters reshape domains.** FineWeb-Edu shifts topics more than DCLM-fasttext, which amplifies more formats and keeps the most Entertainment, Games, Comment Section, and Creative Writing data; both amplify Politics, Health, Science & Tech., History, Knowledge Articles, Tutorials, Academic Writing, and Q&A Forums (§5, Fig. 4). Higher held-out perplexity after quality filtering (14.7, 14.0 vs 12.1) is read by the authors as a larger distribution shift than domain rebalancing (§5; Interpretation).
- **Limits stated by the authors.** Mixture predictions rest on few small runs and are sensitive to noise; transfer across model scales is uncertain; taxonomies are one of many valid choices and single-label (Impact Statement, §7). Some DCLM Core tasks are near random at 1b-1x, and the DCLM baseline numbers were not reproduced exactly (App. E).
- **Distilled annotators.** The 140M classifiers imitate 405B labels with a measured gap (Table 7); App. B also calls them "150M parameter", which conflicts with the 140M stated elsewhere.

## Connections
- [[regmix]] — mixture-regression method adapted here with downstream targets and adaptive KL-penalized search.
- [[dclm]] — provides the 1b-1x pool, training setting, and the DCLM-fasttext filter.
- [[fineweb]] — source of the FineWeb-Edu classifier compared in Tables 1-2.
- [[olmes]] — evaluation suite for all 9 tasks.
- [[dolma]] — Bloom-filter deduplication used to build the 200B corpus.
- [[doremi]], [[aioli-data-mixing]] — mixture-optimization methods cited in §3 and §6.
- [[data-mixing-laws]] — tried as an alternative predictor; lower Spearman (App. C).
- [[qurating]] — earlier LLM-rating quality selection by the first author, cited in §6.
- [[perplexity-correlations-data-selection]] — cited: ~10k URL domains ranked by perplexity-benchmark correlation across 90 models (§6).
- [[pretrainers-guide-training-data]] — cited for curation effects of toxicity, source composition, and data age (§6).
- [[paloma]] — per-domain perplexity evaluation from the same institute.

## Verification
- Created on 2026-09-14 from https://arxiv.org/abs/2502.10341 (v3, 2025-07-16; PDF read including App. B, C, E and Tables 7-10).
- Audit claims not found in the source: none. The audit's unconfirmed "about 35%" DCLM-fasttext recovery is confirmed as the task-average column of Table 2 (MMLU column: 56%); the "up to 84%" figure is the FineWeb-Edu task average.
- Not reported by the source: optimizer and LR schedule of the 1B runs (deferred to DCLM); numeric per-domain token shares were not transcribed from Table 9.
