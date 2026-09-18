<!-- scope: FineWeb2 (arXiv 2506.20920): a FineWeb-derived pre-training data pipeline that sets LID, filter, and upsampling thresholds per language from statistics; 20 TB, 1,868 language-script pairs; early-signal multilingual task selection
     deps: [[fineweb]]
     see-also: [[deduplicating-training-data]], [[datadecide]], [[uberweb-multilingual-curation]], [[tokenizer-language-unfairness]], [[atlas-multilingual-scaling-laws]]
-->

# FineWeb2: One Pipeline to Scale Them All — Adapting Pre-Training Data Processing to Every Language
- **Core Insight:** A pipeline that derives language-identification, filtering, and upsampling thresholds from each language's own data statistics produced models that beat prior multilingual datasets on 11 of 14 evaluated languages, including 5 languages not used to design it (§5, Fig. 3); in six canary languages at 350B tokens the full pipeline raised the aggregate score over LID-only data, for example Arabic 21.7 → 25.2 and French 18.3 → 23.6 (App. A.8, Tables 29-30).
- **Guideline:** When curating web data for a language other than English, compute LID and heuristic-filter thresholds from that language's distributions instead of reusing English values, and test the pipeline on languages held out from its design, because English thresholds ranked below adapted methods or removed too much data (Table 25) and the pipeline still lost to HPLT2 on German and Indonesian (Tables 40-41) and to some hand-built single-language datasets (§5).
- **Authors:** Guilherme Penedo, Hynek Kydlíček, Vinko Sabolčec, Bettina Messmer, Negar Foroutan, Amir Hossein Kargaran, et al. (Hugging Face, EPFL)
- **Year:** 2025 (arXiv v1 2025-06; preprint, under review)
- **URL:** https://arxiv.org/abs/2506.20920
- **Source type:** paper
- **Relevant topics:** multilingual pre-training data, language identification, per-language filter thresholds, MinHash deduplication, duplication-aware upsampling (rehydration), evaluation task selection, low-resource language precision

## Abstract
Open English pre-training datasets have improved, but multilingual data remains hard because filtering and deduplication must be tailored to many languages. The paper introduces a pipeline based on FineWeb that adapts automatically to any language. Design choices are ablated on nine languages using evaluation tasks selected by measurable criteria. Corpora from the pipeline train better models than prior datasets. A rebalancing method that uses duplication counts and quality signals adds a further gain. The pipeline is applied to almost 100 Common Crawl snapshots to produce FineWeb2: 20 terabytes, 5 billion documents, over 1000 languages. Pipeline, training, and evaluation code are released.

## Key Contributions
- A language-adaptive version of the FineWeb pipeline: GlotLID with per-language confidence thresholds, global per-language MinHash, per-language heuristic-filter thresholds, and wordlist precision filtering for low-resource languages (§4.2-4.4).
- A task-selection procedure for "early-signal" evaluation (monotonicity, signal-to-noise, non-random performance, ordering consistency), yielding 84 of 197 tested tasks across nine languages (§3.3, App. A.5.1).
- Rehydration: upsampling weights per MinHash cluster size set from the filtering removal rate of that cluster size (§4.5).
- Validation on five unseen languages and comparison with multilingual and single-language datasets (§5, App. A.9-A.10).
- Release of FineWeb2 (ODC-By), the pre-filtering version, and code (§1, §5).

## Key Figures/Tables to Study
- Fig. 1 and Tables 29-34: aggregate score after each pipeline step, six languages, 350B tokens.
- Fig. 2: removal rate by MinHash cluster size for French (U shape) and the resulting weights.
- Fig. 3, Fig. 7, Tables 40-44: comparison with CC-100, mC4, CulturaX, HPLT, HPLT2, raw Common Crawl, and language-specific datasets.
- Table 15: LID threshold formula versus best-performing threshold range. Table 25: filter-threshold method ranks.
- Table 28: precision and recall of wordlist filtering in a native-speaker audit.

## Technical Details
- **Starting data (§4.1).** URL blocklist and trafilatura extraction as in FineWeb. About 40% of documents passed FineWeb's English fastText threshold of 0.65; FineWeb2 starts from the remaining 60%.
- **Experimental design (§3).** Nine canary languages: Arabic, Chinese, French, Hindi, Russian, Swahili, Telugu, Thai, Turkish (Table 1). One monolingual model per language per ablation, to avoid cross-language confounders (§3).
- **Tokenizer (§3.1, App. A.3).** Gemma tokenizer (vocabulary 256,000), chosen by subword fertility on Wikipedia; average fertility 2.10 tokens per word versus 2.16 for Bloom (Table 2). Mistral-v3, Phi3, Llama3, and Command-R had Telugu fertility of 9.74-10.11 (Table 2). Tokenizers with vocabulary over 256,000 were excluded (App. A.3).
- **Aggregate score (§3.3).** Per task: (score − random baseline)/(1 − random baseline), with scores below baseline set to 0. Scores are averaged within the categories RC, GK, NLU, and CR, then across categories.
- **Task criteria (App. A.5.1).** Monotonicity: mean Spearman ρ between step and score ≥ 0.5. SNR from four seed models ≥ 20, relaxed for generative tasks. Non-randomness (max gain over baseline / final-step std) ≥ 3. Ordering consistency (Kendall τ-a) observed but not thresholded. Cloze formulation replaces multiple-choice letters, which score at random early in training (App. A.5.2).
- **LID (§4.2, App. A.6).** GlotLID V3 (1880 languages) replaces FT176. Without thresholds, GlotLID scored higher on higher-resource canary languages and FT176 slightly higher on lower-resource ones (Fig. 5). Threshold = max{0.3, min{0.9, Med(X) − σ(X)}}, where X is the language's confidence-score distribution (§4.2). Swahili performed best near 0.3, removing almost 65% (§4.2). The formula fell outside the best range for Chinese (0.7415 vs 0.895-0.937) and Hindi (0.6827 vs 0.483-0.557) (Table 15). The text says "median" (§4.2); the Table 15 section says "mean" (App. A.6.2).
- **Deduplication (§4.3).** MinHash with FineWeb's parameters (14 buckets of 8 hashes, 5-grams over language-specific word tokens), global per language, run before filtering; cluster size is stored in metadata. The effect varies by language with no relation to resource level; it lowered the aggregate for French (18.3 → 18.0) and Thai (11.1 → 11.0) (Tables 30, 32).
- **Stopwords (§4.4.1, App. A.7.1).** Stopwords are words above a frequency threshold in cleaned Wikipedia; documents need ≥ 2. Uncleaned low-resource Wikipedias put English words in stopword lists; for Dagbani, recomputed stopwords removed > 99% of misclassified data (App. A.7.1).
- **Filter thresholds (§4.4.2, App. A.7.2).** Five adaptation methods (English, MeanStd, Quantile, 10Tail, MedianRatio) on Common Crawl or Wikipedia statistics; 207 ablation models at 29B tokens. Selected: 10Tail on Wikipedia for FineWeb Quality (rank 3.00), Quantile on Wikipedia for Gopher Quality (3.22), MeanStd on Common Crawl for Gopher Repetition (2.22); no-filter baseline ranks 7.00, 6.33, 6.22 (Table 25).
- **Precision filtering (§4.4.3, App. A.7.3).** Wordlists of words with affinity ≥ γ = 0.85; about a third of 1,900 languages had contamination above 10%; a URL whitelist restores removed in-language pages. Audit of 2,000 documents per language: glk_Arab precision 2.10% → 27.21% (recall 95.24%); bar_Latn 69.45% → 94.90% (97.77%); ary_Arab 1.75% → 4.14% (88.57%) (Table 28).
- **Rehydration (§4.5, Fig. 2).** Weight 10 for the cluster size with the lowest removal rate, weight 1 for every cluster size above the global removal rate (62.4% for French), linear interpolation between. Singletons and the most-duplicated clusters had above-global removal rates in most languages checked.
- **Dataset (§5, App. A.11-A.13).** 96 snapshots (summer 2013 to April 2024); 20 TB; 1,868 language-script pairs, 1,226 with over 100 documents (§5). English is excluded (§5). Russian is 28.0% and Mandarin 11.6% of UTF-8 bytes (Fig. 8). Test split: min{1%, 100k} documents per language before filtering (App. A.13).

## Recipe ledger
All rows: FineWeb2 data-ablation models, Llama architecture, nanotron framework (§3.1). Stage for all training rows: pretrain-stable unless noted.

| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| FineWeb2 ablation model | 1.46B | pretrain-stable | Architecture | 14 layers; 32 heads; 32 KV heads; d_model 2048; tied embeddings; embedding size 256008; RMSNorm ε 1e-05; init std 0.02; Gemma tokenizer | arXiv:2506.20920v1 App. A.4 Table 3 | verified 2026-09-14 | tokenizer by fertility comparison (Table 2); layers reduced for larger vocabulary (§3.1) |
| FineWeb2 ablation model | 1.46B | pretrain-stable | Optimizer (all scales) | Adam β1 0.9, β2 0.95, ε 1.0e-8; weight decay 0.1; grad clip 1.0 | App. A.4 Table 5 | verified 2026-09-14 | no ablation reported |
| 29BT runs | 1.46B | pretrain-stable | Seq length; batch; LR schedule | 2048; 1024 sequences = 2,097,152 tokens; dp 64; peak 3e-4; 14000 steps; 500 linear warmup; cosine decay over 13500 steps to 3.0e-5 | App. A.4 Tables 4-5 | verified 2026-09-14 | no ablation reported |
| 100BT runs | 1.46B | pretrain-decay/anneal | Seq length; batch; LR schedule | 2048; 840 sequences = 1,720,320 tokens; dp 56; peak 8e-4; 59000 steps; 2950 (5%) linear warmup; constant, then linear decay over last 11800 (20%) steps from step 47200 to 0 | App. A.4 Tables 4-5 | verified 2026-09-14 | batch from critical-batch-size rule of DeepSeek-AI et al. 2024; decay placement so checkpoints can be continued (Tables 4-5 captions) |
| 350BT runs | 1.46B | pretrain-decay/anneal | Seq length; batch; LR schedule | 2048; 1280 sequences = 2,621,440 tokens; dp 64; peak 7e-4; 134000 steps; 6700 (5%) linear warmup; linear decay over last 26800 (20%) steps from step 107200 to 0 | App. A.4 Tables 4-5 | verified 2026-09-14 | same as 100BT row |
| All scales | 1.46B | pretrain-stable | Tokens implied by steps × batch | 29.36B; 101.50B; 351.27B | Tables 4-5 | derived (steps × tokens per batch) | — |
| Canary-language dataset comparison | 1.46B | eval-gate | Training tokens | 29 billion (§5) vs "30 billion" (Fig. 7 caption) | §5; App. A.9 Fig. 7 | conflict | not resolved in the paper |
| Other ablations | 1.46B | eval-gate | Training tokens | filter methods 29B (§4.4.2); LID thresholds 30B (App. A.6.2); per-step and dedup runs 350B (§4.3, Fig. 1); unseen languages 100B (§5) | as listed | verified 2026-09-14 | — |

Not reported (checked body and App. A.1-A.13): number of seeds for dataset-comparison runs, compute or GPU hours, mixed precision, data mixture for multilingual training (all runs are monolingual).

## Findings relevant to generality
- **Held-out languages (§5, Tables 40-44).** Aggregate at 100B tokens, FineWeb2 / best other: German 17.0 / 17.1 (HPLT2); Indonesian 20.3 / 22.4 (HPLT2); Italian 18.6 / 17.8 (HPLT2); Japanese 21.8 / 18.0 (CulturaX); Vietnamese 21.1 / 19.7 (HPLT2). The authors state trends on canary and unseen languages agree (§5) (Result, single study).
- **Language-expert pipelines.** In some languages a hand-designed single-language dataset outperforms FineWeb2 (§5, Fig. 3).
- **Measurement.** Generative tasks are noisier; multilingual models can score well on multiple-choice tasks but answer in the wrong language on generative tasks ("accidental translation") (App. A.5.1).
- **Low-resource content diversity.** 70% (1,320) of 1,868 language-script pairs have more than half their documents from Bible- or Wikipedia-related domains, mostly Bible (§5, App. A.12).
- **Limits (§6).** Only a small share of languages tested; short ablation runs; task properties measured early in training may change later; no measurement of cultural alignment, bias, or diversity.

## Connections
- [[fineweb]] — the English pipeline, MinHash parameters, and ablation protocol that FineWeb2 adapts (§4.1, §4.3).
- [[ccnet]], [[c4]] — CC-100 and mC4 are comparison baselines with fixed cross-language pipelines (§3.2).
- [[deduplicating-training-data]] — cited as the standard dedup practice that rehydration departs from (§4.5).
- [[datadecide]], [[benchmark-variance-quantified]] — related work on choosing data with small-model runs and on benchmark noise.
- [[tokenizer-language-unfairness]] — related evidence on tokenizer fertility differences across languages (App. A.3).
- [[uberweb-multilingual-curation]], [[multilinguality-curse-250-languages]], [[atlas-multilingual-scaling-laws]] — later or related multilingual data and scaling studies.
- [[whose-language-counts-quality-filter]] — related critique of quality filters applied across language varieties.
- [[dclm]] — English data-curation benchmark in the same line of data-ablation work.

## Verification
- Created on 2026-09-14 from https://arxiv.org/abs/2506.20920 (arXiv v1, 26 Jun 2025; latest version served on that date). Body and App. A.1-A.13 read.
- Audit claims not found in the source: none. The audit summary is consistent with the abstract; the phrase "how filtering and dedup thresholds must change per language" is this card's reading of §4.2-4.4, and deduplication itself uses one parameter set for all languages (§4.3).
