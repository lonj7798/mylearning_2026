<!-- scope: Dolma paper (Soldaini et al., ACL 2024; manuscript for Dolma v1.6): 3T-token English corpus from six sources, web pipeline order and removal rates, 1.2B-model data ablations, domain-fit, mixture and code-share experiments, and the Dolma toolkit
     deps: [[ccnet]], [[c4]]
     see-also: [[the-pile]], [[fineweb]], [[deduplicating-training-data]], [[minhash-lsh]], [[doremi]], [[olmo-2]]
-->

# Dolma: an Open Corpus of Three Trillion Tokens for Language Model Pretraining Research
- **Core Insight:** Dolma v1.6 contains 3,059B Llama tokens from Common Crawl, GitHub, Reddit, Semantic Scholar, Project Gutenberg, and Wikipedia/Wikibooks (Table 1); in 1.2B-parameter models trained on 150B tokens, Dolma and the Pile gave similar fit to diverse Paloma domains, while single-source web corpora (C4, mC4-en, RefinedWeb) gave higher average perplexity (§9.2, Figure 5).
- **Guideline:** When a pretraining corpus must cover many domains, add curated non-web sources to filtered web text, because at 1.2B scale single-source web corpora fit diverse domains worse (§9.2) and a web-only mix had higher perplexity on code and academic papers (App. M, Figure 12); when a specific domain is needed, a small in-domain share can be enough (4.9% vs 24.2% reference material gave nearly identical S2ORC perplexity, App. M).
- **Authors:** Luca Soldaini, Rodney Kinney, Akshita Bhagia, Dustin Schwenk, David Atkinson, Russell Authur, et al. (36 authors; Allen Institute for AI, UC Berkeley, CMU, Spiffy AI, MIT, University of Washington)
- **Year:** 2024 (arXiv v1 2024-01; ACL 2024)
- **URL:** https://arxiv.org/abs/2402.00159
- **Source type:** paper
- **Relevant topics:** pretraining data curation, web filtering, deduplication, PII and toxicity filtering, data ablations, source mixtures, decontamination

## Abstract
Commercial models rarely describe their pretraining data, and open models are often released without their data or a way to reproduce it, which makes research on how data affects capabilities difficult. The authors curate and release Dolma, a three-trillion-token English corpus built from web content, scientific papers, code, public-domain books, social media, and encyclopedic material. The paper documents design principles, construction, and contents, reports experiments on intermediate states of the corpus, and releases the curation toolkit (Abstract).

## Key Contributions
- The Dolma corpus: 3T tokens, 4,367M documents, 6 sources, curated from about 200 TB of raw text to 11 TB (§1, Table 1); released under ODC-By, toolkit under Apache 2.0 (Conclusion).
- The Dolma Toolkit, which unifies language, quality, and content filters as "filtering" and implements up/down-sampling, deduplication, and decontamination as a Rust "mixing" module with a Bloom filter (§4.1).
- A data-ablation protocol: 1.2B OLMo-family models trained to 150B tokens and evaluated zero-shot on 8 datasets chosen to avoid contamination (§4.2, App. D).
- Per-source pipelines with removal rates and a datasheet (§5–§8, App. N), and OLMo-1B trained on Dolma as a validation (§9.1).

## Key Figures/Tables to Study
- Table 1 (sources and sizes). Figures 1–3 (web quality, content, and stacked-filter ablations on HellaSwag; all tasks in App. O). Figure 4 (Reddit formats).
- Figure 5 (Paloma domain fit across corpora). Table 3 (code share). Table 4 and Figure 12 (mixtures). Figure 9 (filter correlations). Figure 11 (contamination).

## Technical Details
- **Sizes, Llama tokens** (Table 1): Common Crawl 2,479B; GitHub 411B; Reddit 89B; Semantic Scholar 70B; Project Gutenberg 6.0B; Wikipedia and Wikibooks 4.3B; total 3,059B. The text gives the web subset as 2.28T tokens (§5) and Reddit as 80B tokens (§7).
- **Web acquisition.** 25 Common Crawl snapshots from 2020-05 to 2023-06 (§5).
- **Web pipeline order** (§5.5): CCNet output → URL dedup → document dedup → quality filters → content filters → paragraph dedup.
- **CCNet stage.** FastText language ID keeps pages with English score ≥ 0.5 (removed 61.7% by bytes); within-snapshot duplicate-paragraph removal (about 70% of paragraphs, mostly headers and navigation); CCNet overall removes 84.2%, 175.1 TB → 27.7 TB (§5.1).
- **Deduplication** (Bloom filter, §5.4): exact URL removes 53.2% of documents; exact document removes 14.9% of URL-deduped documents; exact paragraph removes 18.7% of paragraphs. Stages (i)–(ii) run first to reduce later processing; paragraph dedup runs last because "paragraph removal risks disrupting content analysis" (§5.4). The datasheet gives 19.1% of UTF-8 characters for paragraph dedup (App. N.4).
- **Quality filters** (§5.2): all Gopher rules (15.23% of characters tagged) plus the single C4 rule removing paragraphs that do not end in punctuation, "C4 NoPunc" (22.73%). Gopher rules as listed (App. N.4): fewer than 50 or more than 100K words; median word length < 3 or > 10; symbol-to-word ratio > 0.10; fraction of words with an alphabetic character < 0.80; fewer than 2 of 8 required words; > 0.90 of lines starting with a bullet; > 0.30 of lines ending with an ellipsis; > 0.30 duplicated lines or characters in duplicated lines; most-common n-gram character fraction > 0.20 / 0.18 / 0.16 (2/3/4-grams); duplicate n-gram fraction > 0.15 to 0.10 (5- to 10-grams). Documents with a token sequence repeating over 100 times are also removed (0.003% of characters).
- **Toxicity** (§5.3, App. H): two FastText classifiers ("hate", "NSFW") trained on Jigsaw Toxic Comments score sentences (BlingFire splitter); sentences above the threshold are removed. τ = 0.4 ("High") removes 5.5–7.3% and τ = 0.0004 ("Low") removes 29.1–34.9%; Low generally performs better, High was adopted to meet the token target (§5.3, Figure 2). The datasheet reports 1.01% of data tagged at score > 0.4 (App. N.4).
- **PII** (§5.3, App. I): regular expressions for email addresses, IP addresses, phone numbers; documents with ≤ 5 spans get special-token replacement (0.02% of documents); documents with ≥ 6 spans are removed (0.001%); removal vs replacement had no measured effect in ablations.
- **Code** (§6): The Stack (deduplicated with MinHash and LSH by Allal et al., 2023), collected March 2023; data files such as JSON and CSV removed; RedPajama v1 rules (41.49% of data tagged, App. N.4) plus StarCoder rules, which together gave lower code perplexity and better evaluation-suite scores than RedPajama v1 rules alone (§6.2); documents with detect-secrets matches removed (§6.3).
- **Reddit** (§7): 378M posts, December 2005–March 2023, via Pushshift; comments < 500 characters, submissions < 400 characters, documents > 40,000 characters, and comments with fewer than 3 votes removed; 26,123 banned or NSFW subreddits excluded; PII documents removed rather than masked; document-level dedup.
- **Other sources** (§8): C4 re-run through the web pipeline except URL dedup; peS2o (about 40M papers) used as-is; English Project Gutenberg books deduplicated by exact title; Wikipedia and Wikibooks (March 2023) with documents of 25 or fewer words removed.
- **Decontamination** (App. L): for OLMo-1B, documents containing a paragraph longer than 13 tokens that appears in Paloma are removed (≤ 0.02% of documents). Dolma v1.6 as released is not decontaminated (App. N.4).
- **Filter overlap** (App. J, Figure 9): document-level Pearson correlations between filters are generally low; Gopher rules correlate negatively with dedup, most in the CCNet "Low" perplexity bucket (−0.36, 43M documents), because they remove random strings that dedup does not catch.

## Recipe ledger
| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| Dolma data-ablation model (OLMo architecture) | 1.2B (16 layers, 16 heads, d=2048) | pretrain | tokens | stopped at 150B; schedule set for 95k steps (about 200B tokens) | arXiv:2402.00159v2 §4.2; App. D.1 | verified 2026-09-14 | no ablation reported |
| Dolma data-ablation model | 1.2B | pretrain | context; tokenizer; position; activation | 2048 tokens; GPT-NeoX tokenizer; ALiBi; SwiGLU | App. D.1 | verified 2026-09-14 | no ablation reported |
| Dolma data-ablation model | 1.2B | pretrain | optimizer; LR; warmup; decay; weight decay | LionW; peak 1e-4; 2000 steps; cosine; 1e-2 | App. D.1 | verified 2026-09-14 | no ablation reported |
| Dolma data-ablation model | 1.2B | pretrain | batch; hardware | 1024 (micro-batch 8 on each of 128 compute units); 64 AMD MI250X | App. D.1 | verified 2026-09-14 (units derived: 8 × 128 = 1024 sequences) | no ablation reported |
| OLMo-1B (released) | 1.2B | pretrain | steps; tokens; batch | 739,328 steps (about 3.1T tokens); batch 2048 on 256 compute units | App. D.4 | verified 2026-09-14 | no ablation reported |
| OLMo-1B (released) | 1.2B | pretrain | optimizer | AdamW (switched from LionW) | App. D.4 | verified 2026-09-14 | "instabilities we found in the LionW optimizer" (App. D.4) |
| Dolma web content filter | n/a (data) | pretrain data | toxicity threshold τ | 0.4 | §5.3 | verified 2026-09-14 | Figure 2: τ = 0.0004 generally better; 0.4 chosen to keep token count |
| Mixture ablation models | 1B | pretrain | mixture, token share web / code / reference / books | Naïve 83.5 / 13.8 / 2.5 / 0.2%; Web Only 100 / 0 / 0 / 0%; Reference+ 81.2 / 13.5 / 4.9 / 0.4%; Gopher-like 68.4 / 5.4 / 24.2 / 2.0% | App. M, Table 4 | verified 2026-09-14 | Figure 12 (Paloma perplexity); no social data in these mixes |
| Code-share ablation models | 1B | pretrain | code share of C4 + Stack mixture | 0%, 5%, 15% | App. M, Table 3 | verified 2026-09-14 | Table 3, 5 seeds |

## Findings relevant to generality
- **Domain fit (Result, single study).** 1.2B models on 150B tokens: the Pile model fits diverse Paloma domains well despite the Pile's smaller size; Dolma and, to a lesser extent, RedPajama v1 give similar coverage; C4, mC4-en, and RefinedWeb give higher average perplexity. The authors conclude that non-web data from diverse curated sources matters (§9.2, Figure 5).
- **Mixtures.** All four mixes perform similarly on C4 100 domains; the Web Only mix has higher perplexity on HumanEval and on M2D2 S2ORC; Reference+ (4.9% reference) and Gopher-like (24.2%) are nearly identical on S2ORC. The authors state that source distribution is linked to downstream capabilities and that users should sample subsets according to their needs (App. M, Figure 12).
- **Code share** (Table 3, 5 seeds): bAbI ICL exact match 0.0 / 8.8 / 10.1 and WebNLG Rouge-2 16.8 / 19.3 / 22.0 at 0 / 5 / 15% code; GSM8K fine-tuned stays 0.0; GSM8K with program-aided fine-tuning 11.8 / 14.2 / 14.7 (App. M).
- **Filter ablations.** C4 NoPunc alone beats C4 All and Gopher All on perplexity and downstream tasks, and Gopher All + C4 NoPunc is best (§5.2, Figure 1). Stacking quality filters, dedup, and content filters has a compounding positive effect on HellaSwag (§5.5, Figure 3). The filter order itself is not ablated.
- **Formatting.** Treating Reddit comments and submissions as independent documents beats partial or full thread linearization (§7.1, Figure 4).
- **OLMo-1B** (Table 2): average over 8 tasks 60.3, vs TinyLlama 59.4, Pythia 54.5, StableLM2 66.5; better than TinyLlama on 4 of 8 tasks (§9.1).
- **Contamination** (App. L, Figure 11): WSC, SICK, GLUE AX, SemEval 2014 Task 1, SuperGLUE COPA, and SuperGLUE AXb are 100% contained in Dolma; HumanEval, WiC, e-SNLI, and SNLI are over 90%; many copies are in the code subset; these datasets are excluded from the paper's evaluations.
- **Classifier audits.** At English score ≥ 0.5, all ICE English documents from nine countries are kept (App. G); toxicity rates across location subreddits differ by < 5% at all thresholds (App. H).
- **Stated limits.** Ablations use one 1B-scale dense architecture and zero-shot tasks, so decisions may not hold at 7B–70B (Limitations). The authors state that Dolma v1.7 improves downstream results with the model held constant; this paper gives no numbers (Conclusion, footnote 15).

## Connections
- [[ccnet]] — Dolma's web pipeline starts from CCNet output (§5.1); CCNet's KenLM buckets (high 21.9%, medium 28.5%, low 49.6%) were not changed by Dolma's heuristic filters (§5.2).
- [[c4]] — C4 is a Dolma source re-run through the web pipeline (§8), and the C4 NoPunc rule is adopted (§5.2).
- [[the-pile]] — 387B tokens per §2; the multi-source baseline in the domain-fit comparison (§9.2).
- [[fineweb]] — listed as released while the paper was under review (§2); not compared.
- [[deduplicating-training-data]] — Lee et al. (2022), cited for token-efficiency gains from dedup (§5.4).
- [[minhash-lsh]] — the code subset inherits The Stack's MinHash + LSH near-dedup (§6.4).
- [[doremi]] — a method for setting domain weights; Dolma does not prescribe a mixture (App. M).
- [[olmo-2]] — later OLMo pretraining mix; see that card for its use of Dolma 1.7.
- Toolkit documentation (github.com/allenai/dolma, docs/getting-started.md on main, fetched 2026-09-14; separate artifact): `dolma tag` runs taggers, `dolma dedupe` writes duplicate-span attributes under `attributes/`, and `dolma mix` builds a dataset from a JSON or YAML config over compressed JSONL files. The paper describes only the "filtering" and "mixing" operations (§4.1).

## Verification
- Checked on 2026-09-14 against: https://arxiv.org/abs/2402.00159 (v2, 2024-06-06); secondary: github.com/allenai/dolma README.md and docs/getting-started.md (main).
- Corrections to the previous card version:
  - "Apply filters in the order URL → document-dedup → language → quality → content → paragraph-dedup" and "URL / document-level deduplication first; language identification second" → language ID (English ≥ 0.5) runs inside CCNet before URL dedup; order is CCNet → URL dedup → document dedup → quality → content → paragraph dedup (§5.1, §5.5).
  - "documents above threshold are dropped" (toxicity) → sentences above the threshold are removed (§5.3).
  - "Ablation-driven defense of the filter order: each stage is shown to improve downstream OLMo eval scores" → filter choices and thresholds are ablated and stacked filters show a compounding gain (Figures 1–3); the order is justified by efficiency and is not ablated (§5.4).
  - "Quality filters ... (line-length, symbol-to-word ratio, stopword ratio, fraction of lines ending in punctuation, duplicate-line fraction)" → exact Gopher rule list plus C4 NoPunc (§5.2, App. N.4).
  - "peS2o uses different quality filters than web — it trusts publication structure" → peS2o is used as-is apart from the repeated-sequence filter (§8, App. N.4).
  - "The Stack code uses near-dedup via MinHash on code tokens" → Dolma starts from the already-deduplicated Stack, deduplicated with MinHash and LSH by Allal et al. (§6.4).
  - "Social media (Reddit) is filtered by subreddit-level quality lists" → a blocklist of 26,123 banned or NSFW subreddits plus length and vote thresholds (§7.2).
  - "Luca Soldaini, ... and ~30 others" → 36 authors.
- Removed as unsupported by the source: `dolma-ngram` paragraph n-gram dedup with default T = 1.0; "every stage's config is published"; "the dolma CLI accepts YAML configs, runs filters as streaming passes over JSONL shards, and emits per-document attribute files ... keep/drop decision is a separate, cheap pass" (not in the paper; see the toolkit pointer); "is itself a scientific contribution, not just an engineering artifact"; "Direct successor to ccnet + c4 pipelines; same lineage, more transparency"; "[[fineweb]] uses a classifier-driven rather than heuristic-driven quality filter" (not compared in the paper).
- Not reported by the source: per-source sampling rates used to train OLMo-1B; OLMo-1B compute in GPU-hours; any ablation of filter order; model-based quality filtering results.
