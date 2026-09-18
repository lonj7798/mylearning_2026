<!-- scope: The Pile — an 825 GiB 22-source English pretraining corpus, with size-controlled evidence that a diverse mixture lowers cross-domain BPB
     deps: [[c4]]
     see-also: [[dolma]], [[ccnet]], [[data-constrained-scaling]]
-->

# The Pile: An 800GB Dataset of Diverse Text for Language Modeling
- **Core Insight:** In a size-controlled comparison at 1.3B parameters and ~40GB of training data, a model trained on the Pile reached 0.9433 bits per UTF-8 byte (BPB) on the Pile test set against 1.3293 for CC-100 (en) and 1.1275 for Raw CC, and was best on all 22 component test sets (§4.2, Table 3; Table 4).
- **Guideline:** When a model must perform on academic, code, legal, and mathematical text, add curated domain sources to the pretraining mixture rather than scaling crawl alone, because the size-controlled comparison in this paper attributes the largest BPB gains to exactly those components (Table 4).
- **Authors:** Leo Gao, Stella Biderman, Sid Black, Laurence Golding, Travis Hoppe, Charles Foster, et al. (EleutherAI)
- **Year:** 2020 (arXiv v1 2020-12)
- **URL:** https://arxiv.org/abs/2101.00027
- **Source type:** paper
- **Relevant topics:** corpus mixture design, source diversity, domain balance, BPB evaluation, dataset documentation

## Abstract
The Pile is an 825.18 GiB English text corpus assembled from 22 sub-datasets, 14 of which the paper introduces. Many components come from academic or professional sources (PubMed Central, ArXiv, GitHub, FreeLaw, Stack Exchange, USPTO, PhilPapers, NIH ExPorter). The authors evaluate GPT-2 and GPT-3 zero-shot on Pile components and find both struggle on several of them, and they train 1.3B-parameter models from scratch on size-controlled subsets of the Pile, CC-100 (en), and raw Common Crawl. The Pile-trained model improves on every Pile component and on WikiText, with little change on LAMBADA. The paper also documents profanity, bias, and consent properties of the constituent datasets.

## Key Contributions
- An 825.18 GiB English corpus combining 22 sources, with published construction code (§1.1).
- 14 new language-modeling datasets, including Pile-CC (jusText extraction from WARC files rather than WET files), OpenWebText2, BookCorpus2, and DM Mathematics-adjacent sources (§1.1, §2.1).
- A size-controlled evaluation showing broad cross-domain BPB improvement of Pile-trained 1.3B models over CC-100 and raw CC models (§4).
- Bits-per-UTF-8-byte (BPB) as the reporting metric, which makes results comparable across tokenizers (§3).
- An exploratory documentation section covering profanity, gender/religion/demographic bias, and consent status of each component (§6, Tables 5, 10-13).

## Key Figures/Tables to Study
- **Table 1** — component raw size, weight, epochs, effective size, mean document size.
- **Table 2** — per-component BPB for GPT-2 (4 sizes) and GPT-3 (ada→davinci).
- **Table 3** — size-controlled comparison: Pile vs CC-100 (en) vs Raw CC on Pile BPB, WikiText PPL, LAMBADA PPL/ACC.
- **Table 4** — per-component BPB for models trained on each of the three corpora.
- **Figure 4** — magnitude of BPB improvement of the Pile model over the CC-100 model per test set.

## Technical Details
- Total size 825.18 GiB raw; 1254.20 GiB effective after up-weighting (Table 1).
- 22 components. Largest by weight: Pile-CC 18.11%, PubMed Central 14.40%, Books3 12.07%, OpenWebText2 10.01%, ArXiv 8.96%, GitHub 7.59% (Table 1).
- Up-weighting is expressed as epochs within one pass over the Pile: Wikipedia (en) 3.0, Gutenberg (PG-19) 2.5, PubMed Central / OpenWebText2 / ArXiv 2.0, Books3 / FreeLaw / OpenSubtitles 1.5, Pile-CC / GitHub 1.0 (Table 1). The appendix states the authors upweighted to at most 3 epochs and avoided more than 2 epochs for most components (App. D.1).
- Mixture weight is a share of bytes in the final dataset, not a sampling temperature (Table 1 caption).
- Pile-CC uses jusText on Web Archive (WARC) files rather than the pre-extracted WET plaintext, which the authors state yields higher-quality output (§2.1).
- Evaluation metric is bits per UTF-8 encoded byte, BPB = (L_T / L_B) · ℓ / ln 2, where L_T is dataset length in tokens, L_B its length in UTF-8 bytes, and ℓ the mean per-token cross-entropy (§3).
- Size-controlled training setup: architecturally identical 1.3B-parameter models following Brown et al. (2020); each corpus deduplicated against the evaluation sets with 13-gram overlap filtering and subsampled to approximately 40GB (§4.1).
- Size-controlled results (Table 3): Pile 0.9281 val / 0.9433 test BPB, WikiText 5.59 PPL, LAMBADA 12.78 PPL / 50.1 ACC; CC-100 (en) 1.3143 / 1.3293, 8.27, 11.78 / 49.7; Raw CC 1.1180 / 1.1275, 11.75, 19.84 / 43.8.
- Largest per-component BPB gaps of the Pile model over the CC-100 model (Table 4): DM Mathematics 1.5206 vs 3.1774, ArXiv 0.7945 vs 1.8159, EuroParl 1.1202 vs 2.7141, GitHub 0.5597 vs 1.6509, Stack Exchange 0.8152 vs 1.5414. On Pile-CC the gap is small (0.9989 vs 1.0873).
- Raw CC beats CC-100 on Pile BPB (1.1275 vs 1.3293) while losing on WikiText and LAMBADA. The authors attribute this to CC-100's perplexity-based filtering against a Wikipedia-trained model, which discards data too dissimilar from Wikipedia and reduces diversity (§4.2, Interpretation by the authors).
- Language composition estimated with fastText: 97.4% English, with the authors noting language identification is unreliable for rare languages (§5).

## Findings relevant to generality
- Cross-domain generality is measured here as per-component BPB on a 22-domain held-out set, alongside two conventional benchmarks (WikiText, LAMBADA). The Pile model wins all 22 component rows and WikiText while being roughly flat on LAMBADA (12.78 vs 11.78 PPL for CC-100), which the authors read as increased cross-domain coverage without loss on traditional benchmarks (§4.2, Tables 3-4).
- The comparison is size-controlled at ~40GB, which the authors note is generous to CC-100 since the real CC-100 (en) is about one third the size of the Pile (§4.1).
- Limits: one model size (1.3B), one seed, no downstream task suite beyond WikiText and LAMBADA, and no ablation isolating individual components or mixture weights. The paper does not measure how diversity scales with model size.
- Measurement caution the paper raises itself: the GPT-2-Pile-to-GPT-3 comparison in Figure 3 is a proxy, could be confounded by dataset-specific scaling effects, and produced results the authors call puzzling (§3, Interpretation).

## Connections
- [[c4]] and [[ccnet]] are the crawl-centric baselines the Pile is compared against; CC-100 is built with the CCNet pipeline.
- [[dolma]] and [[fineweb]] continue the documented-open-mixture line with larger scale and per-step ablations.
- [[data-constrained-scaling]] addresses how many epochs a component may be repeated, which the Pile sets by hand (Table 1).

## Verification
- Checked on 2026-09-18 against: https://arxiv.org/abs/2101.00027 (arXiv v1, 31 Dec 2020)
- Corrections to the previous card version:
  - "Diversity of source domains is itself a scaling variable" → replaced with the paper's measured result: 1.3B models trained on size-controlled Pile reach 0.9433 test BPB vs 1.3293 (CC-100) and 1.1275 (Raw CC), and win all 22 component rows (§4.2, Tables 3-4). The paper makes no scaling-law claim about diversity.
  - "825 GiB ... 22 diverse high-quality subsets" → exact figure is 825.18 GiB raw, 1254.20 GiB effective (Table 1).
  - "22 component datasets with manually chosen mixture weights" → kept, and the actual weights and epoch counts added from Table 1.
  - Author list trimmed to the first six plus "et al." per the card standard; the full list is Gao, Biderman, Black, Golding, Hoppe, Foster, Phang, He, Thite, Nabeshima, Presser, Leahy.
  - Year line now gives the arXiv v1 month (2020-12).
  - Added missing **Source type** field.
- Removed as unsupported by the source:
  - "broad, high-quality mixtures outperform monolithic web corpora on cross-domain generalization" as an unquantified general claim — replaced by the size-controlled numbers above.
  - "Made explicit mixture design a first-class pretraining decision" and "became a baseline for open LMs" — historical claims about the field, not results in this paper.
  - "Showed benefits of curated-domain coverage beyond raw crawl scale" as phrased — the paper's comparison is size-controlled at ~40GB, so it does not test scale of crawl.
  - "helped push later data documentation standards" — an attribution about later work, not a claim of this paper.
- Not reported by the source: downstream task accuracy beyond LAMBADA; per-component ablations of mixture weights; training compute or GPU-hours for the 1.3B runs; results at more than one model size.
