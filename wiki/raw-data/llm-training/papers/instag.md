<!-- scope: open-set instruction tagging used to quantify SFT-data diversity and complexity, and to select a 6K subset
     see-also: [[instag-diversity]], [[deita]], [[lima]], [[evol-instruct]], [[prismatic-synthesis]]
-->

# #InsTag: Instruction Tagging for Analyzing Supervised Fine-tuning of Large Language Models
- **Core Insight:** Tagging SFT queries with an open-set tagger produces 6,398 normalized tags over a 306,044-sample pool, and a 6K subset selected for maximum tag coverage (100%) and highest average tag count (16.56) trains a 13B model to 6.44 on MT-Bench, above open-source 13B baselines fine-tuned on 52K–125K samples (§4.1, §4.2, Table 3).
- **Guideline:** When selecting an SFT subset from a large pool, tag each query with an open-set intention tagger and select by complexity-first diverse sampling, because in this paper's controlled sweeps MT-Bench increases with both average tag count and tag coverage rate at a fixed 6K subset size (§4.3, Figure 4). When the pool is not tagged, this selection signal is unavailable.
- **Authors:** Keming Lu, Hongyi Yuan, Zheng Yuan, Runji Lin, Junyang Lin, Chuanqi Tan, Chang Zhou, Jingren Zhou
- **Year:** 2023 (arXiv v1 2023-08; v2 2023-08-15)
- **URL:** https://arxiv.org/abs/2308.07074
- **Source type:** paper
- **Relevant topics:** instruction tagging, SFT data diversity, SFT data complexity, data selection, MT-Bench

## Abstract
The paper argues that diversity and complexity are treated as critical properties of SFT datasets without quantitative definitions. It proposes InsTag, an open-set fine-grained tagger that labels each query with intention and semantic tags using ChatGPT, followed by a normalization procedure. Diversity is defined as the unique-tag coverage rate of a dataset over the full tag set, and complexity as the average number of tags per query. The authors tag widely used open-source SFT datasets, report that model ability grows with more diverse and complex data, and use the tagger as a selector to draw 6K samples. The resulting models, TagLM, score higher on MT-Bench than open-source models trained on considerably larger SFT sets. Code is released at https://github.com/OFA-Sys/InsTag.

## Key Contributions
- Defines instruction **diversity** as unique-tag coverage rate and **complexity** as average tag count per query, both computed from open-set tags (§3.4).
- Builds an open-set ChatGPT tagging prompt (no predefined ontology) plus a four-step normalization pipeline: frequency filtering, rule aggregation, semantic aggregation, association aggregation (§3.2).
- Evaluates tag quality with GPT-4 and human annotators: 96.1% tag precision and 86.6% tag consistency under GPT-4 annotation (§3.3, Table 2).
- Proposes **complexity-first diverse sampling** (Algorithm 1) and trains TagLM-13b-v1.0 / v2.0 on the selected 6K subset (§4.1–4.2).
- Runs decoupled sweeps that vary complexity at fixed diversity and diversity at fixed complexity, both at a fixed 6K subset size (§4.3, Figure 4).
- Distills the tagger into InsTagger, a 7B LLaMA-2 model, for local tagging (App. §I).

## Key Figures/Tables to Study
- **Table 2** — tagging precision and consistency, with human/GPT-4 agreement (Fleiss κ and Cohen κ).
- **Figure 2a** — open-source SFT datasets plotted as diversity (tag coverage rate) against complexity (average tag count), colored by AlpacaEval score.
- **Table 3** — MT-Bench main results with SFT data size per model.
- **Figure 4a / 4b** — MT-Bench against average tag count, and against tag coverage rate, each with a random-sampling baseline.

## Technical Details
- Raw ChatGPT annotation produces over 12 thousand tags in the abstract's framing and over 100 thousand original unique tags across the full dataset collection (§3.1; §3.2).
- Normalization chain with measured counts: frequency filtering at α = 20 retains 8,541 tags; rule aggregation (lowercasing, special-character replacement, NLTK stemming) reduces to 7,157; semantic aggregation with PhraseBERT embeddings and DBSCAN at minimum semantic similarity 0.05 reduces to 6,587; association aggregation with FP-Growth at minimum support 40 and minimum confidence 99% yields 1,772 association rules that merge tag groups into atomic tags (§3.2).
- The abstract states the resulting tag set as 6.6K tags (Abstract); §3.2 gives 6,587 after semantic aggregation.
- Tag-quality evaluation: 4,000 sampled cases for GPT-4 (2,000 precision, 2,000 consistency); three human annotators label 40 cases (1%) by majority vote, reaching 100% on both metrics. Fleiss κ between humans is 0.47 (precision) and 0.73 (consistency); Cohen κ between human majority vote and GPT-4 is 0.92 and 0.75 (§3.3, Table 2).
- Experiment pool: WizardLM(Alpaca), WizardLM(ShareGPT), UltraChat, and ShareGPT pooled into 306,044 samples with tag-set size 6,398 and average tag number 4.48 (§4.1).
- Selected subset: 6K samples with average tag number 16.56 and tag coverage 100%, drawn by Algorithm 1 (§4.1).
- MT-Bench results, average of three GPT-4 judgments: TagLM-13b-v1.0 (LLaMA, 6K) 6.44±0.04; TagLM-13b-v2.0 (LLaMA-2, 6K) 6.55±0.02; vicuna-13b-v1.3 (125K) 6.39; wizardlm-13b (70K) 6.35; vicuna-13b-v1.1 (70K) 6.31; baize-v2-13b (56K) 5.75; openchat-13b-v1 (8K) 5.22; alpaca-13b (52K) 4.53; Llama-2-13b-chat 6.65; gpt-4 8.99 (Table 3).
- Complexity sweep: 10 subsets of 6K samples each, all at 100% tag coverage, with average tag numbers from 6.7 to 16.6. MT-Bench increases with average tag number at a coarse level; the authors state the trend is not significant between subsets with small differences in tag count. All 10 subsets beat a random 6K baseline whose average tag number is about 4.5 (§4.3, Figure 4a).
- Diversity sweep: subsets of 6K samples at fixed average tag number 5.0 and varying tag coverage. MT-Bench increases with coverage, with a plateau the authors report between 50% and 90% coverage. A random subset at 71.9% coverage performs similarly to the constructed 70%-coverage subset (§4.3, Figure 4b).
- InsTagger distillation: 7B LLaMA-2 fine-tuned on 773,511 tagging samples (1,000 held out for validation), batch size 512, 1 epoch, because more than 1 epoch overfit. Tag-level F1 is 31.8% under exact match and 73.4% under PhraseBERT fuzzy match at 0.8 cosine similarity (App. §I).

## Recipe ledger

| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| TagLM-13b-v1.0 (LLaMA) / v2.0 (LLaMA-2) | 13B | SFT | examples | 6K selected samples | arXiv:2308.07074 §4.1 | verified 2026-09-18 | §4.3 Figure 4a/4b: MT-Bench rises with tag count and coverage at fixed 6K size |
| TagLM-13b-v1.0 / v2.0 | 13B | SFT | epochs | 5 | arXiv:2308.07074 §4.1 | verified 2026-09-18 | no ablation reported |
| TagLM-13b-v1.0 / v2.0 | 13B | SFT | batch size (sequences) | 128 | arXiv:2308.07074 §4.1 | verified 2026-09-18 | no ablation reported |
| TagLM-13b-v1.0 / v2.0 | 13B | SFT | learning rate | 2 × 10⁻⁵ | arXiv:2308.07074 §4.1 | verified 2026-09-18 | no ablation reported |
| TagLM-13b-v1.0 / v2.0 | 13B | SFT | prompt template | Vicuna-style | arXiv:2308.07074 §4.1 | verified 2026-09-18 | no ablation reported |
| InsTagger | 7B (LLaMA-2) | distill-SFT | examples | 773,511 (1,000 held out) | arXiv:2308.07074 App. §I | verified 2026-09-18 | no ablation reported |
| InsTagger | 7B (LLaMA-2) | distill-SFT | batch size / epochs | 512 / 1 epoch | arXiv:2308.07074 App. §I | verified 2026-09-18 | App. §I: "training for more than 1 epoch will lead to over-fitting" (no numbers given) |
| TagLM-13b-v1.0 / v2.0 | 13B | SFT | LR schedule, warmup, optimizer, sequence length | not reported | checked body §4.1, App. §C–§I | not reported | — |

## Findings relevant to generality
- The paper measures breadth only through MT-Bench (eight task categories) and, for the dataset survey in Figure 2a, through AlpacaEval scores collected from the official leaderboard rather than run by the authors (§3.4, Figure 2 caption).
- Per-category MT-Bench results: TagLM-13b-v1.0 is highest among the compared baselines on stem and extraction, second on math, coding, and writing, and lower on roleplay (§4.2, Figure 3).
- Limits stated by the authors: the fine-grained complexity trend is weak, which they attribute to ChatGPT not recalling all applicable tags and to tags removed during normalization (§4.3). The diversity plateau between 50% and 90% coverage is attributed to tags carrying unequal semantic distance (§4.3).

## Connections
- [[instag-diversity]] — a second library card covering the same arXiv entry (2308.07074); treat this card as the primary extract.
- [[deita]] — uses "Instag Complexity", "Instag Diversity", and TagLM as baselines (Deita Tables 2, 4, 5).
- [[lima]] — the small-high-quality-SFT result this paper cites as motivation (§2).
- [[evol-instruct]] — WizardLM data is two of the four pooled datasets used here (§4.1).
- [[prismatic-synthesis]] — later diversity-based selection that cites tag-metadata selection as prior work.

## Verification
- Checked on 2026-09-18 against: https://arxiv.org/abs/2308.07074 (arXiv v2, 2023-08-15)
- Corrections to the previous card version:
  - "Tag vocabulary spans thousands of semantic and intent categories" → 6,587 tags after semantic aggregation, 6.6K in the abstract, and 6,398 in the experimental pool (§3.2, Abstract, §4.1).
  - "Showed strong results from small selected subsets" with no number → TagLM-13b-v1.0 scores 6.44 on MT-Bench from 6K samples versus vicuna-13b-v1.3 at 6.39 from 125K (Table 3).
  - Card title lacked the leading "#" that the paper prints as part of its title on page 1; title corrected.
  - Added source type, key figures, technical loci, and a recipe ledger, none of which the previous version had.
- Removed as unsupported by the source: "Diversity and complexity in SFT data become measurable if you first tag instructions with a large open-ended tag vocabulary" as a bare claim without the counts; "Analyze open SFT datasets through tag coverage and tag complexity" and "Use the tagger as a selector to build stronger small training subsets" as content-free restatements of the method.
- Not reported by the source: SFT learning-rate schedule, warmup, optimizer, sequence length, and hardware; any evaluation outside MT-Bench and the borrowed AlpacaEval numbers; any result at a scale other than 13B.
