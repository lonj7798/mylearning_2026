<!-- scope: Liu et al. (ICLR 2024) controlled study of complexity, quality, and diversity metrics for SFT data selection, and the DEITA score-first, diversity-aware selector with its 6K/10K SFT models
     deps: [[evol-instruct]]
     see-also: [[cherry-llm]], [[ifd]], [[instag]], [[instag-diversity]], [[alpagasus]], [[lima]], [[less]], [[superfiltering]], [[prismatic-synthesis]]
-->

# What Makes Good Data for Alignment? A Comprehensive Study of Automatic Data Selection in Instruction Tuning
- **Core Insight:** Mistral-7B fine-tuned on 6K samples selected by DEITA from a 300K pool scores 7.22 on MT-Bench and 80.78% on AlpacaEval, compared with 5.89 / 56.90% for 10K randomly selected samples and 5.32 / 75.12% for the 200K-sample zephyr-beta-sft checkpoint (Table 6).
- **Guideline:** When a fixed SFT budget must be selected from a large, mixed-quality pool, rank samples by the product of a trained complexity score and a trained quality score and admit a sample only if its embedding cosine similarity to every already selected sample is below a threshold (0.9 in the paper), because on LLaMA-1-13B this selector at 6K samples beat random (6K), Alpagasus (6K), TAGLM (6K), and LIMA (1K) subsets on MT-Bench and AlpacaEval (Table 5); its evidence comes mainly from GPT-4-judged MT-Bench, so confirm on non-judge evaluations before relying on it.
- **Authors:** Wei Liu, Weihao Zeng, Keqing He, Yong Jiang, Junxian He (ShanghaiTech University, BUPT, Meituan, Alibaba Group, HKUST)
- **Year:** 2023 (arXiv v1 2023-12; ICLR 2024)
- **URL:** https://arxiv.org/abs/2312.15685
- **Source type:** paper (with official code and data release at github.com/hkust-nlp/deita)
- **Relevant topics:** SFT data selection, instruction-tuning data, complexity scoring, quality scoring, embedding diversity filter, data efficiency

## Abstract
The paper studies which properties make instruction-tuning data effective for alignment. It runs controlled selection studies along three dimensions, complexity, quality, and diversity, compares existing metrics with new ones, and proposes a simple selection strategy that combines them. DEITA (Data-Efficient Instruction Tuning for Alignment) is a series of models fine-tuned from LLaMA and Mistral on automatically selected data. With 6K SFT samples, over 10x fewer than the baselines, DEITA performs better than or on par with open-source alignment models of the time. DEITA-Mistral-7B with 6K SFT samples plus 10K DPO pairs reaches 7.55 MT-Bench and 90.06% AlpacaEval. The models and selected datasets are released.

## Key Contributions
- Controlled single-dimension studies of complexity, quality, and diversity metrics at a fixed 6K budget on two pools (§2, Tables 2-4).
- EVOL COMPLEXITY and EVOL QUALITY: evolve one seed sample into variants, have ChatGPT rank and score all variants in one prompt, and train a scorer on those scores (§2.3, §2.4).
- REPR FILTER: an iterative embedding-distance filter for diversity (§2.5).
- Algorithm 1, score-first diversity-aware selection, and the DEITA 6K/10K models and datasets (§3.1, Tables 6-7).

## Key Figures/Tables to Study
- Figure 1: the three scoring pipelines. Algorithm 1: the selection loop.
- Tables 2-4: MT-Bench for each complexity, quality, and diversity metric on X_sota and X_base.
- Table 6 (MT-Bench, AlpacaEval) and Table 7 (Open LLM Leaderboard) for DEITA vs other SFT models.
- Figure 2: data scaling from 1K to all 300K. Appendix C.1, Figure 4: threshold and encoder sensitivity.

## Technical Details
- **Pools:** X_sota = WizardLM (Alpaca), WizardLM (ShareGPT), UltraChat, and ShareGPT, "300K samples" (§2.2); Table 1 lists ShareGPT 58K, UltraChat 105K, WizardLM 143K. X_base = Alpaca 52K, Dolly 15K, OAssist 10K, FLAN 2022 23K, "100K samples" (§2.2, Table 1).
- **Study setup:** LLaMA-1 13B, budget m = 6K, MT-Bench judged by GPT-4 (§2.2). "ChatGPT" means gpt-3.5-turbo-0613 (Figure 1 caption).
- **EVOL COMPLEXITY:** seed set = 2K random Alpaca samples (§2.3). Each instruction is evolved M = 5 times with the In-Depth Evolving prompt of [[evol-instruct]] (adding constraints, deepening, concretizing, increasing reasoning steps), giving 6 variants (§2.3; App. E.2 Tables 12-13). ChatGPT ranks and scores the variants together (App. E.2 Table 14); a LLaMA-1 7B model is trained to predict the score from the instruction (§2.3). Multi-turn: per-turn scores are summed (§2.3).
- **EVOL QUALITY:** responses are evolved M = 5 times by enhancing helpfulness, augmenting relevance, enriching depth, fostering creativity, and supplying additional details; ChatGPT ranks and scores them; a LLaMA-1 7B model predicts quality from the instruction-response pair; same 2K seed set (§2.4; App. E.4).
- **Why joint ranking:** scoring variants one at a time ("Direct Scoring") gives similar, high scores to most examples (§2.3; App. B Tables 8-9).
- **REPR FILTER:** sentence embeddings from LLaMA-1 13B; d is called the cosine distance to the nearest selected neighbour and the admit condition is printed as d < τ, τ = 0.9 in all experiments (§2.5). The released code computes d as the dot product of normalized embeddings (a cosine similarity) and drops a candidate if any similarity exceeds the threshold (repo `src/deita/selection/filter/base.py`, `compute_distance` and `filter`).
- **Algorithm 1:** evol score s = c × q (per turn, summed over turns); sort the pool by s; walk the sorted pool and add x if it passes REPR FILTER; stop at m samples (§3.1, Algorithm 1).
- **Complexity results (X_sota / X_base, MT-Bench):** Random 5.84 / 4.93; Perplexity 4.06 / 1.89; IFD 5.91 / 2.46; Instag Complexity 6.18 / 4.98; EVOL COMPLEXITY 6.27 / 5.57 (Table 2). High-perplexity samples "typically exhibit very short responses" (§2.3).
- **Quality results:** Random 5.84 / 4.93; Response Length 5.94 / 5.65; Direct Scoring (50K pool) 5.61 / 4.44; EVOL QUALITY 6.19 / 5.67 (Table 3).
- **Diversity results** (c × q held near the pool mean, App. A): Random 5.82 / 4.34; Instag Diversity 6.10 / 4.46; REPR FILTER 6.17 / 4.68 (Table 4).
- **Selector comparison (LLaMA-1-13B, MT-Bench / AlpacaEval):** Random 6K 5.84 / 73.91; Alpagasus 6K 5.61 / 71.21; LIMA 1K 4.29 / 41.98; TAGLM 6K 6.09 / 72.80; DEITA 6K 6.46 / 77.08 (Table 5).
- **Main models (MT-Bench / AlpacaEval):** DEITA-LLaMA1-13B 10K 6.60 / 78.01; DEITA-LLaMA2-13B 10K 6.79 / 81.09 vs LLaMA2-13B-Chat 6.65 / 81.09; DEITA-Mistral-7B 10K 7.32 / 81.67; DEITA-Mistral-7B 6K + 10K DPO 7.55 / 90.06 vs zephyr-beta 7.34 / 90.60 (Table 6). The abstract and Table 6 label the DPO model as 6K SFT; §3.3 text calls it "DEITA-Mistral-7B10K +DPO".
- **Release:** deita-6k-v0 and deita-10k-v0 datasets, complexity and quality scorer models, and scorer training data (repo README).

## Recipe ledger
| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| DEITA selection (all models) | — | SFT | data budget | 6K and 10K samples from X_sota | arXiv:2312.15685v2 §3.2 | verified 2026-09-14 | Fig. 2 scaling sweep 1K-300K on MT-Bench |
| DEITA selection | — | SFT | REPR FILTER threshold τ; encoder | 0.9; LLaMA-1 13B embeddings | §2.5; App. C.1 | verified 2026-09-14 | App. C.1 Fig. 4: τ swept 0.8-0.9, model-based encoder stable, E5-Large-V2 degrades |
| DEITA scorers | 7B | SFT | complexity / quality scorer backbone; seed data | LLaMA-1 7B; 2K Alpaca samples, M = 5 evolutions | §2.3, §2.4 | verified 2026-09-14 | Tables 2-3 vs baselines; no ablation of seed size |
| DEITA-LLaMA1-13B 6K/10K | 13B | SFT | batch; epochs; peak LR; warmup ratio | 128; 6; 1e-5; 0.03 | App. A | verified 2026-09-14 | no ablation reported |
| DEITA-LLaMA2-13B 6K/10K | 13B | SFT | batch; epochs; peak LR; warmup ratio | 128; 6; 2e-5; 0.1 (following Lu et al. 2023) | App. A | verified 2026-09-14 | no ablation reported |
| DEITA-Mistral-7B 6K/10K | 7B | SFT | batch; epochs; peak LR; warmup ratio; schedule | 512; 6; 2e-5; 0.1; cosine | App. A | verified 2026-09-14 | hyperparameters from Tunstall et al. (2023); epochs raised to 6 "to ensure adequate training", no ablation |
| DEITA-Mistral-7B 6K + DPO | 7B | preference | DPO data; batch; epochs; LR; warmup; schedule | 10K pairs sampled from Zephyr's UltraFeedback data; 32; 9; 5e-7; 0.1; linear | §3.2; App. A | verified 2026-09-14 | no ablation reported |
| DEITA-Mistral-7B 6K + DPO | 7B | preference | DPO β | not reported | checked body, App. A, repo README | not reported | — |
| All DEITA models | 7B, 13B | SFT | max input length; template | 2048; Vicuna-style | App. A | verified 2026-09-14 | no ablation reported |
| All DEITA models | 7B, 13B | SFT | compute | 4 A100 (7B) / 8 A100 (13B); DeepSpeed ZeRO-3; FlashAttention-2 | App. A | verified 2026-09-14 | n/a |
| All DEITA models | 7B, 13B | SFT | optimizer, weight decay, packing, loss masking, seeds | not reported | checked body, App. A | not reported | — |

## Findings relevant to generality
- **Judge-based measurement:** all metric studies (Tables 2-4) use MT-Bench only, which is scored by GPT-4 (§2.2); the number of training seeds is not reported. See [[judge-llm-bias]] for MT-Bench judge biases.
- **Benchmarks disagree:** the authors note DEITA's AlpacaEval gains are "not apparently consistent" with its MT-Bench gains and attribute the MT-Bench gains to coding, math, and reasoning subtasks (§3.3, Fig. 3) (Interpretation).
- **Non-judge evaluation (Open LLM Leaderboard):** DEITA-LLaMA1-13B 10K averages 64.27 vs 60.14 for random 10K, with MMLU 60.60 vs 47.35 (Table 7). On LLaMA-2-13B, random 10K scores higher on ARC (61.52 vs 58.87) and HellaSwag (83.69 vs 82.08), while DEITA has the higher average (62.71 vs 61.32) (Table 7).
- **More selected data can hurt:** under DEITA selection, MT-Bench first rises then declines as m grows; 3K samples are comparable to all 300K (§3.3, Fig. 2).
- **Pool dependence:** quality metrics change results more on the lower-quality X_base pool than on X_sota (§2.4, Table 3).
- **Human evaluation:** 100 LIMA test prompts, 4 annotators; DEITA-LLaMA1-13B 6K vs Vicuna-13B-v1.3 win/tie/lose 12% / 77% / 11%, vs random 6K 34% / 43% / 23%; author-annotator agreement 77% on 50 pairs (App. D, Table 10).

## Connections
- [[evol-instruct]]: source of the In-Depth Evolving prompts reused by EVOL COMPLEXITY.
- [[cherry-llm]], [[ifd]]: the IFD complexity baseline in Table 2 (Li et al., 2023a).
- [[instag]], [[instag-diversity]]: Instag Complexity, Instag Diversity, and TAGLM baselines (Tables 2, 4, 5).
- [[alpagasus]]: the Direct Scoring quality baseline and a Table 5 selector.
- [[lima]]: 1K baseline in Table 5 and source of the human-evaluation prompts.
- [[ultrafeedback]]: origin of the 10K DPO pairs.
- [[less]], [[superfiltering]], [[prismatic-synthesis]]: other data-selection methods; not compared in this paper.

## Verification
- Checked on 2026-09-14 against: https://arxiv.org/abs/2312.15685 (arXiv v2, 16 Apr 2024, PDF incl. appendices); github.com/hkust-nlp/deita README and `src/deita/selection/filter/{base,combined_filter}.py` (main branch; commit hash not shown on the fetched raw files).
- Corrections to the previous card version:
  - Title "DEITA: What Makes Good Data for Alignment?" → full published title above. Affiliation "(HKUST)" → five institutions (title page).
  - "scorers distilled into a 13B LLM" → LLaMA-1 7B scorers; LLaMA-1 13B is the embedding model (§2.3-2.5).
  - Complexity mutations "add constraints, increase depth, breadth" → adding constraints, deepening, concretizing, increasing reasoning steps (§2.3). Quality mutations "clarity, detail, informativeness" → helpfulness, relevance, depth, creativity, details (§2.4).
  - "rankings aggregated into pairwise labels" → ChatGPT scores all variants in one prompt and the scorer regresses those scores (§2.3).
  - "admit if min cosine distance > τ" → admit if nearest-neighbour cosine similarity is below 0.9 (§2.5 prints d < τ; code confirms similarity).
  - "~10K subset beats 10× larger random pools" / "6K superior to 300K baselines on Open LLM Leaderboard" → Table 6/7 comparisons above; 3K matches all 300K on MT-Bench (Fig. 2).
  - "DEITA-Mistral-7B matched Zephyr-7B-beta on MT-Bench with 6K SFT" → 7.22 (6K SFT) vs 7.34 (zephyr-beta SFT+DPO); 7.55 after adding 10K DPO (Table 6).
  - "plateau around 6K-10K" → performance rises then declines with more data (Fig. 2).
- Removed as unsupported by the source: "axis-ablation table" and the claims that removing diversity "collapses" the score, removing complexity weakens reasoning, and removing quality weakens format compliance (no leave-one-dimension-out ablation exists); "lexicographic, not weighted sum, because pure combined-score fails"; transfer to Yi models; "LIMA 1K vs UltraChat 200K puzzle" framing; cost estimate; adaptive-threshold follow-ups; "reference for Tülu 3's data selection"; "superseded by prismatic-synthesis"; the "Risks + gotchas" list.
- Not reported by the source: SFT or DPO β, optimizer settings, number of seeds, AlpacaEval judge version, contamination checks.
