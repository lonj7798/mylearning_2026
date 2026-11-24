<!-- scope: DeepMath-103K (arXiv:2504.11456) — hard (level ≥5), decontaminated, verifiable math RL set; embedding+LLM-judge decontamination; reference answers by 3-way R1 agreement; zero-RL results
     deps: [[big-math]], [[numina-math]]
     see-also: [[math-verify]], [[verifier-pitfalls]], [[polaris]], [[skywork-or1]], [[dapo]], [[rl-design-guide]]
-->

# DeepMath-103K: A Large-Scale, Challenging, Decontaminated, and Verifiable Mathematical Dataset
- **Core Insight:** Before decontamination, 90% of AIME24 and AMC23 problems, 76.6% of MATH500 and 33.6% of OlympiadBench had near-duplicates in the 2,869K-question raw pool (§3, Stage 2); after embedding + LLM-judge decontamination, level filtering (≥5) and a reference answer kept only when three R1 solutions agree, zero-RL on Qwen-2.5-7B reaches 20.4 / 17.5 on AIME24/25 against 15.6 / 14.4 for Open-Reasoner-Zero-7B (Table 3).
- **Guideline:** When the RL prompt pool is mined from the web or from forum data, decontaminate against every evaluation set with embedding retrieval plus an LLM paraphrase judge, and keep a problem only if independently generated solutions (three R1 samples plus the source solution when present) reach the same final answer, because raw web pools contained up to 90% of benchmark items as paraphrases (§3) and answer disagreement is the cheapest signal of an unverifiable or wrong reference.
- **Authors:** Zhiwei He, Tian Liang, Jiahao Xu, Qiuzhi Liu, Xingyu Chen, Yue Wang, Linfeng Song, Dian Yu, Zhenwen Liang, Wenxuan Wang, Zhuosheng Zhang, Rui Wang, Zhaopeng Tu, Haitao Mi, Dong Yu (Tencent, Shanghai Jiao Tong University)
- **Year:** 2025 (arXiv:2504.11456; v2 read, stamped 22 May 2025)
- **URL:** https://arxiv.org/abs/2504.11456
- **Source type:** paper (dataset + RL results)
- **Relevant topics:** decontamination, difficulty filtering, reference-answer creation by consensus, zero-RL data

## Summary
Four-stage pipeline: (1) collect Math StackExchange subsets of MMIQC and WebInstructSub plus NuminaMath-CoT (2,869K questions after basic filtering); (2) decontaminate against 14 benchmarks; (3) keep problems that GPT-4o rates level 5 or higher on an AoPS-style scale; (4) reformat to a single numerical or symbolic answer and verify it by agreement across generated solutions. About 95K hard problems remain; 8K level 3–5 problems from SimpleRL are added for ~103K.

## Key Contributions
- Measured contamination rates of a raw web math pool against standard benchmarks (§3).
- Reference answers created by agreement among three R1 solutions and the source solution (§3, Stage 4).
- RL results across three starting points (base, math base, distilled 1.5B) (Table 3).
- Reported cost: $138,000 GPT-4o API fees and 127,000 H20 GPU hours (§3).

## Key Figures/Tables to Study
- §3 pipeline description (Stages 1–4), Fig. 1a difficulty distribution vs other RL sets, Table 3 (RL results), Table 6 (training config).

## Technical Details
**Decontamination (Stage 2).** Top-5 embedding search with paraphrase-multilingual-MiniLM-L12-v2, then Llama-3.3-70B-Instruct judges duplicate or paraphrase; any match is discarded. Benchmarks include MATH, AIME, AMC, Minerva Math, OlympiadBench, Omni-MATH, GSM8K and GPQA (14 total).

**Difficulty (Stage 3).** GPT-4o with AoPS competition-rating guidelines (the Omni-MATH method), six ratings per problem averaged; keep level ≥ 5.

**Reference answer (Stage 4).** GPT-4o removes unsuitable problems and rewrites the rest toward one numerical or symbolic answer. Three R1 solutions are generated; a problem is kept only if all extracted final answers (including the source solution's, when present) are identical.

**RL setup (Table 6).** LR 1e-6; reward +1 correct, −1 otherwise; 16 rollouts per prompt (18 for the Omn-1.5B run); 500 steps for the 7B zero runs, 1,800 for DeepMath-1.5B.

**Results (Table 3; pass@1 over 16 samples, T = 0.6, 32K max tokens).**

| Model | AIME24 | AIME25 | AMC23 | MATH500 |
|---|---|---|---|---|
| Qwen-2.5-7B base | 7.7 | 5.4 | 35.3 | 54.8 |
| Open-Reasoner-Zero-7B | 15.6 | 14.4 | 58.9 | 81.8 |
| DeepMath-Zero-7B | 20.4 | 17.5 | 64.7 | 85.5 |
| R1-Distill-Qwen-1.5B | 29.4 | 24.8 | 72.0 | 84.7 |
| DeepScaleR-1.5B-Preview | 42.3 | 29.6 | 80.3 | 89.4 |
| DeepMath-1.5B | 37.3 | 30.8 | 82.3 | 89.9 |

GPQA-Diamond (Table 4): DeepMath-Zero-7B 41.7.

## Verification
- Read arXiv HTML v2 on 2026-11-24 (full page). All numbers from §3, Tables 3, 4, 6.
- The paper's text says the 1.5B model is SOTA on all benchmarks, but Table 3 shows DeepScaleR-1.5B-Preview higher on AIME24 (42.3 vs 37.3).
- Intermediate counts after each stage are not given; only 2,869K raw → ~95K hard (Fig. 7 caption).
- No ablation isolates the 3-way-agreement filter or the decontamination step on RL accuracy; the dataset comparison against ORZ also changes the data source, so it is not a controlled test of any single filter.
