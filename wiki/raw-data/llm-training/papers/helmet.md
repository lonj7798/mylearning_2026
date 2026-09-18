<!-- scope: HELMET long-context benchmark (7 application categories, 8K-128K controlled lengths, GPT-4o judge metrics, 2-shot prompting for base models) and its study of 59 long-context LMs
     see-also: [[ruler]], [[infinitebench]], [[nolima]], [[lost-in-the-middle]], [[prolong]], [[longbench]], [[babilong]]
-->

# HELMET: How to Evaluate Long-Context Language Models Effectively and Thoroughly
- **Core Insight:** Across 35 instruction-tuned models at 128K tokens, the Spearman correlation with ∞Bench QA is 0.63 for NIAH, 0.81 for RULER MK, and 0.88 for the RAG task HotpotQA (Fig. 4), and no synthetic task averages above 0.8 against the downstream categories (Fig. 3; §3.1).
- **Guideline:** When a long-context checkpoint needs a fast check during development, use the RAG category, because it is easy to run, works with base models, and correlates more with other downstream categories than synthetic recall (Abstract; §3.1; App. E.1); for model decisions, evaluate all seven categories, because categories such as ICL correlate with others at 0.34-0.61 (Fig. 5).
- **Authors:** Howard Yen, Tianyu Gao, Minmin Hou, Ke Ding, Daniel Fleischer, Peter Izsak, et al. (Princeton Language and Intelligence, Princeton University; Intel)
- **Year:** 2024 (arXiv v1 2024-10; ICLR 2025)
- **URL:** https://arxiv.org/abs/2410.02694 (code and data: https://github.com/princeton-nlp/HELMET)
- **Source type:** paper
- **Relevant topics:** long-context evaluation, synthetic vs downstream correlation, LLM-as-judge metrics, base-model evaluation, positional extrapolation, instruction tuning effects

## Abstract
Long-context language models (LCLMs) are often evaluated with needle-in-a-haystack (NIAH) tests or arbitrary task subsets. The authors find that existing benchmarks give noisy signals because of limited application coverage, insufficient lengths, unreliable metrics such as ROUGE, and incompatibility with base models. HELMET covers seven application-centric categories, supports controllable input lengths up to 128K tokens, uses model-based evaluation for QA and summarization, and adds few-shot demonstrations so base models can be evaluated. In a study of 59 LCLMs, (1) synthetic tasks such as NIAH do not reliably predict downstream performance, (2) the categories show distinct trends and low correlation with each other, and (3) open-source models lag closed ones on tasks that need full-context reasoning or complex instruction following, with a gap that widens with length. The authors recommend RAG tasks for fast development and holistic evaluation across categories.

## Key Contributions
- Seven categories with 128K-capable datasets: RAG, generation with citations (Cite), passage re-ranking (Re-rank), many-shot in-context learning (ICL), long-document QA (LongQA), summarization (Summ), and synthetic recall (§2.1; Table 3).
- Reference-based GPT-4o judging for NarrativeQA and summarization, validated against human judgments (§2.2; App. B.6).
- Two-shot demonstrations for all tasks except ICL and RULER, so base models produce parseable outputs (§2.3, fn. 6; Table 8).
- A correlation analysis of synthetic, RAG, and downstream tasks at 128K (§3.1-3.2; Figs. 3-5, 8-9).
- An evaluation of 59 models of different architectures, sizes, and position-extension methods at five lengths (§3; Table 15).

## Key Figures/Tables to Study
- **Fig. 1 / Table 5** — NIAH, RULER, ∞Bench, and HELMET rankings of six frontier models at 128K.
- **Table 3** — datasets and metrics per category.
- **Figs. 3, 4, 8** — synthetic and RAG tasks vs downstream categories (Spearman ρ).
- **Fig. 5 and Fig. 9** — correlation between categories and between all datasets.
- **Fig. 6** — instruction-tuned model scores per category at 8K-128K (all models: Fig. 10).
- **Fig. 7** — degradation with length as task complexity increases.
- **Table 8** — 0-shot vs 2-shot results for base and instruction-tuned models, 3 seeds.

## Technical Details
### Datasets and metrics (§2.1; Table 3; App. B)
- **RAG:** Natural Questions, TriviaQA, PopQA, HotpotQA; metric SubEM (substring exact match: the gold answer appears in the output). Distractors are hard negatives retrieved with `Alibaba-NLP/gte-large-en-v1.5` from the 2019-08-01 Wikipedia dump split into 100-word passages (§2.1, fn. 3-4). For NQ, TQA, and PopQA the gold passage is placed at six evenly spaced positions; HotpotQA's two gold passages are shuffled with k−2 distractors into three permutations (§2.1). PopQA is filtered to entities with popularity below 1,000 (App. B.1).
- **Cite:** ALCE ASQA and QAMPARI with GTR retrieval over Wikipedia; the score averages correctness and citation quality; MAUVE fluency is omitted (§2.1; App. B.2).
- **Re-rank:** MS MARCO with TREC relevance labels; k passages sampled with balanced labels, three permutations, output the top-10 IDs; metric NDCG@10, a graded ranking metric over the top 10 positions (§2.1; App. B.3).
- **ICL:** TREC Coarse (6 labels), TREC Fine (50), NLU (68), BANKING77 (77), CLINC150 (151); labels mapped to random integers so the model must learn from demonstrations; rounds contain one example per label; 500 evaluation samples (Table 3; App. B.4).
- **LongQA:** NarrativeQA (judge score), ∞Bench QA (ROUGE F1), ∞Bench MC (accuracy); documents truncated from the end to fit L (§2.1).
- **Summ:** ∞Bench Sum and Multi-LexSum, judge score (§2.1). Lengths ("Medium"/"Max" columns): NarrativeQA 73K/518K, ∞Bench QA 191K/835K, ∞Bench Sum 154K/835K, Multi-LexSum 90K/5M; ZeroSCROLLS datasets are 6K-12K/10K-33K (Table 4).
- **Synthetic recall:** JSON KV (UUID keys and values, six evenly spaced query positions), RULER MK Needle, RULER MK UUID, RULER MV; SubEM (Table 3; App. B.5). The four were selected because they correlate more with realistic datasets than other synthetic tasks (App. E.1).

### Model-based metrics (§2.2; App. B.6)
- Judge: `GPT-4o-2024-05-13`. NarrativeQA score = fluency (0 or 1) × correctness (0-3), normalized to [0, 100] (§2.2).
- Summarization: GPT-4o decomposes the gold summary into atomic claims; recall = share of claims supported by the output; precision = share of output sentences supported by the reference; final = F1 × fluency (§2.2; App. B.6).
- Validation: 105 claims from 25 Multi-LexSum summaries were all factually correct, with one of 25 instances missing a key point (App. B.6). Human-judge agreement (Cohen's κ): recall 0.76 (∞Bench Sum) and 0.72 (Multi-LexSum); precision 0.91 and 0.83 (App. B.6). Disagreements come mostly from partially supported key points, where the judge is more lenient than humans (App. B.6).
- ROUGE-L for GPT-4o stays within 2 absolute points across lengths while the judge score rises (§2.2; Fig. 2). A Mistral-7B-Inst-v0.3 summary made of one sentence repeated hundreds of times gets ROUGE-L 12.3 and judge score 0.0 (App. B.6).

### Evaluation setup (§3; App. D)
- Lengths L ∈ {8192, 16384, 32768, 65536, 131072} Llama-2 tokens; greedy decoding; chat templates for instruction-tuned models (§3; App. D).
- Samples: 600 for JSON KV, NQ, PopQA, TQA; 300 for MS MARCO and HotpotQA; 500 for ICL; 100 for the rest (App. D).
- Open models run with HuggingFace, FlashAttention2, and BF16 on H100 80GB GPUs; with at most 8 H100s, Command-R and Jamba-1.5-Large were not run at 128K (App. D).

### Results
- At 128K (Table 5): NIAH is 100.0 for five of six frontier models (Gemini-1.5-Pro 45.3). RULER ranks Gemini-1.5-Flash (86.6) above Gemini-1.5-Pro (65.3); ∞Bench ranks Llama-3.1-8B-Inst (44.1) above Llama-3.1-70B-Inst (39.7); HELMET gives Pro 62.7 vs Flash 50.7 and 70B-Inst 49.3 vs 8B-Inst 47.0.
- NIAH correlations with the downstream categories are all ≤ 0.8 and the RULER average's are all < 0.85 (§3.1). On NIAH most models score perfect or near zero, so it separates models poorly; RULER MK spreads scores between 0% and 100% (§3.1; Fig. 4).
- Category correlations (Fig. 5): ICL vs Recall 0.61, RAG 0.50, Cite 0.34, Re-rank 0.36, LongQA 0.51, Summ 0.38. The two ALCE datasets do not correlate with each other, which the authors read as citation writing being a skill separate from factual answering (App. E.2).
- At 128K closed models outperform open models on every category except ICL; the gap is small on synthetic recall and LongQA and 30-40 absolute points on Cite and Re-rank (§3.3). In v3 Fig. 6, Re-rank is 59.7 for Gemini-1.5-Pro vs 27.9 for Qwen2.5-14B-Inst-1M (31.8 points, derived), and Cite is 44.3 for GPT-4o-08 vs 27.1 for Qwen2.5-14B-Inst-1M (17.2 points, derived) and 7.5 for Llama-3.1-70B (Fig. 6).
- Degradation with length grows with task complexity (Fig. 7): Qwen2-57B-Inst, which relies on position extrapolation, goes from 54.7 to 12.7 on RAG between 64K and 128K.
- Many open models beat closed models on ICL (for example Jamba-1.5-Mini 91.0 vs GPT-4o-08 86.3 at 128K, Fig. 6); the authors suggest heavy instruction tuning may hurt ICL (§3.3). Interpretation.
- Claude-3.5-Sonnet scores low because it often omits citation markers, rankings, or class labels, sometimes refuses for copyright reasons, and hits output-token limits (App. E.6).

## Findings relevant to generality and long context
- **Measurement of breadth.** Long-context ability is not one quantity: category correlations range down to 0.34 at 128K (Fig. 5), and no model leads all categories (§3.3). Result (single study; 35 instruction-tuned models for Figs. 3, 4, and 9; 30 per the Fig. 8 caption).
- **Base-checkpoint evaluation.** Two-shot demonstrations raise Llama-3.1-8B (base) from 77.3 to 98.0 on JSON KV and from 0.1 to 7.5 on MS MARCO at 128K (Table 8, 3 seeds). The effect is not uniform: GPT-4o-05 drops from 99.3 to 36.7 on JSON KV (Table 8).
- **Prompt sensitivity of rankings.** Llama-3.2 1B/3B models score 1.8-2.8 on ∞Bench but 21.2-36.9 on HELMET (Table 9); the authors attribute this to prompting with demonstrations (App. A.1).
- **Post-training.** Instruction tuning generally improves all categories, and Llama-3.1-8B-Inst outperforms Llama-3.1-70B (base) on most tasks (App. E.5). Result (single study).
- **Position extension without training.** Llama-3-Inst with RoPE base changed from 500,000 to 16,000,000 at inference, and Qwen2-Inst with YaRN, drop in performance past 32,768 tokens; changing the position embedding can also lower short-length scores (Llama-3-8B-Inst on ODQA and ICL) (App. E.3).
- **Position bias at 128K.** Models generally favor the most recent context; some (Llama-3, Llama-3.2) favor the start, yet at long inputs their middle-depth scores often exceed start-depth scores (App. E.4).

## Connections
- [[ruler]] — source of the MK Needle, MK UUID, and MV recall tasks; its average correlates < 0.85 with downstream categories.
- [[infinitebench]] — source of the LongQA and Summ novel tasks; HELMET re-runs it and reports ranking inversions (Table 6, Table 9).
- [[lost-in-the-middle]] — origin of the JSON KV task and the gold-passage position protocol.
- [[nolima]] — measures literal overlap between question and context; HELMET RAG has ROUGE-1 precision 0.689 there (NoLiMa Table 1).
- [[prolong]] — ProLong 8B (524,288 training length) is evaluated here; 49.4 average at 128K (Fig. 6).
- [[many-shot-icl]] and [[longcite]] — related single-category evaluations of ICL and citation.
- [[yarn]] and [[position-interpolation]] — extension methods whose models are compared (Table 15; App. E.3).
- [[babilong]] and [[longbench]] — other long-context benchmarks with different length ranges and task types.

## Verification
- Created on 2026-09-14 from https://arxiv.org/abs/2410.02694 (arXiv v3, 2025-03-06; submission history v1 2024-10-03, v2 2024-10-10).
- Audit claims not found in the source: none. Note on the audit's "closed models lead open ones by 30-40 points on re-ranking and citation generation": this is the §3.3 prose; the v3 Fig. 6 Cite gap to the best listed open model is 17.2 points (derived above).
- Inconsistencies inside the source: §3.1 and the Fig. 3 caption use 35 instruction-tuned models, the Fig. 8 caption says 30.
- Not reported by the source: training experiments. The 59-model count is from v3; earlier versions were not checked.
