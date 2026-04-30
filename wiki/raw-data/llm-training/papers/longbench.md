<!-- scope: long-context eval — LongBench benchmark with realistic long-context tasks
     deps: [[needle-in-haystack-data]]
     see-also: [[ruler]], [[babilong]]
-->

# LongBench: A Bilingual, Multitask Benchmark for Long Context Understanding
- **Core Insight:** While synthetic tests (NIAH, RULER, BABILong) probe long-context skills in isolation, realistic long-context evaluation requires a benchmark of natural tasks (document QA, summarization, few-shot learning over long context, code completion) in both English and Chinese; LongBench fills this gap with 21 tasks averaging 6K–18K tokens.
- **Guideline:** Report LongBench alongside synthetic RULER/NIAH; the mix reveals whether a model generalizes beyond the training-data distribution (books, papers) to diverse real tasks.
- **Authors:** Yushi Bai, Xin Lv, Jiajie Zhang, Hongchang Lyu, Jiankai Tang, Zhidian Huang, Zhengxiao Du, Xiao Liu, Aohan Zeng, Lei Hou, Yuxiao Dong, Jie Tang, Juanzi Li (Tsinghua)
- **Year:** 2023 / updated 2024
- **URL:** https://arxiv.org/abs/2308.14508
- **Relevant topics:** long-context evaluation, bilingual, realistic tasks, LongBench

## Abstract
LongBench is a bilingual (English + Chinese) long-context evaluation suite with 21 tasks across 6 categories: single-document QA, multi-document QA, summarization, few-shot learning, synthetic tasks, code completion. Average input length is 6K–18K tokens (max tested ~31K). LongBench-Chat (LongAlign companion) extends to 10K–100K for chat-style long queries. The benchmark is widely used as a "natural-task" complement to synthetic evals.

## Key Contributions
- **21 natural long-context tasks** covering QA, summarization, few-shot, code.
- **Bilingual coverage** — English and Chinese each.
- **LongBench-Chat** extension with longer contexts (10K–100K).
- De facto companion to NIAH / RULER for most long-context releases.

## Task categories (REQUIRED — long-context eval)

### Single-Document QA
- NarrativeQA, Qasper, MultiFieldQA-en/zh.

### Multi-Document QA
- HotpotQA, 2WikiMQA, Musique, DuReader.

### Summarization
- GovReport, QMSum, MultiNews, VCSum (zh).

### Few-shot learning
- TriviaQA, SAMSum, TREC (long-context few-shot).

### Synthetic tasks
- PassageRetrieval-en/zh, PassageCount.

### Code completion
- LCC, RepoBench-P.

## Dataset size
- ~4,750 test samples total across tasks.
- Each task ~200–500 samples.
- Input length distribution: median ~10K, tail to 31K.

## Modality-specific technical details (REQUIRED — long-context)
- **Token-range:** 6K → 31K (LongBench); 10K → 100K (LongBench-Chat).
- **Needle-retrieval difficulty:** varies — NarrativeQA and passage-retrieval tasks are retrieval-like; multi-doc QA requires reasoning.
- **Document-type mix:** news, academic, legal, fiction, dialogue, code.
- **Packing strategy:** N/A (evaluation).
- **Position-encoding stress-test:** less sensitive than RULER; realistic task noise dominates position artifacts.
- **Evaluation metric:** task-specific — F1 (QA), ROUGE (summarization), accuracy (few-shot).

## Quality / diversity evaluation (of benchmark)
- Used by Llama 3, Qwen 2, ChatGLM, Claude, GPT-4-128K in release evals.
- Correlation with human-judged long-context quality: strong for QA and summarization; weaker for few-shot.
- Updated LongBench-v2 (2024) with more Chinese coverage and longer contexts.

## Risks + gotchas
- **Task quality varies:** some tasks (NarrativeQA) are solvable with ~4K context; real long-context signal is strongest on multi-doc QA and synthesis.
- **Contamination:** NarrativeQA, HotpotQA, TriviaQA are older benchmarks likely seen during pretraining.
- **English/Chinese coverage only** — other languages unaddressed.

## Connections
- Sibling synthetic evals: [[ruler]], [[babilong]], [[needle-in-haystack-data]].
- Companion SFT data: [[longalign]] (LongAlign-10K + LongBench-Chat by same Tsinghua group).
- Used to evaluate: [[prolong]], [[long-context-llama3]], [[qwen-long-context-synth]].
