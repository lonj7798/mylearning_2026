---
chapter: ch-51
course: llm-training
phase: read
excerpt_of: "primary source arXiv:2306.05685 (NeurIPS 2023 Datasets and Benchmarks); library card wiki/raw-data/llm-training/papers/judge-llm-bias.md has no Verification section as of 2026-09-15"
source_url: https://arxiv.org/abs/2306.05685
created_at: "2026-04-23"
revised: "2026-09-15 — numbers re-read from the primary text; the card's '20-30% flip rate' and 'reference-guided +10 pp agreement' are not what the paper reports"
---

# Excerpt: Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena (Zheng et al. 2023)

**Authors:** Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin,
Zhuohan Li, Dacheng Li, Eric P. Xing, Hao Zhang, Joseph E. Gonzalez, Ion Stoica. Source type: paper.

ch-51 uses this source for one purpose: judge inconsistency is a noise source that does not shrink when the
evaluation set grows. The full treatment of judge bias and calibration is in ch-49.

## Table 2 — position bias, as printed

Consistency is "the percentage of cases where a judge gives consistent results when swapping the order of two
assistants". The test set is deliberately hard: "we construct two similar answers to each first-turn question in
MT-bench by calling GPT-3.5 twice with a temperature of 0.7" (§3.2).

| Judge | Prompt | Consistency | Biased toward first | Biased toward second | Error |
|---|---|---|---|---|---|
| Claude-v1 | default | 23.8% | 75.0% | 0.0% | 1.2% |
| Claude-v1 | rename | 56.2% | 11.2% | 28.7% | 3.8% |
| GPT-3.5 | default | 46.2% | 50.0% | 1.2% | 2.5% |
| GPT-3.5 | rename | 51.2% | 38.8% | 6.2% | 3.8% |
| GPT-4 | default | 65.0% | 30.0% | 5.0% | 0.0% |
| GPT-4 | rename | 66.2% | 28.7% | 5.0% | 0.0% |

"Only GPT-4 outputs consistent results in more than 60% of cases" (§3.2). Under the default prompt the swap
changes the verdict in 35.0% of pairs for GPT-4, 53.8% for GPT-3.5, and 76.2% for Claude-v1. Appendix D.1 reports
that position bias is less prominent in some categories and less prominent for model pairs whose quality differs
widely (Tables 9-11).

## Mitigations, as reported

- Swapping positions: "a conservative approach is to call a model a winner only when an answer is preferred in
  both orders. If the results are inconsistent after swapping, we can call it a tie" (§3.4).
- Few-shot judge: raises GPT-4 consistency from 65.0% to 77.5%; "however, high consistency may not imply
  [higher] accuracy" (§3.4).
- Reference-guided grading on math questions: Table 4 reports the GPT-4 failure rate on 10 math questions with
  position swaps as 14/20 (default), 6/20 (chain of thought), 3/20 (reference-guided); §3.4 states the
  improvement as "from 70% to 15%".
- Verbosity: under the "repetitive list" attack on 23 MT-Bench answers, the failure rate is 91.3% for Claude-v1,
  91.3% for GPT-3.5, and 8.7% for GPT-4 (Table 3).

## Agreement with humans

Table 5 (MT-Bench, first turn) reports GPT-4 pairwise vs human agreement at 85% under setup S2 (non-tie votes
only) and 66% under setup S1 (ties and position-bias-inconsistent votes included, counted as ties); human vs human
agreement in the same table is 81% (S2) and 63% (S1). Table 6 (Chatbot Arena) reports GPT-4 vs human at 87% (S2)
and 64% (S1). Figure 2 shows agreement rising with the win-rate difference between the two models in a pair. The
paper reports agreement rates; it does not report a measurement of judge calibration drift over time.

## Used in

ch-51 §1.5 (judge inconsistency as an irreducible noise term in a win-rate gate) and Common mistakes. ch-49 holds
the full bias inventory.
