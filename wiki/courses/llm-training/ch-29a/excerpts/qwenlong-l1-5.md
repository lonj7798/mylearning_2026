---
chapter: ch-29a
course: llm-training
phase: read
excerpt_of: arXiv:2512.12967v1 (QwenLong-L1.5), §3 Long-Context Data Construction, §4.1, Tables 1, 7, 8 (chapter-local verified extract; no library card exists yet)
source_url: https://arxiv.org/abs/2512.12967
created_at: "2026-09-15"
---

# Excerpt: QwenLong-L1.5 — synthesized long-context RL data with grounding and robustness checks

- **Authors:** Weizhou Shen et al. (Tongyi Lab, Alibaba Group)
- **Year:** 2025 (arXiv v1 2025-12)
- **Source type:** paper
- **Used in:** ch-29a §4, §6, Negative samples, Generalization lens

## Corpus (§3.1)
Code repositories, academic literature, professional documents, general knowledge and literature, and "a small collection of multi-turn
dialogues simulated by large language models"; after rule-based and LLM-as-a-judge filtering, "82,175 high-quality documents, totaling
approximately 9.2 billion tokens".

## QA synthesis (§3, §3.2)
- Multi-hop QA via knowledge graph: triplet extraction, cross-document aggregation, entity and relation clustering; paths sampled by
  random walk and BFS; "path nodes are deliberately distributed sparsely across multiple documents"; entities obfuscated, for example
  "the year ending with 5 in the late 20th century"; complexity controlled "by regulating path length".
- Corpus-level numerical QA: schema extraction, cross-document table aggregation, natural-language queries translated to SQL; "By
  executing the SQL against the aggregated tables, we accurately simulate complex calculation processes ... thereby deriving the Ground
  Truth Answers."
- General long-context tasks: proposer, solver, and verifier agents; the proposer is conditioned on documents and a buffer of validated
  QA pairs to push "harder and more diverse questions".
- Difficulty: "we further extend the context to our target length by strategically inserting irrelevant documents."

## Verification (§3)
- "(1) Knowledge Grounding Check: We temporarily removed the source document and tested if the model could still answer the question.
  Samples that could be answered correctly (i.e., relying on the model's internal knowledge) were filtered out".
- "(2) Contextual Robustness Check: We expanded the context with irrelevant documents and verified the model's answer. Any sample where
  the answer accuracy (pass@k) dropped to zero was discarded."
- Yield: "14.1k high-quality training samples from an initial pool of 42.7k synthesized examples" after difficulty filtering,
  deduplication, and test-set decontamination.
- Table 1: QwenLong-L1 1.6K samples, average input 11,441 tokens, max 59,563; QwenLong-L1.5 14.1K, average 34,231, max 119,932.
- The value of k in pass@k and the checking model are not reported in §3.

## Results (base Qwen3-30B-A3B-Thinking-2507 → QwenLong-L1.5-30B-A3B)
- Table 7 average 61.92 → 71.82 (+9.90); MRCR 51.27 → 82.99; CorpusQA 71.56 → 81.25; LongBench-V2 49.11 → 55.27; Frames 70.27 → 74.76;
  DocMath 62.26 → 66.26; LongBench-V1-QA 67.10 → 70.40.
- Table 8: MMLU-Pro 81.03 → 81.33; AIME24 90.31 → 90.0; AIME25 82.81 → 86.46; GPQA-Diamond 75.88 → 76.78; LongMemEval 60.80 → 76.40;
  BFCL-V4 Memory-Rec_Sum 41.94 → 40.00.
- The training stages are RL (GRPO-based, multi-stage input lengths 20K/60K/120K) plus merging; data effects are not isolated from the
  RL algorithm changes in Table 7.
