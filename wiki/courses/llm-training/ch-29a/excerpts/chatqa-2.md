---
chapter: ch-29a
course: llm-training
phase: read
excerpt_of: arXiv:2407.14482v3 (ChatQA 2, ICLR 2025), §3.1-3.2, §5, Tables 2, 7, 8, App. B (chapter-local verified extract; no library card exists yet)
source_url: https://arxiv.org/abs/2407.14482
created_at: "2026-09-15"
---

# Excerpt: ChatQA 2 — 128K extension of Llama3-70B and NarrativeQA summary insertion

- **Authors:** Peng Xu, Wei Ping, Xianchao Wu, Chejian Xu, Zihan Liu, Mohammad Shoeybi, Bryan Catanzaro (NVIDIA)
- **Year:** 2024 (arXiv v1 2024-07; v3 2025-02 used here; ICLR 2025)
- **Source type:** paper
- **Used in:** ch-29a §2.4, §3.4, §7

## Continued pretraining (§3.1)
- SlimPajama following Fu et al. (2024), long documents upsampled with hyperparameter 0.1, "to produce 10 billion tokens with
  sequence length of 128k"; RoPE base 500,000 → 150M; "batch size to 32 to fit 4 million tokens in a batch and use a learning rate
  of 3e-5 to train 2000 steps (8B tokens in total)". The 10B and 8B figures are both printed in §3.1.
- "we found it more effective to separate different documents using special characters, such as '<s>', rather than the reserved
  beginning and ending tokens <BOS> and <EOS>. We hypothesize that the <BOS> and <EOS> tokens in Llama3 signal the model to ignore
  previous chunks of text after pretraining". App. B: a 2B-token comparison on NIAH shows "<s>" "much better" (heatmap only, no numbers).

## Long SFT data (§3.2)
- Three stages; stages 1–2 follow ChatQA 1.5 with contexts up to 4K tokens.
- Under 32K: LongAlpaca12k, GPT-4 samples from Open Orca, Long Data Collections.
- 32K–128K: "We utilize NarrativeQA, which includes summary paragraphs, questions, answers, and source long web pages. The summaries
  are human-generated based on the source web pages, while the question-answer pairs are annotated by humans using the summaries.
  To extend the context length, we inserted a summary into the corresponding long web page document at a random location, ensuring
  that sentence structure was not disrupted."
- "Since we used NarrativeQA for synthetic data generation, we intentionally excluded it from the evaluation benchmarks".
- Long and short SFT data are blended; LR 3e-5, batch 32.

## Results
- InfiniteBench (>100K) average, Table 2: Llama3-ChatQA-2-70B 41.04; Llama3.1-70B-Instruct 39.81; Qwen2-72B-Instruct 39.77;
  GPT-4-Turbo-2024-04-09 33.16. En.Sum is 16.08 for ChatQA-2-70B vs 30.94 for Llama3.1-70B-Instruct; the authors attribute this to
  "the lack of summarization data in our SFT recipe" (§5.2).
- Table 7 (8B): three-stage vs all-in-one-stage, Short 52.50 vs 48.55; Long (32K) 39.41 vs 40.69; Ultra-long 35.59 vs 35.07.
- Table 8, Llama3-ChatQA-2-8B (new, 1.6M-sample SFT) vs Llama3.1-8B-Instruct: Long (32K) 42.05 vs 42.42; Ultra long 35.18 vs 33.17;
  HumanEval 66.46 vs 70.73; MMLU 65.73 vs 67.59; MT-bench 8.09 vs 8.42; GSM8K 87.41 vs 83.70. These are different training pipelines,
  not a with/without-long-SFT ablation.
