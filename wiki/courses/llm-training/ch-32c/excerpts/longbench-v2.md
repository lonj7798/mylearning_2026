---
chapter: ch-32c
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/longbench-v2.md (planned card; not present on 2026-09-15)
source_url: https://arxiv.org/abs/2412.15204
created_at: "2026-09-15"
---

# Excerpt: LongBench v2: Towards Deeper Understanding and Reasoning on Realistic Long-context Multitasks

**Authors:** Yushi Bai, Shangqing Tu, Jiajie Zhang, Hao Peng, Xiaozhi Wang, Xin Lv, et al. (Tsinghua University; Zhipu.AI)
**Version read:** arXiv:2412.15204v2 (3 Jan 2025; v1 Dec 2024), PDF text.
**Source type:** paper. Status: no library card existed for this slug on 2026-09-15.

## Data (Abstract; §1; §3)
- 503 multiple-choice questions (four options, random baseline 25%), English only, contexts from 8k to 2M words; median 54k words, average 104k words (§1).
- Six task categories: single-document QA, multi-document QA, long in-context learning, long-dialogue history understanding, code repository understanding, long structured data understanding; 20 subtasks (Abstract; Table 1).
- Collection: 97 annotators, 24 reviewers. Automated review: a question is too easy if three long-context LLMs with 128k windows all answer correctly (inputs truncated from the middle). Manual review: rejected if the human reviewer answers correctly within 3 minutes (§1; §3.2).
- Difficulty split: "Hard" = at least two of three models wrong and the reviewer cannot solve it within 10 minutes; 192 Easy, 311 Hard. Length split by words: Short <32k (180), Medium 32k-128k (215), Long >128k (108) (§3).
- Verification on 70 sampled items: 68/70 correct, 67/70 not found by 15 minutes of web search; estimated error rate about 3% (§3.3).

## Evaluation setup (§4.1)
- 10 open-source LLMs with 128,000-token windows and 7 proprietary LLMs; "middle truncation ... for sequences exceeding the model's context window length"; zero-shot and zero-shot + CoT.
- "For a fair comparison, the Qwen2.5 series models are evaluated without YaRN"; YaRN results are in App. E Table 4.

## Results (Table 2; overall accuracy %, zero-shot / zero-shot + CoT)
- Human experts: 53.7 under a 15-minute limit (8% of items answered "I don't know").
- GPT-4o-2024-08-06 50.1 / 51.2; GPT-4o-2024-11-20 46.0 / 51.4; o1-preview-2024-09-12 57.7 / 56.2; Claude-3.5-Sonnet-20241022 41.0 / 46.7; GLM-4-Plus 44.3 / 46.1.
- Open models: Qwen2.5-72B-Instruct 39.4 / 38.8; Llama-3.1-70B-Instruct 31.6 / 36.2; Llama-3.1-8B-Instruct 30.0 / 30.4; Mistral-Large-Instruct-2411 34.4 / 39.6.
- CoT gives "an average 3.4% improvement for open-source models" (§4.1). The table note states that models "do not show lower scores on subsets with longer length ranges because the distribution of tasks differs significantly across each length range" (Table 2 note).

## Memorization control (§4.3; Table 3; overall % with context / without context)
- GPT-4o 50.1 / 33.1; GLM-4-Plus 44.3 / 27.6; Qwen2.5-72B-Inst. 39.4 / 30.0; Llama-3.1-8B-Inst. 30.0 / 25.8. "Without context, most models achieve an overall accuracy ranging from 25% to 30%."

## RAG baseline (§4.2; Fig. 4)
- Chunks of 512 tokens, top-N with N = 4 ... 256. Qwen2.5 and GLM-4-Plus show no significant improvement beyond 32k retrieval context, and perform better at 32k than with the full 128k window without RAG (Qwen2.5 +4.1%).

## Inference configuration: YaRN (App. E, Table 4; overall zero-shot / CoT)
- Following the Qwen2.5-72B-Instruct model card, "YaRN with a scaling factor of 4.0".
- Qwen2.5-7B-Instruct 27.0 / 29.8 → +YaRN 30.0 / 35.6. Qwen2.5-72B-Instruct 39.4 / 38.8 → +YaRN 42.1 / 43.5.

## Limitations stated (§6)
- "The benchmark's size may not be sufficiently large ... it could also lead to less stable results that are more vulnerable to randomness." English only. Uneven length distribution across tasks; comparisons are recommended per length interval.
