---
chapter: ch-29a
course: llm-training
phase: read
excerpt_of: arXiv:2409.01893v2 (What are the Essential Factors in Crafting Effective Long Context Multi-Hop Instruction Datasets? — MIMG / LongMIT), §1-4 (chapter-local verified extract; the longmit card describes a generic dataset class, not this paper)
source_url: https://arxiv.org/abs/2409.01893
created_at: "2026-09-15"
---

# Excerpt: LongMIT (MIMG) — multi-agent multi-hop long-context instruction data

- **Authors:** Zhi Chen, Qiguang Chen, Libo Qin, Qipeng Guo, Haijun Lv, Yicheng Zou, et al. (Shanghai AI Laboratory; Harbin Institute of Technology; Central South University)
- **Year:** 2024 (arXiv v1 2024-09; v2 2025-05 used here)
- **Source type:** paper
- **Used in:** ch-29a §4, §6

## Problem measured (Abstract, §1)
With Self-Instruct over long documents using Qwen2-72B, "fewer than 35% of samples generated ... are multi-hop, and over 40% exhibit poor
quality"; the introduction states "high-quality examples representing only 60%".

## Framework (§2)
- Quality Verification Agent: scoring or classification verifiers with explicit verification conditions, applied throughout generation.
- Single-hop Question Generation Agent: generating questions before answers.
- Multiple Question Sampling: retrieval-based, intra-document, and inter-document sampling of single-hop questions.
- Multi-hop Question Merger Agent: merges the sampled single-hop questions into one multi-hop question.
- Contexts are then padded with additional documents to the target length (§4.1).

## Reported results
- MIMG yields "over 85% multi-hop, high-quality, and non-duplicative samples" (§1).
- Instruction tuning on the synthesized data "improves long-context QA capabilities across various LLMs, with an average gain of at least
  7.54%", with larger improvements on 2WikiMQA, MuSiQue, and HotpotQA (§4.2, Table 1).
- Input-token cost of the added agent interactions is about 3K tokens per sample against documents averaging 70K tokens (§4.1).

## Independent use
[[context-synthesis-short-to-long]] uses LongMIT data (64.4k samples) as a baseline: LLaMA3.1-8B + UltraChat 23.35 → 28.54 on an 8-task
LongBench average (Table 3 of that paper).
