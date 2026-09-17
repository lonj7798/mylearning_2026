---
chapter: ch-25
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/mt-eval.md (library card not present on 2026-09-15; quotations below are taken from the primary source)
source_url: https://arxiv.org/abs/2401.16745
primary_version: arXiv:2401.16745v1 (2024-01)
created_at: "2026-09-15"
---

# Excerpt: MT-Eval: A Multi-Turn Capabilities Evaluation Benchmark for Large Language Models

Authors: Wai-Chung Kwan, Xingshan Zeng, Yuxin Jiang, Yufei Wang, Liangyou Li, Lifeng Shang, et al. (CUHK; Huawei Noah's Ark Lab; HKUST). Checked against the v1 PDF on 2026-09-15. Used by ch-25 `read.md` §9 and the long-context section.

## Design (Abstract, §1, §3)
> "By analyzing human-LLM conversations, we categorize interaction patterns into four types: recollection, expansion, refinement, and follow-up."

- Patterns derived from LMSYS-Chat-1M (§1). Recollection: recall information from earlier turns; Expansion: varied topics within one subject; Refinement: clarify or revise the initial instruction; Follow-up: questions about the assistant's previous response.
- New documents and instances are generated with GPT-4 and manually reviewed to avoid contamination (§1, §3). Refinement adds one constraint per turn; Expansion asks a different task about the same document; Recollection has a document-classification subtask and a global-instruction subtask; Follow-up extends MT-Bench by three turns (§3).
- Single-turn versions are built for every turn except Follow-up (§3).
- Table 1: 168 dialogues, 1,170 turns, 6.96 average turns per dialogue, 60.63 average words per turn.
- Responses scored by GPT-4 (1–10), except classification accuracy normalized to 10 (§4.3). Greedy decoding (§4.2).

## Single-turn vs multi-turn (Table 3, ST Avg. / MT Avg.)
| Model | ST | MT (change) |
|---|---|---|
| GPT-3.5-Turbo | 8.07 | 7.23 (−0.84) |
| GPT-4 | 9.17 | 8.84 (−0.33) |
| ChatGLM3-6B | 5.71 | 4.52 (−1.19) |
| Vicuna-7B-v1.5 | 6.31 | 5.82 (−0.49) |
| Vicuna-13B-v1.5 | 7.10 | 6.45 (−0.65) |
| Llama-2-chat-7B | 7.21 | 5.31 (−1.90) |
| Llama-2-chat-13B | 7.55 | 5.47 (−2.08) |
| Qwen-chat-7B | 6.86 | 5.91 (−0.95) |
| Qwen-chat-14B | 7.62 | 6.64 (−0.98) |
| Mistral-Instruct-7B | 7.69 | 6.93 (−0.76) |
| Mixtral-Instruct-8x7B | 8.28 | 6.78 (−1.50) |

> "the observed gap between the two scenarios does not appear to be directly correlated with the fundamental capabilities of the models. For instance, while Llama-2-chat models outperform Vicuna models in the single-turn setting, they noticeably lag behind in multi-turn dialogues." (§4.4)

## Distance and error propagation (§4.4–4.6)
> "In the Recollection task, all LLMs except GPT-4 struggle to adhere to the initial global instructions as the conversation length, i.e., distance from their initial instruction, increases. Table 4 also supports this trend, revealing that most models perform better on the first task (i.e., the first six turns) compared to the second (i.e., the final six turns)" (§4.4; Table 4 reports the Refinement task)

- Error analysis: "Noncompliance with Earlier Instructions (49.5%)", "Error Propagation (48%)", "Evaluation (2.5%)" (§4.5).
- > "models conditioned on gold context exhibit significant improvement in Recollection and Refinement tasks." (§4.6, Table 7)
- GPT-4 judge vs 5 human annotators on 180 responses: average Spearman correlation 0.58 (§4.5, Table 5).
