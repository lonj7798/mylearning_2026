---
chapter: ch-25
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/lmsys-chat-1m.md (library card not present on 2026-09-15; quotations below are taken from the primary source)
source_url: https://arxiv.org/abs/2309.11998
primary_version: arXiv:2309.11998v4 (v1 2023-09; v4 2024-03; ICLR 2024 per the PDF header)
created_at: "2026-09-15"
---

# Excerpt: LMSYS-Chat-1M: A Large-Scale Real-World LLM Conversation Dataset

Authors: Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, et al. (UC Berkeley; UC San Diego; CMU; and others). Checked against the v4 PDF on 2026-09-15. Used by ch-25 `read.md` §8 and the Core insight.

## Collection (§2)
> "LMSYS-Chat-1M is collected on our website from April to August 2023. The website offers three types of chat interfaces: Single model, Chatbot Arena (battle), and Chatbot Arena (side-by-side)."

> "The dataset contains raw conversation text without any processing. To ensure the safe release of data, we have made our best efforts to remove conversations that contain personally identifiable information (PII). In addition, we have included the OpenAI moderation API output for each message."

## Statistics (Table 1; tokens counted with the Llama 2 tokenizer)
| Dataset | # Convs | # Models | # Users | # Langs | Avg. # Turns per Sample | Avg. # Tokens per Prompt | Avg. # Tokens per Response | Human Preference |
|---|---|---|---|---|---|---|---|---|
| Anthropic HH | 338,704 | 1 | 143 | 1 | 2.3 | 18.9 | 78.9 | Yes |
| OpenAssistant | 66,497 | - | 13,500 | 35 | - | 36.9 | 214.2 | Yes |
| Chatbot Arena | 33,000 | 20 | 13,383 | 96 | 1.2 | 52.3 | 189.5 | Yes |
| LMSYS-Chat-1M | 1,000,000 | 25 | 210,479 | 154 | 2.0 | 69.5 | 214.5 | No |

- Top five models by conversation count: Vicuna, Koala, Alpaca, ChatGLM, Llama; Vicuna is the site default (§3.1).

## Topic distribution (§3.2)
> "From 100K randomly sampled English conversations, we extract user prompts, which include both the initial and follow-up turns. We remove prompts that are either too short (fewer than 32 characters) or too long (more than 1536 characters)."

> "The majority of questions are related to coding and software (Clusters 1, 2, 6, 16, 18)." (k-means with 20 clusters on all-mpnet-base-v2 embeddings)

## Benchmark prompts from real traffic (§4.4)
- GPT-3.5-Turbo scores each prompt from 1 to 10 for its potential to test problem-solving, creativity, and truthfulness.
- > "we find GPT-4 wins 52% in Top-50 but only 22% in Bottom-50 against GPT-3.5-turbo" (the prose cites "Table 5"; the numbers appear in Fig. 5, "GPT-4 vs GPT-3.5 on top-50 and bottom-50 benchmark", while Table 5 in the PDF is a jailbreak table)
- > "we identified the 200 most challenging prompts that get 9+ score agreed by GPT-3.5-Turbo, Claude-2, and GPT-4." These form Arena-Hard-200, which "reveals larger performance gaps between open and proprietary models (e.g., GPT-4, Claude) than MT-Bench" (Fig. 6).
