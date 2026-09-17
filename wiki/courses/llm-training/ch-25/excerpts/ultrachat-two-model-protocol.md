---
chapter: ch-25
course: llm-training
phase: read
excerpt_of: Ding et al. — "Enhancing Chat Language Models by Scaling High-quality Instructional Conversations" and the thunlp/UltraChat README (library cards [[ultrachat-pipeline]] and [[ultrachat-construction]] are not verified; quotations below are taken from the primary sources)
source_url: https://arxiv.org/abs/2305.14233 ; https://github.com/thunlp/UltraChat (README on main, fetched 2026-09-15)
primary_version: arXiv:2305.14233v1 (2023-05)
created_at: "2026-04-23"
revised_at: "2026-09-15 (rewritten from the primary sources for the 2026-09 revision of read.md)"
---

# Excerpt: UltraChat — sector construction, user simulation, and statistics

Authors: Ning Ding, Yulin Chen, Bokai Xu, Yujia Qin, Zhi Zheng, Shengding Hu, et al. (Tsinghua University). Used by ch-25 `read.md` §1, §3, and the Recipe.

## Paper vs README numbers
| Item | Paper (arXiv v1 §4.1–4.3) | README |
|---|---|---|
| Sector I subtopics | "30 to 50 subtopics or related concepts" per topic | "1100+ subtopics" |
| Sector I questions | "10 different questions for each subtopic ... 10 more questions based on each original question"; "approximately 500,000 questions as opening lines" | "up to 10 specific questions" per subtopic; entity branch "200k specific questions and 250k general questions along with the 50k meta-questions" |
| Sector II instructions | instructions for 20 types; "approximately 80% ... fed back ... to generate more detailed instructions" | "200 different instructions" per type; "80% of the instructions are further expanded" |
| Sector III materials | "we collect 10,000 text pieces from the C4 corpus ... five distinct instructions ... the concatenated set of 500,000 pieces serves as the opening lines" | "~10w diverse materials from C4" (10w = 100,000); "up to 5 questions/instructions" |
| Rounds | Fig. 1: "3~7 rounds of generation" (Sector I) | Sector I "3~7-round"; Sectors II and III "2~4-round" |

The paper's Sector III numbers are internally inconsistent (10,000 × 5 = 50,000, not 500,000). Per-sector dialogue counts are not reported in the paper.

## Design principle (§4)
> "While the core to ensuring data diversity is to ensure the diversity of opening lines and user response style, this section will mainly focus on the construction and design of how to obtain a diverse set of opening lines and how to prompt the user properly."

README: "The general idea of UltraChat is to use separate LLMs to generate opening lines, simulate users and respond to queries."

## Sector I user prompt (§4.1)
> "we provide the user model with carefully crafted prompts that explicitly ask the model to respond concisely and meaningfully, taking into account the context of the ongoing dialogue history."

## Sector III templates (Table 4)
```
{text}\n{instruction}
{text} {instruction}
{instruction} Answer according to: {text}
{text} Based on the passage above, {instruction}
{instruction}: {text}
Given the text: {text}\n{instruction}
{instruction}\nGenerate according to: {text}
```

## User simulation and refinement (§4.4)
> "It has been observed that when the user model is solely provided with the current dialogue history, it tends to assume the role of an AI assistant. This 'role exchange' situation can significantly impact the coherence of the multi-turn conversation. To address this, in addition to presenting the dialogue history, we include prompts explicitly instructing the model to adopt various user personalities."

> "To enhance the realism of user responses, we specifically exclude excessively polite statements such as 'Thank you,' 'Thanks,' and the 'You're welcome' response in the subsequent model-generated output."

## Statistics (Table 5)
| Dataset | #Dialogue | Avg. #Turns | Avg. Dialog Length (tokens) | Avg. Utt. Length (tokens) | Lexical Diversity | Topic Diversity (↓) | Coherence | User Simulation |
|---|---|---|---|---|---|---|---|---|
| SODA | 1,486,869 | 3.6 | 231.8 | 22.5 | 38.6 | 0.797 | 8.48 | No |
| Baize | 210,311 | 3.1 | 293.9 | 52.8 | 67.1 | 0.751 | 9.06 | Yes |
| UltraChat | 1,468,352 | 3.8 | 1467.4 | 309.3 | 74.3 | 0.702 | 9.06 | Yes |

Caption: lexical diversity is MTLD averaged per utterance; topic diversity is average pairwise cosine distance of OpenAI embeddings on 10,000 samples; coherence is scored by ChatGPT from 1 to 10.

## UltraLLaMA (§4.6)
> "we break down each dialogue into smaller sequences, limiting them to a maximum length of 2048 tokens. During the training process, we only calculate the loss for the model's responses. ... The model is trained with 128 A100 GPUs and the total batch size is 512."
