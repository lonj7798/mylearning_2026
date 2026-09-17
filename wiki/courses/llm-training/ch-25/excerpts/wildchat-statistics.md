---
chapter: ch-25
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/wildchat.md (the card has no statistics or Verification section; quotations below are taken from the primary source)
source_url: https://arxiv.org/abs/2405.01470
primary_version: arXiv:2405.01470v1 (2024-05; ICLR 2024 per the PDF header)
created_at: "2026-09-15"
---

# Excerpt: WildChat: 1M ChatGPT Interaction Logs in the Wild — statistics, coverage, and fine-tuning

Authors: Wenting Zhao, Xiang Ren, Jack Hessel, Claire Cardie, Yejin Choi, Yuntian Deng (Cornell; AI2; USC; UW). Checked against the v1 PDF on 2026-09-15. Used by ch-25 `read.md` §1, §6, §8, and the Recipe.

## Collection (Abstract, §2)
> "we offered free access to ChatGPT for online users in exchange for their affirmative, consensual opt-in to anonymously collect their chat transcripts and request headers."

> "This linking process yielded 1,054,528 full conversations. Out of these conversations, 14,743 conversations are reserved for the WildBench benchmark (Lin et al., 2024), resulting in 1,039,785 conversations (2,639,415 turns) in the publicly released version."

## Table 1 (Llama-2 tokenizer; users estimated by unique IP addresses)
| Dataset | #Convs | #Users | #Turns | #User Tok | #Chatbot Tok | #Langs |
|---|---|---|---|---|---|---|
| Alpaca | 52,002 | – | 1.00 | 19.67±15.19 | 64.51±64.85 | 1 |
| Open Assistant | 46,283 | 13,500 | 2.34 | 33.41±69.89 | 211.76±246.71 | 11 |
| Dolly | 15,011 | – | 1.00 | 110.25±261.14 | 91.14±149.15 | 1 |
| ShareGPT | 94,145 | – | 3.51 | 94.46±626.39 | 348.45±269.93 | 41 |
| LMSYS-Chat-1M | 1,000,000 | 210,479 | 2.02 | 69.83±143.49 | 215.71±1858.09 | 65 |
| WildChat | 1,039,785 | 204,736 | 2.54 | 295.58±1609.18 | 441.34±410.91 | 68 |

## Turn distribution and languages (§3)
> "On average, each conversation includes 2.52 user-chatbot interaction rounds (turns). ... approximately 41% of conversations contain multiple turns. While most conversations have fewer than 10 turns, the distribution exhibits a long tail, with 3.7% of conversations extending beyond 10 turns."

> "English being the most prevalent, accounting for 53% of the turns, followed by Chinese and Russian, which constitute 13% and 12% of the dataset, respectively."

(§3 prose also gives 1,009,245 conversations and 2.52 turns, while Table 1 gives 1,039,785 and 2.54.)

## Coverage (§3, Fig. 3, Fig. 4)
> "we fintuned a Llama-2 7B model on each dataset and then used it to measure how likely other datasets are. If a dataset 'covers' another, then we expect the model trained on this dataset to be able to 'explain' data from the other dataset, resulting in a lower negative log-likelihood (NLL)."

Fig. 3 average NLL (rows: trained on; columns: evaluated on Alpaca, Dolly, Open Assistant, ShareGPT, WildChat; first-turn user prompts only, 70/30 split):

| Trained on | Alpaca | Dolly | Open Assistant | ShareGPT | WildChat |
|---|---|---|---|---|---|
| Alpaca | 2.24 | 3.66 | 8.42 | 11.11 | 10.87 |
| Dolly | 3.17 | 2.74 | 6.04 | 8.8 | 8.77 |
| Open Assistant | 3.16 | 3.14 | 3.09 | 7.1 | 6.57 |
| ShareGPT | 3.12 | 3.14 | 3.8 | 5.23 | 7.61 |
| WildChat | 3.28 | 3.29 | 3.28 | 5.91 | 5.18 |

(Column order read from the rotated heatmap labels; each row's minimum falls on its own dataset, which confirms the order.)

> "We embedded 10,000 first-turn user prompts from each dataset using OpenAI's embedding model (text-embedding-ada-002). ... WildChat exhibits close to perfect overlap with other datasets but also covers additional areas, further confirming its diversity."

## Toxicity (§4)
> "10.46% of user turns and 6.58% of chatbot turns are deemed toxic by either Detoxify or Moderation."

## WildLlama (§5, Table 9)
> "we used WildChat collected up until July 16, 2023. ... an effective batch size of 128 conversations, a learning rate of 2e-5, and a maximum sequence length of 2048 tokens. ... We fine-tuned WildLlama for three epochs."

MT-bench average (GPT-4 judge): GPT-3.5 7.94, GPT-4 8.99, Vicuna 6.13, Llama-2 Chat 6.26, WildLlama 6.35.
