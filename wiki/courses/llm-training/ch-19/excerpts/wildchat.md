---
chapter: ch-19
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/wildchat.md (the library card has no Verification section and no numbers; values below are taken from the primary source)
source_url: https://arxiv.org/abs/2405.01470
primary_version: arXiv:2405.01470v1 (2024-05, ICLR 2024)
created_at: "2026-09-15"
---

# Excerpt: WildChat: 1M ChatGPT Interaction Logs in the Wild

Authors: Wenting Zhao, Xiang Ren, Jack Hessel, Claire Cardie, Yejin Choi, Yuntian Deng (Cornell University; Allen Institute for AI; USC; University of Washington). Checked against the v1 PDF on 2026-09-15. Used by ch-19 `read.md` §11.

## Collection (Abstract, §2)
> "we offered free access to ChatGPT for online users in exchange for their affirmative, consensual opt-in to anonymously collect their chat transcripts and request headers. From this, we compiled WildChat, a corpus of 1 million user-ChatGPT conversations, which consists of over 2.5 million interaction turns."

1,054,528 linked conversations; 14,743 reserved for WildBench; 1,039,785 conversations (2,639,415 turns) released (§2). About 24% of conversations use the GPT-4-based API and 76% the GPT-3.5-Turbo-based API (§3, Table 2).

## Table 1 (Llama-2 tokenizer; users estimated by unique IP)
| Dataset | #Convs | #Users | #Turns | #User Tok | #Chatbot Tok | #Langs |
|---|---|---|---|---|---|---|
| Alpaca | 52,002 | - | 1.00 | 19.67±15.19 | 64.51±64.85 | 1 |
| Open Assistant | 46,283 | 13,500 | 2.34 | 33.41±69.89 | 211.76±246.71 | 11 |
| Dolly | 15,011 | - | 1.00 | 110.25±261.14 | 91.14±149.15 | 1 |
| ShareGPT | 94,145 | - | 3.51 | 94.46±626.39 | 348.45±269.93 | 41 |
| LMSYS-Chat-1M | 1,000,000 | 210,479 | 2.02 | 69.83±143.49 | 215.71±1858.09 | 65 |
| WildChat | 1,039,785 | 204,736 | 2.54 | 295.58±1609.18 | 441.34±410.91 | 68 |

## Prompt categories (Table 4; first turn of English conversations)
assisting/creative writing 61.9%; analysis/decision explanation 13.6%; coding 6.7%; factual info 6.3%; math reason 6.1%.

## Coverage (§3, Fig. 3)
> "we fintuned a Llama-2 7B model on each dataset and then used it to measure how likely other datasets are. If a dataset 'covers' another, then we expect the model trained on this dataset to be able to 'explain' data from the other dataset, resulting in a lower negative log-likelihood (NLL)."

Caption: 70% of data for training and 30% for validation; only first-turn user prompts are used. Average NLL (rows: trained on; columns: evaluated on):

| Trained on \ Evaluated on | Alpaca | Dolly | Open Assistant | ShareGPT | WildChat |
|---|---|---|---|---|---|
| Alpaca | 2.24 | 3.66 | 8.42 | 11.11 | 10.87 |
| Dolly | 3.17 | 2.74 | 6.04 | 8.80 | 8.77 |
| Open Assistant | 3.16 | 3.14 | 3.09 | 7.10 | 6.57 |
| ShareGPT | 3.12 | 3.14 | 3.80 | 5.23 | 7.61 |
| WildChat | 3.28 | 3.29 | 3.28 | 5.91 | 5.18 |

Embedding check: 10,000 first-turn prompts per dataset embedded with text-embedding-ada-002 and projected with t-SNE; "WildChat exhibits close to perfect overlap with other datasets but also covers additional areas" (§3, Fig. 4).

## Instruction tuning (§5, Table 9)
WildLlama: Llama-2 7B fine-tuned on WildChat collected up to July 16, 2023, with the Vicuna implementation and hyperparameters: four A100 80G GPUs, effective batch size 128 conversations, learning rate 2e-5, maximum sequence length 2048 (longer conversations split), three epochs.

| Model | MT-bench first turn | second turn | average |
|---|---|---|---|
| GPT-3.5 | 8.06 | 7.81 | 7.94 |
| GPT-4 | 8.96 | 9.03 | 8.99 |
| Vicuna | 6.68 | 5.57 | 6.13 |
| Llama-2 Chat | 6.41 | 6.12 | 6.26 |
| WildLlama | 6.80 | 5.90 | 6.35 |

## Toxicity (§4)
Over 10% of interactions are flagged as toxic by Detoxify or the OpenAI Moderation API (§1, §4); sexual content is 88.51% of toxic user turns under Moderation (§4).
