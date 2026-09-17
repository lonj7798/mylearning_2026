---
chapter: ch-15
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/wildchat.md (library card not verified as of 2026-09-15; values below are taken from the primary source)
source_url: https://arxiv.org/abs/2405.01470
primary_version: arXiv:2405.01470v1 (2024-05-02); ICLR 2024
created_at: "2026-09-15"
---

# Excerpt: WildChat: 1M ChatGPT Interaction Logs in the Wild, statistics and coverage sections

Authors: Wenting Zhao, Xiang Ren, Jack Hessel, Claire Cardie, Yejin Choi, Yuntian Deng. Source type: paper. Read in the v1 PDF text on 2026-09-15 for ch-15 §4.

## Collection (§2)
- A chatbot service backed by the ChatGPT and GPT-4 APIs; users consented before access (§2, App. B).
- 1,054,528 linked conversations; 14,743 reserved for WildBench; 1,039,785 conversations (2,639,415 turns) released (§2).

## Statistics (Table 1; Llama-2 tokenizer; users estimated by unique IP)
| Dataset | #Convs | #Users | #Turns | #User tok | #Chatbot tok | #Langs |
|---|---|---|---|---|---|---|
| Alpaca | 52,002 | – | 1.00 | 19.67 | 64.51 | 1 |
| Open Assistant | 46,283 | 13,500 | 2.34 | 33.41 | 211.76 | 11 |
| Dolly | 15,011 | – | 1.00 | 110.25 | 91.14 | 1 |
| ShareGPT | 94,145 | – | 3.51 | 94.46 | 348.45 | 41 |
| LMSYS-Chat-1M | 1,000,000 | 210,479 | 2.02 | 69.83 | 215.71 | 65 |
| WildChat | 1,039,785 | 204,736 | 2.54 | 295.58 | 441.34 | 68 |
- About 41% of conversations have multiple turns; 3.7% exceed 10 turns (§3).
- English 53% of turns, Chinese 13%, Russian 12% (§3, Fig. 2b).
- Table 4 (1,000 sampled English first turns): assisting/creative writing 61.9%, analysis/decision explanation 13.6%, coding 6.7%, factual info 6.3%, math reason 6.1%.
- Toxicity: 10.46% of user turns and 6.58% of chatbot turns flagged by Detoxify or the OpenAI Moderation API (§4).

## Coverage (Fig. 3)
Llama-2 7B finetuned on first-turn user prompts of one dataset (70% train, 30% validation) and evaluated by average NLL on the others. Rows = trained on, columns = evaluated on (Alpaca, Dolly, Open Assistant, ShareGPT, WildChat):
| Trained on | Alpaca | Dolly | OA | ShareGPT | WildChat |
|---|---|---|---|---|---|
| Alpaca | 2.24 | 3.66 | 8.42 | 11.11 | 10.87 |
| Dolly | 3.17 | 2.74 | 6.04 | 8.8 | 8.77 |
| Open Assistant | 3.16 | 3.14 | 3.09 | 7.1 | 6.57 |
| ShareGPT | 3.12 | 3.14 | 3.8 | 5.23 | 7.61 |
| WildChat | 3.28 | 3.29 | 3.28 | 5.91 | 5.18 |
(Values read from the heatmap labels in the PDF text layer.)

## Instruction tuning (§5, Table 9)
WildLlama: Llama-2 7B on WildChat collected up to July 16, 2023; Vicuna hyperparameters; effective batch 128 conversations; LR 2e-5; max length 2048; 3 epochs. MT-bench average (GPT-4 judge): WildLlama 6.35, Llama-2 Chat 6.26, Vicuna 6.13, GPT-3.5 7.94, GPT-4 8.99.
