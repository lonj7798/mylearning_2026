---
chapter: ch-28
course: llm-training
phase: read
excerpt_of: Bai, Lv, Zhang et al. — "LongAlign: A Recipe for Long Context Alignment of Large Language Models"
source_url: https://aclanthology.org/2024.findings-emnlp.74/
created_at: "2026-04-23"
revised: "2026-09-15 (rewritten against the EMNLP 2024 Findings PDF and arXiv 2401.18058 v1)"
---

# Excerpt: LongAlign — data construction, mixing with short data, and packed-loss weighting

**Paper:** Yushi Bai, Xin Lv, Jiajie Zhang, Yuze He, Ji Qi, Lei Hou, Jie Tang, Yuxiao Dong, Juanzi Li. Findings of EMNLP 2024 (arXiv v1 2024-01).
**Library card:** [[longalign]]. Where the card and the paper differ, this excerpt follows the EMNLP PDF.

## Context extension before SFT (§4.1)

> "This involves expanding the base frequency b of the RoPE position encoding by 200 times (from 10,000 to 2,000,000) and continual training on pretraining data with lengths under 64k, for a total of 10 billion tokens." (§4.1)

Base models: ChatGLM3-6B, Llama-2-7B, Llama-2-13B, all extended to 64K before SFT.

## Data construction (§3.2; App. A)

- Documents from 9 sources: Arxiv, Books3, C4, CLUECorpus2020, CommonCrawl, Github, Stack Exchange, Wikipedia, WuDaoCorpora. "We sample articles with lengths under 64k ... Note that we upsample longer articles to ensure our dataset covers more long texts." (App. A)
- Generator: Claude 2.1. 10k instances, 10% Chinese, 8k-64k tokens measured with the ChatGLM tokenizer (§3.2).
- Four prompt types (general, summary, reasoning, information extraction). Example of the reasoning prompt: "Given the above text, please propose 5 English questions that require multi-hop reasoning, make sure they are diverse and cover all parts of the text" (App. A).
- Selection: "For each long article, we randomly select one of the four task prompts and have Claude generate five questions ... We then randomly choose one of these questions and request Claude for its answer" (App. A).
- Verification: "We recruit 4 Ph.D. students to manually check 100 randomly sampled data ... 94 have correct answers. Among the remaining data, 2 answers are incorrect, 3 answers are incomplete, and 1 answer is irrelevant" (App. A, EMNLP version).
- Shape of the data: ShareGPT short data has a target/sequence token ratio of 19.3 (percent) and 330 target tokens on average; LongAlign-10k has 0.015 and 200 target tokens (App. A).

## Mixing with short data (§4.1-§4.2, Table 3)

Short data: "the entire 76k ShareGPT data" (§4.1). Long data suites are added to it. ChatGLM3-6B-64k results:

| Long data added | LongBench-Chat | S-Doc QA | M-Doc QA | Summ | MT-Bench | ARC | HellaSwag | TruthfulQA | MMLU |
|---|---|---|---|---|---|---|---|---|---|
| LongAlign-0k | 3.73 | 58.7 | 41.1 | 38.4 | 5.34 | 50.3 | 74.7 | 51.6 | 45.5 |
| LongAlign-5k | 5.99 | 61.8 | 42.1 | 42.0 | 5.50 | 50.3 | 75.1 | 52.5 | 46.6 |
| LongAlign-10k | 6.28 | 64.0 | 44.4 | 44.2 | 5.51 | 50.5 | 74.9 | 52.5 | 45.5 |
| LongAlpaca-12k | 4.58 | 65.8 | 45.6 | 44.1 | 4.93 | 51.5 | 75.4 | 53.2 | 47.1 |

Authors' findings (§4.2): "as the amount of long instruction data increases, there is a consistent improvement in the model's performance across all long tasks. Meanwhile ... its performance on short tasks remains comparable". LongAlign-10k is better than LongAlpaca-12k on LongBench-Chat and MT-Bench; LongAlpaca-12k is slightly better on LongBench, which the authors attribute to 2WikiMQA and NarrativeQA being closer to LongAlpaca's sources (§4.2). The paper does not state that gains saturate at 10k.

## Packing and loss weighting (§3.3; App. B)

Equal weight per sequence (Eq. 2), with M sequences in the batch and L_i, N_i the summed loss and target-token count of sequence i:

```
L  = (1/M) Σ_{i=1..M} L_i / N_i
```

Packing averages per pack instead (Eq. 3), which favors sequences with more target tokens and sequences in packs with fewer sequences. The fix scales the loss of sequence i by K/(N_i M), where K is the number of packs, and sums over packs (Eq. 4). Implementation: "a weighted 1D mask for each pack ... the weight is set to 1/N ... the loss is calculated as the summation of the cross entropy loss at each token scaled by K/M N" (App. B). Packing uses `flash_attn_varlen_func` with `cu_seqlens_q` / `cu_seqlens_k` so each sequence attends only within itself (App. B).

Training: 8×A800 80G, 2 epochs, about 1500-2000 steps; a pack holds 12 sequences on average; batch size 8 packs gives global batch 96 (§4.1).

| Model | Naive batching | Sorted batching | Packing | Packing + loss weighting |
|---|---|---|---|---|
| ChatGLM3-6B-64k, LongBench-Chat | 5.87 | 5.40 | 5.76 | 6.21 |
| Llama-2-7B-64k, LongBench-Chat | 5.95 | 6.38 | 5.89 | 6.10 |

(Table 4.) Training time on 8×A800: ChatGLM3-6B-64k naive 45.4 h, packing 20.5 h, sorted batching 19.1 h (Fig. 5).

## LongBench-Chat (§3.4; App. C.1)

50 queries of 10k-100k tokens (40 English, 10 Chinese); 30 written by the authors to mimic user queries and 20 taken from LooGLE long-dependency QA; GPT-4 scores 1-10 with few-shot examples and a reference answer.

## Connections

- [[ch-28]] §2; packing and masking in ch-04; SFT mixture shares in ch-30b.
- [[prolong]] — reports that synthetic long SFT data did not help after 40B tokens of long continued training.
