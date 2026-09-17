---
chapter: ch-19
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/rephrasing-the-web.md (the library card has no Verification section; values below are taken from the primary source)
source_url: https://aclanthology.org/2024.acl-long.757/ ; https://arxiv.org/abs/2401.16380
primary_version: ACL 2024 long paper (pages 14044-14072); arXiv:2401.16380v1 (2024-01) read for version differences
created_at: "2026-04-23"
revised_at: "2026-09-15 (generality revision; rewritten from the primary source)"
---

# Excerpt: Rephrasing the Web: A Recipe for Compute and Data-Efficient Language Modeling (WRAP)

Authors: Pratyush Maini, Skyler Seto, He Bai, David Grangier, Yizhe Zhang, Navdeep Jaitly. Checked on 2026-09-15. Used by ch-19 `read.md` §7.

## Headline claims and a version difference
- ACL abstract: WRAP on C4 "speeds up pre-training by ∼ 3×. At the same pre-training compute budget, it improves perplexity by more than 50% on average across different subsets of the Pile, and improves zero-shot question answer accuracy across 13 tasks by more than 2%."
- arXiv v1 abstract: the same sentence reads "improves perplexity by more than 10% on average". The v1 body (§4) and the ACL body both state "our models improve perplexity by 50% over models trained on real data alone."
- §1: "train equivalent models with 5x lesser data, or 3x lesser compute"; "our 350M parameter model trained on combinations of real and synthetic rephrases on just 15% of the entire C4 corpus, outperforms pre-training a 1.3B parameter on the entire C4."

## Method (§3.1)
- Styles: "(i) Easy (text that even a toddler will understand); (ii) Medium (in high quality English such as that found on Wikipedia); (iii) Hard (in terse and abstruse language); (iv) Q/A (in conversation question-answering format)."
- Rephraser: "a frozen Mistral-7B instruction-tuned model". Medium prompt: "For the following paragraph give me a paraphrase of the same in high-quality English language as in sentences on Wikipedia".
- Chunking: "Each example is chunked into sequences of nearly 300 tokens using the NLTK sentence splitter. This was based on our empirical observation that asking an LLM to rephrase more than 300 tokens often led to a loss of information."
- Mixing: "we sample real and synthetic data in a 1:1 ratio. Note that this means the same raw data is seen twice, once in its original form, and once as its rephrase."
- Fig. 1 caption: "Results in (b,c) use conversation Q/A style rephrasing."

## Training (§3.2)
Decoder-only models at 128M (12 layers, 12 heads, d 768), 350M (24, 16, 1024), 1.3B (24, 16, 2048); max sequence length 1024; no dropout; XL models 300k steps at batch size one million tokens; peak LR 3e-4 (128M, 350M) and 2e-4 (1.3B); minimum LR 1e-5; weight decay 0.01; gradient clipping 1.0; cosine schedule with 1% warmup; Adam β1 0.9, β2 0.999.

## Table 1 (1.3B models; "Real Tok." = web tokens available; WRAP rows averaged over 3 runs)
| Dataset (Real Tok.) | ARC-E | BoolQ | Wino. | PIQA | HellaSwag | TruthfulQA | OBQA | LogiQA | Avg |
|---|---|---|---|---|---|---|---|---|---|
| Half C4 (85B) | 53.2 | 59.1 | 57.3 | 76.0 | 61.0 | 34.1 | 34.2 | 26.6 | 50.2 |
| Full C4 (170B) | 54.6 | 54.2 | 59.0 | 76.1 | 61.2 | 33.5 | 36.8 | 26.9 | 50.3 |
| TinyLlama (1T) | 55.3 | 57.8 | 59.1 | 73.3 | 59.2 | 37.6 | 36.0 | 27.0 | 50.7 |
| Synthetic (85B) | 57.7 | 60.0 | 58.8 | 76.9 | 57.8 | 44.0 | 34.2 | 26.3 | 52.0 |
| Synthetic+C4 (85B) | 57.4 | 62.2 | 58.9 | 76.0 | 60.8 | 40.6 | 35.3 | 27.1 | 52.3 |

| Dataset (Real Tok.) | ARC-C | SciQ | PubMedQA | MathQA | MMLU | Avg |
|---|---|---|---|---|---|---|
| Half C4 (85B) | 29.5 | 76.8 | 57.2 | 22.9 | 24.2 | 42.1 |
| Full C4 (170B) | 29.7 | 77.3 | 57.4 | 23.8 | 23.9 | 42.4 |
| TinyLlama (1T) | 30.1 | 81.8 | 61.4 | 23.7 | 25.8 | 44.6 |
| Synthetic (85B) | 32.3 | 78.4 | 60.2 | 23.2 | 24.6 | 43.7 |
| Synthetic+C4 (85B) | 31.5 | 79.0 | 61.5 | 23.5 | 24.8 | 44.0 |

§5.2: "synthetic data can not impart 'new knowledge'. It can only help pre-train faster". The text gives 40.5% for Synthetic+C4 on TruthfulQA; Table 1 prints 40.6.

## Real data and style (§7)
- RQ2: "synthetic data using the QA prompt are sufficient for strong performance on QA tasks. However, when evaluated on Pile perplexity, we observe significant degradation in perplexity across many sub-domains in Figure 4." Table 2 (1.3B, 150B tokens, 35B real tokens), general average: Med+C4 49.3, QA+C4 51.1, Med 47.9, QA 51.0.
- RQ1: rephrasers T5-base (fine-tuned), Qwen-1.8B-chat, Mistral-7B-chat, Vicuna-13B-chat-v1.3 at 345M / 30B tokens; "all rephrase models reduce perplexity over only real C4 data."
- RQ3 (leakage): cosine similarity of SimCSE sentence embeddings for real–synthetic pairs is higher than for two random C4 samples and other baselines; App. C adds a comparison with MRPC real paraphrases.

## Cleanup and cost
- App. B: a rule-based post-process removes leading segments with unwanted elements; "the error rate (occurrence of sentences with unwanted elements) after the modification is less than 0.1%."
- §9.1: "we are able to generate 3M tokens per hour on a single A100 when using the Mistral-7B. Generating 85B tokens (as in our work) accounts for about 25K GPU hours."
