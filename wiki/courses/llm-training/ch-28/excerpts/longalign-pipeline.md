---
chapter: ch-28
course: llm-training
phase: read
excerpt_of: Bai, Lv, Zhang et al. — "LongAlign: A Recipe for Long Context Alignment of Large Language Models"
source_url: https://aclanthology.org/2024.findings-emnlp.74/
created_at: "2026-04-23"
---

# Excerpt: LongAlign — the 5-question pick-one trick and packed-loss correction

**Source:** `wiki/raw-data/llm-training/papers/longalign.md`
**Paper:** Yushi Bai et al. (Tsinghua + Zhipu AI), 2024 — EMNLP Findings
**arXiv/ACL:** https://aclanthology.org/2024.findings-emnlp.74/

---

## Bibliographic header

> *"Long-context ability is not solved by context-window extension alone; you need dedicated long instruction data, length-aware SFT, and evaluation on realistic 10k-100k-token prompts."*

LongAlign is the first paper to fully frame long-context alignment as a distinct training stage — not a pre-training modification, not a position-encoding hack, but a *post-extension SFT problem* with its own data, batching, and loss-weighting requirements.

---

## The two-stage generation — cross-span coverage by construction

From the raw-data notes:

> *"Self-Instruct-style two-stage synthesis: (1) feed a long document + task-type prompt to Claude, ask for 5 candidate questions covering the whole text; (2) randomly choose one question and ask Claude for the answer."*

The trick is the **random choice**. If you let the teacher pick which question to answer, it picks the locally-answerable one — the question with the lowest synthesis cost for the teacher, which is the retrieval question, not the integration question. Randomly sampling from 5 candidates forces the teacher to occasionally commit to a question that requires reading far across the document. Cross-span coverage is **baked into the sampling, not the prompt**.

**Notice:** This is why LongAlign-10k beats the *larger* LongAlpaca-12k on multi-segment integration. It isn't the token count; it's the sampling gate.

The four task-prompt families used:

1. general questions
2. summarization / multi-part integration
3. multi-hop reasoning
4. information extraction

Nine document sources are sampled upstream (ArXiv, Books3, C4, CLUECorpus2020, CommonCrawl, GitHub, StackExchange, Wikipedia, WuDaoCorpora), with length-tail upsampling so the final dataset is not dominated by the short end of 8k–64k.

---

## Base-model extension *before* SFT

> *"Before SFT, the authors first extend all of them to 64k context: expand the RoPE base frequency by 200x, from 10,000 to 2,000,000; continually train on pretraining data up to 64k for 10B tokens."*

LongAlign is *post-extension* — it is not a substitute for long-context pretraining, it is the alignment layer that sits on top. The 200× RoPE base rescale (`10K → 2M`) is Llama-2-scale; for Llama-3 the equivalent is `500K → 128M` ([[excerpts/prolong-coherence]]). Both are the same NTK-aware trick, tuned to different base models.

---

## Packed loss is biased — the correction

> *"If each pack contributes equally to the batch loss, then packs with fewer sequences, usually the longest ones, get overweighted. Sequences with more target tokens also get overweighted."*

Walk the biases separately:

- **Pack-level bias.** A pack with 2 long sequences contributes the same to the batch-average loss as a pack with 12 short sequences — so the long-tail sequences get 6× their fair share of gradient.
- **Sequence-level bias.** Inside a pack, a sequence with 500 target tokens dominates the sequence with 50 target tokens in the token-average — so verbose targets are over-weighted.

The desired objective is **equal average contribution per sequence**, not per pack, not per target-token. LongAlign's fix is a weighted 1-D mask constructed at preprocessing:

```
for each token t:
    if t is a target token in sequence s with N target tokens total:
        weight[t] = 1 / N
    else:
        weight[t] = 0
```

During training, with `K` packs in the batch and `M` total sequences, token losses are scaled by `K / (M · N)`. Algebraically that recovers the per-sequence objective.

**Reported effect:**
- ChatGLM3-6B-64k on LongBench-Chat: **5.76 → 6.21**
- Llama-2-7B-64k on LongBench-Chat: **5.89 → 6.10**

The LongBench-overall gain is smaller because LongBench is mostly retrieval-like, but the chat-style gain is material.

---

## Packing vs sorted batching — the throughput ablation

Wall-clock on 8×A800 80G:

| Model | Naive | Packing | Sorted batching |
|---|---|---|---|
| ChatGLM3-6B-64k | 45.4 h | 20.5 h | 19.1 h |
| Llama-2-7B-64k | 67.2 h | 23.4 h | 23.3 h |
| Llama-2-13B-64k | 117.2 h | 41.2 h | 44.5 h |

Sorted batching often matches packing on throughput *without* the loss-weighting correction — because batches become length-homogeneous and the within-batch pad-rate is already low. Tradeoff: sorted-batching batches are *distributionally biased* across steps (all-long batch then all-short batch), which packing avoids at the cost of the loss-weight fix.

---

## LongBench-Chat — the eval companion

50 examples at 10k–100k tokens, four task categories (information extraction, multi-segment integration, multi-segment reasoning, full-text comprehension). 30 authored mimicking real user queries (20 EN, 10 ZH), 20 adapted from LooGLE. GPT-4 + few-shot grading, calibrated against humans.

This is the realistic-query complement to synthetic RULER — and the `5.76 → 6.21` numbers above are measured on this benchmark.

---

## Connections

- Chapter synthesis: [[ch-28]]
- Coherence-filter successor: [[excerpts/prolong-coherence]]
- Multi-turn extension: [[excerpts/longmit-multiturn]]
- Position-encoding lane: [[excerpts/longrope-per-dim-search]]
- Evaluation lineage: [[excerpts/ruler-task-family]]
