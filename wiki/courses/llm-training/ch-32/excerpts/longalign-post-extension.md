---
chapter: ch-32
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/longalign.md
source_url: https://aclanthology.org/2024.findings-emnlp.74/
created_at: "2026-04-23"
---

# Excerpt: LongAlign - post-extension alignment as a distinct stage

**Source library:** `wiki/raw-data/llm-training/papers/longalign.md`
**Artifact:** LongAlign-10k pick-one-of-5 synthesis + sequence-level loss weighting + LongBench-Chat

---

## Why this source anchors ch-32

LongAlign is the paper that formalizes **long-context SFT as Job 3** in ch-32's three-job decomposition. Its core claim - that context extension is necessary but insufficient, and that the model still needs dedicated long-instruction alignment - is what forces the ch-32 split between Job 1 + Job 2 (long-context mid-training stage) and Job 3 (SFT stage with a small long sub-mix).

Ch-32's "Job 3 lives in SFT" rule, its "mix them up and you get either a model that has the window but can't reason across it, or a chat model that degrades on short inputs" warning, and its reference to cross-span coverage as the key synthesis trick are all transcribed from LongAlign.

---

## The attested pipeline ch-32 transcribes

From the source (lines 42-69):

- **Size:** 10,000 supervised examples.
- **Length range:** 8k-64k tokens (ChatGLM tokenizer).
- **Language mix:** ~90% English, ~10% Chinese.
- **Seed document sources (9):** ArXiv, Books3, C4, CLUECorpus2020, CommonCrawl, GitHub, StackExchange, Wikipedia, WuDaoCorpora.
- **Teacher model:** Claude 2.1.
- **Generation pattern:** two-stage Self-Instruct-style:
  1. Generate **5 candidate questions** covering the whole document.
  2. Randomly pick one, ask Claude for the answer.
- **Verification:** 4 PhD students checked 100 samples; 94/100 correct.

The **pick-one-of-5** trick is what ch-32 names as "cross-span coverage." Without it, the teacher picks a locally-answerable question from the first few thousand tokens and the model learns long retrieval but not long reasoning. The pick-one-of-5 trick is tiny in implementation and load-bearing in effect.

---

## The post-extension positioning ch-32 inherits

From the source (lines 83-91):

- Base models studied: ChatGLM3-6B, Llama-2-7B, Llama-2-13B.
- Before SFT, all extended to 64k context:
  - Expand RoPE base frequency 200x, from 10,000 to 2,000,000.
  - Continually train on pretraining data up to 64k for 10B tokens.
- LongAlign is a recipe for **post-extension alignment**, not a substitute for long-context pretraining.

Ch-32 uses this to underline the sequencing rule: Job 1 + Job 2 (context extension, long-doc CPT) must happen **before** Job 3 (long SFT), otherwise there is no long window to align in the first place. The 200x RoPE-base expansion is the Job 1 handoff; the 10B-token continued-pretraining is the Job 2 handoff; LongAlign-10k is the Job 3 handoff.

---

## The packed-loss correction ch-32 references

From the source (lines 106-122):

- Naive packed loss is biased: packs with fewer sequences get overweighted, and sequences with more target tokens get overweighted.
- Fix: build a weighted 1D mask where target-token positions for a sequence get weight 1/N (N = number of target tokens for that sequence).
- Scale token losses by K/(M*N) where K packs, M sequences.
- Effect on LongBench-Chat:
  - ChatGLM3-6B-64k: 5.76 -> 6.21.
  - Llama-2-7B-64k: 5.89 -> 6.10.

Ch-32 cites the correction as evidence that long-SFT is an engineering problem as well as a data problem. The throughput numbers (on 8xA800 80G, naive 45-117h; packed 20-41h) show batching strategy changes whether the recipe is usable at all.

---

## The data-diversity finding ch-32 quotes

From the source (lines 155-162):

- Long-task performance improves up to ~10k long examples and saturates.
- LongAlign-10k beats the larger LongAlpaca-12k on multi-segment integration.
- Model scaling still helps: 13B > 7B.

Ch-32 uses the 10k saturation point to anchor its claim that long-SFT data is a *diversity* problem past 10k, not a volume problem. This is consistent with [[front-loading-reasoning]]'s diversity-vs-quality rule: in the SFT stage quality dominates, but the long-instruction sub-mix within SFT saturates on volume around 10k examples.

---

## The LongBench-Chat benchmark ch-32 references

From the source (lines 139-154):

- 50 examples, 10k-100k input length.
- 30 authored to mimic real user queries, 20 adapted from LooGLE.
- Four task categories: Information Extraction, Multi-segment Integration, Multi-segment Reasoning, Full-text Comprehension.
- GPT-4 scores 1-10 with few-shot examples; better correlated with humans than F1 / ROUGE-L.

Ch-32 references LongBench-Chat as the **realistic long-SFT eval gate** - the benchmark that reveals whether the long-SFT data actually installs long instruction-following rather than just long retrieval. It complements RULER (retrieval) and BABILong (reasoning) in ch-32's per-stage eval-gate list.

---

## Connections

- **[[prolong]]** - Stage 2 (SFT) of ProLong is a LongAlign-style pipeline.
- **[[long-context-llama3]]** - Meta's 0.1% long-SFT rule sits at a different operating point than LongAlign's 10k dedicated mix; the trade-off is short-context regression vs long-capability depth.
- **[[longalpaca]]** - the earlier, weaker predecessor; LongAlign beats it on multi-segment integration.
- **[[ruler]]** / **[[babilong]]** - retrieval and reasoning evals that complement LongBench-Chat.
- **[[olmo-3]]** / **[[tulu-3]]** - generalize this lesson to full-pipeline context: long-context capability needs a dedicated stage, dedicated data, and dedicated eval.
