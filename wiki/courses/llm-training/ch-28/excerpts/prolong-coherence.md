---
chapter: ch-28
course: llm-training
phase: read
excerpt_of: Gao, Wettig, Yen, Chen — "How to Train Long-Context Language Models (Effectively)"
source_url: https://arxiv.org/abs/2410.02660
created_at: "2026-04-23"
---

# Excerpt: ProLong — the document-coherence filter and the 20B-token budget

**Source:** `wiki/raw-data/llm-training/papers/prolong.md`
**Paper:** Tianyu Gao, Alexander Wettig, Howard Yen, Danqi Chen (Princeton NLP), 2024
**arXiv:** https://arxiv.org/abs/2410.02660

---

## Bibliographic header

> *"Effective long-context training requires (a) selecting long documents that are genuinely coherent end-to-end (not concatenations of unrelated short texts), (b) upsampling code repositories and textbooks because they have real long-range dependencies, and (c) running long-context continued pretraining at a specific, small token budget."*

ProLong is the paper that resolved the "does volume or quality dominate?" debate for long-context continued pretraining. The answer is **quality**, and the data-coherence filter is the reason.

---

## The coherence filter — what counts as a "long document"

From the raw-data notes:

> *"Filter by document length: keep only documents with ≥ 64K tokens of coherent content. Filter by coherence signal: code = entire repository concatenated in sensible order (README → source → tests); Books = full-book PDFs; Academic = full papers with references; Web = discard."*

The web discard is the paper's sting. It is tempting — and every prior recipe did this — to treat long web documents as just another long-content source. ProLong finds that long web documents are *mostly scraped listings, boilerplate-heavy archives, or concatenated short articles*. The surface feature (length) exists, the substance (cross-span dependency) does not.

Compare the three coherent sources:

| Source | Why it has real long-range dependency |
|---|---|
| **Code repo** | function A calls function B in another file; tests assert behaviours defined in headers; README references specific modules. Cross-file references are dense and *semantic*. |
| **Books** | narrative or argument structure spans chapters; characters / concepts / citations reference across 100K+ tokens. |
| **Academic** | section N.k references section M.j; method references results; bibliography ties back to the body. |

The web shortcut fails this test. ProLong's filter is labour-intensive (authors admit it is not fully automated) but the resulting 30B-token mix has fundamentally different statistics from raw length-upsampled SlimPajama.

---

## The domain reweighting

> *"Domain upweighting: code repos × 4, books × 2, academic × 2, web × 0.5."*

The final **30B-token ProLong-Data mix** is:

- Code repositories: ~40%
- Books: ~25%
- Academic papers: ~15%
- StackExchange / forum long threads: ~10%
- Miscellaneous long web: ~10%

The 40% code weight is aggressive and the paper flags a specific bias: the trained model gains on code-adjacent tasks and may under-perform on natural-language long-document reasoning. The tradeoff is deliberate — code repositories are the cleanest long-coherence data available.

---

## The two-stage training recipe

> *"Stage 1 — Continued pretraining (20B tokens): extend RoPE base from 500K to 128M (Llama-3.1 style NTK-aware). Train at 64K context initially, expand to 512K in second half."*

The RoPE rescale is the key formula operation:

```
θ_base : 500,000 → 128,000,000   (Llama-3 base → ProLong final)
```

That's a **256× increase in base frequency**, which is ~`√(512K/4K)²` — NTK-aware with scale factor `s ≈ 128`. The math: NTK-aware rescales as `θ → θ · s^(d/(d-2))` where `d` is the head dim; for Llama-3's head dim 128, `s^(128/126) ≈ s`, so `s ≈ 256` produces ~256× base rescale and supports ~128× context expansion (8K → 512K).

Stage 2 is **5B tokens of SFT**:

- 70% long-instruction (LongAlign-style, Claude-3-generated)
- 30% short-instruction (UltraChat)
- plus explicit multi-needle NIAH training samples

The NIAH-training-as-SFT move is worth calling out: ProLong doesn't just evaluate on NIAH, it *trains* on synthesized multi-needle samples. This is a deliberate choice to teach retrieval as a learned skill, not hope it emerges.

---

## The ablation that matters — coherence vs concatenation

> *"Ablation: replacing curated long docs with concatenated short docs costs 10+ points on HELMET."*

This is the paper's core evidence. Same token count, same training budget, same position-encoding fix — just swap the data source for concatenated-shorts-of-equal-length — and HELMET drops 10+ points. The implication: **long-context capability is not a property of long sequences, it is a property of sequences with real long-range structure.**

Every subsequent long-context paper cites this ablation when justifying their curation effort.

---

## Final model performance

- **ProLong-8B** (Llama-3-8B base): 512K context with 20B CPT + 5B SFT.
- HELMET @ 512K: leading open 8B at release.
- InfiniteBench @ 128K: beats Llama-3.1-8B-Instruct and Qwen2-7B-Instruct.
- Short-context retention: MMLU / GSM8K within 0.5 point of Llama-3.1-8B-Instruct.
- Total compute ~200K H100-hours.

---

## Connections

- Chapter synthesis: [[ch-28]]
- SFT recipe lineage: [[excerpts/longalign-pipeline]]
- Llama-3's in-house equivalent: [[excerpts/llama3-staged-schedule]]
- Data-engineering contrast (5B tokens only): [[excerpts/fu-within-domain-upsample]] via the ch-28 read.md
