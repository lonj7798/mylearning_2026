---
chapter: ch-32
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/prolong.md
source_url: https://arxiv.org/abs/2410.02660
created_at: "2026-04-23"
---

# Excerpt: ProLong - the document-coherence thesis ch-32 uses

**Source library:** `wiki/raw-data/llm-training/papers/prolong.md`
**Artifact:** 20B CPT + 5B SFT budget + coherence-filtered 30B-token long-doc mix

---

## Why this source anchors ch-32

ProLong is the clearest quantitative evidence that **long-doc quality dominates volume** in long-context training. Its ablation - replacing curated coherent long documents with concatenated short documents at matched token count costs **10+ points on HELMET** - is the single most-quoted empirical result in the long-context-data literature. Ch-32's Job 2 definition ("long-coherent-document training") is named after ProLong's thesis, and the ch-32 caveat that "data quality buys token budget" is ProLong's headline finding.

ProLong also anchors the "small-budget long-context" case: 20B CPT + 5B SFT is achievable on mid-tier compute (~200K H100-hours total). Ch-32's recipe-by-budget section uses this number to argue that long-context extension is not exclusively a frontier-compute problem.

---

## The coherence filter ch-32 transcribes

From the source (lines 28-40):

- Filter by document length: keep only documents with >=64K tokens of coherent content.
- Filter by coherence signal (source-type-specific):
  - **Code**: whole repository concatenated in order (README -> source -> tests), not single files.
  - **Books**: full-book PDFs parsed with structural fidelity.
  - **Academic**: full papers with references.
  - **Web**: discarded - authors find long web docs are mostly scraped listings with weak coherence.

Ch-32 uses this filter as the canonical example of "long and coherent are not the same predicate on web data." The filter's web-discard rule is the sting: the obvious shortcut (use long web docs) doesn't work, and using them at matched token count is strictly worse than using fewer tokens of code/books/academic.

---

## The domain re-weighting ch-32 cites

From the source (lines 34-40):

- Code repos x 4.
- Books x 2.
- Academic x 2.
- Web x 0.5 (downweighted, not upweighted).

Final 30B-token mix:
- Code repositories: ~40%.
- Books: ~25%.
- Academic papers: ~15%.
- StackExchange / forum threads: ~10%.
- Miscellaneous long web: ~10%.

Ch-32 uses the code-heavy weighting as evidence of a trade-off ProLong itself flags: the mix biases the model toward code-adjacent tasks. This is why ch-32 frames Job 2 as a data-curation problem that must be tuned per deployment target - a model aimed at long-doc natural-language reasoning wants a different mix than one aimed at long-code understanding.

---

## The staged training recipe ch-32 uses

From the source (lines 42-49):

- **Stage 1 - CPT (20B tokens):**
  - Extend RoPE base from 500K to 128M (Llama-3.1 NTK-aware style).
  - Train at 64K context initially, expand to 512K in second half.
  - LR 1e-4 -> 1e-5 cosine.
- **Stage 2 - SFT (5B tokens):**
  - 70% long-instruction + 30% short-instruction + synthetic multi-needle NIAH.

Ch-32 uses this to show that even at a smaller budget, the two-stage structure (position-encoding extension + coherent-long-doc CPT as one stage; long-instruction SFT as a separate stage) survives. The 70/30 long/short SFT mix is more aggressive than Llama 3's 0.1% long-SFT rule because ProLong targets 512K (retrieval-focused) rather than 128K (general-purpose).

---

## The headline ablation ch-32 quotes

From the source (lines 61-65, 67-71):

- Replacing curated long documents with concatenated short documents costs 10+ points on HELMET.
- Short-context retention: MMLU / GSM8K within 0.5 point of Llama-3.1-8B-Instruct base.
- ProLong-8B on HELMET (512K): leading open 8B.
- ProLong-8B on InfiniteBench (128K): beats Llama-3.1-8B-Instruct and Qwen2-7B-Instruct.

Ch-32 cites the 10+ HELMET-point ablation as the single best evidence for the coherence thesis. The short-context retention is equally important: it shows the recipe does not sacrifice short-context capability to gain long-context, which is the binding trade-off Llama 3's 0.1% SFT rule also addresses.

---

## Connections

- **ch-28** - long-context modality synthesis; ProLong is one of three production recipes.
- **[[longalign]]** - SFT-side long-context alignment; ProLong's stage 2 is a LongAlign-style pipeline.
- **[[long-context-llama3]]** - Meta's parallel recipe at larger budget.
- **[[longrope-data]]** - position-encoding lane; orthogonal to the data-curation thesis.
