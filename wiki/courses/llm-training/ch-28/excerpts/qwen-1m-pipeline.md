---
chapter: ch-28
course: llm-training
phase: read
excerpt_of: Qwen Team — "Qwen 2.5-1M: Enabling 1M-Token Context Windows" (and Qwen 3 long-context)
source_url: https://arxiv.org/abs/2501.15383
created_at: "2026-04-23"
---

# Excerpt: Qwen 2.5-1M — three-leg pipeline and Dual-Chunk Attention

**Source:** `wiki/raw-data/llm-training/papers/qwen-long-context-synth.md`
**Paper:** Qwen Team (Alibaba DAMO), 2025
**arXiv / blog:** https://arxiv.org/abs/2501.15383 — https://qwenlm.github.io/blog/qwen2.5-1m/

---

## Bibliographic header

> *"Qwen 2.5-1M and Qwen 3 push effective context to 1M tokens via (a) dual-chunk attention (DCA) at inference for training-free length extrapolation, (b) length-upsampled continued pretraining on 256K-token contexts with code + book + synthetic long tasks, and (c) a synthetic long-task SFT mix heavy on multi-needle retrieval, long-doc summarization, and RAG-style multi-passage QA — with the training context capped at 256K while inference supports 1M through DCA."*

Qwen 2.5-1M is the first open model family that ships **1M-token inference support with real retrieval quality**. The structural choice — train at 256K, extrapolate to 1M at inference — is the interesting piece.

---

## Leg 1 — gradual continued pretraining

From the raw-data:

> *"Stage 1 — 32K: continued pretraining on general long docs, ~50B tokens. Stage 2 — 128K: code-repo heavy mix, ~50B tokens. Stage 3 — 256K: document concatenation across topics, synthetic multi-topic mixed-context, ~100B tokens."*

Total CPT: 200B tokens across three stages, with the context-doubling-per-stage pattern already familiar from Llama-3.

**The stage 3 novelty:** synthetic multi-topic concatenation. Real 256K-coherent documents are rare; Qwen synthesises them by *bundling related web documents* into long mixed-context sequences. This is a hybrid between ProLong's "real coherent long docs" and the cheap "concatenate anything" approach ProLong ablates against — Qwen bundles documents that are *topically* linked so the concatenation has semantic structure.

Per-stage document mix from the raw-data:

- Code repositories concatenated in dependency order
- Books concatenated chapter-wise
- Academic papers + citation-linked follow-ups
- Synthetic multi-document cluster: related web docs bundled

Final mix fractions: code ~35%, books ~25%, academic ~15%, multi-doc bundles ~15%, synthetic ~10%.

---

## Leg 2 — the synthetic SFT task mix

> *"Synthetic task mix: Multi-needle retrieval (1–8 needles injected at varied positions, retrieved on demand); Long-doc summarization (50K–200K doc → summary); RAG-QA (5–20 candidate passages, answer requires cross-passage fusion); Long-code understanding (code-repo + question about cross-file dependencies); FILL-IN (document with masked segment to reconstruct)."*

Each synthetic task targets a RULER failure mode:

| SFT task | RULER family it trains |
|---|---|
| Multi-needle retrieval | MK-NIAH, MV-NIAH, MQ-NIAH |
| Long-doc summarization | CWE / FWE aggregation |
| RAG-QA | QA + MK-NIAH with real distractors |
| Long-code understanding | VT-style cross-reference chains |
| FILL-IN | bidirectional context reconstruction (not in RULER, novel) |

**The filter rule worth noting:**

> *"Answer must reference multiple positions of the source document (to avoid shortcut learning)."*

Without this filter, a teacher will generate answers that only use one short span of the document — trivially answerable without long-context reasoning. The multi-position constraint forces the SFT to actually train the long-context skill, not a bypass.

Teacher: Qwen-Max (internal). Generation cost is not disclosed.

---

## Leg 3 — Dual-Chunk Attention (DCA) at inference

> *"Split long queries into chunks of training-size (256K); within-chunk attention is standard; between-chunk uses a special low-rank formulation that extrapolates RoPE smoothly. Enables 1M-token context serving without retraining."*

The core idea: standard attention at 256K works because the model was trained at 256K. To attend across 1M tokens without OOD-position problems, **split** the 1M sequence into four 256K chunks, compute attention normally within each chunk, and use a **modified low-rank attention between chunks** that re-indexes positions so the between-chunk attention never sees "out-of-training-range" position indices.

This is a *training-inference split*: the model is never exposed to 1M positions during training; the inference-time kernel creates the illusion. The tradeoff, per the raw-data:

> *"DCA is inference-only: generalization from 256K training to 1M inference works for retrieval-style tasks but degrades on true long-range reasoning."*

DCA solves the positional-OOD problem but not the data-OOD problem. The model has never seen 1M-token training sequences, so it has no inductive bias for 1M-token reasoning patterns. Retrieval works because retrieval is local; reasoning-in-a-haystack at 1M-token scale is where DCA's seams show.

---

## The 1M evaluation curves

> *"Qwen-2.5-14B-1M: NIAH 1M ~100%, RULER 1M ~85%, InfiniteBench strong."*

The 15-point NIAH-to-RULER gap at 1M is the same reasoning-tax pattern seen at 128K for Llama-3.1-70B. Qwen's gap is smaller in relative terms because Qwen's SFT explicitly trained multi-needle — a deliberate training choice that narrows the gap that would otherwise be closer to Llama-3's 24 points.

Short-context retention: MMLU / GSM8K within 1 point of base Qwen-2.5 — the small-long-SFT-fraction lesson from Llama-3 holds here too, re-expressed in a different form.

---

## Why this matters for ch-28's thesis

Qwen 2.5-1M is the clearest single example of the three-lane co-design:

1. **Position lane:** YaRN during training, DCA at inference.
2. **Data lane (CPT):** gradual 32K → 128K → 256K with topically-bundled synthetic long sequences.
3. **Data lane (SFT):** task-mix explicitly aligned to RULER's failure-mode taxonomy.
4. **Eval lane:** NIAH + RULER + InfiniteBench; the NIAH-RULER gap is reported honestly.

No single lane does the work. Remove DCA and you lose 1M inference. Remove the multi-needle SFT and RULER drops below NIAH's level of usefulness. Remove the CPT stage and the base doesn't have the retrieval pattern to elicit. The recipe is *load-bearing in all four places*.

---

## The open question

> *"Closed training data: specific synthetic-task distribution not released. Alibaba-internal tooling: Qwen-Max teacher and DCA implementation not fully open."*

The three-leg pipeline is documented at the architectural level but not reproducible at the dataset level — Qwen-Max generation is internal, and the synthetic multi-topic bundles are not released. That's the lab-reproducibility ceiling for this recipe in 2025.

---

## Connections

- Chapter synthesis: [[ch-28]]
- Predecessor recipes: [[excerpts/llama3-staged-schedule]], [[excerpts/prolong-coherence]]
- Position lane sibling: [[excerpts/longrope-per-dim-search]]
- Eval taxonomy driving the SFT design: [[excerpts/ruler-task-family]]
- Gemini's product-side complement (context caching): ch-28 §6
