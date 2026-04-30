---
chapter: ch-10
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/rephrasing-the-web.md
source_url: https://aclanthology.org/2024.acl-long.757/
created_at: "2026-04-23"
---

# Excerpt: Rephrasing the Web (WRAP) — the critique that classical filters have a ceiling

**Source library:** `wiki/raw-data/llm-training/papers/rephrasing-the-web.md`
**Paper:** Maini, Seto, Bai, Grangier, Zhang, Jaitly 2024, "Rephrasing the Web: A Recipe for Compute and Data-Efficient Language Modeling" (ACL 2024).

---

## Why this source anchors ch-10

WRAP is the **critique of classical filtering** — the argument that even after CCNet, C4, Dolma, and FineWeb have done their best work, the surviving web text is still noisy and under-structured in ways no filter can fix. The paper's move is to *rewrite* the corpus rather than (only) filter it, and it reports multiplicative gains on top of filtering. Ch-10 §6 uses this as one of the two "orthogonal directions" to keep in view: filtering has a ceiling, and rephrasing is the move past it.

From the source (lines 7–8):

> **Core Insight:** Raw web text is noisy and under-structured; re-expressing the same content into cleaner styles can make each token more useful and cut pretraining compute/data needs by a large margin.
>
> **Guideline:** For noisy web corpora, chunk documents to about 300 tokens, rephrase them with a frozen instruction-tuned model in multiple styles, mix original and synthetic text 1:1, and train the LM on that blend.

The guideline is operationally simple: chunk, rephrase into multiple styles (Easy, Wikipedia-like, Terse, Q&A), train on a 1:1 mix of real and synthetic. The critique embedded in it is: *even filtered web text does not have the register you want the model to learn.*

---

## The empirical claim — what rephrasing does on top of filtering

From the source (lines 19–21):

> - Showed that the same corpus can support roughly 3x faster pretraining on C4 and substantially better perplexity than raw-text baselines.
> - Showed that a 350M model trained on 15% of C4 with WRAP can beat a 1.3B model trained on all of C4.
> - Demonstrated that style matters: question-answer rephrasing is especially useful for zero-shot QA, while Wikipedia-like rephrasing helps readability and general pretraining quality.

Two of these numbers matter for ch-10. **~3× faster pretraining on C4** means the same corpus, rephrased and blended, reaches a target loss in a third of the wall-time — a multiplicative gain on top of C4's own filtering. **350M-at-15%-of-C4 > 1.3B-at-full-C4** means the rephrasing-plus-filtering combination is more compute-efficient than scaling up the model on the same filtered corpus.

This is the specific evidence ch-10 §6 cites when it says "filtering has a ceiling." On C4 specifically, rephrasing unlocks gains that more filtering does not.

---

## Why rephrasing works where filtering does not

The source's technical detail (lines 32–37):

> - **Seed data:** C4 documents are truncated/chunked to nearly 300 tokens with NLTK sentence splitting; the authors note that rephrasing beyond ~300 tokens often causes information loss.
> - **Teacher model:** frozen Mistral-7B-Instruct.
> - **Styles:** Easy, Medium/Wikipedia-like, Hard/terse, and Q/A.
> - **Mixing:** real and synthetic data are sampled 1:1, so each document is seen both as raw text and as a rephrase.

The mechanism: filtering can select documents that are *more* prose-like or *more* educational, but it cannot rewrite a document into a style the source never contained. If C4 contains a blog post about photosynthesis, filtering can keep or drop it; it cannot turn it into a Wikipedia-style explanation of photosynthesis or a Q&A about photosynthesis. Rephrasing can. The **style diversity** introduced by multi-style rephrasing is a signal filtering fundamentally cannot produce.

This is why WRAP is a complement to ch-10's four pipelines rather than a replacement. The pipelines decide which documents survive; rephrasing decides what styles the surviving documents are presented in. Both levers are positive; they act on different axes.

---

## The cost — and why it has come down since

The paper's cost model: rephrase every C4 document in 4 styles with Mistral-7B-Instruct, then pretrain on real + synthetic blend. At C4 scale (~750B tokens), 4-style rephrasing produces 3T synthetic tokens. Teacher inference at Mistral-7B is cheap relative to the pretraining it supports, but it is not free — the paper reports the mix as economically worthwhile only because the downstream training speedup (3×) covers the teacher cost several times over.

For 2026 pipelines, the arithmetic is more favorable: teacher-model inference has become cheaper (Llama-3.1-8B or Qwen-2.5-7B at similar quality, cheaper hardware), and rephrase-once-reuse-many means the synthetic corpus is a one-time cost amortized across all subsequent pretraining. This is the reason synthetic-pretraining recipes (Nemotron-4, Phi-textbooks, Cosmopedia) have become mainstream since 2024.

---

## How WRAP changes the ch-10 critique checklist

Adding WRAP to the mental model extends the §6 checklist:

- **Q1 (quality signal)** — now also: *is there a rephrasing stage, and if so what teacher and what styles?*
- **Q2 (order)** — rephrasing comes *after* filtering (you rephrase only the survivors) and *after* dedup (you do not want to multiply duplicates).
- **Q3 (threshold)** — classical thresholds are now per-style mix ratios. WRAP uses 1:1 real:synthetic; Nemotron uses different ratios per stage.
- **Q4 (ablation)** — the ablation now has two knobs: filter severity × rephrase mix. Both must be swept for the pipeline to be falsifiable.

This is one reason ch-11 (data ops) and the synthetic-data track are the natural continuations of ch-10 — rephrasing is the bridge from curation to generation.

---

## What WRAP does not claim

- **It does not replace filtering.** The paper is explicit that rephrasing *on top of* filtered C4 is the recipe; rephrasing raw Common Crawl is not the move.
- **It does not solve style bias.** If the teacher has a style bias (Mistral-Instruct's over-polite register, its tendency to start with "Certainly!"), the rephrased corpus inherits it. Appendix B's post-cleanup reduces but does not eliminate this.
- **It does not answer the model-collapse question.** Whether synthetic-heavy training eventually collapses is an open question outside this paper's scope; the 1:1 real:synthetic mix is a hedge against it.

---

## What to take from this paper for ch-10

1. **Classical filtering has a ceiling.** Even on C4, the corpus every subsequent pipeline has been trying to improve, rephrasing unlocks 3× pretraining speedup. Filtering is necessary but not sufficient.
2. **Rephrasing is orthogonal to filtering.** They operate on different axes (selection vs style) and both are positive. Composing them multiplies gains.
3. **Style diversity is a signal filters cannot produce.** Any multi-style rephrase produces registers the source does not contain.
4. **The teacher-model cost has fallen fast.** A 2024 paper's expensive demonstration is a 2026 commodity pipeline.
5. **The four ch-10 pipelines are the substrate, not the endpoint.** FineWeb-Edu is the state of the art in filtering; it is not the state of the art in *corpus construction* once rephrasing is on the table.

---

## Connections

- [[excerpts/ccnet]] — the filtering skeleton WRAP composes on top of.
- [[excerpts/c4]] — the specific corpus WRAP demonstrates rephrasing gains on.
- [[excerpts/dolma]] — Dolma-filtered data is a candidate seed for WRAP-style rephrasing; the pipelines compose.
- [[excerpts/fineweb]] — the most aggressive 2024 filter; WRAP's argument is that even FineWeb-Edu has headroom for rephrasing.
- [[excerpts/scaling-laws-data-quality]] — rephrasing presumably raises `Q` at the cost of a larger `D` (real + synthetic); the framework there formalizes the trade.
- [[ch-10]] §6 (filtering ceiling), §7 (where ch-10 leaves you).
