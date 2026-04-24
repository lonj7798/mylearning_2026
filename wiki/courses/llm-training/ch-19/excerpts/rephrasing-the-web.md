---
chapter: ch-19
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/rephrasing-the-web.md
source_url: https://aclanthology.org/2024.acl-long.757/
created_at: "2026-04-23"
---

# Excerpt: WRAP — chunk-and-rewrite as pretraining-scale synthetic data

**Source library:** `wiki/raw-data/llm-training/papers/rephrasing-the-web.md`
**Heritage:** Maini et al. 2024 (Apple + CMU). The first paper to show that re-expressing raw web text into cleaner styles makes each token more useful — flipping the prevailing "filter the crawl" paradigm of [[fineweb]] and [[dolma]] into a "rewrite the crawl" complement.

---

## Why this source anchors ch-19

Ch-19 §7 is the only section of the chapter about pretraining data rather than SFT data. WRAP exists because every other method in ch-19 generates at SFT scale (10K–10M examples); pretraining needs trillions of tokens. WRAP is the method that scales synthetic data generation to the pretraining regime by replacing *open-ended generation* with *constrained rewriting* — a pipeline that can process one document per teacher-call regardless of the teacher's budget.

---

## The chunking rule — ~300 tokens, no longer

The source file is explicit:

> C4 documents are truncated/chunked to nearly 300 tokens with NLTK sentence splitting; the authors note that rephrasing beyond ~300 tokens often causes information loss.

The 300-token chunk is load-bearing. Three reasons from the paper's framing:

1. **Rephrase fidelity.** A 300-token chunk fits comfortably in Mistral-7B-Instruct's effective attention window for a *rewrite* task. Longer chunks force the teacher to summarize rather than rephrase, which drops facts.
2. **Rephrase determinism.** At 300 tokens the teacher's output is stable across reruns — same chunk in, near-identical rephrase out (up to sampling noise). At 1000+ tokens the outputs diverge, making the synthetic set less reproducible.
3. **Pretraining packing.** 300-token chunks pack cleanly into 1024-token training sequences; longer chunks produce awkward residuals.

NLTK sentence-splitting (rather than a fixed token cutoff) matters too — splitting mid-sentence changes the rephrase task from "rewrite this passage" to "continue then rewrite this fragment," and the teacher often produces the former when given the latter.

---

## The four styles — and what each buys

The source:

> **Styles:** Easy, Medium/Wikipedia-like, Hard/terse, and Q/A.

Not a random enumeration. Each style hits a different pretraining failure mode of raw web text:

- **Easy (grade-school vocabulary).** Raw C4 skews toward SEO-inflated prose and jargon-heavy tech blogs. Easy-rephrasing strips both. Useful for improving basic-reading zero-shot signals.
- **Medium / Wikipedia-like (encyclopedic, neutral).** Raw C4 has inconsistent register — colloquial on one domain, formal on another. Wiki-style rephrasing normalizes register, reducing the variance the base model must absorb.
- **Hard / Terse (dense, technical).** Raw C4 is verbose. Terse rephrasing compresses information per token, improving the model's density of signal.
- **Q/A reformulation.** Raw C4 is assertions; instruction-tuned models benefit from seeing the same content as Q/A pairs even during pretraining. The paper's strongest single-style ablation is Q/A, which disproportionately helps zero-shot QA benchmarks.

Mixing the four styles is load-bearing. Any single style applied alone produces a pretrained model with a characteristic prose mode; mixing preserves the register diversity of raw text while filtering its noise.

---

## The 1:1 real:synthetic mix — why not 100% synthetic

The source:

> **Mixing:** real and synthetic data are sampled 1:1, so each document is seen both as raw text and as a rephrase.

The 1:1 ratio is empirical. The paper ablates 0:1, 1:1, 2:1, and 1:0 mixes. At 0:1 (all real, no synthetic) you have the baseline. At 1:0 (all synthetic) you overfit to Mistral-7B-Instruct's voice — the base model starts emitting "Here's a paraphrase..." boilerplate and refuses to continue in the raw web's voice. At 1:1 you keep the distributional diversity of the raw web while still capturing the compute efficiency of the rephrases.

The 1:1 mix has a second property: each document is "seen twice" by the model, once in each form. This is not just repetition — the content is the same but the surface tokens differ, so the model is forced to learn the content-level representation rather than memorize surface text. It's a form of contrastive augmentation applied at pretraining.

---

## The boilerplate cleanup — why it's non-negotiable

The source:

> Appendix B adds a lightweight post-process that strips boilerplate intros such as "Here's a paraphrase..." or "high-quality English"; the residual error rate after cleanup is reported as under 0.1%.

Mistral-7B-Instruct frequently prefixes its rephrases with meta-commentary. If unfiltered, pretraining on these produces a base model that emits "Here's a paraphrase..." when prompted with arbitrary continuations — it has learned that this phrase is a valid text prefix in the pretraining distribution. The regex-based cleanup is a sentinel against this failure, which the paper caught only because they inspected generated outputs post-training.

The 0.1% residual error rate is the post-cleanup pass-through rate. At pretraining scale (trillions of tokens) 0.1% is still millions of contaminated documents — the cleanup is "good enough" only because the base model's subsequent training on non-contaminated data dominates.

This failure mode recurs in every synthetic-data pipeline at scale: the teacher's meta-commentary is invisible to casual inspection but catastrophic to downstream behavior. WRAP caught it via direct output inspection; ch-22 will generalize this into quality auditing as a first-class filter.

---

## The 350M-beats-1.3B result — what it actually means

The source:

> Showed that a 350M model trained on 15% of C4 with WRAP can beat a 1.3B model trained on all of C4.

Unpacking: 350M × 15% = 52.5M effective parameter-data product. 1.3B × 100% = 1.3B. The WRAP model achieves equivalent loss with 25× less compute-data product. This is a larger efficiency multiplier than any single pretraining trick other than full architectural changes (MoE, FlashAttention).

The result is not a free lunch. The teacher (Mistral-7B-Instruct) was trained on its own compute budget. The "350M beats 1.3B" comparison is downstream-of-teacher. If the teacher cost is amortized over many student runs, the efficiency argument holds; if it's counted per-run, WRAP's amortized cost depends on how many students use the same rephrased corpus.

In practice, frontier labs rephrase once and use the corpus for many runs — the amortization is favorable. Open labs without Mistral-7B-Instruct-class teachers had to wait for [[phi-textbooks]] and [[nemotron-4-synthetic]] to scale this to frontier-quality synthetic pretraining.

---

## What WRAP does not claim

The source:

> The core recipe does not use a learned quality filter.

Notable omission: no quality filter. WRAP relies entirely on style-rewriting and the 1:1 mix to improve quality; it does not score documents and drop the worst. This is why the method works *as a complement to* [[fineweb]] or [[dolma]] rather than replacing them. You filter the crawl to get a clean real corpus, then rephrase to add synthetic complements — the two pipelines stack.

Later work (Nemotron, Phi) combines rephrasing with quality scoring, which is where ch-21 will pick up.

---

## Connections

- [[excerpts/self-instruct]] — SFT-scale synthesis; WRAP is the pretraining-scale analog.
- [[excerpts/magpie]] — also uses an open-weight teacher; similar cost-collapse story.
- [[excerpts/persona-hub]] — diversity axis is orthogonal to style; WRAP could plausibly use persona-conditioned rephrases.
- [[ch-19]] — this excerpt is the foundation of §7 and the pretraining-targeted row in §9's comparison table.
