---
chapter: ch-21
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/blogs/hf-cosmopedia.md
source_url: https://huggingface.co/blog/cosmopedia
created_at: "2026-04-23"
---

# Excerpt: Cosmopedia — the open replication that discovered dedup is the bottleneck

**Source library:** `wiki/raw-data/llm-training/blogs/hf-cosmopedia.md`
**Authors / Org:** Loubna Ben Allal, Anton Lozhkov, Daniel van Strien (Hugging Face TB) — 2024.

---

## Why this source anchors ch-21 §5 and §7

Phi-1.5 claimed "20K topics + GPT-3.5 = broad reasoning corpus." Nobody outside Microsoft could verify because the data was not released. Cosmopedia is the open reproduction. It matters for ch-21 for two reasons:

1. It is the only source in this chapter where someone *actually tried to reproduce* the textbook-synthesis recipe and published the gotchas. The raw-data source is short on numbers and long on lessons; the lessons are what ch-21 §5 and §7 pull.
2. The Cosmopedia team's central finding — "prompts must be rewritten carefully to avoid near-duplicate generations" — is the operational rule that taxonomy-driven pipelines must respect. Ch-21 §5 quotes it directly.

---

## The three-source seed strategy

From the source's Data Recipe:

> - **Curated sources:** Stanford course outlines, Khan Academy, OpenStax, and WikiHow.
> - **Web sources:** millions of web samples are clustered into 145 clusters, labeled by topic from sample excerpts, and low-educational-value clusters are removed.
> - **Instruction/story sources:** UltraChat and OpenHermes2.5 are used as seeds for story-like prompts, with unsuitable categories removed before generation.
> - **Math enrichment:** AutoMathText is used to inject more scientific and mathematical content.

This is Cosmopedia's answer to ch-21 §5's curator-bias critique. A single curator's personal taxonomy (the Phi-1.5 move) risks omitting entire topic areas. Cosmopedia's response: three parallel taxonomies.

- **Curated educational** — Stanford / Khan / OpenStax / WikiHow. Gives the "legitimate educational" backbone.
- **Web-cluster-derived (145 clusters)** — gives coverage of whatever the web happens to talk about, including topics the curator did not anticipate.
- **Instruction / story** — gives conversational and narrative surface area.

The claim is that the *union* of these three is closer to "what a broad reasoning corpus should cover" than any single curated taxonomy would be. The 145-cluster number is an audit knob: more clusters = more topical resolution, at the cost of more manual cluster-labeling work.

Ch-21 §5 includes Cosmopedia in the "audit post-hoc" line of defense against single-curator bias.

---

## The audience × format expansion

From the source:

> **Style taxonomy:** the same topic is re-asked for young children, high school students, college students, and researchers, and rendered as textbooks, blog posts, WikiHow articles, or stories.

This is a 4 × 4 expansion on top of each topic seed. One topic becomes 16 (audience × format) prompt variants. The rationale: for a topic like "eigenvalues," the textbook-for-researchers version and the WikiHow-for-children version teach complementary facets. This adds coverage *within* a topic — depth in the GLAN sense, but along audience/format axes instead of syllabus/concept axes.

The risk, also from the source:

> **Diversity trick:** changing audience or format alone is not enough; the team found that prompts must be rewritten carefully to avoid near-duplicate generations.

Cosmetic variation — "explain X to a 10-year-old" vs "explain X to a 12-year-old" — produces near-duplicate outputs. The teacher rewrites the same exposition with slightly different vocabulary. Structural variation — "write a textbook chapter on X for undergraduates" vs "write a WikiHow article on X with steps and images" — produces meaningfully different outputs. Structural is what works.

Ch-21 §5's "Cosmopedia's real bottleneck is dedup" claim traces to this paragraph.

---

## Scale — what actually shipped

From the source's Technical Details:

> - **Generator:** Mixtral-8x7B-Instruct-v0.1.
> - **Scale:** over 30M files and 25B tokens.
> - **Compute:** over 10k GPU hours on H100s.
> - **Generation stack:** prompt curation in HuggingChat, large-scale generation with `llm-swarm`, deduplication with `datatrove`, training with `nanotron`, evaluation with `lighteval`.
> - **Decontamination:** 10-gram overlap retrieval followed by `difflib.SequenceMatcher`; candidates with high overlap to benchmark samples are removed.
> - **Validation model:** `cosmo-1b`, a 1B Llama2-architecture model trained on Cosmopedia, is used to test whether the corpus carries useful pretraining signal.

Two numbers worth memorizing for ch-21 §7:

- **30M prompts → 25B tokens.** Average ~800 tokens of output per prompt. Cosmopedia is in the pretraining-corpus size class, not the instruction-data size class.
- **10K H100-hours just for generation.** Orders of magnitude more than GLAN's published costs. This is the scale at which "synthetic pretraining" lives.

The validation result:

> cosmo-1b beats TinyLlama 1.1B on ARC-Easy, ARC-Challenge, OpenBookQA, and MMLU, but still trails phi-1.5 on some tasks.

Partial reproduction: the recipe works (cosmo-1b > TinyLlama) but does not fully close the gap to Phi-1.5. The gap is ch-21's evidence that *some* of Phi-1.5's advantage is contamination, *some* is curator quality, and the open replication cannot cleanly disentangle them.

---

## The decontamination workflow — reusable

From the source:

> Benchmark decontamination pipeline for synthetic-pretraining safety.
> **Decontamination:** 10-gram overlap retrieval followed by `difflib.SequenceMatcher`; candidates with high overlap to benchmark samples are removed.

Two-stage: fast n-gram filter (10-gram set overlap) pulls candidates; slow sequence-similarity filter (`difflib.SequenceMatcher`) confirms overlap. The n-gram stage makes the search tractable at 30M-sample scale; the sequence-matcher stage avoids false positives from common boilerplate.

This is the reference decontamination pipeline for any synthetic pretraining corpus in 2024+. Ch-21 does not dedicate a section to it, but ch-22 and ch-23 build on it. Readers should file the "10-gram + SequenceMatcher" pattern as the default.

---

## What Cosmopedia explicitly does not fix

From the source's Risks + Gotchas:

> - **Hallucination risk:** Mixtral can generate incorrect historical facts and mathematical statements, especially in the AutoMathText and Khan Academy subsets.
> - **Duplicate-content risk:** cosmetic prompt variation is not enough; the prompt family needs structural variation.
> - **Benchmark contamination:** synthetic datasets inherit contamination risk from both the seed corpus and the generator's own training data.
> - **Teacher ceiling:** the corpus quality is constrained by the generator model's knowledge and style biases.

The first and fourth are the inherited Phi-line limitations; Cosmopedia does not solve them, only inherits them explicitly and honestly. The second is the lesson other replications should learn. The third is the reason the decontamination pipeline exists.

Ch-21 uses this list as the checklist for the chapter's §7 "red flags." The phrasing "your corpus deduplicates by > 40% — cosmetic variation, not structural" is Cosmopedia-derived.

---

## Why blogs count as primary sources here

The source file is under `blogs/` not `papers/`, and the content is a Hugging Face blog post rather than an arXiv paper. For ch-21 this is still a primary source because:

- Cosmopedia is the only *executed* open reproduction of Phi-style synthetic pretraining with published data.
- The blog includes mechanical details (dedup pipeline, prompt-family design) that the Phi papers omit.
- The open-source nature of the corpus means every later synthetic-pretraining paper cites Cosmopedia as the public reference; ch-21 would not be complete without it.

The blog is a primary engineering report, not a secondary commentary.

---

## Connections

- [[excerpts/phi-textbooks]] — the closed original; Cosmopedia is the open replication target.
- [[excerpts/phi-1-5]] — Phi-1.5's 20K-topic list is what Cosmopedia's three-source seed strategy is trying to supersede in terms of curator-bias robustness.
- [[excerpts/glan]] — different scale (instruction data vs pretraining) but same taxonomy-at-the-top move.
- [[ch-22]] — quality + diversity selection; Cosmopedia's dedup workflow is a ch-22 tool applied to a ch-21 corpus.
- [[ch-23]] — model collapse + recursive training; Cosmopedia's "teacher ceiling" is the ch-23 problem framed as a ch-21 recipe limit.
- [[ch-21]] §5 and §7.
