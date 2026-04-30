<!-- scope: Hugging Face Cosmopedia - synthetic pretraining corpus reproducing phi-style textbook data
     deps: [[phi-textbooks]], [[self-instruct]], [[rephrasing-the-web]]
     see-also: [[dolma]], [[openmathinstruct]], [[openmathinstruct-2]], [[open-thoughts]], [[phi-1-5]]
-->

# Hugging Face Cosmopedia - synthetic pretraining for phi-style textbooks
- **Core Insight:** Cosmopedia turns the phi-style synthetic-pretraining idea into an open pipeline: combine curated educational seeds, clustered web seeds, and instruction/story seeds, then vary audience and format to get a large pretraining corpus with broad topical coverage.
- **Guideline:** For synthetic pretraining, control seed source, audience, and output style explicitly; then spend real effort on deduplication and contamination checks, because prompt diversity is the bottleneck, not raw generation throughput.
- **Author/Org:** Loubna Ben Allal, Anton Lozhkov, Daniel van Strien (Hugging Face TB)
- **Year:** 2024
- **URL:** https://huggingface.co/blog/cosmopedia
- **Relevant topics:** synthetic pretraining, phi-style textbooks, prompt curation, web clustering, decontamination, open dataset release

## Summary
Cosmopedia is Hugging Face's open attempt to reproduce the "textbook-quality" synthetic pretraining line associated with Phi. The blog argues that the interesting part is not only generating synthetic text at scale, but designing the prompt space so the generator covers many useful educational modes without collapsing into near-duplicate boilerplate. The release is large enough to matter for pretraining, not just instruction tuning: 30M samples and 25B tokens generated with Mixtral-8x7B-Instruct-v0.1.

## Key Contributions
- Open reproduction attempt of phi-style synthetic pretraining at large scale.
- Mixed seed strategy: curated educational sources, clustered web sources, and instruction/story seeds.
- Explicit style taxonomy across audience and format to reduce duplicate generations.
- Benchmark decontamination pipeline for synthetic-pretraining safety.
- Reference model `cosmo-1b` to sanity-check whether the corpus actually trains a useful model.

## Data Recipe
- **Curated sources:** Stanford course outlines, Khan Academy, OpenStax, and WikiHow.
- **Web sources:** millions of web samples are clustered into 145 clusters, labeled by topic from sample excerpts, and low-educational-value clusters are removed.
- **Instruction/story sources:** UltraChat and OpenHermes2.5 are used as seeds for story-like prompts, with unsuitable categories removed before generation.
- **Math enrichment:** AutoMathText is used to inject more scientific and mathematical content.
- **Prompt scaling:** the blog says prompt creation reached over 30M prompts, with web-derived prompts providing most of the coverage.
- **Style taxonomy:** the same topic is re-asked for young children, high school students, college students, and researchers, and rendered as textbooks, blog posts, WikiHow articles, or stories.
- **Diversity trick:** changing audience or format alone is not enough; the team found that prompts must be rewritten carefully to avoid near-duplicate generations.

## Technical Details
- **Generator:** Mixtral-8x7B-Instruct-v0.1.
- **Scale:** over 30M files and 25B tokens.
- **Compute:** over 10k GPU hours on H100s.
- **Generation stack:** prompt curation in HuggingChat, large-scale generation with `llm-swarm`, deduplication with `datatrove`, training with `nanotron`, evaluation with `lighteval`.
- **Decontamination:** 10-gram overlap retrieval followed by `difflib.SequenceMatcher`; candidates with high overlap to benchmark samples are removed.
- **Validation model:** `cosmo-1b`, a 1B Llama2-architecture model trained on Cosmopedia, is used to test whether the corpus carries useful pretraining signal.
- **Reported effect:** cosmo-1b beats TinyLlama 1.1B on ARC-Easy, ARC-Challenge, OpenBookQA, and MMLU, but still trails phi-1.5 on some tasks.

## Risks + Gotchas
- **Hallucination risk:** Mixtral can generate incorrect historical facts and mathematical statements, especially in the AutoMathText and Khan Academy subsets.
- **Duplicate-content risk:** cosmetic prompt variation is not enough; the prompt family needs structural variation.
- **Benchmark contamination:** synthetic datasets inherit contamination risk from both the seed corpus and the generator's own training data.
- **Teacher ceiling:** the corpus quality is constrained by the generator model's knowledge and style biases.

## Connections
- Direct open counterpart to [[phi-textbooks]] and the phi-1.5 continuation.
- Companion to [[open-thoughts]] and [[openmathinstruct]] as open synthetic-reasoning data efforts.
- Useful contrast with [[self-instruct]]: Cosmopedia is pretraining-grade corpus construction, not only instruction tuning.
- The blog's emphasis on prompt curation fits the broader quality-over-quantity thread in [[lima]] and [[rephrasing-the-web]].
