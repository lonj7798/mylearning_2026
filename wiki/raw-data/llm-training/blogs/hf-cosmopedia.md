<!-- scope: Cosmopedia — Hugging Face's open synthetic pre-training corpus (30M+ files, 25B tokens) generated with Mixtral-8x7B-Instruct-v0.1, and its prompt-curation method
     deps: [[phi-textbooks]], [[self-instruct]], [[rephrasing-the-web]]
     see-also: [[dolma]], [[openmathinstruct]], [[openmathinstruct-2]], [[open-thoughts]], [[phi-1-5]]
-->

# Cosmopedia: how to create large-scale synthetic data for pre-training Large Language Models
- **Core Insight:** Cosmopedia generates over 30 million files and 25 billion tokens with Mixtral-8x7B-Instruct-v0.1 and reports less than 1% duplicate content; the authors state that most of the project time went into prompt engineering rather than generation, and that over 80% of the prompts come from clustered web seeds rather than curated educational sources ("Prompts curation", "Web data").
- **Guideline:** When scaling synthetic generation for pre-training, vary the seed document as well as the audience and style, because the authors report that changing only the audience or the format in the prompt text was insufficient to prevent a high rate of duplicate content ("Leverage diversity in audience and style").
- **Authors:** Loubna Ben Allal, Anton Lozhkov, Daniel van Strien (Hugging Face)
- **Year:** 2024 (published 2024-03-20)
- **URL:** https://huggingface.co/blog/cosmopedia
- **Source type:** official blog
- **Relevant topics:** synthetic pre-training, phi-style textbooks, prompt curation, web clustering, decontamination, open dataset release

## Summary
The post describes building a synthetic pre-training corpus intended to reproduce the training data of Phi-1.5, whose datasets were not released. The authors state the scaling problem directly: the Phi-1.5 report says 20,000 topics were used to produce 20 billion tokens, which at an assumed 1,000 tokens per file implies roughly 20 million distinct prompts, and the method for combining topics with web samples is not described. Cosmopedia builds its prompts from two seed types, curated educational sources and clustered web data, and releases the pipeline code, the dataset, and a 1B validation model, `cosmo-1b`.

## Key Contributions
- An open corpus of synthetic textbooks, blog posts, stories, posts, and WikiHow articles: over 30 million files and 25 billion tokens, described in the post as the largest open synthetic dataset at the time ("Behind the scenes").
- Generation with an open model, Mixtral-8x7B-Instruct-v0.1, rather than a proprietary one.
- A two-source prompt strategy (curated educational seeds; clustered web seeds) plus story seeds from instruction datasets.
- A web-clustering step that labels topics with Mixtral and assigns each cluster an educational score out of 10, used to remove low-value clusters.
- A benchmark-decontamination pipeline with a reported count of removed samples per split.
- `cosmo-1b`, a 1B Llama 2-architecture model trained on the corpus as a quality check.
- Released code: the `cosmopedia` repo, and the `text-clustering`, `llm-swarm`, `datatrove`, `nanotron`, and `lighteval` libraries used in the stack.

## Technical Details
### 1. Seed sources and prompt construction
- **Curated sources:** Stanford course outlines, Khan Academy, OpenStax, and WikiHow. Scalability is the stated limit: 16,000 unique units could be extracted from OpenStax and 250,000 from Stanford, against a target of at least 20 million prompts ("Curated Sources").
- **Audience and style:** four audiences (young children, high school students, college students, researchers) and three generation styles (textbooks, blog posts, WikiHow articles), giving up to 12× the number of prompts from one topic ("Leverage diversity in audience and style"). The post states that changing the audience or style words alone still produced a high duplicate rate, so the prompts specify how the format and content should differ.
- **Web seeds:** millions of RefinedWeb samples were clustered into 145 clusters; each cluster's topic was labeled by giving Mixtral extracts from 10 random samples. Clusters judged low in educational value (explicit adult material, celebrity gossip, obituaries) were removed, leaving 112 retained topics. 23 million prompts were built this way, over 80% of the total ("Web data").
- **Topic conditioning:** the prompt is conditioned on the cluster topic only 50% of the time, to add diversity and to absorb errors in topic labeling ("Web data").
- **Stories:** UltraChat (the "Questions about the world" subset, covering 30 meta-concepts) and OpenHermes-2.5 seed story generation. Sources unsuitable for storytelling were dropped from OpenHermes-2.5, named as `glaive-code-assist` and `camelai` ("Instruction datasets and stories"). The stated motivation is that models trained on the generated textbooks alone lacked common sense and grade-school knowledge.
- **Math:** AutoMathText samples were added to raise scientific content ("Web data").
- Total prompt count: over 30 million, spanning hundreds of topics, with less than 1% duplicate content ("Prompts curation").

### 2. Generation stack
- Generator: Mixtral-8x7B-Instruct-v0.1, deployed locally on H100 GPUs with TGI and driven by `llm-swarm` ("Textbooks generation at scale").
- Total generation compute: over 10k GPU hours ("Textbooks generation at scale").
- Prompt iteration was done in HuggingChat, then a few hundred samples per prompt were generated with `llm-swarm` to look for repeated patterns. The reported pattern was repeated openings, for example "Once upon a time" and "The sun hung low in the sky"; instructing the model to avoid them reduced but did not remove the repetition.
- Deduplication and tokenization with `datatrove`; training with `nanotron`; evaluation with `lighteval`.

### 3. Decontamination
- Candidates are found by 10-gram overlap, following Phi-1, then compared with `difflib.SequenceMatcher`. A sample is discarded when `len(matched_substrings) / len(benchmark_sample)` exceeds 0.5 ("Benchmark decontamination").
- Applied across MMLU, HellaSwag, PIQA, SIQA, Winogrande, OpenBookQA, ARC-Easy, and ARC-Challenge.
- Removed samples per dataset group, with unique benchmark samples in brackets:

  | Dataset group | ARC | BoolQ | HellaSwag | PIQA |
  |---|---|---|---|---|
  | web data + stanford + openstax | 49 (16) | 386 (41) | 6 (5) | 5 (3) |
  | auto_math_text + khanacademy | 17 (6) | 34 (7) | 1 (1) | 0 (0) |
  | stories | 53 (32) | 27 (21) | 3 (3) | 6 (4) |

  Fewer than 4 contaminated samples were found for MMLU, OpenBookQA, and WinoGrande.

### 4. Validation model
- `cosmo-1b`: 1B parameters, Llama 2 architecture, trained on Cosmopedia ("Training stack").
- Reported comparison: better than TinyLlama 1.1B on ARC-Easy, ARC-Challenge, OpenBookQA, and MMLU; comparable to Qwen-1.5-1B on ARC-Challenge and OpenBookQA; below Phi-1.5 on some tasks. The post attributes the remaining gap to the generator model, topic coverage, or the prompts, and does not separate these (Figure 10). **Result (single study).**
- The post gives no number of training tokens, no hyperparameters, and no compute figure for `cosmo-1b`.

## Findings relevant to generality and distillation
- **Breadth by construction, not measured.** Coverage is argued from the prompt design (145 clusters, 112 retained topics, 4 audiences, 3 styles) and from the duplicate rate below 1%. The post reports no held-out or unseen-task evaluation isolating breadth, so the generality claim rests on the four-benchmark `cosmo-1b` comparison above.
- **Teacher ceiling in distillation.** The conclusion states that the accuracy and reliability of the generations depend largely on the generator, and names Mixtral hallucination of historical facts and of mathematical reasoning in the AutoMathText and Khan Academy subsets as a known failure. Proposed mitigations are retrieval-augmented generation from sources such as Wikipedia, and hallucination measurement; neither is reported as implemented.
- **Contamination is inherited twice.** The decontamination section motivates the pipeline by noting that contamination can enter through the seed samples or through the generator's own training data.
- **Prompt-level mode collapse.** The repeated-opening finding is the post's clearest negative observation: explicit instructions reduced the repeated phrases but did not eliminate them.
- The post reports no negative-feedback mechanism: generations are filtered for contamination and deduplicated, not scored or ranked.

## Connections
- [[phi-textbooks]] and [[phi-1-5]] are the closed-data line this post reproduces, and the source of the 20,000-topic / 20B-token figure it reasons from.
- [[self-instruct]] is instruction-tuning-scale synthesis; Cosmopedia targets pre-training-scale corpus construction.
- [[rephrasing-the-web]] is the neighboring approach of rewriting web documents rather than generating from topic seeds.
- [[open-thoughts]] and [[openmathinstruct]] are open synthetic-data efforts in the reasoning and math domains.
- [[lima]] holds the quality-over-quantity argument this post applies at a much larger scale.
- [[dolma]] is the non-synthetic open pre-training corpus to contrast against.

## Verification
- Checked on 2026-09-18 against: https://huggingface.co/blog/cosmopedia (published 2024-03-20; page read 2026-09-18).
- Corrections to the previous card version:
  - Title "Hugging Face Cosmopedia - synthetic pretraining for phi-style textbooks" → the published title, "Cosmopedia: how to create large-scale synthetic data for pre-training Large Language Models".
  - "millions of web samples are clustered into 145 clusters … low-educational-value clusters are removed" → the post also states the number retained: 112 topics.
  - "web-derived prompts providing most of the coverage" → over 80% of prompts, and 23 million web-based prompts specifically.
  - "rendered as textbooks, blog posts, WikiHow articles, or stories" → the three generation styles are textbooks, blog posts, and WikiHow articles; stories are a separate seed type built from UltraChat and OpenHermes-2.5.
  - "Compute: over 10k GPU hours on H100s" → kept, with the locus; the post attributes this to generation only, not to training `cosmo-1b`.
  - "cosmo-1b beats TinyLlama 1.1B … but still trails phi-1.5 on some tasks" → the post also reports comparability to Qwen-1.5-1B on ARC-Challenge and OpenBookQA.
  - Added the decontamination threshold (ratio > 0.5), the removed-sample counts, the OpenStax/Stanford unit counts, the 50% topic-conditioning rate, the <1% duplicate rate, and the RefinedWeb source, all of which were absent.
  - Author field relabeled from "Author/Org" to "Authors" per the card standard; "Hugging Face TB" → the post bylines three authors under Hugging Face.
- Removed as unsupported by the source:
  - "Diversity trick" as a section label and the framing "prompt diversity is the bottleneck, not raw generation throughput" — the post says most time was spent on prompt engineering, which is a statement about effort, not a measured bottleneck.
  - "Duplicate-content risk: cosmetic prompt variation is not enough; the prompt family needs structural variation" as a stated risk — the post reports this as an observation during development, now recorded under Findings with its locus.
  - "Benchmark decontamination pipeline for synthetic-pretraining safety" — the post frames decontamination as evaluation hygiene, not safety.
- Not reported by the source: `cosmo-1b` training token count, hyperparameters, or training compute; the exact number of web samples clustered ("millions"); per-benchmark scores for `cosmo-1b` in text (they appear only in Figure 10); licence terms.
