---
chapter: ch-21
course: llm-training
phase: read
excerpt_of: "Cosmopedia: how to create large-scale synthetic data for pre-training (Hugging Face blog, 2024-03-20)"
source_url: https://huggingface.co/blog/cosmopedia
created_at: "2026-04-23"
revised_at: "2026-09-15"
---

# Excerpt: Cosmopedia — an open replication of Phi-1.5-style synthetic data

This excerpt was rewritten on 2026-09-15 from the blog post itself, because the library card `hf-cosmopedia` had not
been verified. The April 2026 version stated a 4 × 4 audience-by-format grid, a first pass of 30M prompts with many
duplicates, "deduplication is the bottleneck", and ~40% deduplication as a warning sign; none of these is in the post.

- **Authors:** Loubna Ben Allal, Anton Lozhkov, Daniel van Strien (Hugging Face). Published 2024-03-20.
- **Source type:** official blog (Hugging Face built the dataset and trained cosmo-1b; open code, data, and model; no controlled ablation tables).

## Goal and scale
- "Cosmopedia ... aims to reproduce the training data used for Phi-1.5." Phi datasets are not released, and the post
  notes that critics argue the Phi models "may simply be overfitting benchmarks".
- Generator Mixtral-8x7B-Instruct-v0.1; "over 30 million files and 25 billion tokens"; total generation compute "over
  10k GPU hours". Derived: 25B / 30M ≈ 833 tokens per file.
- "most of the time for Cosmopedia was spent on meticulous prompt engineering"; prompts span "hundreds of topics",
  "achieving less than 1% duplicate content".

## Prompt curation
1. **Curated sources:** Stanford course outlines, Khan Academy, OpenStax, WikiHow. Limit: "we can extract only 16,000
   unique units from OpenStax and 250,000 from Stanford".
2. **Audience and style:** four audiences (young children, high school students, college students, researchers) and
   three generation styles (textbooks, blog posts, wikiHow articles) give "up to 12 times the number of prompts".
   Changing only the audience or format words "was insufficient to prevent a high rate of duplicate content"; the fix
   was "providing specific instructions on how the format and content should differ".
3. **Web data:** "over 80% of the prompts". Millions of web samples (a dataset like RefinedWeb) clustered into 145
   clusters; Mixtral labels each cluster from 10 random extracts and gives an educational score out of 10; clusters of
   low educational value are removed (112 topics retained). Prompts are conditioned on the topic only 50% of the time;
   23 million prompts. AutoMathText samples add mathematical content.
4. **Stories:** "In our initial assessments of models trained using the generated textbooks, we observed a lack of
   common sense and fundamental knowledge typical of grade school education." Stories seeded from UltraChat ("Questions
   about the world", 30 meta-concepts) and OpenHermes2.5 (unsuitable categories omitted) were added.
5. Repeated openings ("Once upon a time", "The sun hung low in the sky") were reduced by instructing the model to avoid them.

## Decontamination
"Similar to Phi-1, we identify potentially contaminated samples using a 10-gram overlap. After retrieving the
candidates, we employ difflib.SequenceMatcher ... If the ratio of len(matched_substrings) to len(benchmark_sample)
exceeds 0.5, we discard the sample." Removed samples (unique benchmark items in brackets):

| Dataset group | ARC | BoolQ | HellaSwag | PIQA |
|---|---|---|---|---|
| web data + stanford + openstax | 49 (16) | 386 (41) | 6 (5) | 5 (3) |
| auto_math_text + khanacademy | 17 (6) | 34 (7) | 1 (1) | 0 (0) |
| stories | 53 (32) | 27 (21) | 3 (3) | 6 (4) |

Fewer than 4 contaminated samples for MMLU, OpenBookQA, and WinoGrande.

## Result
cosmo-1b (1B, Llama2 architecture) "performs better than TinyLlama 1.1B on ARC-easy, ARC-challenge, OpenBookQA, and
MMLU and is comparable to Qwen-1.5-1B on ARC-challenge and OpenBookQA", with "some performance gaps compared to
Phi-1.5". Scores are shown only in a figure; cosmo-1b training tokens and hyperparameters are not given in the post.

## Limitations stated
Mixtral "may sometimes hallucinate and produce incorrect information", for example historical facts or mathematical
reasoning in the AutoMathText and KhanAcademy subsets; retrieval augmentation is proposed as a mitigation.

## Verification
- Read on 2026-09-15 at https://huggingface.co/blog/cosmopedia (fetched text).
