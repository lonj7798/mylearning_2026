<!-- scope: reasoning-data curation - open-source question sourcing, multi-answer generation, and filtering for reasoning models
     deps: [[limo]], [[s1]], [[deepseek-r1]]
     see-also: [[lima]], [[openmathinstruct]], [[openmathinstruct-2]], [[front-loading-reasoning]], [[lets-verify]], [[phi-textbooks]]
-->

# OpenThoughts: Data Recipes for Reasoning Models
- **Core Insight:** Reasoning data quality comes from the whole recipe, not just the teacher model. OpenThoughts shows that question sourcing, question filtering, answer multiplicity, and teacher choice can be tuned systematically to beat much larger open-data baselines.
- **Guideline:** Use a small number of strong sources, sample multiple answers per question, prefer LLM-based difficulty/length filters over generic embedding filters, and do not assume a stronger benchmark model is automatically a better teacher.
- **Authors:** Etash Guha et al.
- **Year:** 2025
- **URL:** https://arxiv.org/abs/2506.04178
- **Relevant topics:** reasoning data curation, SFT, teacher selection, verification, open datasets

## Abstract
OpenThoughts is an open-source project for building reasoning datasets for math, code, and science. The first wave, OpenThoughts2-1M, produced OpenThinker2-32B, the first model trained on public reasoning data to match DeepSeek-R1-Distill-32B on standard reasoning benchmarks. The second wave, OpenThoughts3, comes from 1,000+ controlled experiments over the data pipeline; scaling that recipe to 1.2M examples with QwQ-32B as teacher yields OpenThinker3-7B, which reaches 53% on AIME 2025, 51% on LiveCodeBench 06/24-01/25, and 54% on GPQA Diamond.

## Key Contributions
- **OpenThoughts-114K -> OpenThoughts2-1M -> OpenThoughts3-1.2M:** a staged public reasoning-data program rather than a single dataset release.
- **Systematic recipe search:** over 1,000 ablations across math, code, and science.
- **OpenThinker3-7B result:** a Qwen2.5-7B-Instruct finetune that becomes the strongest open-data 7B reasoning model in the project's table.
- **Full release:** data, code, evaluation code, and models are published on the OpenThoughts site and Hugging Face.

## Data Recipe
- **Stage 1, OpenThoughts-114K:** scales the Sky-T1 pipeline with automated verification.
- **Stage 2, OpenThoughts2-1M:** expands question diversity with synthetic question generation.
- **Stage 3, OpenThoughts3-1.2M:** searches the recipe space more systematically and scales to 850k math, 250k code, and 100k science examples.
- **Question sourcing:** the paper finds that using a small number of top-quality sources beats optimizing for source diversity.
- **Answer multiplicity:** sampling multiple answers per question is the easiest way to expand a source by at least 16x.
- **Teacher choice:** QwQ-32B beats DeepSeek-R1 as a teacher even though DeepSeek-R1 scores higher on many target benchmarks.

## Technical Details
- **Base model:** all main OpenThoughts3 models are finetuned from Qwen2.5-7B-Instruct.
- **Answer filtering:** the paper tests many verification and answer-filtering methods, but none beat training on all answers without filtering.
- **Question filtering:** LLM-labeled difficulty and response-length filters outperform embedding-based and fastText-style heuristics.
- **Deduplication policy:** the final pipeline uses exact deduplication for math and science, and no deduplication for code.
- **Evaluation:** OpenThinker3-7B reports 69.0 AIME24, 53.3 AIME25, 90.0 MATH500, 51.7 LiveCodeBench 06/24-01/25, and 53.7 GPQA-Diamond in the project table.

## Risks + Gotchas
- **Teacher quality is not enough:** a stronger benchmark model can be a worse data generator for this task.
- **Filtering can shrink the dataset too much:** the paper repeatedly shows that some "cleanup" steps do not pay for the lost scale.
- **Recipe is domain-sensitive:** the strongest settings are different for math, code, and science, so there is no single universal filter.
- **Open-data advantage is fragile:** later gains depend on keeping the entire pipeline open and reproducible, not just the final model weights.

## Connections
- Strong companion to [[s1]]: both argue that curation and test-time reasoning budget matter more than raw scale.
- Extends [[limo]] and [[lima]] into a larger, cross-domain data-recipe program.
- Related to [[openmathinstruct]] and [[openmathinstruct-2]] as open reasoning-data efforts, but OpenThoughts is more explicit about end-to-end recipe search.
- Links to [[deepseek-r1]] and [[lets-verify]] through teacher distillation and verification, but OpenThoughts emphasizes SFT data engineering over RL.
