<!-- scope: aggregated cross-source insights for the llm-training raw library
     deps: [[README]], [[COLLECTION-PLAN]]
     see-also: [[phi-textbooks]], [[open-thoughts]], [[tulu-3-sft-mix]]
-->

# LLM Training Techniques — Insights Index

This page is the cross-source map for the raw library. It starts with the synthetic-data layer because that is where the current training literature has become most fragmented and most important.

## Synthetic Data

- Synthetic data is not one recipe. There are at least three distinct regimes in the library: synthetic pretraining corpora ([[phi-textbooks]], [[phi-1-5]], [[rephrasing-the-web]], [[hf-cosmopedia]]), synthetic post-training prompts and responses ([[magpie]], [[persona-hub]], [[tulu-3-sft-mix]], [[nemotron-4-synthetic]]), and synthetic reasoning traces ([[openmathinstruct]], [[open-thoughts]], [[s1]], [[limo]], [[sky-t1]]).
- Prompt diversity is now a first-order bottleneck. [[persona-hub]], [[magpie]], and [[hf-cosmopedia]] all arrive at the same lesson from different angles: stronger teachers alone are not enough; you need a wide prompt manifold, explicit style variation, and deduplication that kills near-clones rather than only exact copies.
- Synthetic pretraining works when it raises information density, not when it merely paraphrases noise. [[phi-textbooks]], [[phi-1-5]], [[rephrasing-the-web]], and [[hf-cosmopedia]] all support the same practical claim: a smaller amount of structured, pedagogical, or rewritten text can beat much larger raw-web budgets for small and mid-sized models.
- Synthetic post-training quality comes from filtering and mixture design more than raw sample count. [[deita]], [[ifd]], [[superfiltering]], [[cherry-llm]], [[magpie]], and [[tulu-3-sft-mix]] all show that selection pressure is load-bearing: complexity, quality, diversity, and capability balance matter more than just harvesting another million prompts.
- Preference data is increasingly synthetic too. [[west-of-n]], [[ultrafeedback-construction]], and [[nemotron-4-synthetic]] show the same pattern: once you have a usable judge or reward model, you can manufacture sharper preference pairs than many human pipelines, but calibration risk becomes central because the judge is now part of the data generator.
- The reasoning-data frontier has moved from “more traces” to “better recipe search.” [[openmathinstruct]] and [[openmathinstruct-2]] are scale-heavy; [[open-thoughts]] studies the recipe itself; [[s1]] and [[limo]] show that a carefully curated thousand-or-less traces can beat much larger noisy sets; [[sky-t1]] shows the cost of entry for open reasoning distillation has collapsed.
- Tool and agent data need verification, not just generation. [[toolformer]] uses perplexity reduction as the acceptance test, [[toolllm]] uses trajectory search and execution, and [[apigen-mt]] pushes further toward verifiable multi-turn function-calling. This is the same underlying rule as reasoning traces: synthetic data gets much more valuable once there is an external correctness signal.
- Failure modes are part of the synthetic-data literature, not a separate ethics appendix. [[model-collapse]] and [[strong-model-collapse]] explain why recursive self-training can drift or collapse; the practical counterweight visible across [[tulu-3-sft-mix]], [[nemotron-4-synthetic]], and [[deepseek-r1-distill-synth]] is to anchor synthetic data with stronger teachers, public corpora, executable checks, or narrow task verifiers instead of blindly training on self-generated text.

## Open Gaps

- The source-collection checklist is now filled. The next work is no longer “missing pages,” but cleanup and consolidation.
- Highest-value next pass: repair stale internal wikilinks and normalize cross-namespace links between `papers/`, `blogs/`, `frameworks/`, `labs/`, and `model-reports/`.
- After link cleanup, the next useful upgrade is depth, not breadth: strengthen thinner cards and add richer `insights.md` synthesis beyond the synthetic-data section.
