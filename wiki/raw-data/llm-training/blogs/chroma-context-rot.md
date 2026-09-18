<!-- scope: Chroma technical report (Jul 2025) — inference-only evaluation of 18 LLMs with task difficulty held fixed while input length varies: four NIAH variants, LongMemEval focused vs full prompts, and a repeated-words copy task
     deps: [[needle-in-haystack-data]]
     see-also: [[context-length-alone-hurts]], [[nolima]], [[longmemeval]], [[fiction-livebench]], [[ruler]]
-->

# Context Rot: How Increasing Input Tokens Impacts LLM Performance
- **Core Insight:** Across 18 LLMs tested with the task held fixed and only input length varied, performance degrades as input grows; the drop is faster for needle-question pairs with lower embedding similarity (tested ranges 0.445-0.775 and 0.521-0.829), and on LongMemEval every model family scores higher on ~300-token focused prompts than on ~113k-token full prompts (Needle-Question Similarity; LongMemEval sections).
- **Guideline:** When a long-context evaluation is meant to show that a model uses its whole window, vary input length with the task held fixed, include low-similarity needle-question pairs and topically related distractors, and score abstentions separately from wrong answers, because this report finds that each of these factors changes the length-dependent drop and that model families differ in abstaining versus hallucinating (Impact of Distractors; LongMemEval).
- **Authors:** Kelly Hong, Anton Troynikov, Jeff Huber (Chroma)
- **Year:** 2025 (Chroma Technical Report, July 14, 2025; not on arXiv; footnote update July 16, 2025)
- **URL:** https://www.trychroma.com/research/context-rot ; code: https://github.com/chroma-core/context-rot
- **Source type:** practitioner evidence (vendor technical report with released code; not peer reviewed)
- **Relevant topics:** long-context evaluation, effective context length, distractors, haystack structure, abstention vs hallucination, LLM-as-judge, repeated-output fidelity

## Summary
The report tests whether models handle the 10,000th input token as reliably as the 100th. It extends Needle in a Haystack (NIAH: one fact, the "needle", placed in unrelated text, the "haystack") in four controlled ways: needle-question similarity, distractors, needle-haystack similarity, and haystack structure. It adds a conversational QA test on LongMemEval and a synthetic task where the model copies a sequence of repeated words. Task complexity is held constant and only input length changes. The authors report non-uniform degradation with length in all experiments and conclude that how information is presented in context matters, not only whether it is present. The report does not explain the mechanism and does not train any model.

## Key Contributions
- Evaluation of 18 closed and open-weights LLMs showing non-uniform performance as input length increases (Contributions).
- A write-up of model-family behavior with distractors and with varying needle-question similarity (Contributions).
- Two controls that most NIAH setups lack: embedding-measured needle-question similarity, and original vs sentence-shuffled haystacks (Needle in a Haystack Extension).
- A focused (~300 tokens) vs full (~113k tokens) comparison on the same LongMemEval questions (LongMemEval).
- Released codebase with `experiments/niah_extension/`, `experiments/longmemeval/`, `experiments/repeated_words/` (GitHub README).

## Key Figures/Tables to Study
- "NIAH: Needle-Question Similarity" (arXiv haystack/arXiv needles): high- vs low-similarity needles across input length, grouped by model performance tier.
- "Impact of Distractors: Performance by Number of Distractors", "Performance by Individual Distractors", and "Failure Analysis" (arXiv haystack/PG essay needles).
- "Haystack Structure: Averaged Performance Across 18 Models for Original vs Shuffled Haystacks".
- "LongMemEval Results" for the Claude, GPT, Gemini, and Qwen families, and "Results by Question Type - Claude Opus 4".
- "Repeated Words" per family, with "Position Accuracy" and "Word Count Difference" plots.
- Per-model accuracies for NIAH and LongMemEval appear only in these figures; the text gives no per-model numbers.

## Technical Details
Loci are section names on the web page.

**Common setup**
- Models (18): Claude Opus 4, Sonnet 4, Sonnet 3.7, Sonnet 3.5, Haiku 3.5; o3, GPT-4.1, GPT-4.1 mini, GPT-4.1 nano, GPT-4o, GPT-4 Turbo, GPT-3.5 Turbo; Gemini 2.5 Pro, 2.5 Flash, 2.0 Flash; Qwen3-235B-A22B, Qwen3-32B, Qwen3-8B. Not every model appears in every experiment, because of context-window or thinking-budget limits (Appendix › Models Tested).
- NIAH grid: 8 input lengths and 11 needle positions for each combination of needle type, haystack topic, and haystack structure, across each model's maximum context window (Details).
- temperature=0 except where incompatible (o3) or discouraged (Qwen thinking mode); Qwen models extended from 32,768 to 131,072 tokens with YaRN; standard and thinking modes both run (Details).
- Judge: GPT-4.1, aligned by manually labeling ~500 NIAH outputs and ~600 LongMemEval outputs and iterating the judge prompt until agreement > 0.99 (Appendix › LLM judge alignment).
- Refusals to attempt: 69 of 194,480 LLM calls (0.035%) (Details).

**Needle-question similarity**
- Similarity is cosine similarity averaged over five embedding models: text-embedding-3-small, text-embedding-3-large, jina-embeddings-v3, voyage-3-large, all-MiniLM-L6-v2 (Needle-Question Similarity).
- Haystacks are Paul Graham (PG) essays and arXiv papers. Topics were found by chunking into 1-3 sentences, embedding with text-embedding-3-large, UMAP (n_neighbors=30, min_dist=0.05, n_components=50), HDBSCAN (min_cluster_size=10, min_samples=15), and 20 MMR-selected chunks per large cluster (Experiment).
- The top-10 chunks retrieved for each question were checked manually to confirm the haystack holds no answer, so wrong answers count as hallucinations (Experiment).
- 8 hand-written needles per question, each assigned to the topic cluster with > 0.9 probability; similarity ranges 0.445-0.775 (PG) and 0.521-0.829 (arXiv), standard deviation < 0.1 across embedding models (Experiment).
- Result: lower-similarity pairs degrade faster with length; at short lengths models succeed on low-similarity pairs too; no notable variation across the 11 needle positions (Results).

**Distractors**
- One high-similarity needle (second highest of 8) plus 4 hand-written distractors; conditions: needle only, needle + 1 random distractor, needle + all 4 (Impact of Distractors › Experiment).
- One distractor lowers performance relative to baseline and four lower it further; the effect grows with input length. Distractor 3 causes the largest decline, and distractors 2 and 3 appear most often in hallucinated answers (arXiv haystack/PG needles) (Results).
- Claude models have the lowest hallucination rates; Claude Sonnet 4 and Opus 4 tend to abstain when uncertain. GPT models have the highest hallucination rates (Results).

**Needle-haystack similarity**
- Mean top-5 similarity: PG haystack, PG needles 0.529 and arXiv needles 0.368; arXiv haystack, arXiv needles 0.654 and PG needles 0.394 (Needle-Haystack Similarity › Experiment).
- In the PG haystack, arXiv needles score significantly better; in the arXiv haystack, differences are minimal. The authors state two topics are insufficient for a general conclusion (Results).

**Haystack structure**
- Original excerpts vs sentences randomly reordered. "Across all 18 models and needle-haystack configurations", models perform better on shuffled haystacks; the plotted figure averages over the 18 models (Haystack Structure › Results).

**LongMemEval**
- LongMemEval_s filtered to knowledge-update, temporal-reasoning, and multi-session questions; 38 ambiguous or unanswerable prompts removed, leaving 306 prompts averaging ~113k tokens; focused versions average ~300 tokens (LongMemEval › Experiment).
- All models score significantly higher on focused prompts. The Claude family shows the largest gap, driven mostly by abstentions; Opus 4 and Sonnet 4 score lower on full prompts than older Claude models. Thinking mode raises both conditions, but a gap remains (Results).
- Question-type order without thinking: knowledge update > multi-session > temporal reasoning; with thinking: knowledge update > temporal reasoning > multi-session (Results).

**Repeated words**
- 7 word pairs; 1,090 variants per pair; lengths 25, 50, 75, 100, 250, 500, 750, 1,000, 2,500, 5,000, 7,500, 10,000 words; max_output_tokens = 2 × input tokens; thinking budget 0 or minimum (128 for Gemini 2.5 Pro); o3 excluded; scored by normalized Levenshtein distance (Repeated Words › Experiment).
- GPT-3.5 Turbo excluded after refusing 60.29% of tasks via content_filter (Experiment).
- Performance degrades with length for all models. Claude Sonnet 3.5 beats newer Claude models up to its 8,192-token output limit; Opus 4 degrades slowest but refuses 2.89% of attempts. GPT-4.1 refuses 2.55%, typically from ~2,500 words. Gemini models (except 2.5 Flash on "apples"/"apple") emit words absent from the input from ~500-750 words. Qwen3-8B does not attempt 4.21% and emits unrelated text from ~5,000 words (Results).
- The unique word is placed correctly most often when it is near the start of the sequence (Results).

## Findings relevant to generality, negative feedback, long context, agentic training, distillation
### Long context
- A model that passes short-length versions of a task still loses accuracy as irrelevant content is added within its advertised window; the authors attribute the decline to input size because the needle-question pair is held fixed (Needle-Question Similarity › Results).
- The authors expect larger degradation on real tasks that require synthesis or multi-step reasoning (Interpretation; Limitations & Future Work).
- The report does not explain the mechanism; it names attention sensitivity to input structure as a direction for interpretability work (Haystack Structure; Limitations).
- Output length matters too: when output grows with input, copy fidelity falls, and models over- or under-generate relative to the input word count (Repeated Words).
### Evaluation of calibration under long inputs
- Accuracy alone mixes two failure modes. Under distractors and on LongMemEval full prompts, Claude models lose points mainly through abstention and GPT models through confident wrong answers (Impact of Distractors; LongMemEval).

## Connections
- [[needle-in-haystack-data]] — the original NIAH harness with PG-essay haystacks that this report extends.
- [[nolima]] — non-lexical NIAH variant; the report notes 72.4% of NoLiMa pairs need external world knowledge.
- [[longmemeval]] — source of the LongMemEval_s prompts used for the focused vs full comparison.
- [[michelangelo]] and [[openai-mrcr-graphwalks]] — Latent List, MRCR, and Graphwalks tasks discussed in Related Work.
- [[yarn]] — used to extend Qwen3 models to 131,072 tokens in these runs.
- [[context-length-alone-hurts]] — paper that controls for perfect retrieval and still finds length-driven drops.
- [[lost-in-the-middle]] — position effects; this report finds no notable needle-position variation for its NIAH task.
- [[ruler]] and [[fiction-livebench]] — other evaluations that vary length or go beyond lexical retrieval.
- [[anthropic-context-engineering]] — practitioner guidance on context engineering, the direction this report recommends.

## Verification
- Created on 2026-09-14 from https://www.trychroma.com/research/context-rot (web page dated July 14, 2025, footnote update July 16, 2025); experiment folders checked at https://github.com/chroma-core/context-rot.
- Audit claims not found in the source: (1) "Design implications: keep lexical overlap between question and evidence low, use several distractors, and measure abstention versus hallucination" is the audit's recommendation, not a statement in the report. (2) "All 18 models did better on shuffled haystacks" as a per-model claim: the text says the pattern holds "across all 18 models and needle-haystack configurations", but the only figure averages over models and no per-model values are given. (3) "Gemini 2.5 Pro produced words not in the input starting around 500-750 words": the report states this onset for the Gemini family (all pairs and models except 2.5 Flash on "apples"/"apple"), with 2.5 Pro showing the greatest variability.
- Not reported by the source: per-model accuracy numbers in text, confidence intervals, number of runs per cell, exact token counts for the 8 NIAH input lengths.
