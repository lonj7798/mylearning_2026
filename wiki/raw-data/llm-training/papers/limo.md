<!-- scope: reasoning-trace synthesis — ~800 curated long-CoT traces that activate latent math reasoning
     deps: [[lima]], [[s1]]
     see-also: [[openmathinstruct]], [[star]], [[openmathinstruct-2]]
-->

# LIMO: Less is More for Reasoning
- **Core Insight:** A tiny set of carefully curated long-CoT examples can unlock strong math reasoning in a capable pretrained base; the bottleneck is not volume, but whether the traces provide the right cognitive template.
- **Guideline:** For reasoning SFT, spend effort on problem difficulty, trace quality, and reflective structure; a few hundred to about 1K examples can outperform much larger but flatter distillations.
- **Authors:** Yixin Ye, Zhen Huang, Yang Xiao, Ethan Chern, Shijie Xia, Pengfei Liu
- **Year:** 2025
- **URL:** https://arxiv.org/abs/2502.03387
- **Relevant topics:** reasoning-trace synthesis, long-CoT, SFT curation, latent-capability activation

## Abstract
The current arXiv version argues that complex mathematical reasoning can emerge from minimal but strategically designed demonstrations. The key claim is a "Less-Is-More Reasoning Hypothesis": if the base model already contains the needed domain knowledge from pretraining, then a small number of high-quality post-training examples that show reflective reasoning can elicit strong performance. In the current reported version, the model reaches 63.3% on AIME24 and 95.6% on MATH500 while using only 1% of the training data required by prior approaches.

## Key Contributions
- Small curated reasoning set: the original release uses 817 long-CoT samples across competition math, MATH, GSM8K-hard, and physics Olympiad problems.
- Formalizes the **Less-Is-More Reasoning Hypothesis**: strong latent knowledge from pretraining plus high-quality demonstrations are the two prerequisites.
- Shows that **question difficulty**, **solution-chain quality**, and **trace diversity** matter more than dataset size.
- Release includes data, model, and evaluation harnesses for replication.

## Synthesis pipeline (REQUIRED — concrete, modality-specific)
- **Seed input:** a large candidate pool of math and reasoning problems drawn from public competition-style sources.
- **Question selection:** keep only problems that strong baselines still find hard, so the dataset teaches multi-step reasoning rather than short-answer recall.
- **Solution generation:** collect multiple candidate traces from strong reasoning models and human editing, then keep traces that show reflective structure.
- **Quality scoring:** correctness of the final answer against the gold solution; presence of self-verification and re-checking segments; branching or backtracking markers that expose non-linear reasoning; fine-grained step granularity instead of outline-only answers.
- **Manual curation:** hand-filter down to the final small set, removing lucky guesses and traces with subtly broken intermediate logic.
- **Output shape:** long-CoT traces, typically thousands of tokens each, with explicit reflection and answer formatting.
- **Teacher model(s):** the paper/release uses a mix of strong reasoning teachers and human-edited traces rather than a single distilled teacher.
- **Cost / compute:** labor-heavy curation; the paper emphasizes data quality over large-scale generation.

## Modality-specific technical details (REQUIRED — reasoning-trace)
- **Reasoning length distribution:** long-CoT traces, with some examples reaching many thousands of tokens.
- **Trace style:** reflective long-CoT with verification and backtracking, structurally similar to o1 / R1 style traces.
- **Correctness verifier:** gold-answer exact match for the final answer; no step-level verifier is used.
- **Error-mode filter:** hand review removes traces with subtly wrong intermediate logic even if the final answer is right.

## Quality / diversity evaluation
- The current reported version reaches **63.3% AIME24** and **95.6% MATH500**, while also showing strong out-of-distribution gains.
- Ablations show that random or low-quality samples do not reproduce the effect; the quality gap is not reducible to volume.
- Generalization extends beyond math to broader reasoning benchmarks, which is the strongest evidence that the traces activate a latent capability rather than memorize one dataset.

## Risks + gotchas
- **Curator subjectivity:** the final set is hand-selected, so exact reproducibility depends on matching the curation policy.
- **Base-model dependence:** the result requires a strong reasoning-rich base; weak bases do not reliably activate from a tiny dataset.
- **Benchmark overlap:** competition-style sources make contamination auditing non-trivial.

## Connections
- Twin of [[s1]] — same hypothesis, independent replication with hand curation vs semi-automated filter.
- Extends [[lima]] "Less Is More for Alignment" into reasoning.
- Contrasts mass-distillation ([[openmathinstruct]], [[openmathinstruct-2]], [[bespoke-stratos]], [[openr1]]).
- Connects to the latent-capability debate: [[rlvr-beyond-base-model]], [[front-loading-reasoning]].
