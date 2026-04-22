<!-- scope: content-grounded synthesis for RAG / LFQA / summarization / extraction (ICLR 2024)
     deps: [[self-instruct]]
     see-also: [[persona-hub]], [[ultrachat-construction]]
-->

# Genie: Achieving Human Parity in Content-Grounded Datasets Generation
- **Core Insight:** For tasks where outputs must be grounded in a specific document (RAG, summarization, long-form QA, information extraction), synthesize data in three stages — prepare content, generate task-specific examples grounded in that content, then filter aggressively for faithfulness + well-formedness — the result can reach human-parity on blind quality evaluation.
- **Guideline:** To build a content-grounded SFT corpus (for RAG or tool-use): select a document corpus, prompt a teacher with `(document, task spec)` to produce `(question, answer, citations)` triples, filter with a faithfulness classifier, train on what remains.
- **Authors:** Asaf Yehudai, Boaz Carmeli, Yosi Mass, Ofir Arviv, Nathaniel Mills, Eyal Shnarch, Leshem Choshen (IBM Research)
- **Year:** 2024 (ICLR 2024)
- **URL:** https://arxiv.org/abs/2401.14367
- **Relevant topics:** content-grounded synthesis, RAG training data, faithfulness filtering, LFQA, summarization

## Abstract
Genie is a three-stage pipeline for content-grounded data synthesis: (1) Content Preparation — select and preprocess documents; (2) Generation — few-shot prompt a teacher LLM to produce task-specific examples (question-answer pairs, summaries, extractions) grounded in the content; (3) Filtering — apply faithfulness + well-formedness + quality filters. Across three task families (LFQA, summarization, information extraction), human raters preferred Genie-generated data roughly on par with human-written data. Models trained on Genie data match or beat those trained on human datasets (ELI5, ASQA, CNN-DailyMail).

## Key Contributions
- **Three-stage pipeline** (Prepare → Generate → Filter) as a canonical template for content-grounded synthesis.
- Demonstrated **human-parity** in blind quality evaluation.
- Released three synthetic datasets: LFQA-synth, Summ-synth, InfoExtract-synth.
- Established faithfulness classifiers as a critical filter for content-grounded synthesis.

## Synthesis pipeline (REQUIRED — be concrete)

### Stage 1 — Content Preparation
- Select a corpus (e.g. Wikipedia, news, domain-specific documents).
- Chunk / split documents into contexts of manageable size (≤4K tokens).
- Optionally tag with metadata (topic, genre, date).

### Stage 2 — Generation (few-shot)
- Prompt template (schematic for LFQA):
  ```
  Given the following document, generate a long-form question that requires synthesizing information across multiple parts of the document, and provide a detailed, grounded answer.
  Document: {context}
  Question:
  Answer:
  ```
- Similarly tailored prompts for summarization and information extraction.
- Teacher: a capable LLM (paper uses GPT-4-class).

### Stage 3 — Filtering
- **Faithfulness filter:** a classifier (NLI-style or finetuned faithfulness scorer) checks whether the answer is entailed by the document — drop if not.
- **Well-formedness filter:** format / language / length constraints.
- **Overall quality filter:** LLM-judge quality score.

## Output shape
- Task-specific datasets released per family (LFQA, Summ, Extract).
- Sizes in the tens to hundreds of thousands per task.

## Quality / diversity evaluation
- Human evaluators rated Genie-generated examples against human-written examples — preferences were statistically close to 50-50 (human parity).
- Models fine-tuned on Genie-synth compared favorably to ELI5 / ASQA / CNN-DailyMail-trained baselines.
- Faithfulness-filter ablation: removing it significantly degrades downstream grounding.

## Risks + gotchas
- **Faithfulness classifier is the binding quality gate** — poor classifier = hallucination-contaminated data.
- **Document coverage bias** — if the content corpus is skewed, so is the synthetic data.
- **Teacher-bound ceiling** — answers cannot exceed teacher's understanding of the document.
- **Scope narrow to content-grounded tasks** — not applicable to open-ended generation.

## Connections
- Content-grounded counterpart of [[self-instruct]] / [[ultrachat-construction]] (which are content-free).
- Essential for modern RAG training stacks and tool-use training.
- Shares the **three-stage (prepare, generate, filter)** template with later content-grounded pipelines.
- Faithfulness-filter idea is the conceptual seed for "synthetic-data verification" lines like [[faithful-synth-eval]] discussions.
