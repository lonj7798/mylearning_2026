<!-- scope: long-context synthetic-data/eval protocol built by embedding bAbI reasoning tasks inside long natural-text backgrounds
     see-also: [[ruler]], [[longalign]], [[magpie]]
-->

# BABILong: Testing the Limits of LLMs with Long Context Reasoning-in-a-Haystack
- **Core Insight:** BABILong turns short synthetic reasoning problems into arbitrarily long-context tasks by embedding bAbI facts inside real book text, showing that long-context retrieval is much easier than long-context reasoning.
- **Guideline:** For long-context evaluation, do not stop at needle retrieval; use hybrid generators like BABILong that separately stress retrieval, supporting-fact aggregation, and symbolic reasoning as context length scales.
- **Authors:** Yuri Kuratov, Aydar Bulatov, Petr Anokhin, Ivan Rodkin, Dmitry Sorokin, Artyom Sorokin, Mikhail Burtsev
- **Year:** 2024
- **URL:** https://arxiv.org/abs/2406.10149
- **Relevant topics:** long-context evaluation, synthetic data generation, bAbI, long-context reasoning, retrieval vs reasoning, PG19, templated tasks

## Abstract
BABILong is a scalable long-context benchmark built by taking the 20 bAbI reasoning tasks and hiding their facts inside long natural documents from PG19. This makes the benchmark useful as a controlled synthetic-data generator, not just a static test set: the reasoning task stays fixed while the surrounding context can be extended to thousands or millions of tokens. The paper shows that many long-context LLMs degrade sharply as length and reasoning complexity increase, often using only a small fraction of their advertised context window effectively. The benchmark is explicitly designed to remain extendable to future context lengths.

## Key Contributions
- Converts the classic **20-task bAbI suite** into a long-context benchmark by mixing task facts with long natural-text distractors from **PG19**.
- Defines a **length-scaling protocol** where background sentences are added in natural order until a target token budget is reached, making the benchmark extendable to very large windows.
- Creates a **hybrid synthetic-natural setup**: reasoning structure is templated and contamination-resistant, while the distractor context is real prose rather than artificial filler.
- Shows that **reasoning-in-a-haystack** is materially harder than simple needle retrieval, especially on tasks requiring multiple supporting facts, counting, deduction, or ordering.
- Provides public fixed evaluation splits up to **10M tokens**, with the paper also reporting experiments up to **50M tokens** for some recurrent-memory systems.

## Key Figures/Tables to Study
- **Figure 1** - the core artifact: it visualizes the generation protocol where bAbI facts are hidden inside PG19 book text and shows why the benchmark is about retrieval plus reasoning, not retrieval alone.
- **Table 1** - useful for understanding the first ten bAbI-derived task types, the number of facts per task, and how much difficulty already exists even at `0k`.
- **Table 2** - the main long-context headline: accuracy collapses well before the full advertised window for many models.
- **Section 2 / dataset description** - the most important text for data design because it explains how examples are composed and why the benchmark scales.
- **Appendix N (datasheet)** - useful for exact split construction details, including which source splits are used and how public benchmark subsets are sampled.

## Technical Details
### What is synthetic, and what is natural
- The **reasoning problem** is synthetic and templated. BABILong inherits the bAbI tasks, which are generated from simulated micro-world interactions like movements, object transfers, counting events, temporal ordering, and simple deduction.
- The **background context** is natural text. The paper uses **PG19 books** because they are long, coherent documents that allow construction of very long contexts.
- Each final example is therefore a **hybrid composition**:
  - templated task facts, distractor facts, question, and target answer from bAbI
  - unprocessed background sentences from PG19
- This matters because BABILong is not merely a benchmark card. It is a reusable **data-generation recipe** for long-context stress testing.

### Generation protocol
- The benchmark starts from a standard bAbI sample: a short set of facts plus a question-answer pair.
- Those task sentences are then **hidden between irrelevant sentences** drawn from another distribution, namely PG19.
- The paper describes generation as **gradually adding background sentences in their natural order until the augmented sample reaches the desired length**.
- In the datasheet, the authors describe each sample as **unprocessed bAbI sentences mixed between unprocessed PG19 sentences**.
- The question can appear **at the beginning or at the end** of the resulting sequence.
- Because the background can keep growing while the core reasoning task stays the same, the same task family can be evaluated at many different context lengths.

### Task structure and why it is stronger than simple NIAH
- BABILong inherits **20 reasoning tasks** from bAbI rather than a single retrieval template.
- Early tasks include **single supporting fact**, **two supporting facts**, **three supporting facts**, **yes/no**, **counting**, and **lists/sets**.
- The broader suite also includes reasoning families such as **deduction**, **induction**, **time reasoning**, **path finding**, and related symbolic skills from bAbI.
- This makes the benchmark more informative than a standard needle-in-a-haystack setup:
  - some tasks mainly test retrieval of one supporting sentence
  - others require identifying multiple relevant facts
  - harder tasks require combining those facts with symbolic reasoning
- The paper explicitly argues that current models may retrieve relevant facts yet still fail on the downstream reasoning step.

### Length-scaling protocol
- BABILong is designed to be **extendable to arbitrary lengths** because the background text can continue to grow without changing the underlying task.
- Public benchmark configurations include lengths such as `0k`, `1k`, `2k`, `4k`, `8k`, `16k`, `32k`, `64k`, `128k`, `256k`, `512k`, `1M`, and `10M`.
- The paper positions **10M tokens** as the largest predefined split and reports some model evaluations up to **50M tokens**.
- This protocol is important methodologically:
  - it lets researchers scale **context length** while keeping the **reasoning template** comparable
  - it exposes where models fail because of retrieval/load rather than because the task definition changed
- In that sense, BABILong is closer to a **synthetic scaling harness** than to a conventional fixed benchmark.

### Splits, sampling, and evaluation design
- The benchmark combines **all test samples from bAbI** with **PG19 test text** to form the full test pool.
- For the public evaluation set with **100 samples per task per length**, the authors randomly sample from that larger pool.
- The paper’s datasheet also notes an extended version with **1,000 samples per split** for a subset of tasks.
- For training, the authors inherit **bAbI train splits** and sample background text from **PG19 train**.
- The benchmark is attractive for eval design because it is mostly free of standard human-labeling bottlenecks:
  - answer labels come directly from the bAbI task definition
  - no LLM judge is needed for grading
  - no human annotation pass is required for each new length

### What BABILong teaches about long-context data and eval design
- **Retrieval and reasoning should be separated analytically.** A model can find a single relevant sentence and still fail on multi-fact composition.
- **Hybrid synthetic-natural construction is powerful.** Purely templated tasks give control; natural distractors stop the benchmark from becoming trivial.
- **Length scaling should preserve task identity.** The background grows while the reasoning problem stays stable, which makes comparisons cleaner.
- **Generated tasks help with contamination resistance.** The paper argues that generated benchmarks like bAbI and BABILong are less exposed to train-test leakage than many popular natural benchmarks.
- **Distribution contrast can be a feature and a limitation.** The short bAbI sentences remain distinguishable from book prose, which helps scalability, but it also means the setup is cleaner than real document reasoning.

## Connections
- Complements [[ruler]]: RULER broadens synthetic long-context evaluation across retrieval, tracing, aggregation, and QA, while BABILong goes deeper on symbolic reasoning over inserted facts.
- Connects to [[longalign]] by showing why long-context capability needs structure-aware data and eval, not only larger context windows.
- Relevant to synthetic-data pages like [[magpie]] for a different reason: BABILong is not instruction synthesis, but it is still a strong example of **generator design as methodology**.
- Useful alongside model reports that make long-context claims, because it gives a controlled way to distinguish advertised context size from usable reasoning depth.

## Source URLs
- https://arxiv.org/abs/2406.10149
- https://proceedings.neurips.cc/paper_files/paper/2024/hash/c0d62e70dbc659cc9bd44cbcf1cb652f-Abstract-Datasets_and_Benchmarks_Track.html
- https://proceedings.neurips.cc/paper_files/paper/2024/file/c0d62e70dbc659cc9bd44cbcf1cb652f-Paper-Datasets_and_Benchmarks_Track.pdf
- https://github.com/booydar/babilong
- https://huggingface.co/datasets/RMT-team/babilong
