<!-- scope: The Pile as a diverse multi-source pretraining corpus
     deps: [[c4]]
     see-also: [[dolma]], [[data-constrained-scaling]]
-->

# The Pile: An 800GB Dataset of Diverse Text for Language Modeling
- **Core Insight:** Diversity of source domains is itself a scaling variable; broad, high-quality mixtures outperform monolithic web corpora on cross-domain generalization.
- **Guideline:** When building a pretraining mix, do not rely only on generic web text; add curated academic, code, forum, and book-like sources with explicit mixture control.
- **Authors:** Leo Gao, Stella Biderman, Sid Black, Laurence Golding, Travis Hoppe, Charles Foster, Jason Phang, et al.
- **Year:** 2020
- **URL:** https://arxiv.org/abs/2101.00027
- **Relevant topics:** corpus mixture design, source diversity, domain balance

## Abstract
The Pile is an 825 GiB English text corpus built from 22 diverse high-quality subsets spanning academic text, code, books, web text, and forums. The main claim is that source diversity materially improves cross-domain performance relative to generic crawl baselines.

## Key Contributions
- Made explicit mixture design a first-class pretraining decision.
- Released a broad-source open corpus that became a baseline for open LMs.
- Showed benefits of curated-domain coverage beyond raw crawl scale.

## Technical Details
- 22 component datasets with manually chosen mixture weights.
- Includes sources such as PubMed, arXiv, GitHub, books, StackExchange, and web text.
- Emphasizes domain coverage rather than only crawl cleanup.
- Also documents risks and problematic sources, which helped push later data documentation standards.

## Connections
- Contrasts with crawl-centric stacks like [[c4]] and [[ccnet]].
- A conceptual precursor to transparent open mixtures like [[dolma]].

