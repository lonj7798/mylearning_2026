<!-- scope: Sebastian Raschka synthetic-data overview as a practical reference note
     see-also: [[hf-cosmopedia]], [[rephrasing-the-web]]
-->

# Sebastian Raschka: Synthetic Data Overview
- **Core Insight:** Synthetic data is best understood as a family of augmentation and generation methods, not one monolithic trick.
- **Guideline:** Use synthetic data deliberately: decide whether you are rewriting, backtranslating, bootstrapping, or full-sample generating, and measure what distribution shift each method introduces.
- **Author/Org:** Sebastian Raschka
- **Year:** 2024-2026 reference material
- **URL:** https://sebastianraschka.com/books/ml-q-and-ai-chapters/ch15/
- **Relevant topics:** synthetic data, augmentation, rewriting, backtranslation, fine-tuning

## Summary
Raschka’s synthetic-data discussion is a practical engineering overview rather than a frontier model report. It is useful here because it lays out the main synthetic-data mechanisms plainly: rewrite existing text, backtranslate, or generate new examples from a model, each with different tradeoffs in diversity, faithfulness, and noise.

## Key Points
- Synthetic data can preserve label structure while varying surface form.
- Backtranslation and rewriting are lower-risk than unconstrained generation.
- Full synthetic generation offers more diversity but increases hallucination risk.
- Synthetic data becomes especially useful during fine-tuning when labeled data is scarce.

## Connections
- Practical counterpart to corpus-level methods like [[hf-cosmopedia]] and [[rephrasing-the-web]].
- Conceptually adjacent to instruction-generation pipelines like [[self-instruct]] and [[humpback]].

