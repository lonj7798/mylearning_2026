<!-- scope: instruction backtranslation as self-alignment from web text
     deps: [[self-instruct]]
     see-also: [[genie]], [[openhermes]]
-->

# Humpback: Self-Alignment with Instruction Backtranslation
- **Core Insight:** You can synthesize instruction-following data directly from raw documents by backtranslating likely instructions from the document text, then curating the resulting pairs.
- **Guideline:** For synthetic SFT without a proprietary teacher answering hand-written prompts, start from a web corpus, infer plausible instructions for each document, and curate those instruction-document pairs before fine-tuning.
- **Authors:** Xian Li, Ping Yu, Chunting Zhou, Timo Schick, Luke Zettlemoyer, Omer Levy, Jason Weston, Mike Lewis
- **Year:** 2023
- **URL:** https://arxiv.org/abs/2308.06259
- **Relevant topics:** instruction backtranslation, self-alignment, synthetic SFT, web-grounded synthesis

## Abstract
Humpback proposes instruction backtranslation: generate instructions from existing text rather than generating text from instructions. This allows scalable self-alignment from a web corpus with iterative self-augmentation and self-curation.

## Key Contributions
- Introduced instruction backtranslation as a scalable alignment primitive.
- Reused raw documents as grounding for synthetic instruction generation.
- Showed strong open alignment results without relying purely on proprietary teacher-chat distillation.

## Technical Details
- Start from a seed aligned model.
- Apply the model to web documents to infer likely instruction prompts.
- Curate the generated instruction-document pairs by quality.
- Fine-tune a stronger student on the curated synthetic set.

## Connections
- Alternative synthetic-data direction to [[self-instruct]] and [[alpaca]].
- A conceptual predecessor to richer grounded-generation methods like [[genie]].

