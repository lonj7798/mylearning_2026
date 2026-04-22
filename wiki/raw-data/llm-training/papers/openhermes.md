<!-- scope: OpenHermes open synthetic instruction/chat mixture
     see-also: [[openhermes-2-5]], [[wizardlm]], [[orca]]
-->

# OpenHermes
- **Core Insight:** Strong open instruct datasets can be built as curated mixtures of multiple synthetic sources rather than one monolithic pipeline.
- **Guideline:** For open chat mixtures, keep provenance at the source-dataset level, prune low-quality refusals/disclaimers, and mix several high-signal synthetic corpora instead of overcommitting to one generation recipe.
- **Authors:** Teknium / OpenHermes curation line
- **Year:** 2023
- **URL:** https://huggingface.co/datasets/teknium/openhermes
- **Relevant topics:** dataset mixture, open chat data, GPT-4 distillation, synthetic SFT

## Abstract
OpenHermes is a curated open instruction dataset of roughly 243K entries, primarily GPT-4-generated, assembled from multiple open synthetic sources such as GPTeacher, WizardLM, Airoboros, Camel, and GPT4-LLM. It became one of the most reused open chat mixtures in the 2023 open-model ecosystem.

## Key Contributions
- Demonstrated the value of mixture curation over single-source generation.
- Removed common low-quality artifacts such as boilerplate refusals and “as an AI” disclaimers.
- Served as the training substrate for several strong open instruct models.

## Technical Details
- Main sources include GPTeacher, WizardLM, Airoboros, Camel, CodeAlpaca, and GPT4-LLM-style corpora.
- Primary supervision is synthetic and mostly GPT-4-derived.
- Focus is on mixture cleanliness and usable model behavior rather than a new generation algorithm.

## Connections
- Predecessor to [[openhermes-2-5]].
- Sits in the same mixture-design family as [[tulu-3-sft-mix]].

