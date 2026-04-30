<!-- scope: OpenHermes 2.5 as a large curated synthetic chat mixture
     deps: [[openhermes]]
     see-also: [[smol-talk]], [[tulu-3-sft-mix]]
-->

# OpenHermes 2.5
- **Core Insight:** OpenHermes 2.5 scales the open-mixture idea: keep the dataset heterogeneous, track source provenance, and use broad synthetic composition as the main alignment asset.
- **Guideline:** For large open instruct mixtures, preserve source-level provenance and category metadata so later curation and ablation remain possible.
- **Authors:** Teknium
- **Year:** 2023
- **URL:** https://huggingface.co/datasets/teknium/OpenHermes-2.5
- **Relevant topics:** dataset mixture, synthetic chat data, provenance-aware curation

## Abstract
OpenHermes 2.5 is a roughly 1M-example continuation of OpenHermes, combining many open synthetic datasets and custom-generated data into a broad general-assistant mixture. Its practical contribution is not a novel generation method but a large, provenance-rich, reusable open alignment corpus.

## Key Contributions
- Scaled OpenHermes from hundreds of thousands to roughly one million examples.
- Mixed many open synthetic sources while retaining source/category metadata.
- Became a base dataset for several later open instruct models.

## Technical Details
- Dataset card lists sources such as Airoboros, Camel, WizardLM, MetaMath, SlimOrca, Platypus, ShareGPT GPT-4 slices, and custom Teknium-generated data.
- Stored in ShareGPT-style multi-turn format with source/category metadata.
- Emphasizes curation breadth and reuse rather than a single synthetic algorithm.

## Connections
- Larger continuation of [[openhermes]].
- Another mixture-design reference beside [[tulu-3-sft-mix]] and [[openhermes]].

