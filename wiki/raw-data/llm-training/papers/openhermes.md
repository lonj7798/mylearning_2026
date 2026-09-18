<!-- scope: OpenHermes (v1.0) dataset card — a 242,000-entry compilation of open, primarily GPT-4-generated instruction datasets
     see-also: [[openhermes-2-5]], [[wizardlm]], [[airoboros]], [[camel]], [[alpaca]]
-->

# OpenHermes Dataset
- **Core Insight:** OpenHermes compiles 242,000 entries of primarily GPT-4-generated instruction data from seven named open datasets, filtered to remove OpenAI refusals, disclaimers, and "As an AI" examples; the card gives no per-source counts and no evaluation of the dataset itself.
- **Guideline:** When a per-source ablation or removal is needed, plan to re-match rows against the original datasets, because OpenHermes rows have only `instruction`, `input`, and `output` fields and carry no source label.
- **Authors:** Teknium (Hugging Face user `teknium`); component datasets by the creators listed in the card
- **Year:** 2023 (Hugging Face dataset repository created 2023-09-04; no paper)
- **URL:** https://huggingface.co/datasets/teknium/openhermes
- **Source type:** model/dataset card
- **Relevant topics:** open SFT data mixtures, GPT-4 distillation data, refusal and disclaimer filtering

## Summary
The dataset card (pretty name "OpenHermes-v1.0") describes 242,000 entries of primarily GPT-4-generated data
taken from open datasets. It lists the component datasets with their creators. It states that filtering removed
OpenAI refusals, disclaimers, and "As an AI" type examples "and more". It states that the mix is the original
Nous-Hermes mix minus two private datasets, Nous-Instruct and PDACTL. It names OpenHermes 13B as "the first fine
tune of the Hermes dataset that has a fully open source dataset". The card does not describe a generation method,
detailed filtering rules, deduplication, decontamination, or any evaluation of the dataset.

## Key Contributions
- A public release of the Hermes instruction mixture without its two private components (card, last paragraph).
- A list of seven component sources with attributions (card, bullet list).
- Removal of refusal and disclaimer examples as the only stated filtering step (card, "Filtering included ...").

## Key Figures/Tables to Study
- The component list in the card body.
- The dataset viewer schema: one `train` split with three string columns `instruction`, `input`, `output`.

## Technical Details
- Size: "242,000 entries" (card, first paragraph). The Hugging Face dataset viewer counts 242,831 rows in one
  `train` split (datasets-server `/size`, read 2026-09-14).
- Components, as printed (card, bullet list; the list is introduced with "including", so completeness is not stated):
  - GPTeacher: General Instruct, Roleplay v1, Roleplay v2, and Code Instruct datasets (Teknium)
  - WizardLM (v1, evol_instruct 70k) (WizardLM Team/nlpxucan)
  - Airoboros GPT-4 (v1.0) (JonDurbin)
  - Camel-AI's domain expert datasets (Camel-AI Team)
  - CodeAlpaca (Sahil2801)
  - GPT4-LLM and Unnatural Instructions (attributed to Microsoft in the card)
- Filtering: "removal of OpenAI refusals, disclaimers, and 'As an AI' type examples and more"; no rules or removal
  counts are given (card).
- Relation to Nous-Hermes: "identical to the original Nous-Hermes', minus the Nous-Instruct and PDACTL datasets
  which were private datasets" (card, last paragraph).
- Format: three string fields `instruction`, `input`, `output`; there is no source, category, or conversation-turn
  field (datasets-server `/first-rows` feature list). The repository holds one data file, `openhermes.json`
  (Hugging Face API file list).
- Field lengths in characters over all 242,831 rows (datasets-server `/statistics`): `instruction` median 126,
  mean 228.5; `input` median 0, mean 21.4; `output` median 676, mean 1,000.8, max 126,939.
- License: the card metadata declares no license (Hugging Face API `cardData` contains only language, pretty_name,
  tags, and task_categories).

## Connections
- [[openhermes-2-5]] — successor dataset; its card calls OpenHermes 2.5 "a continuation of the Open Hermes 1
  dataset" and reports about 1M samples.
- [[wizardlm]], [[evol-instruct]] — origin of the evol_instruct 70k component.
- [[airoboros]] — origin of the Airoboros GPT-4 v1.0 component.
- [[camel]] — origin of the Camel-AI domain expert component.
- [[alpaca]] — CodeAlpaca is a listed component, and the `instruction`/`input`/`output` field names follow the
  Alpaca data format.
- [[tulu-3-sft-mix]] — a later open SFT mixture, for comparison of how mixtures are documented.
- Related artifact without a library card: the OpenHermes-13B model card
  (https://huggingface.co/teknium/OpenHermes-13B; base model NousResearch/Llama-2-13b-hf). It lists training
  hyperparameters for that fine-tune ("Training hyperparameters": learning rate 2e-05, total train batch size 128,
  3 epochs, cosine schedule, 300 warmup steps) and reports GPT4All average 0.7036, AGIEval average 0.3556, and
  BigBench average 36.75, described as "a slight improvement on GPT4ALL Suite and BigBench Suite, with a
  degredation in AGIEval compared to the original hermes" ("Benchmark Results").

## Verification
- Checked on 2026-09-14 against: https://huggingface.co/datasets/teknium/openhermes (README at repository sha
  cbba06ae, last modified 2023-09-07), and the Hugging Face datasets-server `/size`, `/first-rows`, and
  `/statistics` endpoints for the same repository.
- Corrections to the previous card version:
  - "roughly 243K entries" → the card states 242,000 entries; the dataset viewer counts 242,831 rows.
  - "Authors: Teknium / OpenHermes curation line" → Teknium; component datasets by the creators listed in the card.
  - Component list omitted the GPTeacher sub-datasets and Unnatural Instructions → full list as printed (card).
  - "keep provenance at the source-dataset level" (old Guideline) → the dataset has no source field and the card
    gives no per-source counts (`/first-rows` feature list).
  - Title "OpenHermes" → card heading "OpenHermes Dataset".
- Removed as unsupported by the source:
  - "It became one of the most reused open chat mixtures in the 2023 open-model ecosystem."
  - "Demonstrated the value of mixture curation over single-source generation" (no comparison is reported).
  - "Served as the training substrate for several strong open instruct models" (the card names only OpenHermes 13B).
  - "GPT4-LLM-style corpora" and "Focus is on mixture cleanliness and usable model behavior".
  - "open chat data" as a topic: each row is one instruction, an optional input, and one output.
- Not reported by the source: per-source row counts, filtering rules and removal counts, deduplication,
  decontamination, which model generated each component beyond "primarily GPT-4", license, evaluation results.
