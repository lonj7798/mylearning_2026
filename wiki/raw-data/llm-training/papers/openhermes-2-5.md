<!-- scope: OpenHermes 2.5 dataset card — a 1M-row compilation of open and custom synthetic instruction/chat datasets used for the OpenHermes 2.5 and Nous Hermes 2 models
     deps: [[openhermes]]
     see-also: [[smol-talk]], [[tulu-3-sft-mix]], [[airoboros]], [[metamath]]
-->

# OpenHermes 2.5: An Open Dataset of Synthetic Data for Generalist LLM Assistants
- **Core Insight:** OpenHermes 2.5 compiles 1,001,551 primarily synthetic instruction and chat samples from 15 listed open datasets plus unnamed custom synthetic datasets, and its card states that it is the training data of the OpenHermes 2.5 and Nous Hermes 2 model series; the card gives no per-source counts, filtering rules, or evaluation of the dataset.
- **Guideline:** When selecting or ablating OpenHermes 2.5 subsets by origin, audit the `source` field first, because it is empty for 496,743 of 1,001,551 rows (49.6%) and its 14 values do not map one-to-one to the 15 source headings in the card.
- **Authors:** Teknium
- **Year:** 2023 (citation year in the card; Hugging Face repository created 2023-11-12; no paper)
- **URL:** https://huggingface.co/datasets/teknium/OpenHermes-2.5
- **Source type:** model/dataset card
- **Relevant topics:** open SFT data mixtures, GPT-4 distillation data, dataset provenance metadata

## Summary
The card describes OpenHermes 2.5 as "a continuation of the Open Hermes 1 dataset, at a much larger scale, much
more diverse, and much higher quality compilation, reaching 1M, primarily synthetically generated instruction and
chat samples". It states that the Open Hermes 2/2.5 and Nous Hermes 2 models "are underpinned by this exact
compilation and curation of many open source datasets and custom created synthetic datasets". It lists the source
datasets with links, describes a ShareGPT record format, and notes that some records keep metadata such as
`category` and many keep the name of their source dataset. The card does not describe filtering, deduplication,
decontamination, per-source sampling ratios, or any evaluation of the dataset.

## Key Contributions
- Public release of the training data of the OpenHermes 2.5 and Nous Hermes 2 models (card, opening line).
- A list of 15 source datasets with creators and links (card, "Dataset Sources").
- One ShareGPT-structured format with optional per-record `source` and `category` metadata (card, "Dataset Structure").

## Key Figures/Tables to Study
- "Dataset Sources": the 15 headings and their links.
- "Dataset Structure": the example record with `conversations`, `source`, and `category`.
- Dataset viewer column statistics for `source`, `category`, and `conversations` (values below).

## Technical Details
- Size: "reaching 1M" samples (card, Dataset Description). The dataset viewer counts 1,001,551 rows in one `train`
  split with 16 columns (datasets-server `/size`, read 2026-09-14).
- Source headings as printed (card, "Dataset Sources"): Airoboros 2.2; CamelAI Domain Expert Datasets (Physics,
  Math, Chemistry & Biology); ChatBot Arena (GPT-4 Only); Collective Cognition (09-11-2023); CoT Alpaca GPT4
  ("I have lost the source page for this dataset"); Evol Instruct 70K && 140K; Glaive Code Assistant; GPT4-LLM;
  GPTeacher; Medical Tasks (CogStack); MetaMath 40k; SlimOrca 550K; Platypus; ShareGPT (GPT4-Only); Unnatural
  Instructions GPT4.
- The card does not state that size labels in headings ("MetaMath 40k", "SlimOrca 550K") are the number of rows
  taken into OpenHermes 2.5 (card, "Dataset Sources").
- The custom synthetic datasets are credited but not named or sized (card, Dataset Description).
- Format: each record has a `conversations` list; each turn has `from` (role) and `value` (text); the example uses
  the roles `system`, `human`, and `gpt` (card, "Dataset Structure").
- Entries per `conversations` list, including any system turn: min 2, median 2, mean 2.40, max 58
  (datasets-server `/statistics`). A record with 2 entries holds one prompt turn and one response turn (derived).
- `source` field over the full split (datasets-server `/statistics`): empty for 496,743 rows; 14 distinct values:
  glaive-code-assist 182,240; CamelAI 78,390; metamath 56,448; EvolInstruct_70k 51,948; cot_alpaca_gpt4 42,026;
  airoboros2.2 35,380; platypus 22,280; GPT-4 Comparison Data 14,928; UnnaturalInstructions 8,610;
  CogStackMed 4,443; LMSys Chatbot Arena 3,136; caseus_custom 2,688; lmsys1m 1,631; Econ_domain_expert 660.
- No `source` value names SlimOrca, ShareGPT, GPTeacher, or Collective Cognition, and the card does not say which
  rows came from them (card; `/statistics`).
- `category` field: present for 35,380 rows, the same count as `source` = airoboros2.2; 28 distinct values, the
  largest being orca 11,022, coding 5,561, and general 4,203 (datasets-server `/statistics`).
- License: the card metadata declares no license (Hugging Face API `cardData` contains only language, pretty_name,
  and tags).
- The card links a hosted copy on Lilac for exploration, embedding search, and clustering (card, "Lilac Integration").

## Connections
- [[openhermes]] — predecessor; its card reports 242,000 entries with instruction/input/output fields.
- [[airoboros]], [[camel]], [[evol-instruct]], [[metamath]] — origins of listed components (Airoboros 2.2, CamelAI
  domain expert data, Evol Instruct 70K/140K, MetaMath 40k).
- [[orca]] — related: the "SlimOrca 550K" heading links to the Open-Orca/SlimOrca dataset.
- [[smol-talk]] — the SmolTalk dataset card states that it adds 100k OpenHermes2.5 samples "since we found that it
  helps preserve and boost benchmarks such as MMLU and WinoGrande, and BBH" (SmolTalk card, "Dataset composition";
  tested by fine-tuning SmolLM2-1.7B; that card defers ablation details to a later blog post).
- [[tulu-3-sft-mix]] — a later open SFT mixture, for comparison of how mixtures are documented.
- Related artifact without a library card: the OpenHermes-2.5-Mistral-7B model card
  (https://huggingface.co/teknium/OpenHermes-2.5-Mistral-7B; base model mistralai/Mistral-7B-v0.1). It states that
  training on a code-instruction share estimated at "around 7-14% of the total dataset" boosted TruthfulQA, AGIEval,
  and the GPT4All suite but reduced BigBench, and that HumanEval pass@1 rose from 43% (OpenHermes 2) to 50.7%
  (benchmarking by the Glaive team) ("Model description"). No controlled ablation is described.

## Verification
- Checked on 2026-09-14 against: https://huggingface.co/datasets/teknium/OpenHermes-2.5 (README at repository sha
  b8203782, last modified 2024-04-15), and the Hugging Face datasets-server `/size` and `/statistics` endpoints.
- Corrections to the previous card version:
  - Title "OpenHermes 2.5" → the title in the card's citation block.
  - "roughly 1M-example" → "reaching 1M" (card); 1,001,551 rows (dataset viewer).
  - "retaining source/category metadata" → `source` is empty for 49.6% of rows and `category` for 96.5%
    (966,171 rows) (`/statistics`).
  - "Became a base dataset for several later open instruct models" → the card names the Open Hermes 2/2.5 and
    Nous Hermes 2 models.
  - "custom Teknium-generated data" → "custom created synthetic datasets", creator and contents not stated.
  - "ShareGPT-style multi-turn format" → ShareGPT structure; median 2 entries per record (`/statistics`).
  - Partial source list (seven names plus "custom") → all 15 headings as printed in the card.
- Removed as unsupported by the source:
  - "use broad synthetic composition as the main alignment asset" (old Core Insight).
  - "Its practical contribution is not a novel generation method but a large, provenance-rich, reusable open
    alignment corpus."
  - "Emphasizes curation breadth and reuse rather than a single synthetic algorithm."
- Not reported by the source: per-source row counts, sampling ratios, filtering and deduplication rules,
  decontamination, which models generated each component, license, evaluation results.
