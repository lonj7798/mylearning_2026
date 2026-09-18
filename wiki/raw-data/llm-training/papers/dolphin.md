<!-- scope: The Dolphin dataset card (Eric Hartford, 2023): an Orca-style FLANv2 instruction set with GPT-4 and GPT-3.5 completions, deduplicated and filtered to remove alignment, refusal, avoidance, and bias responses; later Dolphin artifacts appear only as pointers
     deps: [[orca]]
     see-also: [[airoboros]], [[capybara]], [[openhermes]], [[deepseek-r1-distill-synth]]
-->

# Dolphin (Hugging Face dataset `cognitivecomputations/dolphin`)
- **Core Insight:** The card describes about 1 million FLANv2 prompts with GPT-4 completions and about 3.5 million with GPT-3.5 completions, built to follow the Orca paper's submix and system-prompt distribution, then deduplicated and filtered to remove "instances of alignment, refusal, avoidance, and bias"; the released files hold 892k and 2.84M rows (dataset card "Dataset details"; dataset viewer).
- **Guideline:** When this dataset is reused for SFT, measure refusal rate and task quality on held-out prompts before and after training, because the card names the removed response categories but gives no filtering method, no removal counts, and no evaluation of the filtered data.
- **Authors:** Eric Hartford (card and announcement post); acknowledged contributors: Wing "Caseus" Lian, NanoBit, Rohan, Teknium, Pankaj Mathur, Tom "TheBloke" Jobbins
- **Year:** 2023 (dataset repository initial commit 2023-07-01; card last changed 2023-12-18, commit 673d77e)
- **URL:** https://huggingface.co/datasets/cognitivecomputations/dolphin (now served as QuixiAI/dolphin); announcement post https://erichartford.com/dolphin
- **Source type:** model/dataset card
- **Relevant topics:** Orca reproduction, FLANv2, teacher-generated SFT data, refusal filtering, dataset licensing

## Summary
The card presents Dolphin as an attempt to replicate the results of Microsoft's Orca. It lists two files: about 1 million FLANv2 examples augmented with GPT-4 completions (`flan1m-alpaca-uncensored.jsonl`) and about 3.5 million augmented with GPT-3.5 completions (`flan5m-alpaca-uncensored.jsonl`). The authors followed Orca's submix and system-prompt distribution with two stated exceptions: all 75k chain-of-thought examples were included in FLAN-1m instead of a sample, and duplicates were removed, leaving 3.5M instructions in the GPT-3.5 ("ChatGPT") set. Instances of alignment, refusal, avoidance, and bias were then filtered out "to produce an uncensored model upon which can be layered your personalized alignment LoRA". The dataset is licensed Apache-2.0 for commercial or non-commercial use. The card lists planned model releases and states that each model follows the license of its base model (card; the announcement post repeats the composition, filtering, and license text).

## Key Contributions
- A public reconstruction of Orca-style explanation data (FLANv2 prompts with Orca system prompts and GPT-4 / GPT-3.5 responses) (card).
- Two filtered JSONL files plus deduplicated and ShareGPT-format variants and the conversion scripts in the repository (repo file list).
- A stated design choice: remove refusal and alignment behavior at the data stage so that alignment can be added later (card).

## Key Figures/Tables to Study
- "Token distribution for GPT-3.5 completions" figure on the card (image only; no values in text).
- Dataset viewer: the `instruction` column holds the system prompt (17 distinct values in `flan1m-alpaca-uncensored`), with `input` and `output` columns.

## Technical Details
- **Composition.** ~1M FLANv2 + GPT-4 completions; ~3.5M FLANv2 + GPT-3.5 completions (card).
- **Deviations from Orca.** All 75k CoT examples included in FLAN-1m; duplicates removed from the GPT-3.5 set (card).
- **Filtering.** Removed categories: alignment, refusal, avoidance, bias (card). The filtering rules, tools, and counts are not given on the card, in the announcement post, or as a script in the repository file list.
- **Released rows** (dataset viewer, 2026-09-14): `flan1m-alpaca-uncensored` 892k rows; `flan5m-alpaca-uncensored` 2.84M rows; 3,731,947 rows in total; 18.9 GB.
- **Repository files** (tree at commit 673d77e): `flan1m-alpaca-uncensored.jsonl` 1.6 GB, `flan1m-alpaca-uncensored-deduped.jsonl` 1.52 GB, `flan1m-sharegpt-deduped.json` 1.62 GB, `flan5m-alpaca-uncensored.jsonl` 4.8 GB, `flan5m-alpaca-uncensored-deduped.jsonl` 4.54 GB, `flan5m-sharegpt-deduped.json` 4.84 GB; scripts `convertToShareGpt.py`, `dedupeToShareGpt.py`, `fp32_to_fp16.py`, `llama_flash_attn_monkey_patch.py`.
- **Field lengths** (viewer, `flan1m-alpaca-uncensored`): `input` 17 to 40.6k characters; `output` 1 to 7.78k characters.
- **Planned models** (card): Xgen 7b 8k, LLaMA 13b, MPT 30b 8k, LLaMA 33b, Falcon 40b, LLaMA 65b; LLaMA releases non-commercial.
- **Downstream use** (dataset page, 2026-09-14): 502 models listed as trained or fine-tuned on this dataset, including TheBloke/dolphin-2.7-mixtral-8x7b-GGUF, dphn/dolphin-2.6-mixtral-8x7b, and TheBloke/dolphin-2.2.1-mistral-7B-GGUF.

## Recipe ledger
| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| Dolphin dataset, `flan1m-alpaca-uncensored.jsonl` | n/a (data) | distill-SFT data | examples; teacher | "~1 million" FLANv2 examples; GPT-4 completions; 892k rows released | dataset card "Dataset details"; dataset viewer | verified 2026-09-14 | no ablation reported |
| Dolphin dataset, `flan5m-alpaca-uncensored.jsonl` | n/a (data) | distill-SFT data | examples; teacher | "~3.5 million" after deduplication; GPT-3.5 completions; 2.84M rows released | dataset card; dataset viewer | verified 2026-09-14 | no ablation reported |
| Dolphin dataset, FLAN-1m | n/a (data) | distill-SFT data | CoT examples | all 75k included (not sampled) | dataset card | verified 2026-09-14 | no ablation reported |
| Dolphin dataset, both files | n/a (data) | distill-SFT data | response filter | alignment, refusal, avoidance, bias removed; rules and counts | dataset card | not reported (checked card, announcement post, repository file list) | no ablation reported |
| dolphin-2.1-mistral-7b | 7B | SFT | data; epochs; compute; format | Dolphin, modified "for uncensoring, deduping, cleaning, and quality", + Airoboros 2.2.1; 4 epochs; 48 hours on 4x A100; ChatML | huggingface.co/cognitivecomputations/dolphin-2.1-mistral-7b README (separate artifact) | verified 2026-09-14 | no ablation reported |
| dolphin-2.1-mistral-7b | 7B | SFT | LR, batch, sequence length, loss masking | not on the model card | same README | not reported | n/a |

## Findings relevant to negative feedback and distillation
- **Distillation data.** All responses are teacher outputs from GPT-4 or GPT-3.5 on FLANv2 prompts, following Orca's system prompts (card).
- **Negative samples are discarded.** Refusal, avoidance, alignment, and bias responses are removed before SFT, not used as training signal (card). In the terms of the course standard this is removal of samples judged to have negative marginal value; the card reports no measurement of the effect on capability, refusal rate, or safety.
- **Author's stated consequence** (dolphin-2.1-mistral-7b README, separate artifact): the model "is uncensored", "more compliant", and "will be highly compliant to any requests, even unethical ones"; users are "advised to implement your own alignment layer before exposing the model as a service".
- **Rationale post** ("Uncensored Models", erichartford.com/uncensored-models, page updated 2023-05-22, separate artifact about WizardLM): instruction data generated with ChatGPT carries refusals and bias into fine-tuned models; the stated strategy is "Identify and remove as many refusals and biased answers, and keep the rest", then train "in exactly the same way that the original model was trained". The post does not list filter rules; it says an existing Vicuna uncensoring script was adapted.

## Connections
- [[orca]] — the paper whose FLANv2 submix and system-prompt distribution the dataset follows.
- [[airoboros]] — Airoboros 2.2.1 was added to Dolphin data for dolphin-2.1-mistral-7b "to increase creativity" (model card).
- [[capybara]] — that card records dolphin-2.6-mixtral-8x7b among models fine-tuned on Capybara.
- [[openhermes]] — another community SFT collection; not compared on the Dolphin card.
- [[deepseek-r1-distill-synth]] — Dolphin R1 (separate card, huggingface.co/datasets/cognitivecomputations/dolphin-r1, Apache-2.0): 800k samples "similar in composition to the one used to train DeepSeek-R1 Distill models": 300k reasoning samples from DeepSeek-R1, 300k from Gemini 2.0 flash thinking, 200k Dolphin chat samples.

## Verification
- Checked on 2026-09-14 against: https://huggingface.co/datasets/cognitivecomputations/dolphin (card at commit 673d77e, commit history, file tree, dataset viewer); https://erichartford.com/dolphin; secondary: dolphin-2.1-mistral-7b README, dolphin-r1 dataset README, erichartford.com/uncensored-models.
- Corrections to the previous card version:
  - Title "Dolphin Datasets (Eric Hartford / Cognitive Computations)" covering v1, 2.x, and R1 → the card now describes the one artifact at the URL; later Dolphin artifacts are pointers.
  - "Year: 2023 (v1) → 2024 (...) → 2025 (Dolphin-R1 data)" → 2023 (repository created 2023-07-01).
  - "Dolphin-R1: ~800K CoT traces from R1-class teachers; incorporates filtering for correctness + format + language" → 800k samples = 300k DeepSeek-R1 + 300k Gemini 2.0 flash thinking reasoning samples + 200k Dolphin chat samples; no filtering described (dolphin-r1 README).
  - "Train on the remainder with the same recipe as the original Orca" → the card gives no training recipe; "exactly the same way that the original model was trained" is from the WizardLM "Uncensored Models" post.
  - "~1M samples ... ~3.5M samples" → kept as the card's approximate figures; released row counts 892k and 2.84M added.
- Removed as unsupported by the source: the five-step "uncensoring" recipe (regex + classifier refusal detection, "As an AI language model" and moral-preamble templates, formulaic-disclaimer markers); the Dolphin 2.x composition table (SlimOrca, SynthIA, Magicoder OSS-Instruct, CodeFeedback, Samantha) and the names Dolphin-Llama3 and Dolphin-Qwen; "License: Apache-2.0 (unusual for this scale)" and "rare and valuable"; "'Uncensoring' ≠ unsafe by default"; "Evolving composition — each Dolphin version tracks the community's best open synthetic mixes"; "Popular base for community / indie model fine-tunes" (replaced by the page's model count); the "Risks + gotchas" list (API terms-of-service claim, quality variance, community split); "Direct heir to [[orca-2]] philosophy"; "Frequently combined with [[capybara]] in community mixes".
- Not reported by the source: filtering method and removal counts; FLANv2 submix proportions actually achieved; teacher API model versions and generation dates; any evaluation of models trained on the dataset.
