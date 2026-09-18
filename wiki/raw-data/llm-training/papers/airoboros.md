<!-- scope: airoboros (Jon Durbin, 2023–2024) — Self-Instruct implementation with per-task "instructors", topic injection, embedding dedup, LLM-judge culling; versioned GPT-4-generated SFT datasets and LMoE adapters
     deps: [[self-instruct]]
     see-also: [[alpaca]], [[openhermes]], [[openhermes-2-5]], [[persona-hub]]
-->

# airoboros: using large language models to fine-tune large language models
- **Core Insight:** airoboros generates instruction data with no human-written seed tasks through 28 "instructor" modules (one per task type) conditioned on randomly generated topics, rejects near-duplicates by gte-small embedding distance, and removes refusal-like responses by regex; the checked documents report no controlled evaluation of the resulting datasets (README @169e8a9 L3, L22–31; example-config.yaml).
- **Guideline:** When a synthetic SFT set must cover many task types from one teacher API, separate per-task generators let counts, temperatures, and duplicate thresholds be set per task (example-config.yaml); check the generated data against evaluation benchmarks, because the author found benchmark data in the first airoboros-2.1 model, which inflated its TruthfulQA score (airoboros-2.2 dataset card).
- **Authors:** Jon Durbin
- **Year:** 2023 (repository created 2023-04-29; last commit 169e8a9 on 2024-03-07; datasets through airoboros-3.2)
- **URL:** https://github.com/jondurbin/airoboros/tree/169e8a9693ac09bbb3db18a3348e1a614948da1c ; https://huggingface.co/datasets/jondurbin/airoboros-2.2
- **Source type:** released config/code + model/dataset card (practitioner; no controlled evaluation reported)
- **Relevant topics:** Self-Instruct variants, per-task synthetic generation, topic injection, embedding deduplication, LLM-judge filtering, benchmark contamination, LoRA experts

## Summary
The README describes airoboros as "my take on implementing the Self-Instruct paper", "quite heavily modified", that "does not use any human-generated seeds" (L3). It supports OpenAI completion and chat-completion endpoints, so gpt-4 or gpt-3.5-turbo can generate data (L5, L24). The stated goal is to build datasets focused on specific tasks and train one expert model per task, then route each request to an expert (L37–42). The repository also contains LMoE, which loads a LoRA adapter per request chosen by faiss similarity search or by an agent-based router (L55–81). Datasets and models are released on Hugging Face in numbered versions (gpt4-1.x, 2.0/m2.0, 2.1, 2.2, 3.1, 3.2).

## Key Contributions
- Task-specific instructors: 28 Python modules under `airoboros/instructors/` (excluding `__init__.py`) at commit 169e8a9, including general, contextual, counterfactual_contextual, coding, trivia, orca, cot, riddle, roleplay, rp, gtkm, agent, plan, writing, detailed_writing, editor, awareness, multiple_choice, misconception, stylized_response (repository tree `airoboros/instructors/*.py`).
- Topic injection: the teacher generates random topic lists ("Give me a numbered list of 20 completely random topics"), which are inserted into instruction-generation prompts (example-config.yaml L70–71; README L25, L27).
- Embedding-based duplicate rejection with faiss instead of ROUGE scoring (README L26; self_instruct.py L160, L874–907).
- An LLM-judge culling step that scores clusters of similar items and keeps one per cluster (self_instruct.py L604–652, L673–870).
- Per-category LoRA experts and request routing (README L55–81; scripts/segment_experts.py; scripts/tune_expert.sh).

## Key Figures/Tables to Study
- `example-config.yaml`: per-instructor count, batch size, temperature, and duplicate threshold.
- `airoboros/instructors/prompts/general.txt` and `filter.txt`: generation and judging prompts.
- airoboros-2.2 dataset card: contamination note and per-version changes.

## Technical Details
- **Generator defaults** (example-config.yaml @169e8a9): model "gpt-4" (L2); temperature 1.0, top_p 1.0, frequency and presence penalty 0.0 (L63–67); 100 items and batch size 10 per instructor unless overridden (L74, L77). Overrides include contextual temperature 0.5, roleplay and writing 0.9, experience and riddle temperature 0.9 with top_p 0.4, rp presence and frequency penalty 1.3 with 12 turns.
- **Seeds.** No human seed tasks are used (README L3). Some prompt files still contain fixed examples: `orca.txt` includes worked example questions and answers (L7–30), and seed files exist under `prompts/character_seeds/` (0–4) and `prompts/detailed_writing_seeds/` (0–2) (tree).
- **General prompt** (`general.txt`): asks for exactly `{batch_size}` tasks without coding or math, with one "highly complex" task with 3 or more criteria, one task requesting a random output format, the injected topics, and a Flesch-Kincaid hint. The default hint asks for a readability score of 30 or lower (example-config.yaml L80).
- **Topic avoidance.** The default avoidance string excludes tasks about climate change, green tech, DEI, sex and/or gender, religion, politics, social issues, race, ethnicity, artificial intelligence, baking/cooking, urban development, and emotions or physical senses (example-config.yaml L35).
- **Duplicate rejection.** Instructions are embedded with `thenlper/gte-small` into a faiss `IndexFlatL2`; a new instruction is rejected when the nearest L2 distance is ≤ `min_docsearch_score` (example-config.yaml L29; self_instruct.py L160, L904). Default 0.07 in the example config (L60), 0.35 if unset in code (self_instruct.py L129); per-instructor values range from 0.01 to 0.3 (example-config.yaml).
- **Response regex filters.** 16 case-insensitive patterns, including "openai", "language model", "as an? (ai|generative language|gpt|bot)", "my limitations", and "please note that" (example-config.yaml L38–54).
- **Judge culling** (`cull-instructions`). Within each category, items whose instruction+response embeddings lie within the category threshold are grouped. Each item is scored by the LLM with `filter.txt`: a silent 0–100 score, "GOOD" if the score is at or above the threshold (default 100, self_instruct.py L634), and a score of 0 for warnings, disclaimers, refusals, or statements that the response was written by an AI. One good item is kept per group, the longest if several are good, and the whole group is removed if none are good. Categories stylized_response, rp, detailed_writing, contextual, counterfactual_contextual, plan, song, and wordgame skip scoring (self_instruct.py L604–652, L725–870).
- **Dataset versions** (row counts from the Hugging Face dataset viewer size API, retrieved 2026-09-14). airoboros-gpt4 (1.x): GPT-4 data for trivia, math, "nonsensical math", coding, closed-context QA (including confounding contexts), writing, multiple choice (dataset card). gpt4-1.4: multi-character multi-turn chats, Rosetta Code examples in 10 languages, more roleplay, jokes (card). 2.1: 36,306 rows. 2.2: 44,838 rows. 3.1: 59,277 rows. 3.2: 58,709 rows.
- **2.1 changes** (airoboros-l2-70b-2.1 model card): rp and gtkm conversation categories; `cull-instructions` used to shrink the m2.0 dataset "according to gpt-4"; 1,500 stylized_response samples regenerated with character cards as system prompts; an unpublished "de-alignment" dataset. The card states the model "is a bit broken due to a prompt formatting bug in the training code".
- **2.2 changes** (dataset card): awareness instructor; editor instructor that rewrites generated text with errors and trains correction back to the original; 500 summarization examples from mattpscott/airoboros-summarization; "~1279 multiple choice questions"; GTKM and RP saved one row per conversation round; most typographic UTF-8 characters replaced with ASCII.
- **3.1 changes** (dataset card): ShareGPT format; MathJSON math solutions evaluated by an external library (the card states both "now ~17k items" and "roughly 4k samples"), some adapted from MetaMathQA; log information extraction; anonymization; chat introspection; multi-step instructions with acknowledgement.
- **3.2 changes** (dataset card): MathJSON removed because "it seems to confuse the models at times"; de-censorship data re-added; ~11k SlimOrca instructions extended with a follow-up turn.

## Recipe ledger
| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| airoboros LMoE 2.1 expert adapter (Llama-2 base) | size passed as argument | SFT (QLoRA) | epochs; LR; schedule; warmup ratio | 5; 0.0002; constant; 0.03 | github.com/jondurbin/airoboros@169e8a9 scripts/tune_expert.sh | verified 2026-09-14 | no ablation reported; README calls the script "an example" (L69) |
| same | same | SFT (QLoRA) | LoRA r; alpha; dropout; modules; quantization | 64; 16; 0.1; all; 4-bit nf4, double quant, bf16 | scripts/tune_expert.sh | verified 2026-09-14 | no ablation reported |
| same | same | SFT (QLoRA) | max length; grad clip; adam β2; weight decay | 4096; 0.3; 0.999; 0.0 | scripts/tune_expert.sh | verified 2026-09-14 | no ablation reported |
| same | same | SFT (QLoRA) | per-device batch; GPUs; tokens | positional argument; not reported; not reported | scripts/tune_expert.sh | not reported | — |
| same | same | SFT (QLoRA) | cross-expert mixing | each category is sampled at min(500, 10% of its items), and part of that sample (capped by expert size) is added to every other expert's data | scripts/segment_experts.py | verified 2026-09-14 | README: included "in the event of misrouting" (L67) |
| airoboros-l2-70b-2.1 and other full models | 7B–70B | SFT | all training hyperparameters | not reported | checked README, l2-70b-2.1 model card | not reported | — |
| airoboros data generation (example config) | — | distill-SFT | teacher; temperature; top_p; duplicate threshold; judge threshold | gpt-4; 1.0; 1.0; 0.07 (L2, gte-small); 100 | example-config.yaml L2, L29, L60, L64–65; self_instruct.py L634 | verified 2026-09-14 (example defaults; configs for released versions not published) | no ablation reported |

## Findings relevant to generality, negative feedback, long context, agentic training, distillation
- **Contamination (anecdotal).** The first airoboros-2.1 model included benchmark data, which gave it "a crazy high truthfulqa score"; for 2.2 the author removed items with similarity score < 0.15 to TruthfulQA using gte-small and faiss, and did not check most other benchmarks (airoboros-2.2 card).
- **Experts versus one model.** The README lists as "in progress" a proof of concept that an ensemble of category LoRAs outperforms a same-size model tuned on all data; no result is reported (README L49).
- **Negatives.** Refusal-like and low-scored responses are discarded (negative marginal value). The editor instructor places corrupted text in the input and the original text in the target (negative as content). The config describes counterfactual contextual prompts as "used to de-hallucinate Q&A a bit" (example-config.yaml, counterfactual_contextual comment).
- **Agentic data.** Agent/function-calling examples (function and arguments as JSON or YAML) and reWOO-style execution plans are part of the 2.1 training data; the plans are not executed (l2-70b-2.1 model card).
- **Long context.** gpt4-1.4 was not filtered by token length and "some are well over 2048" (card); the expert script uses model_max_len 4096.
- **Distillation.** Most data is generated with gpt-4; the cards note that OpenAI terms restrict training competing models (2.2 and 3.2 cards).

## Connections
- [[self-instruct]] — the method airoboros implements without human seed tasks (README L3).
- [[alpaca]] — README lists airoboros' differences from "self-instruct/alpaca": chat endpoints, topic injection, embedding similarity instead of ROUGE (L22–31).
- [[openhermes]] — lists "Airoboros GPT-4 (v1.0)" among its source datasets (OpenHermes dataset card).
- [[openhermes-2-5]] — includes Airoboros 2.2 as a source dataset (OpenHermes-2.5 dataset card).
- [[orca]] — the `orca` instructor generates "Orca style reasoning/math" questions with intermediate steps (example-config.yaml; orca.txt).
- [[metamath]] — part of the airoboros-3.1 MathJSON data was adapted from MetaMathQA (3.1 card).
- [[persona-hub]] — conditions synthetic prompts on personas; airoboros conditions them on generated topic lists.

## Verification
- Checked on 2026-09-14 against: github.com/jondurbin/airoboros @169e8a9 (README.md, example-config.yaml, self_instruct.py, instructors tree, prompts/general.txt, prompts/filter.txt, prompts/orca.txt, scripts/tune_expert.sh, scripts/segment_experts.py); HF cards jondurbin/airoboros-gpt4, -gpt4-1.4, -2.1, -2.2, -3.1, -3.2, airoboros-l2-70b-2.1; HF dataset viewer size API; teknium/openhermes and OpenHermes-2.5 cards.
- Corrections to the previous card version:
  - Title "Airoboros: Customizable Self-Instruct Pipeline with Role-Specific Instructors" → README title "airoboros: using large language models to fine-tune large language models".
  - "Seed input: a small instructor-specific seed pool" → no human-generated seed tasks (README L3); some prompt files contain fixed examples or seed files.
  - "Deduplication: ROUGE-L + embedding" → embedding L2 distance only (gte-small, faiss); README says this replaces ROUGE scoring (L26).
  - "Teacher: GPT-4 for 2.x onward; GPT-3.5 in v1" → GPT-4 datasets start at airoboros-gpt4 (1.x); gpt-3.5-turbo datasets and models are listed separately (README L182–234).
  - "Math instructor with MathJSON outputs (v3.1)"; "information-extraction instructor"; "anonymization" → these are data categories added in the airoboros-3.1 dataset (card), not instructor modules at 169e8a9; MathJSON was removed in 3.2.
  - "Year 2023–2024 (versions 1.x through 3.x)" → repository 2023-04-29 to 2024-03-07; datasets gpt4-1.x through 3.2.
- Removed as unsupported by the source: "independent researcher"; "de-facto template for build-your-own synthetic SFT"; "proved one researcher can produce a 70B-competitive dataset"; "airoboros-70B among the first such examples"; "community forks exist" (fork count not used as evidence); "sizes range from tens of thousands to hundreds of thousands" (replaced by viewer row counts); "roleplay over-represented relative to code/math"; "some versions have noticeable near-duplicates"; "each release fixed refusal bias"; "used as seed corpus in [[capybara]] and [[dolphin]]"; ancestry claims for [[nemotron-4-synthetic]], [[glan]], [[llama-3-synthetic-pipeline]]; "competes with [[persona-hub]]".
- Not reported by the source: evaluation results for any airoboros dataset or model; token counts; per-version generation configs; training hyperparameters for the full (non-LMoE) models.
