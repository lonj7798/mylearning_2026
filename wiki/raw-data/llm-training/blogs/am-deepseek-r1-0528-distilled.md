<!-- scope: a-m-team AM-DeepSeek-R1-0528-Distilled dataset card — 2.6M prompts with DeepSeek-R1-0528 responses, per-domain verification and filtering, one reported student result
     deps: [[am-deepseek-r1-distilled-1-4m]]
     see-also: [[deepseek-r1]], [[zhihu-zhi-create-distill]], [[distillation-source-matters]], [[am-thinking-v1]]
-->

# AM-DeepSeek-R1-0528-Distilled
- **Core Insight:** The card releases DeepSeek-R1-0528 responses for 2.6 million unique prompts (general chat 47.3%, math 26.1%, code 16.0%, science 8.5%, instruction following 2.1%) and reports one student result: a model trained from Qwen2.5-32B scores 87.1 on AIME2024 against 91.4 for the teacher ("Dataset Summary"; "Dataset Statistics").
- **Guideline:** When using this corpus as SFT data for a general-purpose model, run separate chat, instruction-following, and knowledge evaluations, because the card reports only one AIME2024 number and gives no training settings, decontamination procedure, or teacher sampling parameters.
- **Authors:** a-m-team
- **Year:** 2025 (citation block "June 2025"; Hugging Face repository created 2025-06-04; no paper)
- **URL:** https://huggingface.co/datasets/a-m-team/AM-DeepSeek-R1-0528-Distilled
- **Source type:** model/dataset card
- **Relevant topics:** reasoning distillation, long-CoT SFT data, per-domain verification, perplexity filtering, teacher refresh

## Summary
The card describes a reasoning corpus "distilled from DeepSeek-R1-0528", which it calls an improved version of
DeepSeek-R1 with advances in reasoning, instruction following, and multi-turn dialogue. The team collected and
distilled 2.6 million queries across domains with DeepSeek-R1-0528 as the teacher. The card states that the
teacher's outputs are longer than earlier versions, "especially in mathematics", where some outputs are 1.5 to 2
times longer. All outputs "underwent automated verification" chosen by task category, and each subset is
filtered by perplexity, n-gram repetition, and structure checks. The card states that the dataset "follows a
unified format and verification pipeline, enabling direct comparison with other open-source distillation
corpora", without naming those corpora.

## Key Contributions
- Release of DeepSeek-R1-0528 responses for 2.6M unique prompts in five categories ("Dataset Statistics").
- Per-record verification metadata: `verify_score`, `ppl`, `model_name`, `ground_truth`, `test_case` ("Data Fields").
- A per-category verification list and three filters applied to every subset ("Verification and Quality Control").
- One student-versus-teacher comparison on AIME2024 ("Dataset Summary" table).

## Key Figures/Tables to Study
- "Dataset Summary" table: Qwen2.5-32B student vs DeepSeek-R1-0528 on AIME2024.
- "Dataset Statistics": category counts (an image `AM-distilled.png` follows; its content was not transcribed).
- "Data Fields": metadata available for filtering and subset selection.

## Technical Details
- Category counts: general chat 1,223K (47.3%); math 674K (26.1%); code 412K (16.0%); science 220K (8.5%);
  if 54K (2.1%) ("Dataset Statistics"). The five counts sum to 2,583K (derived). General chat "includes both
  multiturn and other types of data" ("Dataset Statistics" note).
- Files: `code.jsonl`, `if.jsonl`, `math.jsonl`, `multiturn.jsonl`, `other.jsonl`, `science.jsonl` (Hugging Face API
  file list at sha 8d94d362). Total file size 80 GB (dataset page).
- Languages: English and Chinese (card metadata). No license field is present in the card metadata; the
  "Limitations" section restricts use to "research purposes only" and prohibits commercial use.
- Record format: `system` (distillation system prompt with `<think>` and `<answer>` tags, empty in some records)
  and `conversations` (turns with `from` in {human, assistant}, `value`, `info`) ("Data Fields").
- The card states: "The 'system' field is not used in training" ("Data Fields" note).
- `info` fields: `source` (example `OpenHermes-2.5`), `category`, `ground_truth`, `test_case`,
  `instruction_constrain`, `think_content`, `answer_content`, `verify_score` ("float ≥ 0.9"), `model_name`
  (`deepseek-r1-0528`), `ppl` (perplexity of the assistant output) ("Data Fields").
- The dataset viewer preview shows code records with `source` "opencoder" and a `test_case` string with
  `call_type` "assert" (dataset page preview rows).
- Verification by category ("Verification and Quality Control"):
  - Math: Math-Verify, binary pass/fail.
  - Code: test-case validation in sandbox environments.
  - Science: answer similarity scored by an LLM (model not named).
  - Instruction following: `IFEval` validator.
  - General chat: a reward model, "e.g., Decision-Tree-Reward-Llama-3.1-8B".
- Filters applied to each subset: perplexity filtering "using a strong 32B LLM" (model not named); n-gram
  repetition filtering; structural checks such as presence of `<think>` and `<answer>` ("Verification and Quality
  Control"). Thresholds for perplexity and repetition are not given.
- Student result: "Performance on this dataset training with Qwen2.5-32B" links the Qwen/Qwen2.5-32B repository;
  AIME2024 87.1 for the student vs 91.4 for DeepSeek-R1-0528 ("Dataset Summary" table). Sampling, number of
  samples, and evaluation length for this number are not given.

## Recipe ledger
| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| unnamed student trained on this dataset | 32B | distill-SFT | initial checkpoint | Qwen2.5-32B (link to Qwen/Qwen2.5-32B) | dataset card, "Dataset Summary" | verified 2026-09-14 | AIME2024 87.1 vs teacher 91.4, no ablation |
| same | 32B | distill-SFT | training data | this dataset (subset or full set not stated) | dataset card, "Dataset Summary" | verified 2026-09-14 | — |
| same | 32B | distill-SFT | objective, LR, schedule, batch, epochs, max length, packing, loss masking | not reported | checked card sections, API metadata, a-m-team GitHub README | not reported | — |
| AM-DeepSeek-R1-0528-Distilled | — | distill-SFT data | teacher temperature, top-p, max length, samples per prompt | not reported | checked card sections | not reported | — |
| AM-DeepSeek-R1-0528-Distilled | — | distill-SFT data | minimum verification score kept | `verify_score` float ≥ 0.9 | dataset card, "Data Fields" | verified 2026-09-14 | no ablation reported |

## Findings relevant to distillation and generality
- Stage: long-CoT reasoning SFT data; the card does not describe how the student was trained ("Dataset Summary").
- Prompt types: five categories dominated by general chat (47.3%), including multi-turn data; prompt sources are
  named only through the per-record `source` field ("Dataset Statistics"; "Data Fields").
- Teacher: DeepSeek-R1-0528 for all responses; longer math outputs than earlier generations (1.5–2× "for some math
  problems") ("Dataset Summary"). The card gives no token-length statistics.
- Quality control: category-specific verifiers plus perplexity, repetition, and structure filters; "Each sample is
  verified and filtered", and no rejected outputs are described as included ("Verification and Quality Control").
- Generality measurement: only AIME2024 is reported, although 47.3% of prompts are general chat and 2.1% are
  instruction following ("Dataset Summary"; "Dataset Statistics").
- Third-party use: the dataset page lists Zhihu-ai/Zhi-Create-Qwen3-32B among models trained or fine-tuned on this
  dataset (Hugging Face model-tree listing, read 2026-09-14).

## Connections
- [[am-deepseek-r1-distilled-1-4m]] — earlier a-m-team corpus from the original DeepSeek-R1 with a paper describing
  a similar verification pipeline.
- [[deepseek-r1]] — family of the teacher model.
- [[zhihu-zhi-create-distill]] — Zhi-Create-Qwen3-32B, listed as trained or fine-tuned on this dataset.
- [[distillation-source-matters]], [[am-thinking-v1]] — related reasoning-distillation work (see those cards).
- [[ifeval]] — the validator named for instruction-following verification.
- [[open-thoughts]], [[mixture-of-thoughts]], [[nvidia-llama-nemotron-post-training-dataset]] — other released
  reasoning-distillation corpora for comparison.
- [[openhermes-2-5]] — named as an example prompt `source`.

## Verification
- Created on 2026-09-14 from https://huggingface.co/datasets/a-m-team/AM-DeepSeek-R1-0528-Distilled (README at
  repository sha 8d94d362, last modified 2025-06-09; Hugging Face API metadata, createdAt 2025-06-04).
- Audit claims not found in the source: "validated by SFT of Qwen2.5-32B base" (the card says "training with
  Qwen2.5-32B" and links the base repository, but does not name SFT as the method); "Verify confidence ≥0.9" is
  stated only as the `verify_score` field range; "Same org also releases AM-Qwen3-Distilled,
  AM-Thinking-v1-Distilled and AM-DeepSeek-Distilled-40M" (not named on this card); "Later reused as general SFT
  data" (the Zhi-Create listing does not state how the data was used).
