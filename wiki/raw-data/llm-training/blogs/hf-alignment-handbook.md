<!-- scope: The Alignment Handbook — Hugging Face's released YAML recipes and scripts for continued pretraining, SFT, DPO and ORPO, including the Zephyr-7b-β reproduction
     deps: [[dpo]]
     see-also: [[sequence-packing]], [[loss-masking-prompt]], [[allenai-tulu-sft-recipe]], [[fsdp-sft]], [[hf-dpo-zoo]]
-->

# The Alignment Handbook
- **Core Insight:** The handbook publishes one YAML file per training run, so a recipe such as Zephyr-7b-β is fully specified by `recipes/zephyr-7b-beta/sft/config_full.yaml` and `.../dpo/config_full.yaml` plus a two-line `accelerate launch` command; the scripts are thin wrappers over TRL trainers and add no custom training loop (repo README, "How to navigate this project"; `scripts/sft.py`).
- **Guideline:** When reproducing a published open recipe, read the released config rather than the paper, because the maintainers state that after fixing an UltraFeedback labeling problem and a TRL learning-rate-scheduler bug they moved DPO from 3 epochs at β = 0.1 to 1 epoch at β = 0.01 while keeping comparable performance (`recipes/zephyr-7b-beta/README.md`, "Note").
- **Authors:** Lewis Tunstall, Edward Beeching, Nathan Lambert, Nazneen Rajani, Shengyi Huang, Kashif Rasul, et al. (Hugging Face H4)
- **Year:** 2023 (first full Zephyr-7b-β release 2023-11-10 per the README "News" section; repo still updated 2025-07-24 with the SmolLM3-3B recipe)
- **URL:** https://github.com/huggingface/alignment-handbook
- **Source type:** released config/code
- **Relevant topics:** SFT recipe, DPO, ORPO, chat templates, TRL integration, reproducible alignment

## Summary
The repository states its purpose as filling a gap: SFT resources were common, but few public resources described how to train models on human or AI preferences end to end. It contains `scripts/` covering four steps — continued pretraining, SFT for chat, DPO, and ORPO — each supporting full-weight distributed training with DeepSpeed ZeRO-3 or LoRA/QLoRA, and `recipes/` holding one YAML per run. The stated initial scope also lists reward modeling and rejection sampling. The `gpt2-nl` recipe is included to show language or domain adaptation by continued pretraining followed by SFT and DPO.

## Key Contributions
- One YAML per training run, containing every parameter for that run, with a matching `accelerate launch` command in the recipe README.
- Chat-template handling: the Zephyr SFT config sets an explicit Jinja `chat_template` string with `<|user|>`, `<|system|>`, and `<|assistant|>` turn markers and an `eos_token` after each turn.
- `No Robots`, a dataset of 10,000 instructions and demonstrations written entirely by human annotators, released with the Zephyr code (README "News", 2023-11-10).
- A preference-method comparison recipe, `recipes/pref_align_scan/`, evaluating DPO vs. KTO vs. IPO (README "News", 2024-01-18).
- A documented correction of the published Zephyr hyperparameters after two defects were found (see Guideline above).
- Later recipes reusing the same structure: StarChat2 15B, Zephyr 7B Gemma (RLAIF), Zephyr 141B (A35B) with ORPO, SmolLM/SmolLM2-Instruct, SmolLM3-3B.

## Technical Details
- `scripts/sft.py` imports `SFTTrainer` and `TrlParser` from `trl` and `SFTConfig`, `get_dataset`, `get_model`, `get_tokenizer` from the local `alignment` package; `src/alignment/configs.py` defines `SFTConfig(trl.SFTConfig)`, `DPOConfig(trl.DPOConfig)`, and `ORPOConfig(trl.ORPOConfig)` as thin subclasses (`scripts/sft.py` L47-48; `src/alignment/configs.py` L134-152).
- `packing` appears only as a command-line example in the `scripts/sft.py` docstring (L26). It is not set in the Zephyr configs, so those runs use the TRL default.
- Neither `completion_only_loss` nor `assistant_only_loss` is set in the Zephyr SFT config, so the run does not enable TRL's response-only masking. There is no `train_on_response_only` key anywhere in the repo.
- The SFT data is specified as a `dataset_mixture` of `HuggingFaceH4/ultrachat_200k` splits `train_sft` and `test_sft`, each with `weight: 1.0`, and `test_split_size: 1000`.
- The DPO data is `HuggingFaceH4/ultrafeedback_binarized` splits `train_prefs` and `test_prefs`, `test_split_size: 2000`, with the `chosen` and `rejected` columns.
- Full training of the 7B model requires 8 GPUs with 80 GB of VRAM and is launched with `recipes/accelerate_configs/zero3.yaml` (`recipes/zephyr-7b-beta/README.md`). DeepSpeed ZeRO-3 is the documented path for the full model; the README does not give an FSDP command for this recipe.
- QLoRA variants run on `--num_processes=1` with `recipes/accelerate_configs/ddp.yaml` and `--load_in_4bit=true`.
- Pinned environment: Python 3.11, `torch==2.6.0` (cu126), `flash-attn==2.7.4.post1`; the README states the precise PyTorch version matters for reproducibility.
- Known defects the maintainers document: Argilla found a few thousand incorrect GPT-4 preference labels in the source UltraFeedback dataset, and TRL's `SFTTrainer` had a learning-rate-scheduler bug that terminated training early (`recipes/zephyr-7b-beta/README.md`, "Note").

## Recipe ledger

| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| zephyr-7b-sft-full | 7B | SFT | base model | `mistralai/Mistral-7B-v0.1` | `recipes/zephyr-7b-beta/sft/config_full.yaml`@main | verified 2026-09-18 | no ablation reported |
| zephyr-7b-sft-full | 7B | SFT | data | `HuggingFaceH4/ultrachat_200k`, `train_sft` + `test_sft`, weight 1.0 each | same file | verified 2026-09-18 | no ablation reported |
| zephyr-7b-sft-full | 7B | SFT | max sequence length | 2048 | same file | verified 2026-09-18 | no ablation reported |
| zephyr-7b-sft-full | 7B | SFT | peak LR / schedule / warmup | 2.0e-05, cosine, `warmup_ratio: 0.1` | same file | verified 2026-09-18 | no ablation reported |
| zephyr-7b-sft-full | 7B | SFT | epochs | 1 (`num_train_epochs: 1`, `max_steps: -1`) | same file | verified 2026-09-18 | no ablation reported |
| zephyr-7b-sft-full | 7B | SFT | global batch (sequences) | 128 = 16 per device × 8 GPUs × grad-accum 1 | `config_full.yaml` (`per_device_train_batch_size: 16`, `gradient_accumulation_steps: 1`) + `recipes/zephyr-7b-beta/README.md` ("You will require 8 GPUs") | derived | formula and inputs shown in this row |
| zephyr-7b-sft-full | 7B | SFT | precision / attention / checkpointing | `bf16: true`, `flash_attention_2`, `gradient_checkpointing: true` (`use_reentrant: False`) | `config_full.yaml` | verified 2026-09-18 | no ablation reported |
| zephyr-7b-sft-full | 7B | SFT | packing, response-only masking | not set in the config | `config_full.yaml` | not reported | checked: SFT config, `scripts/sft.py`, `src/alignment/configs.py` |
| zephyr-7b-sft-full | 7B | SFT | parallelism | DeepSpeed ZeRO-3 (`recipes/accelerate_configs/zero3.yaml`) | `recipes/zephyr-7b-beta/README.md` | verified 2026-09-18 | not applicable |
| zephyr-7b-dpo-full | 7B | preference | starting checkpoint | `alignment-handbook/zephyr-7b-sft-full` | `recipes/zephyr-7b-beta/dpo/config_full.yaml`@main | verified 2026-09-18 | not applicable |
| zephyr-7b-dpo-full | 7B | preference | data | `HuggingFaceH4/ultrafeedback_binarized`, `train_prefs` + `test_prefs` | same file | verified 2026-09-18 | no ablation reported |
| zephyr-7b-dpo-full | 7B | preference | β | 0.01 | same file | conflict | `recipes/zephyr-7b-beta/README.md`: 1 epoch at β = 0.01 reaches performance comparable to the released `zephyr-7b-beta`; the repo config is the reproduction path |
| zephyr-7b-beta (released checkpoint) | 7B | preference | β and epochs as described in the technical report | β = 0.1, 3 epochs | `recipes/zephyr-7b-beta/README.md` "Note", referring to arXiv:2310.16944 | conflict | the released `zephyr-7b-beta` checkpoint came from this setting; the current config supersedes it for reproduction |
| zephyr-7b-dpo-full | 7B | preference | LR / schedule / warmup | 5.0e-7, cosine, `warmup_ratio: 0.1` | `dpo/config_full.yaml` | verified 2026-09-18 | no ablation reported |
| zephyr-7b-dpo-full | 7B | preference | epochs | 1 | same file | verified 2026-09-18 | `recipes/zephyr-7b-beta/README.md`: 1 epoch found sufficient after the UltraFeedback and TRL fixes |
| zephyr-7b-dpo-full | 7B | preference | global batch (pairs) | 128 = 8 per device × 8 GPUs × grad-accum 2 | `dpo/config_full.yaml` + recipe README GPU count | derived | formula and inputs shown in this row |
| zephyr-7b-dpo-full | 7B | preference | max length / max prompt length | 1024 / 512 | `dpo/config_full.yaml` | verified 2026-09-18 | no ablation reported |
| zephyr-7b-dpo-full | 7B | preference | optimizer | `adamw_torch` (betas not set) | `dpo/config_full.yaml` | verified 2026-09-18 | not applicable |

## Findings relevant to generality
- The repository is a set of configs, not a study. It reports no held-out generalization measurement, no benchmark table, and no ablation for any value in the ledger above. Any generality claim attributed to this source is unsupported; use [[allenai-tulu-sft-recipe]] or the Zephyr technical report for measured results.
- The one measured statement the repo does make is a negative result about reproduction: two upstream defects (dataset labels, scheduler) changed which hyperparameters were best, which is evidence that a recipe copied from a paper may not reproduce against a later library version (`recipes/zephyr-7b-beta/README.md`).

## Connections
- [[dpo]] is the preference objective these configs set `beta` for.
- [[hf-dpo-zoo]] covers the `pref_align_scan` comparison of DPO, KTO, and IPO run from this repo.
- [[allenai-tulu-sft-recipe]] is the comparable open recipe with published ablations.
- [[sequence-packing]] and [[loss-masking-prompt]] hold the mechanisms this repo's configs leave at their TRL defaults.
- [[fsdp-sft]] covers the distributed runtime; note this repo documents DeepSpeed ZeRO-3 for the 7B full run.

## Verification
- Checked on 2026-09-18 against: https://github.com/huggingface/alignment-handbook (branch `main`; citation block version `0.4.0.dev0`), files `README.md`, `recipes/zephyr-7b-beta/README.md`, `recipes/zephyr-7b-beta/sft/config_full.yaml`, `recipes/zephyr-7b-beta/dpo/config_full.yaml`, `scripts/sft.py`, `src/alignment/configs.py`.
- Corrections to the previous card version:
  - "β | 0.1" → `beta: 0.01` in `dpo/config_full.yaml`; the 0.1 value belongs to the technical-report run with 3 epochs, which the recipe README explicitly supersedes.
  - "Global batch size | 32 pairs" → 128 pairs (8 × 8 GPUs × grad-accum 2), derived from the config and the recipe README.
  - "FSDP / ZeRO | FSDP FULL_SHARD" → the recipe README launches the full run with DeepSpeed ZeRO-3; no FSDP config is given for it.
  - "Optimizer | AdamW (β = 0.9, 0.95)" → betas are not set in either config; the DPO config sets `optim: adamw_torch`.
  - "Packing | true" and "Train on response only | true" → neither key is set in the Zephyr SFT config.
  - Author list → the repo's own citation block lists Tunstall, Beeching, Lambert, Rajani, Huang, Rasul, Bartolome, M. Patiño, M. Rush, Wolf. Belkada, von Werra, Fourrier, Habib, Sarrazin, and Sanseviero were not in it.
  - "It powers the Zephyr-7B, StarChat2, and Llama-3.1-Tulu checkpoints" → the README lists Zephyr 7B β, Zephyr 7B Gemma, Zephyr 141B (A35B), StarChat2 15B, SmolLM, SmolLM2-Instruct, and SmolLM3-3B. Tülu is not among them.
  - Recipe directories "`sft/`, `dpo/`, `orpo/`, `kto/`" → the README lists four script steps: continued pretraining, SFT, DPO, ORPO. KTO appears only inside the `pref_align_scan` comparison recipe.
- Removed as unsupported by the source:
  - The `SFTConfig(..., train_on_response_only=True)` code snippet. `train_on_response_only` is not a field of TRL's `SFTConfig` and does not appear in this repo.
  - "NEFTune (α=5) stacks cleanly; toggled via SFT config" — no NEFTune setting in the README or the Zephyr configs.
  - "Prescribed eval: MT-Bench + AlpacaEval + IFEval" — the README prescribes no evaluation suite.
  - "Multi-turn: only the final assistant turn gets gradient" — no such rule in the repo.
  - "DPO runs for < 1 epoch; longer causes chosen-side collapse" — the config runs 1 epoch and the repo makes no collapse claim.
  - "Use ZeRO-3 for ≥ 13B; FSDP for ≤ 13B (in 2024 the throughput crossover shifted)" — no throughput comparison in the repo.
  - "Template mismatch is the #1 silent bug" — the repo makes no such statement and reports no bug ranking.
  - "UltraChat-200K filtered → ~200K multi-turn dialogues … Pack to 2048 tokens" — the config sets `max_seq_length: 2048` and does not pack; the dataset size is a property of the dataset card, not this repo.
- Not reported by the source: wall-clock or GPU-hour cost of either stage; evaluation numbers for any checkpoint; optimizer betas; weight decay; gradient-clipping value.
