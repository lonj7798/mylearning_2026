---
chapter: ch-14a
course: llm-training
phase: read
excerpt_of: "huggingface/smollm text/pretraining/smollm3 nanotron configs at commit a48fa612fd3e1c28da16b7e12ae5c50ab80b4cf7, and the diff of commit 4bc94684c32bf3eaeaeef2f0c899e2ea27c6778a (no library card as of 2026-09-15; chapter-local verified extract)"
source_url: https://github.com/huggingface/smollm/tree/a48fa612fd3e1c28da16b7e12ae5c50ab80b4cf7/text/pretraining/smollm3
created_at: "2026-09-15"
---

# Excerpt: SmolLM3 pre-training configs (stage1_8T.yaml, stage2_8T_9T.yaml, stage3_9T_11T.yaml)

Source type: released config. Commit history of the directory: 4c54ad1 (2025-07-21, "fix configs"), 4bc9468 (2025-08-03, "fix nb of steps in early configs"), a48fa61 (2025-08-12, "add hf links on stage1 config"). The blog announcing SmolLM3 is dated July 8, 2025, so the step counts below were edited after release. The files fetched from `main` on 2026-09-14 are byte-identical to the files at a48fa61.

## Optimizer, schedule, batch (stage1_8T.yaml at a48fa61)
```yaml
# L204-221
  clip_grad: 1.0
  learning_rate_scheduler:
    learning_rate: 0.0002
    lr_decay_starting_step: 4198001
    lr_decay_steps: 522000
    lr_decay_style: linear
    lr_warmup_steps: 2000
    lr_warmup_style: linear
    min_decay_lr: 0
  optimizer:
    adam_beta1: 0.9
    adam_beta2: 0.95
    adam_eps: 1.0e-08
    name: adamW
    torch_adam_is_fused: true
  weight_decay: 0.1
  weight_decay_exclude_named_params:
  - .*token_embedding.*
# L225, L228, L231
  dp: 192
  pp: 1
  tp: 2
# L250-255
  batch_accumulation_per_replica: 1
  ...
  micro_batch_size: 3
  sequence_length: 4096
  train_steps: 4720000
```
The same scheduler and batch values appear in stage2_8T_9T.yaml (L355-363, L376, L401-406) and stage3_9T_11T.yaml (L538-546, L559, L584-589). Each file resumes a checkpoint with `load_lr_scheduler: true` and `load_optimizer: true` (L5-7); the resume paths are under `s3://smollm3/tp-fix-final-pre-training/`.

## Values before commit 4bc9468 (diff, "fix nb of steps in early configs")
- stage1_8T.yaml: `lr_decay_starting_step: 2600000` → `4198001`; `lr_decay_steps: 600000` → `522000`; `train_steps: 3200000` → `4720000`.
- stage2_8T_9T.yaml: `lr_decay_starting_step: 4000000` → `4198001`; `lr_decay_steps: 0` → `522000`; `train_steps: 4000000` → `4720000`.
- stage3_9T_11T.yaml was not changed by this commit.

## Stage boundaries
- stage2_8T_9T.yaml adds a data stage `name: stable stage 2` with `start_training_step: 3450001` (L300-301); stage3_9T_11T.yaml adds `name: decay stage` with `start_training_step: 4198001` (L483-484).
- Derived tokens per step: dp × accumulation × micro batch × sequence length = 192 × 1 × 3 × 4,096 = 2,359,296. Derived GPUs: dp × tp × pp = 192 × 2 × 1 = 384.
- Derived boundaries: 3,450,000 × 2,359,296 = 8.14T; 4,198,000 × 2,359,296 = 9.90T; 4,720,000 × 2,359,296 = 11.14T. Derived decay share: 522,000 / 4,720,000 = 11.06% of steps (1.23T tokens).
- Derived pre-fix plan in stage1_8T.yaml: decay from step 2,600,000 (6.13T tokens) over 600,000 steps (18.75% of 3,200,000 steps) to step 3,200,000 (7.55T tokens).

## Sampling weights grouped by this chapter
Weights are per-dataset sampling weights (stage 1: 41 datasets, L99-140; decay stage: 56 datasets). The grouping is this chapter's: English web = fineweb-edu, dclm, pes2o, wiki; multilingual web = fw2-* and multilingual_wiki; code = stack-v2 and stack-edu subsets, stackexchange, pull-requests, kaggle, jupyter-scripts, github-issues, open-codereasoning-4k; math = infiwebmath, finemath, megamath, openmathinstruct-2, openmathreasoning-4k, tiny-gsm-mind, dolmino math sets; other = cosmopedia2, natural_reasoning.

| Data stage | English web | Multilingual web | Code | Math | Other | Weight sum |
|---|---|---|---|---|---|---|
| stable (stage1_8T.yaml) | 72.4% | 12.0% | 12.9% | 2.7% | 0% | 1.000 |
| stable stage 2 (stage2_8T_9T.yaml) | 64.7% | 11.7% | 14.5% | 9.1% | 0% | 1.00005 |
| decay stage (stage3_9T_11T.yaml) | 49.9% | 13.1% | 23.3% | 13.1% | 0.5% | 1.00565 |

Percentages are normalized by the weight sum (derived). Decay-stage raw weight sums: English web 0.5022, multilingual 0.13205 (fw2-* alone 0.12405), code 0.2348 (0.2343 without open-codereasoning-4k), math 0.1316, other 0.005.

The decay-stage math group includes `dolmino_math_synth_gsm_gsm8k` (stage3_9T_11T.yaml L359) with weight `0.0004 # Dolmino synth math gsm8k` (L473), next to `tiny-gsm-mind-problem-solving` and `tiny-gsm-mind-2students` (L357-358).

## Verification
- Read on 2026-09-15: the three YAML files at commit a48fa61 (compared with the `main` copies), the GitHub commits API for the directory, and the diff of commit 4bc9468 from the GitHub commits API.
- Not in the configs: which checkpoint was released; whether the run was launched with the pre-fix step counts; evaluation results.
