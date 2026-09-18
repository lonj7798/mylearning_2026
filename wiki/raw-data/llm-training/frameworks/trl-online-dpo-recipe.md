<!-- scope: recipe ledger for the TRL OnlineDPOTrainer card: framework defaults at commit a08e713 and the documented example and benchmark runs
     deps: [[trl-online-dpo]]
     see-also: [[dpo]], [[ipo]]
-->

# Recipe ledger — TRL `OnlineDPOTrainer` (commit a08e713)
Companion to [[trl-online-dpo]]. Framework defaults and documented run values are separate facts.
Loci: `cfg` = `trl/experimental/online_dpo/online_dpo_config.py`, `trn` = `trl/experimental/online_dpo/online_dpo_trainer.py`,
`docs` = `docs/source/online_dpo_trainer.md`, all at github.com/huggingface/trl@a08e7139f933b770177fc2abc0b43118e26b260b.
Units: one "prompt" yields 2 completions and 1 preference pair (trn L339, L1196–1201).

## Framework defaults (`OnlineDPOConfig`)

| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| TRL OnlineDPOConfig default | any | preference | peak learning rate (AdamW) | 5e-7 | cfg L160–163 (docstring L153) | verified 2026-09-14 | no ablation reported |
| TRL OnlineDPOConfig default | any | preference | preference loss and β | sigmoid (DPO), β 0.1; option "ipo" where β is IPO's τ | cfg L239–254 (docstring L51–60); trn L1223–1226 | verified 2026-09-14 | no ablation reported |
| TRL OnlineDPOConfig default | any | preference | β schedule | a list of floats selects β per epoch; a one-element list becomes a scalar | cfg L239–247, L367–368; trn L547–552 | verified 2026-09-14 | no ablation reported |
| TRL OnlineDPOTrainer (hard-coded) | any | preference | samples per prompt | 2 | trn L339 | verified 2026-09-14 | no ablation reported |
| TRL OnlineDPOConfig default | any | preference | sampling temperature / top_p / top_k | 0.9 / 1.0 / 0 (disabled) | cfg L185–202 | verified 2026-09-14 | no ablation reported |
| TRL OnlineDPOConfig default | any | preference | max new tokens per completion | 64 | cfg L173–176 | verified 2026-09-14 | no ablation reported |
| TRL OnlineDPOConfig field default | any | preference | `max_length` (prompt + completion, left-truncated) | 512 | cfg L177–184; used trn L1053–1057 | conflict | the dataclass field (512) is the value the code uses; no ablation reported |
| TRL OnlineDPOConfig docstring | any | preference | `max_length` | 256 | cfg L41–44 | conflict | docstring disagrees with the field default above |
| TRL OnlineDPOConfig default | any | preference | missing-EOS penalty | None (off) | cfg L231–238; trn L1192–1193 | verified 2026-09-14 | no ablation reported |
| TRL OnlineDPOConfig default | any | preference | reward weights (multiple reward functions) | None → all weights 1.0, combined by `nansum` | cfg L356–362; trn L240–248, L1044–1047 | verified 2026-09-14 | no ablation reported |
| TRL OnlineDPOConfig default | any | preference | generation backend | `use_vllm` False; if True, `vllm_mode` "colocate", `vllm_gpu_memory_utilization` 0.55 | cfg L259–265, L278–295 | verified 2026-09-14 | no ablation reported |
| TRL OnlineDPOConfig default | any | preference | dropout / gradient checkpointing / bf16 / logging_steps | disabled / True / True if fp16 unset / 10 | cfg L255–258; docstring L148–152 | verified 2026-09-14 | no ablation reported |

## Documented runs

| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| trl-lib/Qwen2-0.5B-OnlineDPO (from Qwen/Qwen2-0.5B-Instruct) | 0.5B | preference | reward model / prompts | trl-lib/Qwen2-0.5B-Reward / trl-lib/ultrafeedback-prompt (train split) | docs L17, L34–37, L56 | verified 2026-09-14 | no ablation reported |
| trl-lib/Qwen2-0.5B-OnlineDPO | 0.5B | preference | hyperparameters | `OnlineDPOConfig(output_dir=...)` only, so framework defaults of the TRL version used | docs L39 | verified 2026-09-14 | TRL version of the run not reported |
| trl-lib/Qwen2-0.5B-OnlineDPO | 0.5B | preference | compute | 8 GPUs, "approximately 1 hour" | docs L52 | verified 2026-09-14 | GPU type not reported |
| Pythia TL;DR Online DPO (trl-lib/pythia-{1b,2.8b,6.9b}-deduped-tldr-sft) | 1B, 2.8B, 6.9B | preference | reward models / dataset | trl-lib/pythia-{size}-deduped-tldr-rm / trl-lib/tldr | docs L138–140, L156–158, L174–176 | verified 2026-09-14 | SFT/RM from arXiv:2403.17031 (docs L132) |
| Pythia TL;DR Online DPO | 1B, 2.8B, 6.9B | preference | learning rate / β / epochs | 5.0e-7 / 0.1 / 3 | docs L141–146, L159–164, L177–182 | verified 2026-09-14 | no ablation reported |
| Pythia TL;DR Online DPO | 1B, 2.8B, 6.9B | preference | max new tokens / warmup_steps / missing-EOS penalty | 53 / 0.1 / 1.0 | docs L147–149, L165–167, L183–185 | verified 2026-09-14 | no ablation reported |
| Pythia TL;DR Online DPO | 1B | preference | per-device batch × gradient accumulation; launcher | 8 × 2; `multi_gpu.yaml` | docs L135–145 | verified 2026-09-14 | no ablation reported |
| Pythia TL;DR Online DPO | 2.8B | preference | per-device batch × gradient accumulation; launcher | 8 × 2; `deepspeed_zero2.yaml` | docs L154, L162–163 | verified 2026-09-14 | no ablation reported |
| Pythia TL;DR Online DPO | 6.9B | preference | per-device batch × gradient accumulation; launcher | 4 × 4; `deepspeed_zero2.yaml` | docs L172, L180–181 | verified 2026-09-14 | no ablation reported |
| Pythia TL;DR Online DPO | 1B, 2.8B, 6.9B | preference | hardware | single node of 8 × H100 | docs L132 | verified 2026-09-14 | no ablation reported |
| Pythia TL;DR Online DPO | 1B, 2.8B, 6.9B | eval-gate | win rate | not reported (docs state only that win rate increases with model size) | docs L190–195 | not reported | checked docs L130–195; the W&B report linked at L193 was not checked |
| Pythia TL;DR Online DPO | 1B, 2.8B, 6.9B | preference | process count, global batch, TRL version, judge for win rate | not reported | docs L130–195 | not reported | — |

## Verification
- Checked on 2026-09-14 against the files above at commit a08e713 (2026-04-21).
- This file is new; the previous card listed no hyperparameters.
