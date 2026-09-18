<!-- scope: recipe ledger for [[openrlhf-dpo]]: OpenRLHF DPO CLI defaults, DPOTrainer class defaults, and the example Llama-3-8B script, at commit 64c1cc4
     deps: [[openrlhf-dpo]]
     see-also: [[dpo]], [[ipo]], [[llama-3]]
-->

# OpenRLHF DPO recipe ledger (commit 64c1cc4)
- **Core Insight:** OpenRLHF's DPO CLI defaults are β = 0.1, AdamW LR 1e-5, 1 epoch, global batch 128, and max length 512; its example Llama-3-8B script uses β = 0.1, LR 5e-7, global batch 256, and max length 8192 (train_dpo.py L189, L236–237, L266, L318; train_dpo_llama.sh L9, L14, L16–17).
- **Guideline:** When a run uses OpenRLHF defaults, record them as framework defaults, not as a tuned recipe, because the repository reports no ablation for any of these values.
- **Authors:** OpenRLHF project contributors (GitHub organization OpenRLHF)
- **Year:** 2026 (commit 64c1cc4, 2026-04-19)
- **URL:** https://github.com/OpenRLHF/OpenRLHF/tree/64c1cc4f30d4c0dab772c8aa1dc11288c425e0da
- **Source type:** released config/code
- **Relevant topics:** DPO hyperparameters, framework defaults, example configuration

## Summary
This file holds the recipe ledger for [[openrlhf-dpo]]. Three kinds of values are recorded separately (course standard §5.3 rule 4): CLI defaults in `openrlhf/cli/train_dpo.py`, constructor defaults of `DPOTrainer` in `openrlhf/trainer/dpo_trainer.py` (overridden by the CLI), and the values set in `examples/scripts/train_dpo_llama.sh`. Unit notes: each dataset item is one chosen/rejected pair (dpo_trainer.py L151), so `train.batch_size` and `train.micro_batch_size` count pairs; the forward pass holds 2× that many sequences (L350–360). `train.batch_size` has the help text "Global training batch size" and `train.micro_batch_size` has "batch size per GPU" (train_dpo.py L188–189).

## Recipe ledger
All loci refer to commit 64c1cc4. No row has an ablation in the source, so the evidence column reads "no ablation reported" throughout.

| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| OpenRLHF `train_dpo.py` CLI default | any | preference | preference loss, β | sigmoid DPO (IPO off), β = 0.1 | train_dpo.py L237–240 | verified 2026-09-14 | no ablation reported |
| OpenRLHF `train_dpo.py` CLI default | any | preference | label smoothing ε (cDPO) | 0.0 | train_dpo.py L241–243 | verified 2026-09-14 | no ablation reported |
| OpenRLHF `train_dpo.py` CLI default | any | preference | NLL coefficient on chosen responses | 0 | train_dpo.py L245–247 | verified 2026-09-14 | no ablation reported |
| OpenRLHF `train_dpo.py` CLI default | any | preference | MoE auxiliary loss coefficient | 0 | train_dpo.py L244 | verified 2026-09-14 | no ablation reported |
| OpenRLHF `train_dpo.py` CLI default | any | preference | epochs | 1 | train_dpo.py L236 | verified 2026-09-14 | no ablation reported |
| OpenRLHF `train_dpo.py` CLI default | any | preference | global batch / micro batch per GPU (pairs) | 128 / 8 | train_dpo.py L188–189 | verified 2026-09-14 | no ablation reported |
| OpenRLHF `train_dpo.py` CLI default | any | preference | optimizer | AdamW (`--optim adam`), LR 1e-5, betas (0.9, 0.95), eps 1e-8, weight decay 0.0 | train_dpo.py L255, L266–269 | verified 2026-09-14 | no ablation reported |
| OpenRLHF `train_dpo.py` CLI default | any | preference | LR schedule | `cosine_with_min_lr`, warmup ratio 0.03, min LR ratio 0.1 | train_dpo.py L271–273 | verified 2026-09-14 | no ablation reported |
| OpenRLHF `train_dpo.py` CLI default | any | preference | gradient clipping | max_norm 1.0 | train_dpo.py L275 | verified 2026-09-14 | no ablation reported |
| OpenRLHF `train_dpo.py` CLI default | any | preference | max sequence length | 512 | train_dpo.py L318 | verified 2026-09-14 | no ablation reported |
| OpenRLHF `train_dpo.py` CLI default | any | preference | DeepSpeed ZeRO stage; param dtype | 2; bf16 | train_dpo.py L205, L206–212 | verified 2026-09-14 | no ablation reported |
| OpenRLHF `train_dpo.py` CLI default | any | preference | reference model path; reference offload | same as policy path; off | train_dpo.py L342–343, L213 | verified 2026-09-14 | no ablation reported |
| OpenRLHF `train_dpo.py` CLI default | any | preference | max training samples | 1000000 | train_dpo.py L308 | verified 2026-09-14 | no ablation reported |
| OpenRLHF `DPOTrainer` constructor default (overridden by CLI) | any | preference | β; max_norm; max_epochs | 0.01; 0.5; 2 | dpo_trainer.py L42–44 | verified 2026-09-14 | no ablation reported |
| OpenRLHF example `train_dpo_llama.sh` (`OpenRLHF/Llama-3-8b-sft-mixture`) | 8B | preference | β; LR | 0.1; 5e-7 | train_dpo_llama.sh L16–17 | verified 2026-09-14 | no ablation reported |
| OpenRLHF example `train_dpo_llama.sh` | 8B | preference | global batch / micro batch per GPU (pairs) | 256 / 1 | train_dpo_llama.sh L9–10 | verified 2026-09-14 | no ablation reported |
| OpenRLHF example `train_dpo_llama.sh` | 8B | preference | epochs; max length | 1; 8192 | train_dpo_llama.sh L13–14 | verified 2026-09-14 | no ablation reported |
| OpenRLHF example `train_dpo_llama.sh` | 8B | preference | ZeRO stage; dtype; packing; gradient checkpointing | 3; bf16; on; on | train_dpo_llama.sh L15, L12, L24–25 | verified 2026-09-14 | no ablation reported |
| OpenRLHF example `train_dpo_llama.sh` | 8B | preference | preference data | `OpenRLHF/preference_dataset_mixture2_and_safe_pku`, chat template applied | train_dpo_llama.sh L18–21 | verified 2026-09-14 | no ablation reported |
| OpenRLHF example `train_dpo_llama.sh` | 8B | preference | commented optional flags | `--model.ipo_enable` "[for IPO]"; `--model.label_smoothing 0.1` "[for cDPO]"; `--ref.offload`; `--model.nll_loss_coef` | train_dpo_llama.sh L28–32 | verified 2026-09-14 | no ablation reported |
| OpenRLHF example `train_dpo_llama.sh` | 8B | preference | compute, number of pairs, total tokens, checkpoint selection | not reported (checked: script, `train_dpo.py`, `dpo_trainer.py`) | — | not reported | — |

## Connections
- [[openrlhf-dpo]] — code description of the loss and trainer these values configure.
- [[llama-3]] — report cited by the `nll_loss_coef` help text.

## Verification
- Checked on 2026-09-14 against: https://github.com/OpenRLHF/OpenRLHF at commit 64c1cc4f30d4c0dab772c8aa1dc11288c425e0da; `train_dpo_llama.sh` is byte-identical at main b117b2b5.
- Corrections to the previous card version: none (new file split from [[openrlhf-dpo]]).
- Removed as unsupported by the source: none.
