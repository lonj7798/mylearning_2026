<!-- scope: HF Alignment Handbook — reference implementation of SFT + DPO pipelines for Zephyr-style models
     deps: [[sequence-packing]], [[loss-masking-prompt]], [[dpo]]
     see-also: [[allenai-tulu-sft-recipe]], [[fsdp-sft]]
-->

# HF Alignment Handbook — SFT Recipe
- **Core Insight:** The handbook distills the Zephyr / Mistral Instruct recipe into reproducible YAML configs: chat-template-aware SFT with response-only loss + packing, then DPO — no custom RL code, all on TRL + Accelerate + DeepSpeed/FSDP.
- **Guideline:** Use the handbook's `run_sft.py` as a starting template; override `dataset_mixer`, `chat_template`, and `max_seq_length`; keep `packing: true` and `train_on_response_only: true` unless you have a specific reason to deviate.
- **Authors:** Lewis Tunstall, Edward Beeching, Nathan Lambert, Nazneen Rajani, Kashif Rasul, Younes Belkada, Shengyi Huang, Leandro von Werra, Clémentine Fourrier, Nathan Habib, Nathan Sarrazin, Omar Sanseviero, Alexander M. Rush, Thomas Wolf (Zephyr lineage)
- **Year:** 2023–present
- **URL:** https://github.com/huggingface/alignment-handbook
- **Relevant topics:** SFT recipe, DPO, chat templates, TRL integration, reproducible alignment

## Overview
The Alignment Handbook is HuggingFace's reference stack for instruction-tuning + preference-optimization. It powers the Zephyr-7B, StarChat2, and Llama-3.1-Tulu checkpoints. Every recipe is a YAML + a shell one-liner.

## Key Contributions
- Fully reproducible YAML config per stage (`sft/`, `dpo/`, `orpo/`, `kto/`).
- First-class chat template handling via `tokenizer.apply_chat_template`.
- Packing + response-only loss by default (TRL's `SFTTrainer` with `packing=True`).
- DeepSpeed ZeRO-3 and FSDP wrappers with tested 8×A100 configs.
- Prescribed eval: MT-Bench + AlpacaEval + IFEval.

## SFT Recipe (Zephyr-7B-β reference)

### Data
- UltraChat-200K filtered → ~200K multi-turn dialogues.
- Apply chat template → single string per turn pair.
- Pack to 2048 tokens.

### Loss
- Response-only (user tokens → labels = -100).
- Multi-turn: only the final assistant turn gets gradient (or unroll for per-turn training).

### Hyperparameters
| Knob | Value |
|------|-------|
| Model | Mistral-7B-v0.1 |
| Max seq length | 2048 |
| Packing | true |
| Train on response only | true |
| Optimizer | AdamW (β = 0.9, 0.95) |
| Learning rate | 2e-5 |
| LR schedule | cosine, 10% warmup |
| Epochs | 1 |
| Global batch size | 128 |
| Precision | BF16 |
| FSDP / ZeRO | FSDP FULL_SHARD |
| Gradient checkpointing | true |

## DPO Recipe (continues from SFT)

### Data
- UltraFeedback-Binarized → chosen / rejected pairs.
- Chat template applied independently to both.

### Hyperparameters
| Knob | Value |
|------|-------|
| β | 0.1 |
| Learning rate | 5e-7 |
| LR schedule | cosine |
| Epochs | 1 |
| Global batch size | 32 pairs |
| π_ref | SFT checkpoint (loaded separately, frozen) |

## Code patterns to lift

```python
# SFT: TRL-aware response-only masking
from trl import SFTTrainer, SFTConfig
config = SFTConfig(
    packing=True,
    max_seq_length=2048,
    train_on_response_only=True,  # masks user tokens automatically
    dataset_kwargs={"add_special_tokens": False},
)
```

```python
# DPO: reference model auto-disabled for PEFT; frozen for full FT
from trl import DPOTrainer, DPOConfig
cfg = DPOConfig(beta=0.1, loss_type="sigmoid")
```

## Lessons captured in the handbook
- Always verify chat template by decoding a packed batch — template mismatch is the #1 silent bug.
- DPO runs for < 1 epoch; longer causes chosen-side collapse (motivating [[ipo]], [[simpo]]).
- Use ZeRO-3 for ≥ 13B; FSDP for ≤ 13B (in 2024 the throughput crossover shifted; re-benchmark).
- NEFTune (α=5) stacks cleanly; toggled via SFT config.

## Connections
- SFT mechanics: [[sequence-packing]], [[loss-masking-prompt]], [[neftune]].
- DPO foundation: [[dpo]].
- Distributed runtime: [[fsdp-sft]].
- Sister recipe: [[allenai-tulu-sft-recipe]].
- Preference-method zoo: [[hf-dpo-zoo]] (IPO, KTO, SimPO, ORPO configs).
