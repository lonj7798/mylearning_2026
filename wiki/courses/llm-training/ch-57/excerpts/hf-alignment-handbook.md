---
chapter: ch-57
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/blogs/hf-alignment-handbook.md
source_url: https://github.com/huggingface/alignment-handbook
created_at: "2026-04-23"
---

# Excerpt: HF Alignment Handbook — TRL in production form

**Source library:** `wiki/raw-data/llm-training/blogs/hf-alignment-handbook.md`
**Artifact:** The Alignment Handbook is HF's reference stack for SFT + preference optimization. It powered Zephyr-7B, StarChat2, and Llama-3.1-Tulu. Every recipe is a YAML + a shell one-liner wrapping TRL.

---

## Why this source anchors ch-57 §2 and §4

Ch-57 §2 describes `SFTTrainer` with `packing=True` and `train_on_response_only=True` as the default SFT entry point; §4 describes the HF-ecosystem integration story (PEFT, datasets, transformers). The Alignment Handbook is the single most-copied set of TRL configs in open source. It is the canonical example of "what a good TRL run actually looks like" at the small-to-mid scale TRL targets.

---

## The SFT config block ch-57 §2 quotes

```python
from trl import SFTTrainer, SFTConfig
config = SFTConfig(
    packing=True,
    max_seq_length=2048,
    train_on_response_only=True,  # masks user tokens automatically
    dataset_kwargs={"add_special_tokens": False},
)
```

Three knobs that together encode the Zephyr recipe:

- **`packing=True`** — concatenate short examples into fixed-length chunks via `ConstantLengthDataset`. Eliminates wasted compute from padding. Requires Flash Attention 2 with packed-sequence support for correct attention masking; TRL expects this.
- **`train_on_response_only=True`** — user tokens get label `-100` (ignored by cross-entropy). Without this, the model learns to reproduce the user's turn — a subtle but frequent bug.
- **`add_special_tokens=False`** in `dataset_kwargs` prevents double-adding BOS/EOS when the chat template already inserts them.

---

## The DPO config block

```python
from trl import DPOTrainer, DPOConfig
cfg = DPOConfig(beta=0.1, loss_type="sigmoid")
```

β=0.1 is the Zephyr default and remains the field-wide modal value two years later. Under PEFT, `ref_model=None` — the reference policy is computed by disabling adapters on the base model.

---

## Zephyr-7B-β reference hyperparameters (ch-57 §4 references this table)

| Knob | Value |
|------|-------|
| Model | Mistral-7B-v0.1 |
| Max seq length | 2048 |
| Packing | true |
| Train on response only | true |
| Optimizer | AdamW (β = 0.9, 0.95) |
| Learning rate | 2e-5 (SFT), 5e-7 (DPO) |
| LR schedule | cosine, 10% warmup |
| Epochs | 1 |
| Global batch size | 128 (SFT), 32 pairs (DPO) |
| Precision | BF16 |
| FSDP / ZeRO | FSDP FULL_SHARD |
| Gradient checkpointing | true |

These numbers were tuned on 8×A100. The handbook ships an 8×A100 FSDP YAML that works out of the box — one of the reasons TRL dominates the prototyping niche. Ch-57 §3's "Accelerate excels at single-node" claim is operationalized by this config set.

---

## The eval triad ch-57 §4 mentions

The handbook prescribes MT-Bench + AlpacaEval + IFEval as the minimum eval suite after DPO. All three are HF-ecosystem-native — `lm-evaluation-harness` or `fastchat` runs directly on the HF checkpoint with no conversion step. This is ch-57 §4's "the thing you train is the thing you serve" — compared to verl, which requires a checkpoint-conversion script before vLLM inference.

---

## Lessons ch-57 §4 inherits

- **Always verify the chat template.** Decode one packed batch and diff it against a real inference string. Template mismatch is the #1 silent bug. TRL does not enforce this; it passes through `tokenizer.apply_chat_template(...)` verbatim.
- **DPO runs < 1 epoch.** Longer causes chosen-side collapse — the motivating observation behind [[ipo]] and [[simpo]]. The handbook caps epochs at 1.
- **FSDP vs ZeRO-3 crossover.** ZeRO-3 for ≥ 13B; FSDP for ≤ 13B — but the handbook notes the crossover shifts every few months; always re-benchmark.
- **NEFTune (α=5) stacks cleanly.** Exposed as an SFT config toggle. Small SFT improvement, zero cost.

---

## The dataset-mixer pattern (ch-57 §4 references)

The handbook introduced `dataset_mixer` as a YAML convention for blending multiple datasets with explicit weights:

```yaml
dataset_mixer:
  HuggingFaceH4/ultrachat_200k: 1.0
  HuggingFaceH4/no_robots: 0.3
```

TRL does not implement the mixer itself — the handbook's `run_sft.py` wrapper reads the YAML, calls `concatenate_datasets` with interleave probabilities, and hands the resulting `Dataset` to `SFTTrainer`. This is the pattern where the HF datasets library does the heavy lifting and TRL just consumes.

---

## Why this matters for the "TRL decision gate" (ch-57 §6)

Every recipe in the handbook is a TRL run that fits on 8×A100. None of them scales to 64+ GPUs. This is the concrete demonstration of ch-57 §6's "TRL is optimized for the single-node productivity niche." Once Zephyr went past 70B (StarChat2-Mixtral, Llama-3.1-Tulu-405B), the recipes either dropped back to LoRA or moved to a different framework. The handbook itself is a proof of TRL's working range.

---

## Attested implementation notes

- The handbook uses TRL's `SFTConfig`/`DPOConfig` rather than raw `transformers.TrainingArguments`. This is important: TRL config subclasses add trainer-specific knobs (packing, response-only, max_prompt_length) that the parent class does not expose.
- `run_sft.py` / `run_dpo.py` are ~100-line scripts; the heavy lifting happens inside TRL. The handbook's own code is deliberately thin — it is a *recipe book*, not a framework.
- Per-variant recipes exist for `orpo/`, `kto/` — same structure, different TRL trainer.

---

## Connections to the rest of the track

- [[dpo]] — the loss; handbook's DPO stage implements it with default β=0.1.
- [[sequence-packing]] — the packing technique `packing=True` invokes.
- [[loss-masking-prompt]] — the response-only mask `train_on_response_only=True` implements.
- [[neftune]] — small-SFT improvement stacked in the handbook.
- [[allenai-tulu-sft-recipe]] — sister recipe from AllenAI, same TRL backbone.
- [[hf-dpo-zoo]] — the wider preference-optimization variant set this recipe lives inside.
