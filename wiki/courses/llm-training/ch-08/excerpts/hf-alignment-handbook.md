---
chapter: ch-08
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/blogs/hf-alignment-handbook.md
source_url: https://github.com/huggingface/alignment-handbook
created_at: "2026-04-23"
---

# Excerpt: HF Alignment Handbook — the reference trainer the lab imitates

**Source library:** `wiki/raw-data/llm-training/blogs/hf-alignment-handbook.md`
**Artifact:** Zephyr-7B SFT recipe + `SFTTrainer` configuration

---

## Why this source anchors ch-08

Ch-08's decision to use TRL's `SFTTrainer` as the lab's reference trainer comes directly from this handbook. The handbook is the only widely-reproduced end-to-end SFT recipe in 2024–25 where *every* load-bearing knob is documented as a YAML + command-line one-liner. If the lab's three silent-failure lines (masking, packing, ordering) matter, this is the document that already tripped on each of them in production.

---

## The attested `SFTConfig` — what the lab inherits

From the source (lines 70-78):

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

Ch-08's "Full-budget path" config extends this with the Zephyr hparam table (source lines 36-50):

| Knob | Handbook value | Why it survives unchanged into the lab |
|------|---------------|----------------------------------------|
| Max seq length | 2048 | lab raises to 4096 for Llama-3.1 but keeps the packing contract identical |
| Packing | true | 2× SFT throughput per [[sequence-packing]]; zero quality delta |
| Train on response only | true | the response-only baseline from [[loss-masking-prompt]] |
| Optimizer | AdamW (0.9, 0.95) | Llama-3 and OLMo-2 use the same β pair |
| Learning rate | 2e-5 | stays unchanged even at SmolLM-135M — SFT LR is size-insensitive at these ranges |
| LR schedule | cosine, 10% warmup | the lab's §7 concept-mapping lands here |
| Epochs | 1 | [[allenai-tulu-sft-recipe]] later revises to 2 epochs at 939K prompts; 1 is fine for the lab's 2K–50K |
| Global batch size | 128 | reached via `per_device=1 × grad_accum=16 × n_gpus=8` |
| Precision | BF16 | no loss scaler, no scaler-state silent bug ([[mixed-precision]]) |
| FSDP / ZeRO | FSDP FULL_SHARD | matches [[fsdp-sft]] §"Typical SFT recipe" line-for-line |
| Gradient checkpointing | true | mandatory at 7B + 4096; optional at 135M |

Every row in ch-08's full-budget config is either lifted verbatim or justified by a delta to this table.

---

## The one explicit silent-failure warning the handbook gives

From the source (line 87):

> Always verify chat template by decoding a packed batch — template mismatch is the #1 silent bug.

This one sentence is the reason ch-08 §Deliverables mandates `chat_template_check.py`. The handbook authors shipped at least one public model (per the Zephyr postmortem thread they reference) whose chat quality regressed silently because the base model's tokenizer had been updated and the handbook-shipped template no longer matched. Loss curves were healthy. Eval was degraded. The cause was three rendered special tokens.

Ch-08's Acceptance gate 1 ("chat_template_check.py output is hand-verified — every special token renders correctly, no stray `<s>` or doubled BOS") exists entirely because of this warning.

---

## What the handbook does NOT say — the silent-failure gaps ch-08 fills

The handbook is recipe-first, not diagnostics-first. It does not cover:

- **Masking unit test.** The handbook assumes `train_on_response_only=True` works. It does not ship a test that asserts the prompt-token gradient is zero. Ch-08 §Deliverables adds this (lifted from [[loss-masking-prompt]]).
- **Packing unit test.** The handbook ships `packing=True` and moves on. It does not ship a test asserting that `cu_seqlens` is threaded through attention. Ch-08 adds the cross-contamination delta test (lifted from [[sequence-packing]]).
- **Clipping-order audit.** The handbook inherits ordering from HF `Trainer` and does not surface it. Ch-08 §6 forces the learner to name the three-line ordering explicitly (`backward → clip → step → scheduler`).

The handbook's assumption is "trust TRL." The lab's assumption is "verify TRL." Both are defensible; the lab is the training wheel.

---

## The 13B / ZeRO-3 vs FSDP note — what changes if you scale past the lab

From the source (line 89):

> Use ZeRO-3 for ≥ 13B; FSDP for ≤ 13B (in 2024 the throughput crossover shifted; re-benchmark).

For ch-08 this means: *if you are on 7B the lab config is correct as written*. If the learner chooses to scale to 13B+ as a stretch goal, the FSDP vs ZeRO-3 decision becomes an explicit benchmark, not a default. [[allenai-tulu-sft-recipe]] shows the 70B answer: `HYBRID_SHARD` (intra-node FULL, inter-node REPLICATE), which is neither of the two choices the handbook's note enumerates. Point being: the SFT-scale default in the lab is only stable up to ~13B; past that, there is no universal recipe and benchmarking is mandatory.

---

## The DPO knob that justifies keeping the lab trainer alive past SFT

From the source (lines 51-66, DPO recipe), the handbook extends the *same* trainer object (the Accelerate-wrapped Trainer parent) to run DPO via `DPOTrainer`. For ch-08 this is a forward-looking note: the SFT trainer the lab produces is *one method swap* away from the DPO trainer in ch-13/14 of the RL track. The masking contract, the packing contract, the clip-and-step ordering — all unchanged. Only the loss function differs.

This matters for the lab memo: `failure-mode-checklist.md`'s §4 ("Reproduction recipe") doubles as a contract the learner can reuse unchanged when Track 4 asks for a DPO run.

---

## Connections

- [[excerpts/sequence-packing]] — the packing algorithm `ConstantLengthDataset` / `DataCollatorWithPacking` implements.
- [[excerpts/loss-masking-prompt]] — the masking contract `train_on_response_only=True` implements.
- [[excerpts/fsdp-sft]] — the `fsdp="full_shard auto_wrap"` knob's memory arithmetic.
- [[excerpts/gradient-clipping]] — `max_grad_norm=1.0` under FSDP dispatches to `FSDP.clip_grad_norm_`.
- [[excerpts/karpathy-training-neural-net-recipe]] — the "decode one batch before training" instinct predates this handbook by four years.
- [[ch-04]] — packing + masking mechanics; ch-08 is the lab that unit-tests ch-04's claims.
- [[ch-06]] — checkpointing; the handbook recipe does not cover DCP, ch-06 does.
