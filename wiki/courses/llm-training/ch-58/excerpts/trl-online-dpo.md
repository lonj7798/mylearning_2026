---
chapter: ch-58
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/frameworks/trl-online-dpo.md
source_url: https://github.com/huggingface/trl
created_at: "2026-04-23"
---

# Excerpt: TRL OnlineDPOTrainer — the one feature that makes §5 Q2 unconditional

**Source library:** `wiki/raw-data/llm-training/frameworks/trl-online-dpo.md`
**Artifact:** `trl/experimental/online_dpo/online_dpo_trainer.py` L1089–1275 (`training_step`), L585–893 (rollout engines).

---

## Why this source defines ch-58 matrix row 7 + §5 Q2 "mandatory TRL"

Ch-58 §5 has exactly one unconditional branch: if the algo is online DPO / SPIN / Nash-MD / XPO, route to TRL regardless of scale. That branch exists because this source is the only attested implementation in the raw-data library.

## Row 7 — "Online DPO / judge-driven: trl/experimental/online_dpo/ + nash_md + xpo + self_distillation"

Source:

> Online DPO (Guo 2024 / Dong 2024) closes the gap between DPO and PPO by making the preference distribution *on-policy*; this trainer is the reference implementation used by many self-play / Nash-LM experiments.

And the "what to notice" box:

> `loss_type` switch exposes both `sigmoid` (canonical DPO) and `ipo` (Azar 2023 L2-on-margin); extensions for self-rewarding and Nash-MD live in sibling files (`nash_md`, `xpo`, `self_distillation`).

The sibling-file list is attested. Ch-58 §5 Q2 cites "no mainline equivalent in OpenRLHF or verl" — the source backs this directly:

> No equivalent online trainer in OpenRLHF mainline — typically users run OpenRLHF PPO or switch to TRL for online DPO.

## The "two samples per prompt" architectural signature

Source code:

> ```python
> rewards = self._calculate_rewards_from_functions(
>     prompts=2 * prompts, completions=completions, ...
> )
> first_half, second_half = rewards.split(batch_size)
> mask = first_half >= second_half
> chosen_indices   = batch_range + (~mask * batch_size)
> rejected_indices = batch_range + ( mask * batch_size)
> ```

This is why online DPO is structurally different from offline DPO: the pair is produced and consumed in the same step. Ch-58 §5 does not elaborate — the §5 tree just routes and moves on — but the "two samples per prompt" is the architectural fact a learner should be able to recite when asked why this chapter exists as a TRL branch.

## Judge flexibility — the "pluggable reward" fact

Source:

> Judge is pluggable: `_calculate_rewards_from_functions` accepts reward models, verifiable reward functions, or a `Judge` interface that calls a frozen LLM to pick the preferred completion.

Ch-58 §5 Q2 implicitly covers SPIN / self-rewarding / RLAIF — all three are "swap the judge" variants of this same trainer. The Judge interface attestation is why ch-58 routes all four (DPO-online, SPIN, Nash-MD, XPO) to the same leaf.

## Comparison to offline DPO — why graduate paths can't avoid TRL for this

Source:

> vs offline DPO: the only difference is where the pair comes from — offline from a dataset, online from `.generate()`. The DPO loss algebra is identical.

This is the "algebra is the same" thesis applied within TRL itself. Ch-58 §1 invokes this for cross-framework; here it's intra-framework: offline DPO (`DPOTrainer`) and online DPO (`OnlineDPOTrainer`) share algebra but differ on data source. A learner who thinks they can "just add a generate() call" to OpenRLHF's `DPOTrainer` has to re-build the rollout harness, judge wiring, EOS-penalty logic, and KL-canary instrumentation — OpenRLHF has none of this.

## vLLM rollout integration (matrix row 1 evidence for TRL)

Source file reference:

> `_generate_vllm` / `_generate_vllm_colocate` ≈ lines 585–893 (rollout engines)

These are the two TRL vLLM paths ch-58 matrix row 1 cites: `_generate_vllm_server` (RPC-style external server) and `_generate_vllm_colocate` (in-process). Neither is async in the verl/OpenRLHF sense.

## What ch-58 inherits verbatim

- `trl/experimental/online_dpo/` + `nash_md` + `xpo` + `self_distillation` sibling-file list (matrix row 7).
- `_calculate_rewards_from_functions` as the judge-pluggable surface.
- The "no mainline equivalent in OpenRLHF" attestation that makes §5 Q2 unconditional.
- `_generate_vllm` and `_generate_vllm_colocate` as the TRL matrix-row-1 subtype.

## Connections

- **[[trl-grpo]]** — the TRL-family companion; ch-58 §5 Q5 cites both for "zoo breadth".
- **[[openrlhf-dpo]]** — the offline-only counterpart; matrix row 6.
- **[[trl-ppo]]** — the actor-critic sibling; ch-58 §1 "algebra is the same" uses all three TRL trainers as evidence.
- **[[self-rewarding-lm]]** / **[[self-play-preference]]** — the algorithms this trainer is the reference implementation for.
