<!-- chapter: ch-08
     track: foundations
     kind: lab
     title: Lab — Systems Tour and Minimal Trainer
     deps: [ch-07]
     sources: [[hf-alignment-handbook]], [[allenai-tulu-sft-recipe]], [[sequence-packing]], [[loss-masking-prompt]], [[fsdp-sft]], [[mixed-precision]], [[gradient-clipping]], [[lr-schedules]], [[olmo-2]], [[karpathy-training-neural-net-recipe]]
     figures: figures/trainer-map.html
-->

# Chapter 8 — Lab: Systems Tour and Minimal Trainer

> **Core insight.** You do not understand a trainer until you can point at the line where masking is applied, the line where the grad-norm is computed, and the line where the optimizer actually steps. Every production post-training failure I have seen reduces to *someone assumed the trainer did the right thing at one of those three lines and did not check*. This chapter closes the assumption.
>
> **Guideline.** Pick TRL's `SFTTrainer` as your reference, read it end-to-end, map each Track-1 concept (ch-01..ch-07) to a specific function, produce a reproducible run — single-GPU 125M at the minimum, 8×H100 on a 7B if you have it — and ship a one-page failure-mode memo enumerating the three silent failure points you would instrument next.

---

## Goal

Finish this chapter with three artifacts that a peer can reproduce:

1. **A map.** Every Track-1 concept in chapters 1–7 is pinned to a concrete call in TRL's `SFTTrainer`. No hand-waving "the trainer handles it."
2. **A run.** A working SFT job — one optimizer step minimum, 100 steps preferred — that uses FSDP FULL_SHARD (or DDP on one node), bf16, packed sequences, response-only masking, and `max_grad_norm = 1.0`.
3. **A memo.** A one-page `failure-mode-checklist.md` listing the three places most likely to fail silently, what you instrumented, and what you would instrument next.

The lab discipline is Karpathy's ([[karpathy-training-neural-net-recipe]]): *"if you can't overfit a single batch, you can't overfit the training set."* Overfit one batch to near-zero loss **before** you touch multi-GPU, dataset mixing, or evaluation. If one-batch overfit does not converge, stop and debug — one of your three silent-failure lines is wrong.

---

## Pick a trainer

The outline lists four options. Pick TRL's `SFTTrainer`. Reasons:

- **Smallest end-to-end reference.** One class (`trl.SFTTrainer`) extends HF `Trainer` and wires packing, response-only masking, Accelerate/FSDP, and the training loop in ~1k lines. `torchtune` is larger and distributed-first; `nanotron` is pretraining-oriented; HF `examples/` scripts are distributed across files with no masking story.
- **Cited upstream.** [[hf-alignment-handbook]] uses it to produce Zephyr-7B; [[allenai-tulu-sft-recipe]] uses a fork (`open-instruct`) with the same loss surface. You are reading the same code whose numerical behavior produced two open-source SOTA chat models.
- **Extends HF `Trainer`.** Optimizer construction, LR scheduler, checkpointing (ch-06) live in the parent class. You get one object where both stacks are visible.
- **All three silent-failure lines are in one repo.** Masking (`DataCollatorWithPacking` / `DataCollatorForCompletionOnlyLM` in `trl/trainer/`), clipping (inherited from `Trainer._maybe_log_save_evaluate` and `accelerator.clip_grad_norm_`), optimizer step (`training_step` in `transformers/trainer.py`). You can read all three in one afternoon.

Other choices are legitimate — if you want RL next chapter to feel continuous, `torchtune` is arguably a better long-term bet. For this lab, stay with TRL.

---

## Full-budget path

Target: 8 × H100 (80 GB), 7B base model, ~1 GPU-hour SFT on a 50K-prompt subset.

**Model.** `meta-llama/Llama-3.1-8B` or `Qwen2.5-7B` base. The choice does not matter for the lab; the trainer is identical.

**Data.** Take 50K prompts from a single source — UltraChat-200K-filtered or Tülu-3-SFT-mix ([[allenai-tulu-sft-recipe]]). Do **not** mix sources for the lab; mixing adds one more debug surface. Apply the model's chat template via `tokenizer.apply_chat_template`. Pack to 4096 tokens with `packing=True`. Mask prompts — `train_on_response_only=True`.

**Config.** Quoting the attested Zephyr-7B recipe ([[hf-alignment-handbook]]):

```python
# attested from hf-alignment-handbook.md lines 70-78
from trl import SFTTrainer, SFTConfig
config = SFTConfig(
    packing=True,
    max_seq_length=2048,             # raise to 4096 for Llama-3.1
    train_on_response_only=True,     # masks user tokens automatically
    dataset_kwargs={"add_special_tokens": False},
    # hparams from the handbook's Zephyr table
    learning_rate=2e-5,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    bf16=True,
    max_grad_norm=1.0,               # ch-01 clipping default
    gradient_checkpointing=True,
    # FSDP wiring (attested: FULL_SHARD per [[fsdp-sft]] §"Typical SFT recipe")
    fsdp="full_shard auto_wrap",
    fsdp_config={"transformer_layer_cls_to_wrap": ["LlamaDecoderLayer"]},
)
```

Run with `accelerate launch --config_file fsdp.yaml` on 8 GPUs. A clean 100-step run on 7B + packed 4096 takes ~15 minutes on 8×H100 at bf16.

**Acceptance for the full-budget path.** One optimizer step completes without OOM; loss on step 1 is ~`ln(V)` ≈ `ln(128k)` = 11.76 for a properly initialized base model; `pre_clip_grad_norm` is O(1) — if it is O(1e2) your LR is too high, your warmup is too short, or masking is off.

---

## Resource-constrained path

Target: 1 × GPU (any ≥ 16 GB), or CPU with patience.

**Model.** `HuggingFaceTB/SmolLM-135M` or `Qwen2.5-0.5B`. 135M fits bf16 fine-tune on 8 GB; 500M fits on 16 GB with `gradient_checkpointing=True`.

**Data.** 2K prompts from Alpaca-cleaned or a 2K slice of UltraChat. Enough to overfit. The goal is **not** a useful model — it is a legible trainer.

**Config changes vs full-budget.** Keep `packing=True`, `train_on_response_only=True`, `bf16=True`, `max_grad_norm=1.0`, `lr_scheduler_type="cosine"`, `warmup_ratio=0.1`. Drop FSDP to DDP (`--num_processes=1`) or FSDP-1-node if you want to exercise the sharding path. Skip pipeline/tensor parallel entirely. Set `max_seq_length=1024`, `gradient_accumulation_steps=4`.

**Karpathy's one-batch check.** Before the 2K-prompt run, take *one* batch, set `num_train_epochs=200`, and verify loss drops from ~11 to < 0.1 within 200 steps. If it does not, one of masking / tokenization / packing / chat-template is wrong — fix it before the real run.

**The memo requirement is unchanged.** A 135M-on-CPU run exercises the same three silent-failure lines as the 7B-on-8xH100 run. You are buying the memo, not the model.

---

## Mapping code to concepts (§1..§7)

This is the load-bearing section. For every Track-1 concept, pin the concrete call in TRL. The companion HTML ([figures/trainer-map.html](figures/trainer-map.html)) renders the same mapping as a clickable call-graph — open it alongside this list.

### §1 — Tokenization + chat template (ch-04)

- `tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)` — consumed by `SFTTrainer._prepare_dataset`.
- **Silent failure:** wrong template. [[hf-alignment-handbook]] is explicit: *"verify chat template by decoding a packed batch — template mismatch is the #1 silent bug."* You **must** decode `tokenizer.decode(batch["input_ids"][0])` in a one-shot script before training; eyeball that the special tokens (`<|im_start|>`, `<|eot_id|>`, etc.) are present and unambiguous.

### §2 — Packing (ch-04 / [[sequence-packing]])

- `DataCollatorWithPacking` / `ConstantLengthDataset` in `trl/trainer/sft_trainer.py` — activated by `packing=True`.
- Produces `input_ids` with `cu_seqlens` (cumulative sequence-start offsets) and `position_ids` reset per sub-sequence ([[sequence-packing]] §Mechanics).
- Attention uses `flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen)` — block-diagonal causal attention with O(Σ L_i) memory, not O(L_max²).
- **Silent failure:** `cu_seqlens` not threaded through to attention, so tokens attend across sub-sequences. The unit test is in §Deliverables.

### §3 — Loss masking (ch-04 / [[loss-masking-prompt]])

- With `train_on_response_only=True`, `DataCollatorForCompletionOnlyLM` walks each sub-sequence, finds the instruction-template boundary, sets `labels[:response_start] = -100`.
- The canonical masking snippet ([[loss-masking-prompt]] §Implementation):

```python
# attested from loss-masking-prompt.md lines 46-52
labels = input_ids.clone()
labels[:prompt_len] = -100            # mask prompt
loss = F.cross_entropy(logits[..., :-1, :].reshape(-1, V),
                       labels[..., 1:].reshape(-1),
                       ignore_index=-100)
```

- **Silent failure:** multi-turn mismatch. Prior assistant turns must also be masked ([[loss-masking-prompt]] §Multi-turn). If only the very first user turn is masked, the model gradient-matches past assistant outputs — turning SFT into a "copy yourself" task.

### §4 — Mixed precision (ch-02 / [[mixed-precision]])

- `bf16=True` in `SFTConfig` → HF `Trainer` wraps the forward in `torch.autocast("cuda", dtype=torch.bfloat16)`.
- Optimizer state stays fp32 (AdamW keeps fp32 `m`, `v`, and a master copy — [[fsdp-sft]] confirms this is the FSDP `MixedPrecision` default).
- **Silent failure:** logging loss in bf16. `loss.item()` will upcast, but a custom callback that does `loss.to(torch.bfloat16).cpu().numpy()` produces a quantized-jaggy curve. Log in fp32 always.

### §5 — Gradient clipping (ch-01 / [[gradient-clipping]])

- `max_grad_norm=1.0` → HF `Trainer.training_step` calls `self.accelerator.clip_grad_norm_(model.parameters(), max_grad_norm)`. Under FSDP, Accelerate dispatches to `FSDP.clip_grad_norm_` — the sharded-aware version that computes the global norm before rescaling ([[fsdp-sft]] §"Distributed-training pitfall").
- **Silent failure:** custom code paths that loop `clip_grad_norm_(p, c)` per parameter bias the optimizer toward small tensors ([[gradient-clipping]] §"Per-tensor norm clip"). If you monkey-patch clipping, this is the bug.

### §6 — Optimizer step, scheduler, clipping ordering (ch-01/ch-02/ch-03)

The ordering that must hold, quoting [[mixed-precision]] and [[gradient-clipping]]:

```
loss.backward()           # produces grads (scaled by S if fp16)
[unscale_ if fp16]
clip_grad_norm_(params, max_norm)           # direction-preserving, global
optimizer.step()
scheduler.step()                            # AFTER optimizer.step()
optimizer.zero_grad(set_to_none=True)
```

- HF `Trainer.training_step` implements this. Under bf16 there is no `unscale_`. Under FSDP the clip is sharded-aware.
- **Silent failure:** `scheduler.step()` called *before* `optimizer.step()` — first step runs at step-(k+1)'s LR, cosine phase permanently off by one (see [[ch-06]] §5.3).

### §7 — LR schedule (ch-03 / [[lr-schedules]])

- `lr_scheduler_type="cosine"` + `warmup_ratio=0.1` → `transformers.optimization.get_cosine_schedule_with_warmup`. Formula ([[lr-schedules]] §Technical Details): `lr(t) = min_lr + 0.5*(peak_lr - min_lr)*(1 + cos(pi*(t - warmup)/(T - warmup)))`.
- **Silent failure:** total `T` computed from `num_train_epochs * steps_per_epoch`, but packing changes `steps_per_epoch` by the pack ratio (~2.5× per [[allenai-tulu-sft-recipe]]). If the scheduler's `T` is set before packing is applied, cosine finishes at step `T/2.5` and the last 60% of training runs at `min_lr` — survivable, not fatal, but measurably worse.

---

## Deliverables checklist

Everything below lives in your lab output directory — **not** in this wiki. Ship them as a gist, a run directory, or an attached PR.

- [ ] `run.sh` — `accelerate launch` invocation, with `fsdp.yaml` (or single-GPU) next to it.
- [ ] `sft_config.py` — the `SFTConfig` above, verbatim, with hyperparameter deltas annotated.
- [ ] `chat_template_check.py` — decodes a packed batch and prints the first 200 tokens; run once, commit the output.
- [ ] `masking_unit_test.py` — see below; must pass.
- [ ] `packing_unit_test.py` — see below; must pass.
- [ ] `overfit_one_batch.py` — 200 steps on one batch, plots loss; must reach loss < 0.1.
- [ ] `failure-mode-checklist.md` — the memo, 1 page, structure below.

**Masking unit test (from [[loss-masking-prompt]]).** Construct one batch, run `loss.backward()`, check that `embed_tokens.weight.grad` has zero mass on the prompt-token rows. In one line:

```python
prompt_ids = batch["input_ids"][labels == -100]
assert model.get_input_embeddings().weight.grad[prompt_ids].abs().sum() == 0.0
```

If this fails, masking is wrong. Do **not** proceed to the full run.

**Packing unit test (from [[sequence-packing]]).** Run the same batch twice: once packed with `cu_seqlens` threaded to attention, once packed with `cu_seqlens` replaced by `[0, L_total]` (i.e. no block-diagonal mask — the cross-contamination case). Losses must differ by > 1e-3 (if they do not, either your packing is not threading `cu_seqlens` or your batch has only one sub-sequence — rebuild the batch).

**Failure-mode memo (structure).** One page, four sections:

1. *Picks.* The three silent failures most likely in this trainer. Mine: (a) chat-template mismatch against the base model's tokenizer; (b) `cu_seqlens` not wired to attention, causing cross-contamination; (c) `scheduler.step()` order in a custom callback. Yours may differ; justify.
2. *What you instrumented.* List of metrics logged per step: `loss`, `pre_clip_grad_norm`, `lr`, `tokens/sec`. Output path for the packed-batch decode; output of the masking unit test.
3. *What you would instrument next.* Per-shard loss breakdown ([[ch-06]] §4); embedding-norm per checkpoint ([[olmo-2]]); loss on a canary set that is byte-identical to a known-good batch (detects data-loader drift across resumes).
4. *Reproduction recipe.* `git rev-parse HEAD`, `pip freeze`, the exact `accelerate launch` line, total wall-clock, final loss.

---

## Acceptance criteria

Hard gates, in order. Do not skip.

1. `chat_template_check.py` output is hand-verified — every special token renders correctly, no stray `<s>` or doubled BOS.
2. `masking_unit_test.py` passes — prompt-token embedding gradient is exactly zero.
3. `packing_unit_test.py` passes — cross-contamination loss delta > 1e-3.
4. `overfit_one_batch.py` reaches loss < 0.1 within 200 steps on a batch of 1.
5. The 100-step (or ≥ 1 optimizer-step) real run completes without OOM, `pre_clip_grad_norm` stays < 10, step-1 loss is within 20% of `ln(V)`.
6. `failure-mode-checklist.md` exists, is one page, and lists three distinct silent failures with a specific metric next to each.

If any gate fails, the lab is incomplete; regress to the gate before it and debug. This is Karpathy's ([[karpathy-training-neural-net-recipe]]) "never skip a step because this time is different."

---

## Connections

- **ch-04 (SFT mechanics)** — packing + masking + chat-template surface. The lab is the unit-test for that chapter's claims.
- **ch-05 (FSDP)** — `fsdp=full_shard` + `LlamaDecoderLayer` wrap policy. Same code path, different knobs.
- **ch-06 (checkpointing)** — the 100-step run must produce a DCP checkpoint; resume-bit-exact is a stretch gate for the full-budget path.
- **ch-07 (silent failure catalog)** — this lab's memo is the chapter-specific instantiation of ch-07's broader catalog.
- **ch-09 (first real end-to-end SFT)** — leverages this trainer as the default. Everything below assumes the three unit tests pass.
- **Track 2 (synthetic data) / Track 3 (SFT-at-scale) / Track 4 (RL)** — all three tracks inherit this trainer. RL in Track 4 swaps the loss function but keeps masking, packing, clipping, optimizer-step ordering identical.

## Further reading

- [[hf-alignment-handbook]] — the reference `SFTConfig` and FSDP wiring; authoritative for lab-scale defaults.
- [[allenai-tulu-sft-recipe]] — what changes at 939K-prompt / 8B–70B scale (LR down, epochs up, HYBRID_SHARD at 70B); memo extensions.
- [[sequence-packing]] — Krell 2021; `cu_seqlens` contract, cross-contamination unit-test idea.
- [[loss-masking-prompt]] — Shi 2024 + Alpaca/InstructGPT canon; response-only loss, multi-turn masking.
- [[fsdp-sft]] — Zhao 2023; `FULL_SHARD` + `MixedPrecision` + `clip_grad_norm_` sharded contract.
- [[mixed-precision]] — Micikevicius 2017; bf16 default, fp32 accumulation rules.
- [[gradient-clipping]] — Pascanu 2013; global-norm clip 1.0.
- [[lr-schedules]] — cosine + warmup 10%; the lab default.
- [[olmo-2]] — reference 2025 post-training numbers; use as sanity-check target if you scale up.
- [[karpathy-training-neural-net-recipe]] — the lab-memo tradition; "overfit a single batch" is gate 4.

## Companion visualization

**[figures/trainer-map.html](figures/trainer-map.html)** — interactive call-graph of TRL `SFTTrainer`. Hover any box (`apply_chat_template`, `DataCollatorWithPacking`, `DataCollatorForCompletionOnlyLM`, `FSDP wrap`, `autocast bf16`, `clip_grad_norm_`, `optimizer.step`, `scheduler.step`) to see the concept-chapter it implements (ch-01..ch-07) and the one-line silent-failure mode. Click a box to lock the detail panel. The colour band on each node indicates which of the three most-likely-to-fail-silently lines it sits on; the legend doubles as the memo's §1 short-list template.
