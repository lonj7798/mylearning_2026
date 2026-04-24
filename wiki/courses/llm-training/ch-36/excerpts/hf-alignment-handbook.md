---
chapter: ch-36
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/blogs/hf-alignment-handbook.md
source_url: https://github.com/huggingface/alignment-handbook
created_at: "2026-04-23"
---

# Excerpt: HF Alignment Handbook — the reference recipe ch-36 forks

**Source library:** `wiki/raw-data/llm-training/blogs/hf-alignment-handbook.md`
**Authors:** Lewis Tunstall, Edward Beeching, Nathan Lambert, Nazneen Rajani, Kashif Rasul, Younes Belkada, Shengyi Huang, Leandro von Werra, Clémentine Fourrier, Nathan Habib, Nathan Sarrazin, Omar Sanseviero, Alexander M. Rush, Thomas Wolf
**Venue:** HuggingFace repository / Zephyr / StarChat2 / Llama-3.1-Tulu lineage
**Year:** 2023–present

---

## Why this source anchors ch-36

Ch-36 is a lab, not a research artifact. Labs need a *reference* — a known-good recipe whose hyperparameters, framework choices, and defaults are the null-hypothesis baseline. The Alignment Handbook is that reference for 2024-era SFT: Zephyr-7B-β, StarChat2, and Llama-3.1-Tulu were all trained from its YAML configs. When ch-36 says "use packing true and response-only true unless you have a reason not to," the reason that's good is because *the Handbook's ablations already closed those axes for all non-exotic use cases*, freeing ch-36 to spend its 8-run budget on axes where the answer is still open (NEFTune) rather than axes where it's settled (packing, masking).

The Handbook's second contribution to ch-36 is *chat-template discipline*. It names "template mismatch" as the #1 silent bug and bakes `apply_chat_template` into the trainer. Ch-36's `§2 Chat-template unit tests` exists because the Handbook said this bug is the #1 silent one, and ch-36 believed it.

---

## The Zephyr-7B-β SFT recipe verbatim

Source lines 36–49 list the hyperparameters that ch-36 adopts near-wholesale:

| Knob | Handbook (Zephyr-7B-β) | Ch-36 full path | Notes |
|------|------------------------|-----------------|-------|
| Model | Mistral-7B-v0.1 | Llama-3.2-3B | Same BF16 base-model pattern; ch-36 smaller for budget |
| Max seq length | 2048 | 2048 | Identical |
| Packing | true | true | Identical |
| Train on response only | true | true | Identical |
| Optimizer | AdamW (β=0.9, 0.95) | AdamW (β=0.9, 0.95) | Identical |
| Learning rate | 2e-5 | 2e-5 | Identical |
| LR schedule | cosine, 10% warmup | cosine, 10% warmup | Identical |
| Epochs | 1 | 1 | Identical |
| Global batch size | 128 | 128 | Identical |
| Precision | BF16 | BF16 | Identical |
| FSDP / ZeRO | FSDP FULL_SHARD | FSDP FULL_SHARD | Identical |
| Gradient checkpointing | true | true | Identical |

**Notice:** every row is identical except the base model. This is intentional. Ch-36's experimental question is *not* "are these hyperparameters good" — the Handbook answered that across several 7B-scale runs. Ch-36's question is "do our unit tests catch the regressions this recipe would silently tolerate." Keeping the rest of the recipe identical eliminates confounds.

---

## The SFTTrainer config — code ch-36 lifts directly

Source lines 70–78:

```python
from trl import SFTTrainer, SFTConfig
config = SFTConfig(
    packing=True,
    max_seq_length=2048,
    train_on_response_only=True,  # masks user tokens automatically
    dataset_kwargs={"add_special_tokens": False},
)
```

Three details matter:

1. **`packing=True`** — routes data through TRL's `DataCollatorWithPacking`, which wraps FlashAttention-2's `flash_attn_varlen_func`. This is the API ch-36's `test_varlen_attention_zeros_cross_sample_leakage` validates. If TRL's version drifts (e.g., switches from SPFHP to FFD packing), the test still passes as long as the `cu_seqlens` contract holds — which is the whole point of testing the contract, not the implementation.

2. **`train_on_response_only=True`** — tells TRL to apply `apply_chat_template(..., return_assistant_tokens_mask=True)` and set labels to `-100` everywhere the assistant-mask is 0. This is the moment ch-36's `§3 Loss-mask unit tests` is validating. If TRL's chat-template logic has a bug for a specific tokenizer (Llama-3.2 uses a different template than Llama-3.1), TRL silently emits a mask that covers the wrong tokens.

3. **`dataset_kwargs={"add_special_tokens": False}`** — prevents double-BOS. The chat template already inserts BOS; if TRL tokenizer adds another, you get two BOS tokens, and the model's next-token prediction for position 1 is trained to predict BOS-after-BOS, which breaks inference. Ch-36's `test_empty_system_prompt_does_not_duplicate_bos` is the exact test for this regression.

**Notice:** the snippet is four lines. The coverage surface is massive. Each of the three keys above is a silent failure waiting for a tokenizer upgrade to reactivate.

---

## The Handbook's "lessons captured" — ch-36's prior

Source lines 87–90:

> - Always verify chat template by decoding a packed batch — template mismatch is the #1 silent bug.
> - DPO runs for < 1 epoch; longer causes chosen-side collapse (motivating [[ipo]], [[simpo]]).
> - Use ZeRO-3 for ≥ 13B; FSDP for ≤ 13B (in 2024 the throughput crossover shifted; re-benchmark).
> - NEFTune (α=5) stacks cleanly; toggled via SFT config.

Ch-36 takes three of these as priors:

- **"Verify chat template by decoding a packed batch"** → `§2` unit tests + a mandatory decode-and-eyeball step in `§8 Failure diagnosis`.
- **"FSDP for ≤ 13B"** → 3B model uses FSDP FULL_SHARD (ch-36 full path); explicitly *not* DeepSpeed ZeRO-3. The "re-benchmark" caveat is logged in `sft-run-memo.md` as future work.
- **"NEFTune (α=5) stacks cleanly"** → this is the hypothesis ch-36 tests in the 2×2×2 ablation. If the off-diagonal cell (packed+masked+NEFTune vs packed+masked+no-NEFTune) shows "NEFTune doesn't stack as cleanly as claimed," that's a surprise worth a paragraph in the memo.

The DPO bullet is not used directly (ch-36 is SFT-only, capstone for the SFT track) but is a link-forward to the RL track ch-40+.

---

## Why the Handbook is "one epoch, cosine, 2e-5" — the archaeology

The Handbook doesn't derive these hyperparameters; it inherits them from Zephyr-7B-β, which inherited them from InstructGPT-style SFT runs, which inherited them from Ouyang 2022. Two details are worth knowing because ch-36 inherits them too:

- **1 epoch.** More than 1 epoch on instruction data causes measurable overfitting to response templates. LIMA ([[lima]]) runs 15 epochs — an outlier that works only because the dataset is 1K and LR is decayed aggressively.
- **2e-5 LR with 10% warmup.** This is a compromise. 1e-5 is safer but converges slower than 1-epoch budget allows; 5e-5 is faster but degrades response diversity on small sets. 2e-5 is the equilibrium the community landed on circa 2023 and hasn't moved off.

Ch-36 holds both constant across all ablation cells. If a cell blows up — e.g., `packed=1, masked=0, NEFTune=1` diverges — the hypothesis is *not* "LR was wrong" but "the broken mask interacted with NEFTune noise to destabilize the embedding loss." Holding LR constant is what lets the lab attribute divergence to the mask/noise interaction rather than to a hyperparameter confound.

---

## DPO hyperparameters — forward reference from ch-36

Source lines 57–66 spell out the Handbook's DPO defaults:

| Knob | Handbook DPO | Purpose |
|------|--------------|---------|
| β | 0.1 | Implicit reward temperature |
| LR | 5e-7 | 40× smaller than SFT |
| Epochs | 1 | Same rule: >1 collapses the chosen side |
| Batch (pairs) | 32 | Half of SFT batch; each pair is 2× the forward cost |
| π_ref | SFT checkpoint, frozen | Ch-36's SFT output feeds the next lab |

These matter for ch-36 because they specify what the *next* chapter (RL track) will need from ch-36's output: a *frozen-compatible* SFT checkpoint, not a PEFT LoRA adapter, not a fused model. Ch-36's `checkpoint-final/` must be loadable as `π_ref` in TRL's `DPOTrainer`. The unit tests don't check this directly, but the `§9 Deliverables` section of read.md requires a round-trip load test.

---

## What ch-36 does NOT take from the Handbook

- **Dataset.** Handbook uses UltraChat-200K; ch-36 uses 1K LIMA + 79K ch-29 synthetic + 20K No-Robots/OpenAssistant-2. Different mix → different skill balance → different eval slices. The [[tulu-3-sft-mix]] is a closer fit for ch-36's mix than the Handbook's.
- **Eval.** Handbook uses MT-Bench + AlpacaEval + IFEval. Ch-36 uses MT-Bench only (single-turn, 80 prompts, judge `gpt-4o-mini`), sliced by skill. This is a budget-driven reduction, not a methodology disagreement.
- **Base model.** Mistral-7B vs Llama-3.2-3B. The Handbook's chat template won't apply cleanly; ch-36 uses Llama-3.2's.

---

## Connections

- Loss-side primitive: [[excerpts/loss-masking-prompt]] — `train_on_response_only=True`'s guts.
- Attention-side primitive: [[excerpts/sequence-packing]] — what `packing=True` activates.
- Regularizer axis: [[excerpts/neftune]] — `neftune_noise_alpha=5` in SFTConfig.
- Baseline-thesis source: [[excerpts/lima]] — the 1K regime the Handbook scales up from.
- Skill-balance companion recipe: [[allenai-tulu-sft-recipe]].
- Runtime primitive: [[fsdp-sft]].
- Lab host: [[ch-36]] — this source defines the full-path recipe.
