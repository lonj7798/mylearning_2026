<!-- chapter: ch-08
     track: foundations
     kind: lab
     title: Lab — Minimal Trainer with a Target-versus-General Capability Measurement
     deps: [ch-07]
     sources: [[trl-sft-trainer]], [[hf-trainer-loop]], [[nanochat]], [[loss-masking-prompt]], [[packing-with-flash-attention]], [[prompt-format-sensitivity-formatspread]], [[hf-alignment-handbook]], [[paloma]], [[neftune]], [[signal-and-noise-eval]], [[tulu-3]], [[fsdp-sft]], [[trl-grpo]]
     figures: figures/trainer-map.html, figures/target-vs-general.html
     revised: 2026-09 (generality revision)
-->

# Chapter 8 — Lab: Minimal Trainer with a Target-versus-General Capability Measurement

> **Core insight.** A supervised fine-tuning run changes two quantities at once, and a lab that reports one of
> them cannot tell a good run from a bad one. On LLaMA-2-7B fine-tuned on Alpagasus Alpaca 5k (5,305 examples),
> response-only SFT raised AlpacaEval 1.0 from 0.01 to 16.29 and lowered the mean over 18 NLP tasks from 49.32
> to 45.29, a drop of 4.03 points ([[loss-masking-prompt]], Table 1). The correctness tests that decide whether
> such a number is meaningful must be tests a correct run passes: two of the six gates in the 2026-04 version of
> this lab fail on correct code, because they compare the step-1 loss against `ln(V)` and require prompt-token
> embedding gradients to be exactly zero.
>
> **Guideline.** When a fine-tuning run is evaluated, measure the base checkpoint and the fine-tuned checkpoint
> on the same target task and the same broad suite, because the target gain and the general regression come
> from the same optimisation and neither is interpretable alone. When a proposed acceptance gate can fail on a
> correct run, replace it with a test whose failure has one cause: packed-versus-unpacked equivalence for
> attention masking, decoded inspection of the positions where `labels != -100` for loss masking, and the base
> model's own loss on the first batch for the input pipeline. When a benchmark difference is smaller than the
> spread across the final checkpoints of one run — 1.7 accuracy points on ARC Challenge for 1B models
> ([[signal-and-noise-eval]] §3.1) — report it as unresolved rather than as an improvement.

---

## Why this chapter matters for a general-purpose model

This lab closes the foundations phase (ch-00 to ch-08) and comes before the pretraining phase. It is placed
here because supervised fine-tuning is the cheapest stage that produces a *pair* of checkpoints — a base model
and a trained model that differ by one controlled amount of data — and a pair is what a generality measurement
needs. Every later stage in the pipeline (mid-training, SFT at scale, preference optimisation, RL) reuses the
same three operations: which tokens are attended to, which tokens carry loss, and how the update is formed and
clipped. Errors in those three operations do not raise exceptions. They change the distribution the model is
trained on, and they show up only as a number that is worse than it should be.

The generality question for this stage is narrow and answerable: does a fine-tuning run that improves its
target task keep the ability the base model already had? [[loss-masking-prompt]] measures a case where it does
not, and names the drop an instruction tuning tax (§4.3 "#3"). Its Fig. 4 plots the 18-task mean against
epochs from 2 to 10 on five datasets for LLaMA-2-7B-BASE, and the paper's stated reading of that figure is that
instruction modelling has a lower tax than response-only loss. The deliverable of this lab is that measurement
for a run the reader controls, together with the evidence that the run was correct.

---

## §1 What the lab produces

Four artifacts, in the reader's own run directory rather than in this wiki.

1. **`symbol-map.md`** — every operation in §2 mapped to a file, a symbol, and a line range at one pinned
   commit of one trainer, with the commit hash recorded.
2. **Three test scripts** — `packing_equivalence.py`, `decode_labels.py`, `step1_loss.py` (§3). Each writes its
   output to a file that is kept with the run.
3. **A run and a small ablation grid** — one training run at the reader's compute scale, plus the levers in §5
   at whatever subset the budget allows, each with at least two seeds or an explicit statement that seeds were
   not run.
4. **`target-vs-general.md`** — one table with one row per checkpoint and these columns: target-task metric,
   broad-suite scores (MMLU, GSM8K, HumanEval, IFEval), held-out bits-per-byte on at least two domains, and the
   same target metric under a second prompt format. The base checkpoint is the first row.

**Acceptance gates.** (1) `symbol-map.md` cites a commit hash and line numbers that exist in that commit.
(2) The packed and unpacked forward passes agree on the same tokens to the tolerance recorded in §3.1.
(3) The decoded text of the positions with `labels != -100` is exactly the assistant turns, including the
end-of-turn token. (4) The step-1 training loss equals the base model's loss on the same batch to within
logging precision. (5) `target-vs-general.md` contains a base-model row and a second-format column. A run that
fails gate 2, 3 or 4 is not evaluated until the cause is found; the evaluation would measure the bug.

---

## §2 Reading one trainer end to end at a pinned commit

**Definition.** Pinned-commit reading means every claim about a framework is checked against a named commit of
the source, and the commit hash is written down next to the claim.

**The problem it addresses, measurably.** Framework symbols move and disappear between releases. In the same
repository as the trainer read below, `GRPOTrainer._compute_loss` sits at L2418 in commit `a08e713`
(2026-04-21) and at L3113 in commit `a04ffd3` (2026-09-14), and the default `max_completion_length` changed to
512 between those commits ([[trl-grpo]], Verification). A chapter or a configuration copied from a blog post
therefore decays silently: an argument that no longer exists is either rejected at construction or, when the
class accepts arbitrary keyword arguments, ignored.

**Mechanism — the data path of TRL's `SFTTrainer` at commit `aa89588`** ([[trl-sft-trainer]]):

1. `_prepare_dataset` (L1438) converts the dataset to ChatML if needed, appends the EOS token for
   non-conversational examples, and tokenizes through the chat template. For prompt-completion data it applies
   the template twice — prompt with `add_generation_prompt=True`, then prompt + completion — and warns when the
   second tokenization does not start with the first (L1536-1556).
2. With `assistant_only_loss=True` the tokenizer is called with `return_assistant_tokens_mask=True` (L1543);
   a template without `{% generation %}` markers is replaced by `get_training_chat_template` (L1251-1258), and
   an example that yields no assistant token raises (L1585-1591).
3. `build_labels` (L1606-1632) writes `labels = input_ids` with `-100` wherever an applicable mask bit is 0.
4. Truncation runs only when packing is off; examples left fully masked are dropped (L1634-1660).
5. `pack_dataset` (L1678) packs rows to `max_length` and adds the column `seq_lengths`.
6. `DataCollatorForLanguageModeling` (L403) pads labels with `-100`, derives `position_ids` from `seq_lengths`
   (L528-552), flattens the batch into one row when padding-free, and sets
   `labels[position_ids == 0] = -100` (L516), so the first token of each packed document carries no loss.

The update itself is inherited from `transformers.Trainer` ([[hf-trainer-loop]]): clipping at L2697-2718,
`optimizer.step()` at L2740, `lr_scheduler.step()` at L2750, all inside `_inner_training_loop`, in that order.
`training_step` (L3981) performs only the forward pass, the gradient-accumulation division, and
`accelerator.backward` (L4059-4070).

**Concept-to-symbol map** (verified at the loci above; see
[figures/trainer-map.html](figures/trainer-map.html) for the same map as a clickable path, with the test that
covers each stage):

| Concept (chapter) | Symbol at `trl@aa89588` / `transformers@v4.57.1` | Locus |
|---|---|---|
| Chat template (ch-04) | `_tokenize(..., chat_template=self.chat_template)` in `tokenize_fn` | `sft_trainer.py` L1519-1603 |
| Loss masking (ch-04) | `assistant_only_loss` / `completion_only_loss` → `build_labels` | `sft_config.py` L259-280; `sft_trainer.py` L1606-1632 |
| Packing (ch-04) | `packing`, `packing_strategy` → `pack_dataset`, column `seq_lengths` | `sft_config.py` L223-238; `sft_trainer.py` L1662-1678 |
| Block-diagonal attention (ch-04) | `padding_free` → `position_ids` from `seq_lengths`, FlashAttention variant required | `sft_trainer.py` L1169, L394-398, L528-552 |
| Mixed precision (ch-02) | `bf16` in `TrainingArguments`; FSDP keeps the optimizer step in full precision | [[fsdp-sft]] §4.4 |
| Gradient clipping (ch-01) | `args.max_grad_norm` → `accelerator.clip_grad_norm_` | `trainer.py` L2697-2718 |
| Optimizer and scheduler order (ch-03) | `optimizer.step()` then `lr_scheduler.step()` | `trainer.py` L2740, L2750 |
| Schedule horizon (ch-03) | `max_steps = ceil(num_train_epochs × num_update_steps_per_epoch)` from the packed dataloader | `trainer.py` L5675-5689 |
| In-loop metrics (ch-06) | `entropy`, `mean_token_accuracy`, `num_tokens` | `sft_trainer.py` L1815-1876 |

**Names that do not exist at this commit**, and are therefore signs that a configuration was copied rather
than read: `train_on_response_only`, `max_seq_length` (now `max_length`), `DataCollatorWithPacking`,
`DataCollatorForCompletionOnlyLM`, `ConstantLengthDataset` ([[trl-sft-trainer]]). The library card
[[hf-alignment-handbook]] carried such a snippet; the repository's own Zephyr configuration contains no
`packing` key, no masking flag, and no FSDP block, and launches with a DeepSpeed ZeRO-3 accelerate config
([[hf-alignment-handbook]] excerpt, `config_full.yaml` and `scripts/sft.py` L20-35).

**A second trainer, for contrast.** `scripts/chat_sft.py` in nanochat performs the same operations in one file
and uses different conventions: the ignore index is `-1`, not `-100`; the loss mask comes from
`tokenizer.render_conversation` rather than from template markers; and packing pads instead of truncating so
that "no tokens are ever discarded" ([[nanochat]], L180-187, L285-296). The two trainers give the same three
concepts different spellings, so a script written against one and run against the other trains the model on
the wrong tokens without raising an error.

**Implication for a general-purpose model.** Each of these stages selects which tokens the gradient sees. A
mask that includes prior assistant turns, a packed row whose documents attend to each other, or a template
whose end-of-turn token is outside the loss all change the trained distribution while leaving the loss curve
smooth.

---

## §3 Three tests that a correct run passes

### 3.1 Packed versus unpacked equivalence

**Definition.** Run the same examples twice — packed into one row with per-document position IDs, and
unpacked one row per document — and compare the per-token log-probabilities at the same tokens.

**Mechanism.** Packing concatenates documents into a fixed-length row. Correct packing restricts attention to
within-document pairs and restarts position IDs at 0 per document; a dense causal mask over the row allows
every earlier position to be attended.

**Worked example.** Two documents of 3 and 2 tokens packed into a row of 5. Under a causal mask restricted to
each document, the allowed query-key pairs number 3·4/2 = 6 in the first and 2·3/2 = 3 in the second, so 9 in
total. Under a dense causal mask over the row, 5·6/2 = 15 pairs are allowed. Six of the fifteen pairs, 40%,
cross the document boundary. Those six pairs enter the forward pass and the gradient, and the loss value does
not separate them from the within-document pairs.

**Evidence.** Fine-tuning Mistral-7B for one epoch on a 20K FLAN subset, validation loss was 1.129 with
padding (no packing), 1.306 with packing and no position IDs, 1.284 with offline packing *with* position IDs,
and 1.127 with online minibatch packing with position IDs ([[packing-with-flash-attention]] Table 2; 8×A100,
max sequence length 4096, one seed). For Llama-2-7B the same rows are 1.266 / 1.579 / — / 1.262, and for
Falcon-7B 1.842 / 2.585. **Result (single study).** The authors attribute the residual gap of offline packing
to the smaller number of optimisation steps in one epoch rather than to attention (§4.1, §4.3), which is why
the lab's test compares forward passes at fixed tokens instead of comparing end-of-run losses.

**Test.** Build one batch both ways, run the model in `float32` on CPU or with a deterministic kernel, and
compare `log p(token)` at the response positions. With correct position IDs and a FlashAttention variant, the
difference is at the level of floating-point reassociation; record the maximum absolute difference you
observe and treat that as the tolerance for later runs. If the difference is of order 0.1 nats, attention is
crossing document boundaries. TRL warns rather than raises when BFD packing is combined with a non-Flash
attention implementation ([[trl-sft-trainer]] L1236-1244), so the warning is not a substitute for this test.

### 3.2 Decoded inspection of the positions where `labels != -100`

**Test.**

```python
row = batch["labels"][0].tolist()
kept = [i for i, lab in enumerate(row) if lab != -100]
print(tokenizer.decode([batch["input_ids"][0][i] for i in kept]))
```

The printed text must be the assistant turns and nothing else, and it must end with the end-of-turn token. TRL
warns when the chat template attributes that token to the next message: "the model may not learn to stop"
([[trl-sft-trainer]] L1260-1266). A model trained without its stop token in the loss generates past the end of
the turn at inference, which is a capability failure with no training-time symptom.

**Why the 2026-04 version's masking test is invalid.** That version asserted
`model.get_input_embeddings().weight.grad[prompt_ids].abs().sum() == 0.0`. Two mechanisms make this false for
correct code.

1. Response positions attend to prompt positions. The key and value vectors at prompt position *j* are
   functions of the input embedding row of the prompt token, so `∂L/∂E[prompt token]` is non-zero through the
   attention path even when the prompt positions carry no loss.
2. When input and output embeddings are tied, the same tensor is the output projection. For softmax
   probabilities `p` and target `y`, the gradient with respect to logit `z_j` is `p_j − 1[j = y]`, where `p_j`
   is the model's probability for vocabulary item *j* and `1[j = y]` is 1 for the target and 0 otherwise.
   With `p = (0.5, 0.3, 0.2)` and target 0, the gradient is `(−0.5, +0.3, +0.2)`: rows 1 and 2 receive
   gradient although neither token appears in the batch. Every vocabulary row receives gradient at every
   supervised position.

The quantity that *is* zero under correct masking is the per-position loss contribution, which is what 3.2
inspects directly.

### 3.3 Step-1 loss against the base model's own loss

**Test.** Before training, compute the base model's loss on the first prepared batch with the same masking.
Then start training and read the step-1 logged loss. The two are the same forward pass, so they must agree to
logging precision.

**Why `ln(V)` is the wrong reference.** `ln(V)` is the loss of a uniform predictor over a vocabulary of size
`V`, which describes a randomly initialised model. For `V = 128,000`, `ln V = 11.76`. A pretrained base model
on instruction data sits near 1-2 nats per token: the one-epoch SFT validation losses measured in
[[packing-with-flash-attention]] Table 2 are 1.129 (Mistral-7B), 1.266 (Llama-2-7B), 1.169 (Granite-8B-code)
and 1.842 (Falcon-7B). A gate of "step-1 loss within 20% of `ln(V)`" therefore fails every correct run and
passes a run whose labels are random. A step-1 loss near 11.76 on a pretrained checkpoint is itself a
diagnostic: the label shift, the template, or the tokenizer is wrong.

**Making the loss comparable — worked example.** Cross-entropy in nats per token depends on the tokenizer.
Bits per byte removes that dependence: `bpb = ℓ / (B · ln 2)`, where `ℓ` is the summed negative
log-likelihood in nats and `B` the number of UTF-8 bytes in the text ([[paloma]] App. B). For a loss of 1.30
nats per token on text that averages 3.8 bytes per token, `bpb = 1.30 / (0.6931 × 3.8) = 0.49`. The same
conversion lets the lab compare a 135M run against a 7B run, and against the base model, on one axis.

---

## §4 Measuring target capability against general capability

**The measurement, stated.** Four quantities per checkpoint: (a) the target-task metric; (b) a broad suite the
training data does not target — MMLU, GSM8K, HumanEval, IFEval; (c) held-out loss as bits-per-byte on more
than one domain; (d) the target metric under a second prompt format. The base checkpoint is measured with the
identical harness, on the same day, with the same decoding parameters.

**Why more than one held-out set.** A single held-out set is a mixture whose composition is unstated. Among
six 1B models that differ only in pretraining corpus, per-domain perplexity ranged to 391,171 on the RedPajama
arXiv domain for the C4-only model, and the C4 and mC4-en models got *worse* between the ~20B and ~150B-token
checkpoints on 65 and 43 domains respectively, while one aggregate number would have continued to improve
([[paloma]] §4.1, App. D.1.1). **Result (single study)**, at 1B scale and English plus code only. Two
procedural details from the same source carry over to the lab: score each document separately after a BOS
token rather than concatenating (Pythia-1.4B at 2B tokens: 92.23 ± 17.33 concatenated versus 42.57 ± 0.29
separate, Table 17), and decontaminate the evaluation text against the training set before believing the
number (§3 G1).

**Noise, and what counts as a difference.** The final 30 checkpoints of a 1B model span 1.7 accuracy points on
ARC Challenge ([[signal-and-noise-eval]] §3.1). Across 30 benchmarks, scoring with bits-per-byte instead of
the primary metric raised the average signal-to-noise ratio from 10.0 to 31.5 and small-scale decision accuracy
from 77.0% to 83.7% (Fig. 6). Two consequences for this lab: report bits-per-byte alongside accuracy for every
ablation, and treat a one-point accuracy difference with one seed as unresolved.

**Format robustness.** Sampling 10 meaning-preserving prompt formats per task across 53 tasks, the median
spread between the best and worst format was 7.5 accuracy points; 20% of tasks spread at least 15 points in
every LLaMA-2 setting; the maximum observed for LLaMA-2-13B was 76 points; and the sensitivity was not removed
by larger models, more few-shot examples, or instruction tuning
([[prompt-format-sensitivity-formatspread]] §4.2, abstract). Model comparisons reverse under a format change
with probability 0.141 for LLaMA-2-13B against -70B at d = 0.02 (§4.2, Fig. 4). A single-format score for an
SFT checkpoint measures the checkpoint and the format together, which is why gate 5 requires two formats.

**The trade-off this measurement exposes.** [[loss-masking-prompt]] Table 1 gives the pattern in one table:
LLaMA-2-7B base scores 49.32 on the 18-task mean and 0.01 on AlpacaEval 1.0; after response-only SFT on
Alpagasus Alpaca 5k it scores 45.29 and 16.29; after the same SFT on Alpagasus Dolly 9k, 45.54 and 21.54. The
open-ended metric rises by 16-21 points while the broad-suite mean falls by about 4. The interactive version
of this table, with every dataset and method from Table 1 plotted on the two axes, is at
[figures/target-vs-general.html](figures/target-vs-general.html); the memo in §1 is the same plot for the
reader's own two checkpoints.

---

## §5 The ablation grid

Each lever below has measured evidence in at least one setting. The lab runs whichever subset the budget
allows, with seeds, and reports both axes of §4 for each cell.

| Lever | Settings to compare | What a source measured | Locus |
|---|---|---|---|
| Epochs | 1 / 2 / 3 | Llama 3.0 on the Tülu 2 mixture, sum loss, LR 5e-6: average performance peaks at 2 epochs and does not improve to 7 | [[tulu-3]] §4.3.2, Fig. 6 |
| Epochs (second setting) | 1 / 3 / 5 / 7 / 9 | LLaMA-2-7B + Alpaca with NEFTune, AlpacaEval (ChatGPT judge): 55.09 / 62.55 / 62.24 / 60.50 / 58.14 | [[neftune]] Table 14 |
| Epochs, general side | 2 to 10 | LLaMA-2-7B-BASE on five datasets: the 18-task NLP mean is plotted against epochs 2 to 10 for response-only and instruction-modelling loss; the paper names the drop an instruction tuning tax and reports instruction modelling having the lower one | [[loss-masking-prompt]] §4.3 "#3", Fig. 4 |
| Learning rate and loss reduction | sum versus mean loss at 2e-6, 5e-6, 1e-5, 2e-5 | Llama 3.0 on the Tülu 2 mixture: sum loss at 5e-6 gave the best average | [[tulu-3]] §4.3.2, Fig. 5 |
| Response-only versus instruction-modelling loss | mask prompt tokens or not | LLaMA-2-7B, Alpagasus Alpaca 5k: 18-task mean 45.29 → 47.47, AlpacaEval 1.0 16.29 → 19.52 with instruction modelling | [[loss-masking-prompt]] Table 1 |
| NEFTune | off / α = 5 | LLaMA-2-7B + Alpaca, AlpacaEval (GPT-4 judge) 29.79 → 64.69; ARC, HellaSwag, MMLU, TruthfulQA "remain stable"; α was selected on AlpacaEval itself | [[neftune]] Table 1, §4, App. A.1 |
| Prompt format at evaluation | at least two formats | Median spread 7.5 points over 10 formats, 53 tasks | [[prompt-format-sensitivity-formatspread]] §4.2 |
| Seeds | ≥ 2 seeds, or state that none were run | Final-checkpoint span of 1.7 points on ARC Challenge at 1B | [[signal-and-noise-eval]] §3.1 |

**Worked example: why the loss reduction is a lever, not a detail.** Two micro-batches are accumulated into one
update. Micro-batch A has 10 loss-bearing tokens with summed token loss 20 nats; micro-batch B has 90 tokens
with summed loss 90 nats. A single forward pass over both would report `(20 + 90) / (10 + 90) = 1.10`.
Averaging the per-micro-batch means reports `(2.0 + 1.0) / 2 = 1.50`, and weights each token of A by
`1/(2 × 10) = 0.05` against `1/(2 × 90) = 0.0056` for each token of B — a factor of 9 between tokens that
should be equal. Tülu 3 states this effect as Eqs. 1-2 and uses a sum loss with an adjusted learning rate to
remove it ([[tulu-3]] §4.3.2). The lab reports which reduction its trainer used; TRL's `chunked_nll` reduces the loss as
`sum / num_items_in_batch` when the trainer supplies that count, and as a mean over local valid tokens
otherwise (`sft_trainer.py` L161-164, L225-231; [[trl-sft-trainer]]).

**Training-format diversification.** Whether training under several chat templates improves robustness at this
scale is an **open question** in the sources read for this chapter: [[prompt-format-sensitivity-formatspread]]
measures evaluation-time sensitivity and states that instruction tuning does not remove it, but does not test
template diversification during fine-tuning. The lab therefore treats diversification as an experiment with an
unknown sign, and the evaluation under two formats as a requirement. ch-30 takes the training-side question up
with the SFT-specific literature.

---

## §6 Two compute paths

**Full-budget path.** One node of 8 H100s, a 7-8B base model, a single-source instruction set of tens of
thousands of prompts. Two published reference points bound the cost: the Tülu 3 8B model was trained on 32 GPUs
for 6 hours with effective batch 128 and maximum sequence length 4,096 for 2 epochs on a 939K-prompt mixture,
and the 70B model on 64 GPUs for 50 hours ([[tulu-3]] §4.3); nanochat's whole pipeline — tokenizer, pretraining
of a depth-24 model, SFT and evaluation — is documented as approximately 1.5 hours on one 8×H100 node, with the
README pricing that node at about $24 per hour ([[nanochat]] `speedrun.sh` L3-4, README L341). This lab does
not predict the reader's throughput: `SFTTrainer` logs `num_tokens` per step ([[trl-sft-trainer]] L1854-1866),
and the measured tokens per second belongs in the memo.

**Resource-constrained path.** One GPU of 16 GB or less, a 125M-500M base model, a few thousand prompts. Every
gate in §1 and every test in §3 applies unchanged, and so does the before/after generality table; the broad
suite is scored with the same harness on the same two formats. What changes is the expected result: at this
scale the target gain may be small and the general regression may dominate. Reporting that honestly is the
deliverable. nanochat is the reference for this path because its SFT script is readable end to end and it
already evaluates a held-out bits-per-byte and a composite chat metric every 200 steps ([[nanochat]] L59-62).

**Distributed settings.** Use FSDP full sharding when the model does not fit in one device's memory with
optimizer state; parameters and gradients are computed in low precision while the optimizer step runs in full
precision, which lowers parameter peak memory from `K_full·Σψ_i/F + K_full·max_i ψ_i` to
`K_full·Σψ_i/F + K_low·max_i ψ_i` bytes, where `ψ_i` is the element count of FSDP unit *i*, `F` the sharding
factor, and `K_low`, `K_full` the bytes per element in each precision ([[fsdp-sft]] §4.4). Otherwise use
single-device training: the lab's questions do not require sharding, and sharding adds a second explanation for
every anomaly.

---

## Recipe

Lab values are this chapter's choices; every source value below was read at the stated locus.

| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value | Lab value | Reason for difference |
|---|---|---|---|---|---|---|---|---|---|
| zephyr-7b-sft-full | 7B | SFT | base model; precision; attention | `mistralai/Mistral-7B-v0.1`; bf16; `flash_attention_2` | alignment-handbook@1de1fc9 `recipes/zephyr-7b-beta/sft/config_full.yaml` | verified 2026-09-17 | no ablation reported | SmolLM2-135M or Qwen2.5-0.5B; bf16; FlashAttention variant if available | fits one GPU; packing tests need a Flash variant |
| zephyr-7b-sft-full | 7B | SFT | peak LR; schedule; warmup; epochs | 2.0e-5; cosine; `warmup_ratio: 0.1`; 1 | same file | verified 2026-09-17 | no ablation reported | same LR and schedule; epochs 1/2/3 as an ablation | epochs are a lever in §5 |
| zephyr-7b-sft-full | 7B | SFT | sequence length; per-device batch; accumulation | `max_seq_length: 2048`; 16; 1 | same file | verified 2026-09-17 | no ablation reported | `max_length: 1024`; per-device 4; accumulation to match 128 sequences | memory at 16 GB |
| zephyr-7b-sft-full | 7B | SFT | global batch (sequences) | 128 | derived: 16 × 1 × 8 devices, device count from `scripts/sft.py` L20-21 | derived | not printed in the file | 128 sequences | keep the batch, change the split |
| zephyr-7b-sft-full | 7B | SFT | packing; loss masking; grad clip | no `packing` key; no masking flag; no `max_grad_norm` | same file | not reported (checked: the YAML, `scripts/sft.py`) | — | `packing=True`, `assistant_only_loss=True`, `max_grad_norm=1.0` | the lab tests packing and masking explicitly |
| Tülu 3 8B | 8B | SFT | epochs; peak LR; loss reduction | 2; 5e-6; sum loss | arXiv:2411.15124 §4.3 and Table 11 (epochs, LR); §4.3.2 (sum loss) | verified 2026-09-17 | §4.3.2 Fig. 5 (sum loss at 5e-6 best of 4 LRs × 2 reductions) and Fig. 6 (2 epochs best of 2-7), Llama 3.0 on the Tülu 2 mixture, 1 seed | 2 epochs; LR from the handbook row | 939K prompts versus a few thousand; LR does not transfer across data size |
| Tülu 3 8B | 8B | SFT | effective batch; max sequence length; compute | 128; 4,096; 32 GPUs for 6 hours | arXiv:2411.15124 §4.3 | verified 2026-09-17 | no ablation reported | 128 sequences; 1,024 tokens | scale |
| Tülu 3 70B | 70B | SFT | peak LR; compute | 2e-6; 64 GPUs for 50 hours | arXiv:2411.15124 §4.3 | verified 2026-09-17 | "found after a hyperparameter search"; the search is not reported | not used | out of lab scope |
| LLaMA-2-7B (IT and IM runs) | 7B | SFT | LR; batch; epochs; schedule; weight decay | 2e-5; 128 sequences; 2; linear, warmup ratio 0.03; 0 | arXiv:2405.14394v2 App. C Table 6 | verified 2026-09-17 | Table 1 reports the outcome for 7 datasets; 1 seed | same LR and batch | direct comparability with the Table 1 baseline |
| LLaMA-2-7B + Alpaca (NEFTune) | 7B | SFT | LR; epochs; batch; max length; α | 5e-5; 3; 128; 512; 5 | arXiv:2310.05914v2 App. A.1, Table 7 | verified 2026-09-14 ([[neftune]]) | Table 14 epoch sweep; α chosen on AlpacaEval itself | α = 5 as an on/off ablation | α selection on the target metric is the failure mode §5 warns about |
| nanochat SFT (`chat_sft.py`) | depth-24 | SFT | initial LR fraction; warmup; warmdown; eval cadence | 0.8 of base LR; 0.0; 0.5; bits-per-byte and ChatCORE every 200 steps | github.com/karpathy/nanochat@f527f76 `scripts/chat_sft.py` L54-62 | verified 2026-09-17 | no ablation reported | eval cadence copied | in-loop general measurement is a gate, not an option |
| TRL `SFTConfig` (framework default, not a trained model) | — | SFT | `max_length`; `packing`; `packing_strategy`; `assistant_only_loss`; `loss_type` | 1024; `False`; `"bfd"`; `False`; field default `None`, resolved to `"chunked_nll"` | trl@aa89588 `sft_config.py` L203-293 | verified 2026-09-17 | framework defaults; no ablation | every one set explicitly in the lab config | defaults change between commits ([[trl-grpo]] Verification) |

**Starting point for a small general-purpose run.** For a 0.5B base model on a few thousand instruction
examples on one GPU: bf16, cosine schedule with `warmup_ratio = 0.1`, peak LR 2.0e-5, 1 epoch first and 2 and 3
as ablations, `max_length = 1024`, global batch 128 sequences reached by accumulation, `packing=True` with the
BFD strategy and a FlashAttention variant, `assistant_only_loss=True`, `max_grad_norm = 1.0`. The LR, schedule,
warmup and epoch count come from the verified Zephyr row (Mistral-7B, UltraChat-200K, one 8×H100 node) and the
Tülu 3 rows (8B, 939K prompts, 32 GPUs); neither source ran at 0.5B or at a few thousand examples, so these
values are a starting point to ablate, not a recommendation supported at this scale.

---

## Generalization lens

**(a) What increases breadth.** Measuring both axes is itself the intervention that keeps breadth in view: the
levers that protect the 18-task mean are visible only when it is measured (instruction-modelling loss recovers
2.18 points of that mean on Alpagasus Alpaca 5k, [[loss-masking-prompt]] Table 1). Stopping at the epoch count
selected on a broad average rather than on the target metric is the second: 2 epochs on the Tülu 2 mixture,
chosen on an average over the evaluation suite ([[tulu-3]] Fig. 6). Scoring with bits-per-byte raises the
reliability of every small-scale decision that follows (30-task average SNR 10.0 → 31.5,
[[signal-and-noise-eval]] Fig. 6).

**(b) What causes narrowing.** Training longer on one narrow distribution: the 18-task mean is plotted against
2 to 10 epochs for LLaMA-2-7B-BASE and the paper names the drop an instruction tuning tax
([[loss-masking-prompt]] §4.3 "#3", Fig. 4). Selecting a hyperparameter on the target metric alone: the
NEFTune α values were chosen on AlpacaEval, and safety was not evaluated ([[neftune]] App. A.1, §7). Training
and evaluating under one prompt format: format sensitivity survives instruction tuning, and a single format
hides a median spread of 7.5 points ([[prompt-format-sensitivity-formatspread]] §4.2). Silent input-pipeline
errors also narrow: packing without position IDs raised Mistral-7B validation loss from 1.129 to 1.306
([[packing-with-flash-attention]] Table 2).

**(c) How to measure it at this stage.** Per-domain held-out bits-per-byte on documents scored separately
after BOS, decontaminated against the training set ([[paloma]] §3, App. B, Table 17); a broad accuracy suite
(MMLU, GSM8K, HumanEval, IFEval) run identically on the base and trained checkpoints; the target metric under
two prompt formats; and a noise floor stated with the result, from seeds or from the spread across the final
checkpoints of one run ([[signal-and-noise-eval]] §3.1).

---

## Common mistakes and how to detect them

| Mistake | Observable symptom | Check |
|---|---|---|
| Gating on "step-1 loss ≈ `ln(V)`" | Every correct run fails the gate; a run with shuffled labels passes it | Compare step-1 loss with the base model's loss on the same batch (§3.3); pretrained bases sit near 1-2 nats ([[packing-with-flash-attention]] Table 2) |
| Asserting prompt-token embedding gradients are exactly zero | The assertion fails on correct masking | Inspect the decoded `labels != -100` positions instead (§3.2); attention and tied embeddings both route gradient to unsupervised rows |
| Copying a config with a non-existent argument (`train_on_response_only`, `max_seq_length`) | Argument rejected, or silently ignored; masking never applied | Grep the pinned commit for the symbol ([[trl-sft-trainer]]) before running |
| Instrumenting clipping in `training_step` or `_maybe_log_save_evaluate` | The hook never fires; `grad_norm` logged as `None` | Clipping is at `trainer.py` L2697-2718 in `_inner_training_loop` ([[hf-trainer-loop]]) |
| Assuming packing breaks the cosine horizon | Time spent on a non-existent bug | `max_steps` is computed from the packed dataloader (`trainer.py` L5675-5689); a hand-built scheduler with a pre-packing step count is the case that does break |
| Packing with a non-FlashAttention implementation | A warning in the log, a slightly worse loss, no error | The packed-versus-unpacked test (§3.1); TRL only warns ([[trl-sft-trainer]] L1236-1244) |
| `assistant_only_loss=True` with a template lacking `{% generation %}` or excluding the stop token | Model does not stop generating at inference | Trainer warnings at L1251-1266; decode the kept positions and confirm the end-of-turn token is present |
| Evaluating only the target task | A run that lost 4 points of broad-suite mean is reported as a success | The base-model row in `target-vs-general.md` ([[loss-masking-prompt]] Table 1 is the reference pattern) |
| Scoring one prompt format | Result not reproducible by a reader who writes the prompt differently | Two formats, both reported ([[prompt-format-sensitivity-formatspread]] §4.2) |
| Concatenating documents for held-out perplexity | Held-out loss has large variance between runs | Score each document separately after BOS (92.23 ± 17.33 versus 42.57 ± 0.29, [[paloma]] Table 17) |
| Reporting a 1-point benchmark change from one seed | The "improvement" does not survive a rerun | Compare against the final-checkpoint spread, 1.7 points on ARC Challenge at 1B ([[signal-and-noise-eval]] §3.1) |
| Selecting α, LR or epochs on the target metric | Target metric rises, broad suite falls, and the selection hid it | Select on the broad suite or on a held-out average; state which was used |

---

## Check your understanding

1. The masking test in §3.2 inspects decoded tokens rather than gradients. Give the two mechanisms that make a
   zero-gradient assertion on prompt-token embedding rows false, and say which one still applies to a model
   whose input and output embeddings are not tied.
2. A run reports step-1 loss 11.8 on a pretrained 7B base with a 128,000-token vocabulary. List the candidate
   causes in the order you would test them, and say what each would do to the decoded `labels != -100` output.
3. Packing without position IDs raised Mistral-7B validation loss from 1.129 to 1.306 in one epoch, but offline
   packing *with* position IDs also raised it, to 1.284. Explain why those two increases have different causes,
   and why the lab's test compares forward passes rather than end-of-run losses.
4. A checkpoint gains 6 points on the target task and loses 3 points of 18-task mean. What additional
   measurement would let you decide whether the 3-point loss is a real regression, and what result would make
   you report it as unresolved?
5. Why does bits-per-byte make an ablation comparable across a 135M and a 7B run in a way that nats-per-token
   does not, and what property of the evaluation text must hold for the comparison to be valid?
6. Tülu 3 selected 2 epochs on an average over its evaluation suite, while the NEFTune α values were selected
   on AlpacaEval. Explain, in terms of the two axes in §4, why those two selection procedures have different
   risks, and what the NEFTune paper's own reported numbers do and do not license.
7. A reader reproduces this lab and reports a 7-point target gain under their prompt format. What is the
   smallest set of additional numbers that would let a second reader decide whether the gain is a property of
   the model or of the prompt?

---

## Connections

- **ch-07 — Training Failure Modes: Numerical, Masking, and Capability-Level Failures** (previous chapter, and
  this chapter's dependency): supplies the failure taxonomy; this lab instantiates its masking and
  capability-level entries as executable tests.
- **ch-04 — Sequence Packing, Loss Masking, and Chat Templates**: the mechanisms that §2 and §3 verify in
  source.
- **ch-05 — Distributed Training Choices That Change Batch Size, Sequence Length, and Tokens Seen**: the
  reason the recipe table separates per-device batch, accumulation, and global batch.
- **ch-06 — Checkpointing, In-Loop Evaluation, and Checkpoint Selection**: the in-loop measurement cadence this
  lab adopts from nanochat, and the checkpoint-selection rule behind the epoch ablation.
- **ch-08a — Scaling Laws and Compute Allocation: From Pretraining Loss to Downstream Capability** (next
  chapter): moves from measuring one run to predicting what a compute budget buys.
- **ch-30 — SFT Design Choices and Their Effect on Generalization: Masking, Packing, Templates, Epochs, and
  Learning Rate**: the full treatment of the levers this lab ablates, including training-side template
  diversification.
- **ch-36 — Lab: SFT Run with Masking Tests, a Forgetting Report, and a Held-Out Evaluation Split**: the
  production-scale version of this lab.
- **ch-47 — Evaluation Harness and Suite Design for General Capability**: the harness questions this lab
  answers provisionally with one suite and two formats.

---

## Sources

- [[trl-sft-trainer]] — chapter excerpt: TRL `SFTTrainer` and `SFTConfig` at commit `aa89588`; every symbol,
  default and guard used in §2, §3 and the recipe table.
- [[hf-trainer-loop]] — chapter excerpt: `transformers` v4.57.1 loci for clipping, `optimizer.step()`,
  `lr_scheduler.step()`, and the `max_steps` computation.
- [[nanochat]] — chapter excerpt: the second trainer read in §2 (ignore index `-1`, pad-not-crop packing) and
  the costed 8×H100 reference run in §6.
- [[loss-masking-prompt]] — chapter excerpt of Shi et al. 2024: Table 1 target-versus-general numbers, the
  instruction tuning tax, and the response-only versus instruction-modelling lever.
- [[packing-with-flash-attention]] — chapter excerpt of Kundu et al. 2024: Table 2 packed-versus-padded
  validation losses and throughput, used in §3.1 and §3.3.
- [[prompt-format-sensitivity-formatspread]] — chapter excerpt of Sclar et al. 2023: format spread numbers
  behind the two-format requirement.
- [[hf-alignment-handbook]] — chapter excerpt: the Zephyr SFT configuration as written in the repository, and
  the corrections to the snippet the library card carried.
- [[paloma]] — per-domain held-out loss, bits-per-byte definition, decontamination, and separate-document
  scoring variance.
- [[neftune]] — the NEFTune lever, its α selection procedure, and its reported effect on broad benchmarks.
- [[signal-and-noise-eval]] — noise floor for benchmark differences and the case for bits-per-byte as the
  decision metric.
- [[tulu-3]] — SFT epochs, learning rate, loss reduction, batch and compute at 8B and 70B.
- [[fsdp-sft]] — the mixed-precision memory contract quoted in §6.
- [[trl-grpo]] — verified evidence that symbols and defaults in this repository move between commits, which is
  the reason §2 pins one.
