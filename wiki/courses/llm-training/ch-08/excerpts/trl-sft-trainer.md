---
chapter: ch-08
course: llm-training
phase: read
excerpt_of: https://github.com/huggingface/trl/blob/aa89588dc63ad5f4354614e94c2851759ebb617e/trl/trainer/sft_trainer.py
source_url: https://github.com/huggingface/trl
created_at: "2026-09-17"
revised: 2026-09 (generality revision)
---

# Excerpt: TRL `SFTTrainer` and `SFTConfig` at commit `aa89588`

**Artifact:** `huggingface/trl` at commit `aa89588dc63ad5f4354614e94c2851759ebb617e` (authored 2026-09-09,
"Match config field order to the docstring order", PR #7151); files `trl/trainer/sft_trainer.py` (1927 lines)
and `trl/trainer/sft_config.py` (334 lines). Repository `main` on 2026-09-14 was `a04ffd3`.
**Read on:** 2026-09-17, from the file text cached at
`scratchpad/sources/trl-sft-trainer-aa89588.txt` and `scratchpad/sources/trl-sft-config-aa89588.txt`.

This excerpt exists because ch-08 is a "read the trainer in source" lab and the library card
[[hf-alignment-handbook]] contains a `SFTConfig(...)` snippet with a `train_on_response_only` argument that
does not exist in TRL. Every symbol below was located in the file text at the stated line numbers.

---

## `SFTConfig` fields used by the lab (file `sft_config.py`, commit `aa89588`)

| Field | Default | Docstring statement (locus) |
|---|---|---|
| `max_length` | `1024` | "Maximum length of the tokenized sequence… When packing is enabled, this value sets the sequence length." (L68-71, field L203-211) |
| `packing` | `False` | "Whether to group multiple sequences into fixed-length blocks… Uses `max_length` to define sequence length." (L77-79, field L223-229) |
| `packing_strategy` | `"bfd"` | `"bfd"` (best-fit decreasing, truncates overflow), `"bfd_split"` (splits overflow), `"wrapped"` (cuts mid-sequence) (L80-82, field L230-238) |
| `padding_free` | `False` | Flattens a batch into one sequence; "only supported with the FlashAttention 2 or 3"; forced on when `packing` uses `bfd` (L83-88, field L239-251) |
| `completion_only_loss` | `None` | `None` means: completion-only for prompt-completion datasets, full sequence for language-modeling datasets (L96-101, field L259-270) |
| `assistant_only_loss` | `False` | Loss on assistant turns only; "supported only for conversational datasets" (L102-105, field L271-280) |
| `loss_type` | `None` → `"chunked_nll"` | `"nll"`, `"dft"` (arXiv 2508.05629), `"chunked_nll"` (same math as `nll`, `lm_head` computed on non-ignored tokens only) (L106-115, field L281-293) |
| `truncation_mode` | `"keep_start"` | `"keep_end"` deprecated, removal in v2.0.0 (L72-74, field L212-221) |
| `shuffle_dataset` | `False` | (L75-76, field L219-222) |
| `chat_template_path` | `None` | Path to a tokenizer or a Jinja file; added special tokens must be added to the tokenizer and the embedding resized (L49-53) |

`max_seq_length` is not a field of this class; the lab uses `max_length`.

## Symbols named in the pre-revision chapter, checked against this commit

| Symbol in the old chapter | Status at `aa89588` |
|---|---|
| `train_on_response_only=True` | not a field of `SFTConfig`; the nearest real fields are `assistant_only_loss` and `completion_only_loss` |
| `max_seq_length` | not a field of `SFTConfig`; the field is `max_length` |
| `DataCollatorWithPacking` | no such class in `sft_trainer.py`; the collator is `DataCollatorForLanguageModeling` (L403) |
| `DataCollatorForCompletionOnlyLM` | not present in this file at this commit |
| `ConstantLengthDataset` | not present in this file at this commit; packing is done by `pack_dataset` (imported L66, called L1678) |

## Data path, in call order (file `sft_trainer.py`)

1. `_prepare_dataset` (L1438) — converts to ChatML if needed (L1489-1495), appends the EOS token for
   non-conversational examples (L1498-1514), then tokenizes with `_tokenize` inside `tokenize_fn` (L1519-1603).
   For conversational prompt-completion data it calls the chat template twice: once on the prompt with
   `add_generation_prompt=True`, once on prompt + completion, and warns when the first token ids of the second
   call do not match the first (L1536-1556).
2. Assistant masks — with `assistant_only_loss=True`, `_tokenize` is called with
   `return_assistant_tokens_mask=True` (L1543-1546). If an example produces no assistant token the trainer
   raises: "at least one example has no assistant tokens… it may be missing the `{% generation %}` keyword"
   (L1585-1591).
3. `build_labels` (L1606-1632) — labels are `input_ids` with `-100` at every position where an applicable mask
   bit is 0. Applicable masks are `completion_mask` (only when `completion_only_loss`) and `assistant_masks`
   (always when present). With no applicable mask, `labels == input_ids`.
4. Truncation (L1634-1660) — applied only when not packing; examples left fully masked are dropped (L1657-1660).
5. Packing (L1662-1678) — `dataset.select_columns(["input_ids", "labels"])`, then
   `pack_dataset(dataset, args.max_length, args.packing_strategy, map_kwargs)`. Comment at L1677:
   "Packing adds new column `seq_lengths` needed for document aware FlashAttention".
6. Collation — `DataCollatorForLanguageModeling` (L403-553) pads `labels` with `-100` (L509-511), and in the
   padding-free branch builds `position_ids` from `seq_lengths` (L488-489, L528-552), concatenates the batch
   into one row (L497-500), and then sets `output["labels"][output["position_ids"] == 0] = -100` (L516), so the
   first token of every packed document contributes no loss.

## Guards the trainer applies at construction (`__init__`)

```python
# L1169
self.padding_free = args.padding_free or (args.packing and args.packing_strategy in {"bfd", "bfd_split"})
```

- `attn_implementation` is read from `model.config._attn_implementation` and compared against
  `FLASH_ATTENTION_VARIANTS = {"flash_attention_2", "flash_attention_3", "kernels-community/flash-attn2",
  "kernels-community/flash-attn3", "kernels-community/vllm-flash-attn3"}` (L394-398, L1172-1173).
- With BFD packing and a non-FlashAttention implementation the trainer logs a warning, not an error:
  "Using other implementations may lead to cross-contamination between samples" (L1236-1244).
- `assistant_only_loss=True` with a non-conversational dataset raises `ValueError` (L1245-1250).
- With `assistant_only_loss=True` and a template without `{% generation %}` markers the trainer swaps in
  `get_training_chat_template(processing_class)` (L1251-1258).
- If the template attributes the assistant end-of-turn token to the next message, the trainer warns:
  "the model may not learn to stop" (L1260-1266).
- `padding_free=True` without packing and with `max_length` set raises: "`max_length` is not enforced" (L1269-1275).

## Metrics logged per step (`compute_loss`, L1763-1893)

- `entropy`: with `loss_type="chunked_nll"` from `outputs.entropy_sum / outputs.num_valid_tokens` (L1815-1822);
  otherwise from `entropy_from_logits(shift_logits)` over positions where `shift_labels != -100` (L1823-1852).
- `mean_token_accuracy`: `argmax(logits) == shift_labels` over the same non-ignored positions (L1846-1852, L1873-1876).
- `num_tokens`: cumulative, from `attention_mask.sum()` or from `position_ids.size(1)` in the padding-free path
  (L1854-1866).

## Connections

- [[hf-trainer-loop]] — where clipping, `optimizer.step()`, and `lr_scheduler.step()` actually run.
- [[packing-with-flash-attention]] — measured effect of packing with and without position IDs.
- [[trl-grpo]] — library card for the same repository's GRPO trainer, pinned at `a08e713`, with a Verification
  section listing defaults that changed between two 2026 commits.
