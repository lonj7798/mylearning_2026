---
chapter: ch-04
course: llm-training
phase: read
excerpt_of: huggingface/trl — SFTConfig, SFTTrainer collator, packing, and chat-template utilities
source_url: https://github.com/huggingface/trl/tree/aa89588dc63ad5f4354614e94c2851759ebb617e
created_at: "2026-09-15"
note: "Released code, read at commit aa89588 (last commit touching trl/trainer/sft_config.py, 2026-09-09). Line numbers refer to that commit."
---

# Excerpt: TRL SFT packing, padding-free collation, and loss masks (commit aa89588)

## `trl/trainer/sft_config.py`

- `max_length` defaults to 1024; when packing is enabled it sets the row length (L68-71).
- `packing` defaults to `False` (L77-79). `packing_strategy` defaults to `"bfd"`: `"bfd"` (best-fit decreasing, truncates overflow), `"bfd_split"` (best-fit decreasing, splits overflow sequences), or `"wrapped"` (aggressive, cuts mid-sequence) (L80-82).
- `padding_free` defaults to `False`, is supported only with FlashAttention 2 or 3, and is enabled regardless of its value when packing uses `"bfd"` (L83-88).
- `completion_only_loss` defaults to `None`: loss on the completion for prompt-completion datasets, on the full sequence for language-modelling datasets (L96-101).
- `assistant_only_loss` defaults to `False`: when `True`, loss only on assistant responses, supported only for conversational datasets; when `False`, loss on the entire sequence (L102-105).
- `loss_type` defaults to `"chunked_nll"` (same objective as `"nll"`, with the `lm_head` projection computed only on positions whose label is not −100) (L106-114).

## `trl/data_utils.py`

- `_pack_bfd` (L739-826): filters empty sequences; truncates to `seq_length` or splits overflow; records `seq_lengths` "to later construct `position_ids`"; sorts by length descending; bins greedily with a segment tree that finds the best-fit remaining space.
- `pack_dataset` docstring (L859-898): `"bfd"` "truncates sequences that exceed `seq_length`, discarding overflow tokens"; `"bfd_split"` "splits overflow sequences … Prevents token loss for pre-training or long documents, but may break conversation structure in SFT datasets"; `"wrapped"` "Ignores sequence boundaries and will cut sequences in the middle". Example with lengths 5, 2, 3, 1 and `seq_length=4`: `"bfd"` gives `seq_lengths` [[4], [3, 1], [2]].

## `trl/trainer/sft_trainer.py` (collator and trainer)

```python
# L484-L486
# For padding-free, we should NOT create attention_mask as it causes FlashAttention to ignore position_ids and
# compute wrong cu_seq_lens from the all-1s mask
if self.padding_free or self.return_position_ids:
# L511-L516
if self.padding_free:
    output["position_ids"] = pad(position_ids, padding_value=0, padding_side="right", ...)
    output["labels"][output["position_ids"] == 0] = -100
```

- `get_position_ids_from_packed_seq_lengths` rebuilds per-example position ids that restart at 0 from `seq_lengths` (L528-552).
- "BFD packing requires padding-free mode; otherwise, the collator outputs padded attention masks, causing FlashAttention to ignore position_ids and recompute them incorrectly from the padded attention mask" (L1167-1169).
- PEFT with tokens added by `clone_chat_template`: added ids become trainable token indices, and TRL warns that without `lm_head` in `modules_to_save` "the model may not learn to generate outputs with these new tokens", then adds it (L1088-1106).
- With `assistant_only_loss=True`, TRL swaps in a training chat template with `{% generation %}` markers if the current template lacks them (L1251-1256), and warns "The chat template does not include the assistant turn's end-of-turn token in the loss mask; the model may not learn to stop." when the check fails (L1258-1266).
- Tokenization passes `return_assistant_tokens_mask=assistant_only_loss` to `apply_chat_template` (L1540, L1574) and raises: "You're using `assistant_only_loss=True`, but at least one example has no assistant tokens. … it may be missing the `{% generation %}` keyword" (L1585-1591).

## `trl/chat_template_utils.py`

- `has_generation_markers` searches for `{% generation %}` including whitespace-trim variants (L37-43).
- `clone_chat_template` copies a template from a source tokenizer, adds tokens missing from the target vocabulary, sets EOS on tokenizer and model config, resizes embeddings to `len(tokenizer.vocab)` rounded up to a multiple of 64 by default, and adds dummy `<extra_id_i>` tokens so vocabulary and embedding sizes match (L45-136).
- `is_chat_template_prefix_preserving`: "A prefix-preserving chat template renders earlier messages identically regardless of what messages follow" (L829-831); implemented by comparing token ids of a conversation with and without an appended tool message (L825-883).
- `is_chat_template_stop_token_trained`: renders user / assistant / user with `return_assistant_tokens_mask=True` and checks that the last non-whitespace masked token is an added (special) token; "Some templates attribute an assistant turn's end-of-turn token to the message that follows it, so … the model is never trained to stop" (L886-952).
- `get_training_chat_template` docstring, Qwen3-0.6B example (L1071-1077): the assistant tool-call message renders as `<|im_start|>assistant\n<think>\n\n</think>\n\n<tool_call>…` when it is the last message, and as `<|im_start|>assistant\n<tool_call>…` when a tool message follows, with the tool result rendered as `<|im_start|>user\n<tool_response>\n6\n</tool_response>`. TRL's patched training templates exist for families including LLaMA 3, Qwen2.5, Qwen3, DeepSeek-V3, Gemma, and GPT-OSS (L1038-1041).
