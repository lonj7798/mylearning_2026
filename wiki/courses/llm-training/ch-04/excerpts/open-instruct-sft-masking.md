---
chapter: ch-04
course: llm-training
phase: read
excerpt_of: allenai/open-instruct — SFT label masking in open_instruct/dataset_transformation.py
source_url: https://github.com/allenai/open-instruct/blob/d8a7f1ce97087795c19a170691b4de15a7c228a6/open_instruct/dataset_transformation.py
created_at: "2026-09-15"
note: "Released code read at commit d8a7f1c (the commit also used by the verified card frameworks/allenai-olmo3-open-instruct-scripts.md). Line numbers refer to that commit. OLMo 3 template settings are in the verified card [[open-instruct-allenai-recipes]] (docs/olmo3.md at 098424c)."
---

# Excerpt: open-instruct SFT masking functions (commit d8a7f1c)

## `mask_labels` (L1110-1173)

Masks the token span of each message for which `should_mask(message_idx, message, messages)` is true, by setting labels to −100. The span boundaries are computed by re-rendering conversation prefixes with the chat template:

```python
# L1159-L1171 (abridged)
message_start_idx = tokenizer.apply_chat_template(
    conversation=messages[:message_idx], add_generation_prompt=False, **chat_template_kwargs
).shape[1]
# Compute end of this message's token span. If the next turn is an
# assistant turn, include the generation prompt header in the masked
# region so it's excluded from the loss.
next_is_assistant = message_idx < len(messages) - 1 and messages[message_idx + 1]["role"] == "assistant"
message_end_idx = tokenizer.apply_chat_template(
    conversation=messages[: message_idx + 1], add_generation_prompt=next_is_assistant, **chat_template_kwargs
).shape[1]
labels[:, message_start_idx:message_end_idx] = MASKED_TOKEN_VALUE
```

Consequence for ch-04 (derived from the code): every token that the template emits as part of the generation prompt is masked. If a template's generation prompt ends with `<think>`, that `<think>` token is not trained. `docs/olmo3.md` at 098424c reports this as "our code incorrectly masks the first <think> token as part of the prompt" (L39).

The function also defers masking until a user turn appears, because some templates (the comment names Qwen3.5) fail when rendering a prefix with only system or tool turns (L1125-1128).

## `sft_tulu_tokenize_and_truncate_v1` (L1176-1199)

Renders the full conversation with `add_generation_prompt=False`, clones `input_ids` into `labels`, and calls `mask_labels` with `lambda idx, msg, _msgs: msg["role"] != "assistant"`. Every non-assistant message (system, user, tool) is masked and every assistant message is trained. The docstring states it is "taken directly from" `finetune.py` at commit ba11286.

## `last_turn_tulu_tokenize_and_truncate_v1` (L1202-1225)

Same rendering, with `lambda idx, _msg, msgs: idx < len(msgs) - 1`: every message except the final one is masked, so only the last message is trained.

## `sft_tulu_filter_v1` (L1228-1229)

Keeps an example only if at least one label is not −100, which removes examples whose trained tokens were all truncated.
