---
chapter: ch-30
course: llm-training
phase: read
excerpt_of: arXiv:2411.15124v5; alignment-handbook@1de1fc9; arXiv:2407.21783v3; Qwen3-8B model card; arXiv:2512.02556v1; open-instruct@098424c
source_url: https://arxiv.org/abs/2411.15124
created_at: "2026-04-23"
revised: "2026-09-15 (generality revision; rewritten from primary sources)"
---

# Excerpt: Chat templates and reasoning-history rules as printed by their sources

Used by [[read]] §1.2, §6, §7. The earlier version of this excerpt attributed ChatML to Mistral, stated that base models were conditioned by post-training on their chat tokens, and quoted a handbook lesson that was not found in the handbook files checked. Those claims are removed.

## 1. Tülu 3 template experiment (arXiv:2411.15124v5 §4.3.1, Table 13; card [[tulu-3]])

Verbatim: "We made a small change to the chat template used in previous Tülu versions, specifically removing the new line at the end of the template (before the model response). … We found that replacing the newlines at the end of assistant messages with an eos token resulted in the best performance, but we opted not to use this to avoid generation inconsistency with later steps in our post-training pipeline."

| Chat template (intermediate SFT mixture, Llama 3.0) | Avg. |
|---|---|
| Tülu (replace \n w/ eos) | 53.0 |
| Zephyr | 52.9 |
| Tülu 3 (no \n) | 52.8 |
| Tülu 2 template | 52.6 |
| Llama 3 template | 51.6 |

The `tulu` template string in open-instruct@098424c `open_instruct/dataset_transformation.py` L237-254 renders `<|system|>\n`, `<|user|>\n`, and `<|assistant|>\n` + content + `eos_token`, with a trailing `\n` after every assistant message except the last.

## 2. Zephyr-7B-β SFT recipe (alignment-handbook@1de1fc9, `recipes/zephyr-7b-beta/sft/config_full.yaml`; card [[hf-alignment-handbook]])

```yaml
model_name_or_path: mistralai/Mistral-7B-v0.1
chat_template: "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'system' %}\n{{ '<|system|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}"
learning_rate: 2.0e-05
lr_scheduler_type: cosine
max_seq_length: 2048
num_train_epochs: 1
per_device_train_batch_size: 16
gradient_accumulation_steps: 1
warmup_ratio: 0.1
```

The Trainer-generated model card of the Hub model `alignment-handbook/zephyr-7b-sft-full` (huggingface.co/alignment-handbook/zephyr-7b-sft-full, README, Transformers 4.36.2) lists: learning_rate 2e-05, train_batch_size 16, num_devices 8, total_train_batch_size 128, optimizer "Adam with betas=(0.9,0.999) and epsilon=1e-08", lr_scheduler_type cosine, warmup ratio 0.1, num_epochs 1, and 1,090 steps with validation loss 0.9353. The file has no `packing`, assistant-only-loss, or NEFTune key. Its `dataset_mixture` lists `HuggingFaceH4/ultrachat_200k` splits `train_sft` and `test_sft`, each with weight 1.0, and `test_split_size: 1000`. The commit is dated 2026-04-08; the file used for the original run may differ, but the model card values above agree with this file on LR, schedule, warmup ratio, epochs, and per-device batch. FILM-7B, fine-tuned from Mistral-7B-Instruct-v0.2, uses the Mistral `[INST] … [/INST]` format in its training template ([[in2-film]] §2.2, App. D Example 9).

## 3. Llama 3 chat protocol (arXiv:2407.21783v3 §4.1.1, §4.1.4; card [[llama-3]])

§4.1.1 (verbatim): "we design a new multi-message chat protocol which uses various special header and termination tokens. The header tokens are used to indicate the source and destination of each message in a conversation. Similarly, the termination tokens indicate when it is the time to alternate between human and AI to speak." §4.1.4: formatting tokens "including header and termination tokens" are masked from both chosen and rejected responses in the DPO loss because "having these tokens contribute to the loss may lead to undesired model behaviors such as tail repetition or abruptly generating termination tokens."

## 4. Reasoning-history rules

- **Qwen3-8B model card, Best Practices** (huggingface.co/Qwen/Qwen3-8B): "No Thinking Content in History: In multi-turn conversations, the historical model output should only include the final output part and does not need to include the thinking content. It is implemented in the provided chat template in Jinja2. However, for frameworks that do not directly use the Jinja2 chat template, it is up to the developers to ensure that the best practice is followed."
- **DeepSeek-V3.2 (arXiv:2512.02556v1 §3.2.1)**: "We observed that replicating DeepSeek-R1's strategy—discarding reasoning content upon the arrival of the second round of messages—results in significant token inefficiency. … Historical reasoning content is discarded only when a new user message is introduced to the conversation. If only tool-related messages (e.g., tool outputs) are appended, the reasoning content is retained throughout the interaction. When reasoning traces are removed, the history of tool calls and their results remains preserved in the context." The report adds that agent frameworks that simulate tool interactions through user messages "may not fully benefit" and recommends non-thinking models for them.
- **open-instruct@098424c `olmo_thinker_remove_intermediate_thinking`** (L427-471): for assistant messages that are not last, `content.split('</think>')[-1]` keeps only the text after the thinking span (L451-452). When this template renders a training example, earlier-turn thinking is absent from both the input and the labels.

## Connections

- [[read]] §1.2 (per-turn expansion), §6 (consistency), §7 (agentic history).
- [[minimax-m2-interleaved-thinking]], [[glm-5]], [[smol-training-playbook]] — three sources that retain thinking across turns.
