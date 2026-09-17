---
chapter: ch-04
course: llm-training
phase: read
excerpt_of: huggingface/transformers — chat template documentation, NEFTune hook, and embedding resizing
source_url: https://github.com/huggingface/transformers/tree/05e078a1d276d0f9dcfc08d1f2964cb8abb33b04
created_at: "2026-09-15"
note: "Documentation and released code read at commit 05e078a (2026-09-01). Line numbers refer to that commit."
---

# Excerpt: Transformers chat templates, NEFTune hook, and `resize_token_embeddings` (commit 05e078a)

## `docs/source/en/chat_templating.md`

- Chat models receive messages converted to a token sequence "often with control tokens like `<|user|>` or `<|assistant|>` or `<|end_of_message|>`"; "different models may use different formats or control tokens, even if they were fine-tuned from the same base model" (L23-26).
- Mistral-7B-Instruct-v0.1 renders `<s>[INST] … [/INST]…</s>`, while Zephyr-7B-β renders `<|user|>\n…</s>\n<|assistant|>\n…` for the same chat; "with the wrong control tokens, these models would have drastically worse performance" (L34-74; documentation statement without numbers).
- Warning: "Some tokenizers add special `<bos>` and `<eos>` tokens. Chat templates should already include all the necessary special tokens, and adding additional special tokens is often incorrect or duplicated … make sure you set `add_special_tokens=False` if you tokenize later" (L125-127).
- `add_generation_prompt=True` appends the tokens that start an assistant message, e.g. `<|im_start|>assistant\n`; without them the model may continue the user's message. For templates without an assistant prefix, such as Llama's, the argument has no effect (L129-171).
- `continue_final_message` removes end-of-sequence tokens so generation continues the final message; reasoning models may expose a separate field (`reasoning_content` on Qwen, `thinking` on Gemma) that can be prefilled (L173-200).

## `docs/source/en/chat_templating_writing.md`

- A chat template is a Jinja template stored in the tokenizer's `chat_template` attribute and saved as `chat_template.jinja` (L19, L36-37).
- "The chat template should always match the format the model was trained with" (L86); extra whitespace not present in training "can harm performance" (L90).
- When a repository has both `chat_template.jinja` and a `chat_template` field in `tokenizer_config.json`, the standalone `.jinja` file takes priority (L168).

## `src/transformers/integrations/neftune.py`

```python
# L47-L51
if module.training:
    dims = torch.tensor(output.size(1) * output.size(2))
    mag_norm = module.neftune_noise_alpha / torch.sqrt(dims)
    output = output + torch.zeros_like(output).uniform_(-mag_norm, mag_norm)
return output
```

`output.size(1)` is the sequence dimension of the embedding output, so the scale uses the row length of the batch tensor. The NEFTune paper computes the scale per sequence when lengths differ ([[neftune]], Algorithm 1 footnote).

## `src/transformers/modeling_utils.py` — `resize_token_embeddings`

- Signature default `mean_resizing: bool = True` (L2714).
- "Whether to initialize the added embeddings from a multivariate normal distribution that has old embeddings' mean and covariance or to initialize them with a normal distribution that has a mean of zero and std equals `config.initializer_range`. Setting `mean_resizing` to `True` is useful when increasing the size of the embeddings of causal language models, where the generated tokens' probabilities won't be affected by the added embeddings because initializing the new embeddings with the old embeddings' mean will reduce the kl-divergence between the next token probability before and after adding the new embeddings." (L2734-2741; the docstring cites nlp.stanford.edu/~johnhew/vocab-expansion.html).
- The output head is resized with the same `mean_resizing` setting when it is not tied (L2796-2798).
