---
chapter: ch-30
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/model-reports/llama-3.md
source_url: https://arxiv.org/abs/2407.21783
created_at: "2026-04-23"
---

# Excerpt: Chat templates as frozen contracts with the base model

**Source libraries:**
- `wiki/raw-data/llm-training/model-reports/llama-3.md`
- `wiki/raw-data/llm-training/model-reports/qwen-3.md`
- `wiki/raw-data/llm-training/model-reports/deepseek-v3.md`
- `wiki/raw-data/llm-training/model-reports/deepseek-r1.md`

**Practitioner voice:** [[hf-alignment-handbook]] — "verify chat template by decoding a packed batch — template mismatch is the #1 silent bug."

---

## Why this source set anchors ch-30

Chat templates are axis #3 of the five SFT design axes, and the only axis where the decision is not a hyperparameter sweep — it is a lookup. Each base model ships with a frozen set of delimiter tokens that its continued-pretraining or prior post-training already conditioned on. Using a different set during SFT is mathematically possible and practically silent-broken: loss trains fine, downstream model refuses to emit the end-of-turn token because its config lists a different one as the stop token.

This excerpt is an explicit side-by-side of four families' templates, pulled verbatim from their model reports and tokenizer configs, so ch-30's matrix is traceable.

---

## Llama-3 — header + eot_id

From `llama-3.md`, §Post-Training Pipeline / SFT:

> **Training:** LR 1e-5 (405B), cosine decay, context 8K-32K (extended), loss on response tokens only.

The Llama-3 chat template is not quoted in prose inside the technical report (it lives in the tokenizer's `tokenizer_config.json`), but the format is attested in the model card and reproduced here verbatim:

```
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are helpful.<|eot_id|><|start_header_id|>user<|end_header_id|>

Hi<|eot_id|><|start_header_id|>assistant<|end_header_id|>

Hello.<|eot_id|>
```

## Notice: the double newline after `<|end_header_id|>` is load-bearing

The blank line between the role header and the turn body is *part of the template*, not cosmetic whitespace. It is the literal two-character sequence `\n\n` that the tokenizer emits as specific tokens. If you hand-build the string with only one newline, the tokenization differs and the base model has never been conditioned on the modified delimiter. This is exactly the class of silent bug that [[hf-alignment-handbook]] warns about.

## Llama-3.1 tool extension — `<|python_tag|>`

From `llama-3.md`, §Key Contributions:

> Heavy synthetic-data generation for coding, math, multilingual, reasoning, long-context, tool use, and factuality — each capability gets a dedicated synthetic pipeline.

Llama-3.1 added `<|python_tag|>` (single token) as a tool-call marker, appearing inside the assistant turn. It is not its own role; it is a *marker* inside an assistant span. The mask row for tool-call in ch-30's matrix ("assistant current → Y; tool_call → Y") treats `<|python_tag|>...` as part of the assistant-current span — the model emits it, so it gets loss.

---

## Qwen-3 — ChatML with native thinking tokens

From `qwen-3.md`, §Post-training:

> Stage 1-2: build reasoning ability with **long-CoT cold-start finetuning** and **RL** focused on math and coding.
> Stage 3-4: merge data with and without reasoning paths, then run **general-domain RL**.

Qwen inherits ChatML's `<|im_start|>role\n ... <|im_end|>` delimiters and adds `<think>...</think>` as an assistant-scoped span:

```
<|im_start|>user
Solve 2+2<|im_end|>
<|im_start|>assistant
<think>2 plus 2 is 4.</think>4<|im_end|>
```

Notice what this template does architecturally. The thinking content sits *inside* the assistant turn, between `<|im_start|>assistant\n` and `<|im_end|>`. The `<think>` and `</think>` tags are emitted by the model as ordinary tokens; they carry loss in reasoning SFT (see ch-30's mask table — `<think>` column is `Y` for the reasoning regime).

Qwen's "hybrid thinking" design follows directly: the model can be prompted to emit a `<think>...</think>` span or not, so a single model supports both fast and deep modes (Qwen's §Core Insight). The template accommodates both; the training data's ratio of thinking-vs-not samples determines which mode the model prefers at inference.

---

## DeepSeek-V3 — half-width Unicode delimiters

From `deepseek-v3.md`, §Post-training:

> Runs **SFT + RL** on the base model.
> Distills reasoning behaviors from a **DeepSeek-R1-series long-CoT model** into DeepSeek-V3.

DeepSeek uses half-width Unicode pipes and triangular brackets as delimiters:

```
<｜begin▁of▁sentence｜><｜User｜>Hi<｜Assistant｜>Hello.<｜end▁of▁sentence｜>
```

These are literal Unicode tokens (`U+FF5C` for `｜`; `U+2581` for `▁`). The visual similarity to ASCII pipes and spaces is intentional — the DeepSeek tokenizer emits them as distinct single tokens that no other family shares. This is a subtle protection against template leakage: if you copy a DeepSeek-templated string into a Llama-3 SFT run, the Llama-3 tokenizer will split the delimiters into many subword pieces and the resulting gibberish is visually obvious in the loss curve (step-1 loss far above `ln(V)`).

---

## DeepSeek-R1 — the `<think>`/`<answer>` template chosen for reward parseability

From `deepseek-r1.md`:

> **Template:** `A conversation between User and Assistant... Assistant reasons inside <think>...</think> and answers inside <answer>...</answer>.`

Notice the reasoning here: R1's RL uses **rule-based rewards only** (accuracy + format). The accuracy reward needs the reward parser to find the final answer; the format reward enforces that the answer appears inside `<answer>...</answer>`. The template is not aesthetic; it is the interface the reward function consumes. This is what ch-30 §7 calls "the template pre-determines what RL can later fix" — if R1 had used a free-form template, the rule-based reward could not have been evaluated.

---

## The verification procedure — `apply_chat_template` round-trip

From [[hf-alignment-handbook]], §Lessons captured in the handbook:

> Always verify chat template by decoding a packed batch — template mismatch is the #1 silent bug.

The attested procedure:

```python
messages = [{"role": "user", "content": "Hi"}]
ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
print(tokenizer.decode(ids))
# Confirm every role header, every delimiter, every newline is what you expect.
```

Ch-30's HTML companion renders the verbatim template for each family so this check can be done visually before the real run.

---

## What ch-30 keeps, changes, drops

| Model family source | Ch-30 matrix row | Reason |
|---------------------|------------------|--------|
| Llama-3 report + tokenizer_config | "Llama-3" row verbatim | The delimiters are frozen contracts |
| Qwen-3 report | "Qwen-3" row with `<think>` column populated | Hybrid thinking is the attested design |
| DeepSeek-V3 report | "DeepSeek-V3" row with half-width pipes | Verbatim from the model card |
| DeepSeek-R1 template | Mentioned as the sibling that adds `<think>`/`<answer>` | Shows template-reward coupling |

---

## Connections

- [[excerpts/loss-masking-regimes]] — the template decides which token spans are "system / user / assistant"; the mask decides which get loss.
- [[excerpts/sft-pre-determines-rl]] — template choice bounds RL; R1 is the clearest example.
- [[ch-30]] — §3 and the HTML companion's template renderer both depend on this excerpt.
- [[ch-32]] (reasoning SFT) — extends `<think>` coverage; Qwen and R1 templates are the reference.
- [[ch-33]] (tool-call SFT) — extends `<tool_call>` coverage; Qwen-3 and Llama-3.1 are the reference.
