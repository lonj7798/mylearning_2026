<!-- scope: GPT-2 as zero-shot multitask language-modeling foundation
     deps: [[attention-is-all-you-need]]
     see-also: [[gpt-3-language-models-are-few-shot-learners]], [[neural-text-degeneration]], [[hf-generation-strategies]]
-->

# Language Models are Unsupervised Multitask Learners
- **Core Insight:** Large autoregressive language models trained only on next-token prediction over diverse web text can perform many NLP tasks zero-shot through natural-language prompting.
- **Guideline:** For inference, prompts are the task interface; model behavior is shaped by context, decoding settings, and pretraining distribution rather than task-specific heads.
- **Authors:** Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever
- **Year:** 2019
- **URL:** https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf
- **Relevant topics:** GPT-2, prompting, zero-shot evaluation, WebText, autoregressive decoding

## Abstract
GPT-2 scales the decoder-only Transformer trained on next-token prediction and evaluates it across language modeling, question answering, summarization, translation, and reading comprehension without supervised fine-tuning. The paper argues that sufficiently broad unsupervised language modeling induces task behavior that can be elicited by text context.

## Key Contributions
- Demonstrated strong zero-shot transfer from next-token prediction to multiple NLP benchmarks.
- Introduced WebText, a web-scale dataset filtered by outbound Reddit links as a quality prior.
- Scaled GPT-style decoder-only Transformers up to 1.5B parameters.
- Framed downstream tasks as text-to-text patterns in the input context.
- Highlighted both the promise and risk of general-purpose text generation.

## Key Figures/Tables to Study
- **Table 2:** Model sizes from 117M to 1.5B parameters.
- **Language modeling benchmark tables:** Zero-shot perplexity/accuracy gains across datasets.
- **Task examples:** Prompt formats for translation, QA, and summarization.
- **Release discussion:** Important historical context for staged model release and misuse concerns.

## Technical Details
GPT-2 is a causal decoder-only Transformer: each token predicts the next token under a left-to-right language modeling loss. At inference, the model conditions on all tokens in the prompt and samples or searches one next token at a time.

The important inference lesson is that prompt text acts as implicit task specification. There is no separate classifier head or task adapter in the zero-shot setup. The same next-token distribution must serve completion, answer extraction, translation, and summarization depending on context.

The paper predates modern chat templates, but the same mechanism underlies system/user/assistant formatting: a serialized conversation is still a token sequence, and generation continues from that sequence.

## Connections
- [[gpt-3-language-models-are-few-shot-learners]]: extends prompt-based inference to in-context examples at much larger scale.
- [[neural-text-degeneration]]: GPT-2 exposed the need for better open-ended decoding than plain greedy or beam search.
- [[hf-generation-strategies]]: practical knobs for controlling GPT-style generation.
- [[prefill-vs-decode]]: prompt processing is the prefill phase; continuation is decode.
