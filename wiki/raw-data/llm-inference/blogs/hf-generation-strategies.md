<!-- scope: Hugging Face generation controls for practical decoding
     deps: [[beam-search]], [[neural-text-degeneration]]
     see-also: [[language-models-are-unsupervised-multitask-learners]], [[gpt-3-language-models-are-few-shot-learners]]
-->

# Hugging Face Text Generation Strategies
- **Core Insight:** `generate()` exposes decoding as configuration: greedy, beam, sampling, top-k, top-p, temperature, penalties, and stopping controls are inference-time behavior.
- **Guideline:** Start from task intent: deterministic for extraction/translation-like tasks, sampled for open-ended generation, and always set explicit length/stopping limits.
- **Authors:** Hugging Face Transformers team; Patrick von Platen for the generation blog
- **Year:** 2020-present
- **URL:** https://huggingface.co/docs/transformers/generation_strategies ; https://huggingface.co/blog/how-to-generate
- **Relevant topics:** generate, GenerationConfig, greedy search, beam search, top-k, top-p, temperature

## Abstract
Hugging Face's generation documentation and blog explain how Transformers selects output tokens. The key practical message is that model weights do not fully determine outputs: decoding strategy and logits processors strongly shape quality, diversity, repetition, latency, and token count.

## Key Contributions
- Documents `GenerationConfig` as the central place for generation defaults.
- Shows greedy search as the default next-token argmax strategy.
- Exposes beam search with `num_beams`.
- Exposes multinomial sampling with `do_sample=True`.
- Connects top-k/top-p/temperature to distribution shaping.
- Provides a practical API vocabulary used by most open-source inference stacks.

## Key Figures/Tables to Study
- **Generation strategy guide:** Which parameter combination selects which decoding method.
- **How-to-generate blog examples:** Side-by-side greedy, beam, top-k, and top-p outputs.
- **GenerationConfig docs:** Length, sampling, repetition, and stopping parameters.
- **Logits processor list:** Where penalties and constraints enter before sampling.

## Technical Details
Common settings:

```text
greedy: do_sample=False, num_beams=1
beam:   do_sample=False, num_beams>1
sample: do_sample=True, num_beams=1
top-p:  do_sample=True, top_p<1.0
top-k:  do_sample=True, top_k>0
```

Temperature rescales logits before sampling. Top-k keeps a fixed number of candidates. Top-p keeps the smallest candidate set above a cumulative probability threshold. Repetition penalties and no-repeat n-gram processors modify logits before token selection.

For inference libraries, these controls affect not just text style but also runtime: `num_beams` multiplies active hypotheses; longer `max_new_tokens` increases decode steps; sampling can change expected output length.

## Connections
- [[neural-text-degeneration]]: top-p sampling comes directly from the degeneration diagnosis.
- [[beam-search]]: HF exposes beam search through `num_beams`, but open-ended use is risky.
- [[prefill-vs-decode]]: generation length controls decode duration.
- [[batching-for-inference]]: heterogeneous generation parameters complicate continuous batching.
