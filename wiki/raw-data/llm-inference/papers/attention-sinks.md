<!-- scope: StreamingLLM paper introducing attention sinks and stable windowed KV cache
     deps: transformer-attention
     see-also: [[h2o]], [[snapkv]], [[infllm]]
-->

# Efficient Streaming Language Models with Attention Sinks
- **Core Insight:** Sliding-window KV eviction fails unless initial "attention sink" tokens are kept; preserving a few sink tokens stabilizes long streaming generation.
- **Guideline:** For streaming contexts, keep initial sink-token KV plus a recent window rather than using a pure recent-token cache.
- **Authors:** Guangxuan Xiao, Yuandong Tian, Beidi Chen, Song Han, Mike Lewis
- **Year:** 2023
- **URL:** https://arxiv.org/abs/2309.17453
- **Relevant topics:** attention sinks, StreamingLLM, sliding window, KV eviction, infinite streaming

## Abstract
The paper studies LLM deployment in streaming settings where interactions can exceed the training context length. A naive sliding-window KV cache bounds memory but degrades sharply when early tokens are evicted. The authors identify attention sinks: initial tokens that receive large attention mass even when semantically unimportant. StreamingLLM keeps these sink tokens plus a rolling window, allowing stable generation over much longer streams without fine-tuning.

## Key Contributions
- Identifies the attention-sink phenomenon in pretrained LLMs.
- Shows pure window attention fails after sequence length exceeds cache size.
- Proposes keeping initial sink-token KV entries alongside recent-window KV.
- Demonstrates stable streaming on Llama-2, MPT, Falcon, and Pythia over very long streams.
- Suggests adding a dedicated sink token during pretraining for better streaming deployment.

## Key Figures/Tables to Study
- Attention-map visualizations showing heavy mass on initial tokens.
- Perplexity curves for sliding window versus StreamingLLM.
- Cache-layout diagram: sink tokens plus recent window.
- Long-stream evaluations beyond training context.

## Technical Details
The active KV cache is split into two retained sets: a small fixed prefix of sink tokens and the most recent window. Older middle tokens are evicted. This keeps memory bounded while preserving the attention distribution shape the model expects.

Unlike compression methods that choose important tokens per query or per head, attention sinks are mostly positional and stable. The method is simple to implement in a cache manager but changes what "sliding window" should mean for decoder-only models.

## Connections
- [[h2o]] and [[snapkv]] also evict KV entries, but choose them by attention importance.
- [[infllm]] extends long-context handling with memory retrieval rather than only a fixed window.

## Notes
The source is often cited under the system name StreamingLLM as well as the attention-sink phenomenon.
