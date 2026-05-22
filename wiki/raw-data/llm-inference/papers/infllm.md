<!-- scope: InfLLM training-free long-context method with efficient context memory
     deps: [[attention-sinks]]
     see-also: [[snapkv]], [[quest-kv]], [[cachegen]]
-->

# InfLLM: Training-Free Long-Context Extrapolation for LLMs with an Efficient Context Memory
- **Core Insight:** A pretrained LLM can handle much longer streams by retrieving relevant context memory rather than extending the full active KV cache.
- **Guideline:** For extreme context streams, combine bounded local attention with memory retrieval instead of keeping every prior token active.
- **Authors:** Chaojun Xiao, Pengle Zhang, Xu Han, Guangxuan Xiao, Yankai Lin, Zhengyan Zhang, Zhiyuan Liu, Maosong Sun
- **Year:** 2024
- **URL:** https://arxiv.org/abs/2402.04617
- **Relevant topics:** long-context extrapolation, training-free memory, streaming, KV memory, retrieval

## Abstract
InfLLM addresses the mismatch between finite training context and very long streaming inputs. Instead of continual pretraining, it introduces a training-free memory method that stores and retrieves context information so the model can reason over much longer sequences. The method aims to reduce distraction and out-of-domain effects from simply extending attention over all tokens.

## Key Contributions
- Proposes training-free long-context extrapolation for existing LLMs.
- Uses an efficient context memory to support very long input streams.
- Retrieves relevant memory rather than attending over all prior tokens.
- Avoids changing model weights or requiring long-context fine-tuning.
- Evaluates on long-context tasks with sequence lengths beyond standard windows.

## Key Figures/Tables to Study
- Method diagram for local context plus memory retrieval.
- Long-context benchmark results as context length scales.
- Ablations on memory size/retrieval policy.
- Comparisons with sliding-window and other training-free baselines.

## Technical Details
InfLLM can be understood as moving some context from the active attention window into an external memory. At generation time, the model attends to recent/local tokens and retrieved memory relevant to the current state. This bounds active KV work while retaining access to older information.

For serving, the important question is where memory representations live and how retrieval interacts with KV-cache layout. Unlike pure cache eviction, InfLLM relies on recovering relevant older context through a retrieval mechanism.

## Connections
- [[attention-sinks]] provides the sink-plus-window baseline for stable streaming.
- [[snapkv]] and [[quest-kv]] reduce active KV through selection; InfLLM adds retrieval-style memory.
- [[cachegen]] is relevant if stored memory/KV must be loaded across systems.

## Notes
InfLLM is best read as a long-context memory method, not a serving scheduler by itself.
