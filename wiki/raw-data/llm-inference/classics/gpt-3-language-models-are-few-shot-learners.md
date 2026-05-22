<!-- scope: GPT-3 and in-context few-shot inference as a scaling result
     deps: [[language-models-are-unsupervised-multitask-learners]]
     see-also: [[hf-generation-strategies]], [[openai-streaming-and-token-usage]], [[prefill-vs-decode]]
-->

# Language Models are Few-Shot Learners
- **Core Insight:** Scaling autoregressive language models enables in-context learning: task examples placed in the prompt can steer behavior without gradient updates.
- **Guideline:** Budget prompt tokens as part of the inference program; demonstrations can improve accuracy but increase prefill latency, KV-cache memory, and cost.
- **Authors:** Tom B. Brown et al. (OpenAI)
- **Year:** 2020
- **URL:** https://arxiv.org/abs/2005.14165
- **Relevant topics:** GPT-3, few-shot prompting, in-context learning, scaling, prompt length

## Abstract
GPT-3 scales decoder-only language modeling to 175B parameters and evaluates tasks in zero-shot, one-shot, and few-shot settings. Instead of fine-tuning on each task, the model receives instructions and demonstrations in the context window. The paper shows broad gains from scale and few-shot prompting while documenting weaknesses in reasoning, calibration, and some specialized tasks.

## Key Contributions
- Established few-shot prompting as a central inference pattern for LLMs.
- Compared zero-shot, one-shot, and few-shot performance across many benchmarks.
- Scaled dense decoder-only Transformers to 175B parameters.
- Showed smooth scaling trends for loss and broad task performance.
- Made prompt design a first-class evaluation variable.

## Key Figures/Tables to Study
- **Figure 1.1:** Few-shot prompting setup: instruction plus examples plus query.
- **Figure 3.1:** Test loss improves predictably with model size and compute.
- **Benchmark tables:** Compare zero-shot vs one-shot vs few-shot.
- **Appendix prompt examples:** Useful raw material for understanding prompt serialization.

## Technical Details
At inference, GPT-3 receives a context such as:

```text
Task instruction
Input: example 1
Output: label 1
Input: example 2
Output: label 2
Input: query
Output:
```

The model does not update weights. It conditions on the examples through self-attention and continues the sequence. This makes task adaptation a context-management problem.

Few-shot prompts trade accuracy against latency and memory. More examples increase prompt tokens, which increases prefill compute roughly quadratically for full attention and increases KV-cache memory linearly for subsequent decoding.

## Connections
- [[language-models-are-unsupervised-multitask-learners]]: GPT-2 established zero-shot prompting; GPT-3 made few-shot prompting central.
- [[prefill-vs-decode]]: long few-shot prompts make prefill cost visible.
- [[kv-cache-memory-formula]]: every prompt token remains as cached KV state during decode.
- [[openai-streaming-and-token-usage]]: API token accounting separates input and output tokens for this tradeoff.
