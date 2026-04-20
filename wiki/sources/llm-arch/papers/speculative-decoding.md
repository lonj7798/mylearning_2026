# Fast Inference from Transformers via Speculative Decoding
- **Authors:** Yaniv Leviathan, Matan Kalman, Yossi Matias
- **Year:** 2022
- **URL:** https://arxiv.org/abs/2211.17192
- **Core Insight:** Small draft model proposes, large model verifies in parallel -- mathematically equivalent sampling.
- **Guideline:** To accelerate autoregressive inference, use a small draft model to propose multiple tokens, then verify them in parallel with the large model. This produces identical output distributions while achieving 2-3x speedup by amortizing the cost of large model forward passes.
- **Relevant chapters:** Inference optimization, Decoding strategies, Serving systems, Latency reduction

## Abstract
Inference from large autoregressive models like Transformers is slow - decoding K tokens takes K serial runs of the model. In this work we introduce speculative decoding - an algorithm to sample from autoregressive models faster without any changes to the outputs, by computing several tokens in parallel. At the heart of our approach lie the observations that (1) hard language-modeling tasks often include easier subtasks that can be approximated well by more efficient models, and (2) using speculative execution and a novel sampling method, we can make exact decoding from the large models faster, by running them in parallel on the outputs of the approximation models, potentially generating several tokens concurrently, and without changing the distribution. Our method can accelerate existing off-the-shelf models without retraining or architecture changes. We demonstrate it on T5-XXL and show a 2X-3X acceleration compared to the standard T5X implementation, with identical outputs.

## Key Contributions
- Introduced speculative decoding: a small "draft" model generates candidate continuations, then the large "target" model verifies/rejects them in a single parallel forward pass
- Proved mathematically that the sampling method produces an identical output distribution to standard autoregressive decoding -- this is not an approximation
- Achieved 2-3x wall-clock speedup on T5-XXL without any retraining, architecture changes, or quality degradation
- Drew the analogy to speculative execution in CPU design, applying the same principle (speculate on likely outcomes, verify cheaply) to neural network inference
- Demonstrated that the method works with off-the-shelf models, requiring only a compatible smaller model for drafting

## Why This Paper Matters
Speculative decoding is one of the most practical inference optimization techniques available today. Because autoregressive generation is fundamentally bottlenecked by sequential forward passes (not compute), this technique exploits the fact that verification is parallelizable even when generation is not. It is now widely deployed in production LLM serving systems (vLLM, TensorRT-LLM, etc.) and has spawned a family of related techniques (Medusa, EAGLE, staged speculative decoding). The key insight -- that a small model's guesses are often correct and a large model can verify cheaply -- is elegant and broadly applicable.
