---
chapter: ch-47
course: llm-training
phase: read
excerpt_of: primary source Thinking Machines Lab blog post, Sep 2025 (no library card as of 2026-09-15)
source_url: https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/
created_at: "2026-09-15"
---

# Excerpt: Defeating Nondeterminism in LLM Inference

**Post:** Horace He and Thinking Machines Lab, "Defeating Nondeterminism in LLM Inference", Thinking Machines Lab: Connectionism, September 2025. Source type: official blog of the organization that wrote the kernels (practitioner evidence for the measurements).

## Claim and mechanism ("Batch invariance", "How do we make kernels batch-invariant?")
- The common explanation for nondeterministic inference is floating-point non-associativity combined with concurrent execution. The post argues this is not the cause for LLM inference: the forward pass contains no atomic adds and is run-to-run deterministic for a fixed input batch.
- The cause given is lack of *batch invariance*: matrix multiplication, RMSNorm, and attention kernels change their reduction order with the batch size, so an individual request's numerics depend on how many other requests share its batch. Server load varies, so from a user's perspective the result is nondeterministic even at temperature 0.
- Fix: make the reduction order for each element fixed regardless of batch size, for the three reduction operations (RMSNorm, matmul, attention).

## Measurements ("How nondeterministic are completions?", "Performance", "True on-policy RL")
- Qwen/Qwen3-235B-A22B-Instruct-2507, non-thinking mode, prompt "Tell me about Richard Feynman", temperature 0, 1,000 completions of 1,000 tokens each: 80 unique completions; the most common appears 78 times; all completions agree for the first 102 tokens; at the point of divergence 992 continue with "Queens, New York" and 8 with "New York City".
- With the batch-invariant kernels enabled, all 1,000 completions are identical.
- Throughput, Qwen3-8B on one GPU, 1,000 sequences of 90-110 output tokens: vLLM default 26 s; unoptimized deterministic 55 s; with an improved attention kernel 42 s.
- RLVR on Bigmath, policy initialized from Qwen 2.5-VL instruct 8B, maximum rollout length 4,096: without an importance-weighting correction the reward collapses partway through training; with the correction, or with bitwise-identical sampler and trainer, training proceeds. The KL divergence in log-probabilities between sampler and trainer stays near 0.001 with importance weighting and at 0 with bitwise-identical numerics.

## Verification
- Read on 2026-09-15 against the cached text of the post (sections listed above). The post is not peer reviewed and reports single runs.
