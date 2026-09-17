---
chapter: ch-02
course: llm-training
phase: read
excerpt_of: "Horace He and Thinking Machines Lab — Defeating Nondeterminism in LLM Inference (Connectionism blog, 2025-09-10)"
source_url: https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/
created_at: "2026-09-15"
revised: "2026-09-15 (created from the published post; generality revision)"
---

# Excerpt: Defeating Nondeterminism in LLM Inference (Thinking Machines Lab, 2025-09)

Created on 2026-09-15 from the post at the URL above (doi 10.64434/tml.20250910). Source type: official blog
of the lab that ran the experiments, with released code (thinking-machines-lab/batch-invariant-ops). The
library card with this slug was planned but had not been written when this excerpt was created. Section names
below are the post's headings.

## Claims about the cause
- Floating-point addition is not associative: `(0.1 + 1e20) - 1e20` gives 0 while `0.1 + (1e20 - 1e20)` gives
  0.1; summing the 8 values ±1e-10, ±1e-5, ±1e-2, ±1 in 10,000 random orders gives 102 distinct results
  ("The original sin: floating-point non-associativity").
- The same matmul on the same data repeated 1000 times on a GPU returns bitwise-equal results; the forward pass
  of an LLM contains no operation that needs atomic adds, so it is "run-to-run deterministic". FlashAttention
  backward is named as the common LLM operation that needs atomic adds or extra recomputation ("When are atomic
  adds needed?").
- Kernels are not batch-invariant: `torch.mm(a[:1], b)` and `torch.mm(a, b)[:1]` differ by 1669.25 in the
  printed example. Because server load changes the batch size, a user's output depends on other requests
  ("Batch invariance and 'determinism'").
- Only three reduction operations need changes: RMSNorm, matrix multiplication, and attention. A kernel breaks
  batch invariance when the batch size changes its reduction strategy (split reductions, a different
  tensor-core instruction, or split-KV attention that divides the KV length into a number of splits that
  depends on the request) ("How do we make kernels batch-invariant?").
- Attention must also be invariant to how a sequence is chunked (chunked prefill, prefix caching); the fix
  updates the KV cache before the attention kernel and uses a fixed split size rather than a fixed number of
  splits ("Batch-invariant attention").

## Experiments
- Qwen/Qwen3-235B-A22B-Instruct-2507, prompt "Tell me about Richard Feynman", 1000 completions of 1000 tokens
  at temperature 0: 80 unique completions (most common 78 times). All agree for 102 tokens; at token 103, 992
  continue "Queens, New York" and 8 continue "New York City". With batch-invariant kernels all 1000 completions
  are identical ("How nondeterministic are completions?").
- Cost, Qwen-3-8B on one GPU, 1000 sequences of 90-110 output tokens: vLLM default 26 s; unoptimized
  deterministic vLLM 55 s; with improved attention kernel 42 s ("Performance"). The batch-invariant matmul loses
  "about 20% performance compared to cuBLAS" ("Batch-invariant matrix multiplication").
- RL ("True on-policy RL"): "the different numerics between training and inference implicitly turns our
  on-policy RL into off-policy RL." RLVR on Bigmath, policy initialized from Qwen 2.5-VL instruct 8B, maximum
  rollout length 4096:
  - without off-policy correction (importance weighting), reward collapses partway through training, with a
    loss spike and a sampler-trainer KL spike around step 318;
  - with importance weighting, sampler-trainer KL "stays around 0.001 with occasional spikes" and training
    proceeds;
  - with bitwise-identical sampler and trainer, KL stays at 0 and training proceeds.
- Numbers of seeds, evaluation accuracies, and the importance-weighting variant are not reported in the post.

## Used in ch-02
- §6 (cause of evaluation nondeterminism, batch invariance, cost), §7 (bitwise on-policy RL result), Common
  mistakes.
