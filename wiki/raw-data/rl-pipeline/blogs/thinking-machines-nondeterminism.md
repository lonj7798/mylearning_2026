<!-- scope: Thinking Machines Lab, "Defeating Nondeterminism in LLM Inference" (Horace He, Sep 2025): floating-point non-associativity, why atomics are not the cause, batch invariance, batch-invariant RMSNorm / matmul / attention (fixed split-size KV), vLLM demo, true on-policy RL experiment
     deps: [[grpo]]
     see-also: [[mai-rl-infrastructure]], [[nemo-rl-tp-accuracy]], [[flash-attention-stability]], [[deepseek-train-infer-consistency]], [[truncated-importance-sampling]], [[fp16-train-infer-mismatch]]
-->

# Defeating Nondeterminism in LLM Inference (Thinking Machines Lab)
- **Core Insight:** LLM inference is nondeterministic for a user mainly because kernels are not **batch-invariant** and the batch size depends on server load, not because of atomic adds racing on concurrent cores. The forward pass is "run-to-run deterministic", but a token's numerics change with how many other tokens share the batch.
- **Guideline:** Make every reduction kernel (RMSNorm, matmul, attention) use a reduction order that does not depend on batch size: one kernel configuration for all shapes (no split-K), and in attention a fixed split *size* (not a fixed number of splits) plus a KV cache layout updated before the kernel. With batch-invariant sampler kernels, the trainer can be matched bitwise and RL becomes truly on-policy (0 sampler–trainer KL).
- **Authors:** Horace He and Thinking Machines Lab.
- **Year:** 2025 ("Thinking Machines Lab: Connectionism, Sep 2025"; doi 10.64434/tml.20250910).
- **URL:** https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/
- **Source type:** lab engineering blog
- **Relevant topics:** floating-point non-associativity, batch invariance, split-K / split-KV, FlashDecoding, deterministic inference, sampler–trainer parity, on-policy RL

## Summary
At temperature 0, LLM APIs and open engines (vLLM, SGLang) still return different outputs. The common explanation is "concurrency + floating point". The post shows that this is incomplete: repeating the same `torch.mm` on the same BF16 data 1,000 times gives bitwise equal results, and "in the typical forward pass of an LLM, there is usually not a single atomic add present". The real cause is that kernels change their reduction strategy with batch size; since the batch size seen by a request depends on concurrent load, output depends on other users' requests. The post then shows how to build batch-invariant RMSNorm, matmul and attention, releases `thinking-machines-lab/batch-invariant-ops`, runs vLLM deterministically, and shows that a bitwise-matched sampler and trainer give a flat 0 KL in RL.

## Key Contributions
- Separates *run-to-run determinism* (same inputs, same kernel) from *batch invariance* (result for one element independent of batch size and composition).
- Identifies the three reduction ops that matter (RMSNorm, matmul, attention) and the batch-dependent strategies that break invariance (split reduction, split-K, stream-K, smaller tensor-core instructions, split-KV / FlashDecoding with a fixed number of splits).
- Fixed split-size Split-KV: e.g. KV length 1000 → three splits of 256 and one of 232, instead of four splits of 250.
- Demonstration on vLLM (FlexAttention backend + `torch.Library` operator substitution) and an RLVR experiment.

## Key Figures/Tables to Study
- Section "Batch invariance and 'determinism'": the matmul example where one row computed alone vs inside the full batch differs (`(out1 - out2).abs().max()` prints 1669.2500 for the post's linspace inputs).
- Section "How do we make kernels batch-invariant?": data-parallel vs split-reduction diagrams for RMSNorm, matmul, attention.
- Section "Experiments": completion counts and the runtime table; "True on-policy RL" KL plot.

## Technical Details
**Non-associativity.** (a + b) + c ≠ a + (b + c) in floating point; e.g. `(0.1 + 1e20) - 1e20` = 0 vs `0.1 + (1e20 - 1e20)` = 0.1. Summing one 8-element array in random orders 10,000 times gives 102 unique results.

**Why atomics are not the explanation.** Atomic adds are nondeterministic, but there is usually enough parallelism along the batch dimension, and libraries use split (tree) reductions with a clean-up step or semaphores. The one common LLM op that needs atomics for speed is FlashAttention backward; the Triton implementation avoids atomics "costing 40% more FLOPs". The forward pass contains none.

**Batch invariance by op.**
- *RMSNorm*: data-parallel (one row per core) is invariant; switching to split reduction at small batch breaks it. Simplest fix: ignore the small-batch case and accept slower kernels there.
- *Matmul*: tiles of the output are assigned to cores; with small M and N a split-K kernel is needed for occupancy (example: [1024, K] × [K, 1024] with [128, 128] tiles yields only 64 tiles). Stream-K is not even batch-position-invariant. Fix: "compile one kernel configuration and use that for all shapes"; measured loss "about 20% performance compared to cuBLAS" for an unoptimized Triton kernel.
- *Attention*: reduces over both feature and sequence; must give identical numerics whether a sequence is processed in chunks (chunked prefill, prefix cache) or at once. Handling cached and new KV separately changes block boundaries (block size 32, 80 cached + 48 new → 5 blocks instead of 4); fix by updating the KV cache and page table before the kernel. Decode needs Split-KV; FlashInfer's balanced scheduling picks split size from occupancy and is not invariant; fixed split size is.
- Parallelism reductions are out of scope, with one note: NVLink-Sharp in-switch reductions are deterministic on Blackwell and on Hopper with CUDA 12.8+.

**Experiments.**
- Qwen/Qwen3-235B-A22B-Instruct-2507, 1,000 completions at temperature 0, 1,000 tokens each, prompt "Tell me about Richard Feynman": 80 unique completions, most common 78 times; identical for the first 102 tokens; at token 103, 992 say "Queens, New York" and 8 say "New York City". With batch-invariant kernels all 1,000 are identical.
- Runtime, Qwen-3-8B on one GPU, 1,000 sequences of 90–110 output tokens: vLLM default 26 s; unoptimized deterministic vLLM 55 s; + improved attention kernel 42 s.
- RLVR on Bigmath from Qwen 2.5-VL instruct 8B, max rollout 4096. Without off-policy correction (importance weighting) reward collapses, with a loss spike around step 318 and a matching KL spike. With importance weighting KL stays "around 0.001 with occasional spikes". With bitwise-identical sampler and trainer, KL is a flat line at 0 and training is smooth.

**Relevance to RL infrastructure.** The post frames sampler–trainer numerical difference as what "implicitly turns our on-policy RL into off-policy RL"; [[mai-rl-infrastructure]] cites it for the TP row-parallel reduction issue and adopts a shared, fixed-split attention kernel across learner forward, sampler prefill and decode.

## Verification
Fetched the HTML with curl on 2026-11-24 and extracted the article text. All numbers are quoted from the article text (code comments and prose). Figures (diagrams and the RL reward/KL plots) were not read as images; plot values above are those stated in the prose.
