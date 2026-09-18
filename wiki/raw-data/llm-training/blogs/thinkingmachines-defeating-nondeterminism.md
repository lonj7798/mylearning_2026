<!-- scope: the cause of run-to-run nondeterminism in LLM inference servers (lack of batch invariance in reduction kernels), how to build batch-invariant RMSNorm / matmul / attention, their cost, and the consequence for on-policy RL
     see-also: [[rollout-training-mismatch-tis]], [[async-rollout]], [[verl-rollout]]
-->

# Defeating Nondeterminism in LLM Inference
- **Core Insight:** LLM inference endpoints are nondeterministic because their reduction kernels are not batch-invariant and server load changes the batch size, not because of "concurrency + floating point": the forward pass of an LLM uses no atomic adds and is run-to-run deterministic ("When are atomic adds needed?"), yet 1000 temperature-0 completions of one prompt from Qwen3-235B-A22B-Instruct-2507 produced 80 unique completions under default kernels and one identical completion 1000 times under batch-invariant kernels ("Experiments").
- **Guideline:** When a result must be reproducible across server load — an eval at temperature 0, or an RL loop whose sampler and trainer must agree — use batch-invariant kernels (fixed matmul tile configuration, fixed split-size Split-KV attention, KV cache updated before the attention kernel), accepting about 20% matmul slowdown versus cuBLAS and 55s versus 26s end to end on the reported Qwen-3-8B serving test ("Batch-invariant matrix multiplication", "Performance"). Otherwise, correct the sampler-trainer mismatch with importance weighting, because the reported run without correction collapsed in reward ("True on-policy RL").
- **Authors:** Horace He and Thinking Machines Lab
- **Year:** 2025 (published 2025-09-10; Thinking Machines Lab: Connectionism; doi 10.64434/tml.20250910)
- **URL:** https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/
- **Source type:** official blog (the organization that produced the kernels and ran the experiments; reliability: official for its own experiments, practitioner-evidence for the general kernel claims)
- **Relevant topics:** inference determinism, batch invariance, floating-point non-associativity, atomic adds, split-K matmul, FlashDecoding / Split-KV attention, vLLM, sampler-trainer mismatch, on-policy RL, importance weighting, RLVR

## Summary
The post rejects the common "concurrency + floating point" explanation of LLM inference nondeterminism. Floating-point non-associativity is the reason numerical differences exist at all, but on a GPU the same matmul on the same data returns bitwise-equal results every time, and the forward pass of an LLM contains no atomic adds, so it is run-to-run deterministic. The nondeterminism a user observes comes from batch invariance: a kernel's reduction strategy depends on the batch size, server load determines the batch size, and load is nondeterministic from any single user's perspective. The post then shows how to make the three reducing operations — RMSNorm, matrix multiplication, and attention — batch-invariant, measures the resulting completion determinism and slowdown on vLLM, and applies the same machinery to RL, where sampler and trainer numerics differing turns nominally on-policy RL into off-policy RL.

## Key Contributions
- Separates three claims that are all true at once: some GPU kernels are nondeterministic; every kernel in an LLM forward pass is deterministic; the inference server is deterministic given its exact requests; and results are still nondeterministic from a user's perspective ("Batch invariance and 'determinism'").
- Identifies lack of batch invariance as the cause, and notes it is not GPU-specific — CPU- and TPU-served endpoints have the same source ("Batch invariance and 'determinism'").
- Gives the per-operation recipe for batch invariance: data-parallel RMSNorm with a fixed reduction strategy, one compiled matmul kernel configuration for all shapes, and fixed-split-size Split-KV attention with the KV cache and page table updated before the kernel.
- Releases batch-invariant kernels (`thinking-machines-lab/batch-invariant-ops`) and a deterministic-mode vLLM example built on vLLM's FlexAttention backend and `torch.Library` ("Implementation").
- Demonstrates true on-policy RL: with bitwise-identical sampler and trainer, sampler-trainer KL divergence is flat at 0 ("True on-policy RL").

## Key Figures/Tables to Study
- The RMSNorm data-parallel versus split-reduction diagrams, and the matmul data-parallel versus Split-K diagrams — they show exactly which condition (batch too small to saturate cores) breaks invariance.
- The FlashAttention-with-KV-cache boundary-condition figure (block size 32, 80 cached elements plus 48 new) and the fixed-number versus fixed-size Split-KV comparison.
- The performance table under "Performance" (three configurations, seconds).
- The two "True on-policy RL" plots: reward versus step for three runs, and sampler-trainer logprob KL versus step for the same three runs.

## Technical Details
- **Floating-point non-associativity:** `(0.1 + 1e20) - 1e20` gives 0 while `0.1 + (1e20 - 1e20)` gives 0.1; summing the 8-element array `[1e-10, 1e-5, 1e-2, 1]` and its negations in 10,000 random orders yields 102 unique results ("The original sin").
- **Atomic adds:** a 2048×2048 bf16 `torch.mm` repeated 1000 times is bitwise identical every run ("The original sin"). Atomics are avoided by parallelism along the batch dimension and by split/tree reductions with a clean-up pass or semaphore. The only commonly used LLM operation with a real penalty for avoiding them is FlashAttention backward; the standard Triton FlashAttention backward avoids atomics by recomputation at the cost of 40% more FLOPs ("When are atomic adds needed?").
- **Batch non-invariance, demonstrated:** with `a` of shape [2048, 4096] and `b` of shape [4096, 4096] from `torch.linspace(-1000, 1000, ...)`, `torch.mm(a[:1], b)` and `torch.mm(a, b)[:1]` differ by a maximum absolute value of 1669.25 ("Batch invariance and 'determinism'"). Each result is itself run-to-run deterministic; it is not hardware- or library-version invariant.
- **Which operations matter:** pointwise operations are assumed batch-invariant, so only the three reducing operations need work — RMSNorm, matrix multiplication, attention ("How do we make kernels batch-invariant?"). NVLink-Sharp in-switch reductions are stated to be deterministic on Blackwell and on Hopper with CUDA 12.8+.
- **RMSNorm:** invariance requires the reduction order per element to be fixed regardless of batch size; the break occurs when a small batch leaves cores idle and the kernel switches to atomics or split reductions. The recommended handling is to ignore the small-batch case or to use one reduction strategy with enough parallelism at all sizes ("Batch-invariant RMSNorm").
- **Matmul:** breaks come from Split-K (needed when M and N are small) and from switching tensor-core instruction sizes (for example `wgmma.mma_async.sync.aligned.m64n128k16`) at small batch. The fix is to compile one kernel configuration and use it for all shapes; the measured cost is "about 20% performance compared to cuBLAS", with larger loss at very small batch sizes and a jigsaw pattern from tile and wave quantization ("Batch-invariant matrix multiplication"). Stream-K is noted as not even batch-position-invariant.
- **Attention:** must be invariant both to how many requests are processed at once and to how each request is sliced (chunked prefill, prefix caching). Reducing over cached K/V separately from current K/V breaks invariance through block boundary conditions; the fix is to update the KV cache and page table before the attention kernel. Split-KV / FlashDecoding is unavoidable at decode-time query lengths, so invariance requires a fixed split *size* rather than a fixed split *count*: for a KV length of 1000, three splits of 256 and one of 232 instead of four of 250. FlashInfer's balanced scheduling algorithm, which picks the largest split size that saturates the cores, is named as not batch-invariant ("Batch-invariant attention"). The required FlexAttention changes were not in the code release at publication.
- **Completion experiment:** Qwen/Qwen3-235B-A22B-Instruct-2507, non-thinking mode, prompt "Tell me about Richard Feynman", 1000 completions at temperature 0, 1000 tokens each. 80 unique completions, the most common occurring 78 times; all identical for the first 102 tokens; the first divergence is at token 103, where 992 completions continue "Queens, New York" and 8 "New York City". With batch-invariant kernels all 1000 completions are identical ("How nondeterministic are completions?").
- **Performance experiment:** one GPU serving Qwen-3-8B, 1000 sequences of output length 90-110. vLLM default 26 s; unoptimized deterministic vLLM 55 s; with the improved attention kernel 42 s. The post attributes most of the gap to vLLM's unoptimized FlexAttention integration ("Performance").
- **RL experiment:** RLVR on Bigmath, policy initialized from Qwen 2.5-VL instruct 8B, maximum rollout length 4096. Without off-policy correction (importance weighting), "our reward collapses partway through training", with a significant loss spike around step 318 and a matching spike in sampler-trainer logprob KL. With importance weighting, that KL "stays around 0.001 with occasional spikes" and training proceeds. With bitwise-identical sampler and trainer, the run is "fully on policy (i.e. 0 KL divergence)", the KL trace is flat at 0, and training also proceeds ("True on-policy RL").
- **Framing of the RL problem:** "the different numerics between training and inference implicitly turns our on-policy RL into off-policy RL" ("True on-policy RL").

## Recipe ledger
| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| Qwen 2.5-VL instruct (policy init) | 8B | RL | RLVR dataset | Bigmath | blog §"True on-policy RL" | verified 2026-09-18 | no ablation reported |
| Qwen 2.5-VL instruct (policy init) | 8B | RL | max rollout length | 4096 tokens | blog §"True on-policy RL" | verified 2026-09-18 | no ablation reported |
| Qwen 2.5-VL instruct (policy init) | 8B | RL | off-policy correction | three runs compared: none / importance weighting / bitwise-identical sampler and trainer | blog §"True on-policy RL" | verified 2026-09-18 | reward collapse without correction; loss and KL spike near step 318 |
| Qwen 2.5-VL instruct (policy init) | 8B | RL | sampler-trainer logprob KL | ~0.001 with occasional spikes (importance weighting); 0 (bitwise identical) | blog §"True on-policy RL" | verified 2026-09-18 | the KL-versus-step plot for the three runs |
| Qwen3-235B-A22B-Instruct-2507 | 235B total / 22B active | eval-gate | sampling for the determinism test | temperature 0, 1000 completions, 1000 tokens each, non-thinking mode | blog §"How nondeterministic are completions?" | verified 2026-09-18 | 80 unique completions default, 1 with batch-invariant kernels |
| Qwen-3-8B | 8B | eval-gate | serving benchmark | 1000 sequences, output length 90-110, one GPU | blog §"Performance" | verified 2026-09-18 | 26 s / 55 s / 42 s across three configurations |

Not reported: RL algorithm name, learning rate, batch size, number of steps trained, group size, reward definition, GPU type, and the absolute reward values in the plots.

## Findings relevant to generality and to negative feedback
- Measurement: a temperature-0 evaluation is not a deterministic measurement under default kernels, so a before-and-after comparison of two checkpoints mixes the model change with batch-composition noise ("Experiments").
- RL stability: uncorrected sampler-trainer numerical mismatch was followed by reward collapse in the one reported run; the post's two remedies are importance weighting or removing the mismatch outright ("True on-policy RL"). This is a **Result (single study)** on one dataset and one policy initialization.
- The post reports no downstream benchmark scores, so it gives no evidence about breadth of capability, forgetting, or output diversity.

## Connections
- [[rollout-training-mismatch-tis]] — independent treatment of the same sampler-trainer mismatch with truncated importance sampling; agrees on the direction of the effect.
- [[async-rollout]] — asynchronous rollout systems, where staleness adds a second off-policy source on top of the numerical one described here.
- [[verl-rollout]] — the rollout side of a training framework, where sampler log-probabilities would be recorded to measure this mismatch.

## Verification
- Checked on 2026-09-18 against: https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/ (post dated Sep 10, 2025; no version history published)
- Created on 2026-09-18 from https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/ (2025-09-10 published version).
- Corrections to the previous card version: none (no prior card; ch-02 and ch-58 noted on 2026-09-15 that the planned library card was not present).
- Removed as unsupported by the source: none.
- Chapter claims not found in the source: none. Each cited chapter claim was checked at the locus — ch-02 L196 (definitions, `0.1 + 1e20` example), L201 (no atomic adds in the forward pass), L208 (80 completions, 102 tokens, 992/8 split), L222 (26/55/42 s, ~20% matmul cost), L252 (Bigmath RLVR numbers), L281 (FlashAttention backward), and ch-58 L26, L27, L91, L342, L344. The source states a *loss* spike around step 318 with a corresponding KL spike; ch-02 L252 and ch-58 L91 describe it as a KL spike near step 318, which the source supports.
- Not reported by the source: RL hyperparameters beyond the three listed above; GPU model for either experiment; whether batch invariance holds across changes of tensor-parallel size, GPU type, or library version (the post explicitly limits its matmul claim to run-to-run behavior on one setup).
