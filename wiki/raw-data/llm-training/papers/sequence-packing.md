<!-- scope: Sequence packing — concat multiple short sequences into one fixed-length block with attention masking
     deps: []
     see-also: [[packed-vs-unpacked-ablation]], [[fsdp-sft]], [[loss-masking-prompt]]
-->

# Efficient Sequence Packing without Cross-Contamination
- **Core Insight:** Instruction datasets have 50–89% padding tokens; packing multiple short sequences into fixed-length blocks plus attention masks that prevent cross-sequence attention yields 2× SFT throughput with zero accuracy impact.
- **Guideline:** Always pack SFT data — use FlashAttention's varlen API (`flash_attn_varlen_func`) with `cu_seqlens` to enforce per-sequence causal masking without materializing a full block-diagonal mask; pad to the longest packed block, not to global max length.
- **Authors:** Mario Michael Krell, Matej Kosec, Sergio P. Perez, Andrew Fitzgibbon
- **Year:** 2021
- **URL:** https://arxiv.org/abs/2107.02027
- **Relevant topics:** SFT throughput, FlashAttention varlen, bin packing, attention masking, padding-free training

## Abstract
Up to 50% (and in extreme cases 89%) of tokens in BERT/GLUE-style fine-tuning are padding. We formalize sequence packing as a bin-packing problem and present efficient algorithms (NNLS, shortest-pack-first) that produce packed batches with minimal padding. Crucially, we design attention masks and position IDs that make the packed model mathematically equivalent to the unpacked one — no accuracy loss — while achieving a 2× speedup on BERT phase-2 pretraining. This approach requires no accelerator-specific custom kernels.

## Key Contributions
- Formalizes SFT batching as bin-packing (1D offline bin-packing).
- Provides two practical algorithms: SPFHP (Shortest-Pack-First Histogram-Packing) and NNLSHP (Non-Negative Least Squares Histogram-Packing).
- Specifies attention mask and positional-ID corrections preserving mathematical equivalence to unpacked training.
- Demonstrates 2× throughput on BERT phase 2 with no metric change.
- Directly enables FlashAttention-2's varlen kernels used in modern SFT.

## Key Figures/Tables to Study
- **Figure 1:** Length histograms across GLUE tasks — motivates packing.
- **Figure 3:** Block-diagonal attention mask for packed sequences.
- **Table 3:** Throughput gains on Wikipedia phase 2 (~2×).

## Technical Details

### Mechanics of a packed block
Given n short sequences s_1, …, s_n with lengths L_1, …, L_n (Σ L_i ≤ L_max), concatenate along the sequence axis:
`packed = [s_1 | s_2 | … | s_n | PAD]`
with:
- `cu_seqlens = [0, L_1, L_1+L_2, …, Σ L_i]` — cumulative start offsets.
- `position_ids` reset to 0 at each sequence boundary.
- Attention mask: token t in sequence i can attend to tokens in sequence i only, with causal structure inside.

### FlashAttention varlen interface
Modern implementations use `flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen)` which computes block-diagonal causal attention **without materializing the full (L_max × L_max) mask**. Memory: O(sum L_i) not O(L_max²).

### SPFHP algorithm (paper's preferred)
1. Compute length histogram of dataset.
2. Greedy: iterate over bins, fill each with the longest remaining sequence first; for each subsequent fit, pick the longest sequence that still fits (shortest gap).
3. Achieves near-optimal packing ratio (>99% fill) in O(N log N).

### NNLSHP
Solves a non-negative least squares problem matching a desired length histogram → provides mathematically optimal packing strategies given histogram constraints.

### Why cross-contamination matters
Without masking, token t in sequence 2 can attend to sequence 1's tokens → the softmax's partition function leaks across documents → changes gradients even on sequence 1 tokens. The block-diagonal mask restores exactness.

### Loss handling
Labels for padding tokens must be set to -100 (ignore_index); loss masks for prompt tokens (see [[loss-masking-prompt]]) are applied per sub-sequence inside the pack.

## Connections
- Downstream enabler: [[fsdp-sft]] — packing reduces per-step memory so FSDP's sharding is effective.
- Required companion: [[loss-masking-prompt]] for SFT loss definition.
- Ablations on quality: [[packed-vs-unpacked-ablation]].
- HF implementation: [[hf-alignment-handbook]].
- Paper adapting this to 2024 SFT stacks: TRL's `DataCollatorWithPacking`, Axolotl's `sample_packing`.
