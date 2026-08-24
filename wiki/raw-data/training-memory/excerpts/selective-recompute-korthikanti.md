# Reducing Activation Recomputation in Large Transformer Models
<!-- slug: selective-recompute-korthikanti · type: paper · source: https://arxiv.org/abs/2205.05198 -->

**Core Insight.** Full activation recomputation (Chen 2016) saves memory but costs 30–40% extra compute; selective recomputation keeps only the small, cheap-to-store activations (LayerNorm outputs, linear projections) and discards only the large, cheap-to-recompute activations (attention matrices), achieving a 5× activation memory reduction with over 90% less recompute overhead — and sequence parallelism removes the remaining tensor-parallel activation duplication.

**Guideline.** In Megatron-style tensor-parallel training, use selective activation recomputation as the default (not full recompute). Pair it with sequence parallelism to partition the LayerNorm and Dropout activation regions across tensor-parallel ranks, eliminating their duplication across GPUs.

## Technical Details

- **Two novel techniques introduced together:**
  1. **Selective activation recomputation:** Within each transformer layer, identifies which operations have high memory cost but low recompute cost. Attention score computation (softmax, dropout over `s×s` matrices) is discarded and recomputed; MLP, LayerNorm, and projection outputs are retained.
  2. **Sequence parallelism:** Partitions the non-tensor-parallel operations (LayerNorm and Dropout, which are independent across the sequence dimension) across tensor-parallel ranks, eliminating activation duplication that tensor parallelism alone leaves intact.
- **Memory result:** 5× reduction in activation memory consumption.
- **Compute result:** Execution time overhead from recomputation reduced by over 90% (from ~30–40% to <4%).
- **GPT-3 175B analogue at 70% memory saving:** Corresponds to the playbook's "70% activation memory reduction at 2.7% compute cost" figure — the 2.7% is what remains after selective recompute on attention.
- **Scale validation:** 530B parameter GPT-3-style model, 2,240 NVIDIA A100 GPUs — MFU improved from 42.1% (full recompute) to 54.2% (selective recompute + seq parallelism) = **29% faster training**.
- **Why attention matrices are the right thing to discard:** The `s×s` attention score matrix is large (quadratic in sequence length) but recomputing it is cheap (just a matmul + softmax) — especially since FlashAttention already recomputes it in its backward pass. MLP activations are the opposite: smaller but expensive to recompute (large FFN matmuls).
- **Framework integration:** Released in Megatron-LM and NeMo-Megatron; sequence parallelism requires matching the tensor-parallel communicators.
- **Training-memory angle:** Bridges [[gradient-checkpointing-chen]]'s general √n technique and production Transformer training. The key move is asymmetric selection — not "checkpoint some layers" but "checkpoint only the cheap-to-store parts of every layer" — which recovers 5× memory at essentially no throughput cost in practice.

## Citation
Vijay Korthikanti, Jared Casper, Sangkug Lym, Lawrence McAfee, Michael Andersch, Mohammad Shoeybi, Bryan Catanzaro. "Reducing Activation Recomputation in Large Transformer Models." arXiv:2205.05198, 2022. https://arxiv.org/abs/2205.05198
