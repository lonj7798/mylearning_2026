<!-- scope: distributed training — parallelism strategies for GPU clusters
     deps: [[attention-is-all-you-need]]
     see-also: [[flash-attention-2]], [[megatron-lm]], [[zero]]
-->

# The Ultra-Scale Playbook: Training LLMs on GPU Clusters
- **Core Insight:** Parallelism is multi-dimensional; memory usage, compute efficiency, and communication overhead are the three axes to optimize jointly.
- **Guideline:** Enumerate all five parallelism dimensions (DP, TP, PP, SP, EP) and benchmark thousands of configs to find the sweet spot for your hardware.
- **Authors:** Nouamane Tazi, Ferdinand Mom, Haojun Zhao, Phuc Nguyen, Mohamed Mekkouri, Leandro Werra, Thomas Wolf
- **Organization:** Hugging Face
- **Published:** Feb 19, 2025
- **URL:** https://huggingface.co/spaces/nanotron/ultrascale-playbook
- **Reading time:** 2-4 days
- **Source file:** `llm-arch/The_Ultra-Scale_Playbook_Training_LLMs_on_GPU_Clusters.pdf`
- **Extracted text:** `llm-arch/ultra-scale-playbook.txt` (5891 lines)
- **Relevant chapters:** Ch 13 (Distributed Training), Ch 11 (Pre-training), Ch 25 (KV-Cache & Serving)

## Summary

Comprehensive open-source book on scaling LLM training from 1 GPU to thousands. Based on 4000+ scaling experiments on up to 512 GPUs. Covers all parallelism strategies with theory, code examples, and reproducible benchmarks. Uses nanotron framework for all experiments.

## Table of Contents

1. **High Level Overview** — Memory, compute efficiency, communication overhead
2. **First Steps: Training on One GPU**
   - Memory usage in Transformers (parameters, gradients, optimizer states, activations)
   - Activation recomputation (gradient checkpointing)
   - Gradient accumulation
3. **Data Parallelism** — Revisiting global batch size
4. **ZeRO (Zero Redundancy Optimizer)** — Sharding optimizer states, gradients, parameters
5. **Tensor Parallelism** — Splitting layers across GPUs, TP in a Transformer block
6. **Sequence Parallelism** — Splitting along sequence dimension
7. **Context Parallelism**
   - Ring Attention
   - Zig-Zag Ring Attention (balanced compute)
8. **Pipeline Parallelism**
   - Splitting layers across nodes
   - All-forward-all-backward
   - One-forward-one-backward (1F1B)
   - LLaMA 3.1 schemes
   - Interleaving stages
   - Zero Bubble and DualPipe
9. **Expert Parallelism** — MoE-specific distribution
10. **5D Parallelism in a Nutshell** — Combining DP, TP, PP, SP, EP
11. **Finding the Best Training Configuration**
    - Step 1: Fitting a training step in memory
    - Step 2: Achieving target global batch size
    - Step 3: Optimizing training throughput
    - Benchmarking thousands of configurations
    - Lessons learned on benchmarking
12. **GPU Internals** — Fusing, threading, mixing

## Key Insights

- Three axes of optimization: memory usage (hard limit), compute efficiency (GPU utilization), communication overhead (GPU idle time)
- Training step = forward pass + backward pass + optimization step
- Activations are by far the largest memory burden for large batch sizes/sequences
- Activation recomputation trades compute for memory — essential technique
- 4000+ experiments provide empirical data on which parallelism strategies work best at different scales
- Practical methodology for finding optimal training configuration given hardware constraints

## Why This Is Critical

This is the most comprehensive open-source treatment of distributed LLM training. For architecture research at Anthropic, understanding how architectural choices (number of layers, hidden dimension, attention heads, FFN width, MoE expert count) interact with parallelism strategies is essential — architecture decisions determine what parallelism is possible and efficient.
