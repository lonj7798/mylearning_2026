<!-- scope: CUDA graph capture as an inference latency optimization for stable execution shapes
     deps: transformer-inference-loop
     see-also: flashinfer, tensor-parallel-inference, pipeline-parallel-inference
-->

# CUDA Graphs for Inference
- **Core Insight:** For repeated inference steps with stable shapes, replaying a captured CUDA graph can remove CPU launch overhead and improve latency predictability.
- **Guideline:** Use CUDA graphs for steady-state decode or fixed-shape batches, but design around static memory addresses, fixed shapes, warmup, and dynamic-request bucketing.
- **Authors:** NVIDIA CUDA and PyTorch documentation; representative serving use in vLLM and FlashInfer
- **Year:** synthesis card, 2021-2025 sources
- **URL:** https://docs.pytorch.org/docs/stable/notes/cuda.html#cuda-graphs ; https://docs.pytorch.org/TensorRT/tutorials/runtime_opt/cuda_graphs.html ; https://arxiv.org/abs/2501.01005
- **Relevant topics:** CUDA graphs, kernel launch overhead, static shapes, graph replay, decode latency, serving bucketing

## Abstract
This is a synthesis card rather than one paper. CUDA graphs capture a sequence of GPU operations and replay it with lower CPU overhead. In LLM inference, they are most relevant to decode loops where the same kernels run repeatedly. The hard part is that serving systems have dynamic batch sizes and sequence lengths, while graph replay expects stable shapes and memory addresses.

## Key Contributions
- PyTorch exposes raw `torch.cuda.CUDAGraph` plus `torch.cuda.graph` and `make_graphed_callables`.
- Torch-TensorRT documents graph use for inference latency and dynamic-output limitations.
- vLLM and FlashInfer show the serving pattern: bucket or pad dynamic requests into graph-compatible shapes.
- CUDA graphs complement kernel optimization by reducing dispatch overhead between kernels.

## Key Figures/Tables to Study
- PyTorch CUDA semantics docs: capture, replay, warmup, and memory-pool requirements.
- Torch-TensorRT CUDA graphs tutorial: runtime optimization constraints for inference.
- FlashInfer CUDAGraph-compatible scheduling section: serving-specific dynamic batching issue.
- vLLM distributed/optimization docs: graph capture appears alongside KV-cache and batch scheduling constraints.

## Technical Details
Graph capture records operations on fixed tensors. Replay reuses the same memory addresses, so inputs are copied into static buffers before replay and outputs are read from static output buffers. Warmup must happen on a side stream before capture so allocations and autotuning do not occur inside the graph.

In LLM serving, graphing the whole system is usually unrealistic because requests arrive and finish dynamically. Practical systems graph repeated subgraphs for common batch sizes or decode shapes, use padding/bucketing, and fall back to eager execution for unusual shapes.

CUDA graphs do not reduce FLOPs or HBM traffic. They reduce host-side launch overhead and make small-kernel sequences more predictable, which matters in low-latency decode.

## Connections
- [[flashinfer]] explicitly designs scheduling around CUDAGraph compatibility.
- [[flashdecoding]] reduces decode attention kernel cost; CUDA graphs reduce launch overhead around repeated decode steps.
- [[tensor-parallel-inference]] adds communication kernels, which may also be captured when shapes and collective topology are stable.
- [[pipeline-parallel-inference]] may graph per-stage steady-state decode when microbatch shapes are bucketed.
