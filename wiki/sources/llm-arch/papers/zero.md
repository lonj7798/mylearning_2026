# ZeRO: Memory Optimizations Toward Training Trillion Parameter Models
- **Authors:** Samyam Rajbhandari, Jeff Rasley, Olatunji Ruwase, Yuxiong He
- **Year:** 2019
- **URL:** https://arxiv.org/abs/1910.02054
- **Core Insight:** Shard optimizer states, gradients, and parameters across GPUs; each GPU only stores 1/N.
- **Guideline:** To maximize model size for a given GPU cluster, progressively shard optimizer states (ZeRO-1), then gradients (ZeRO-2), then parameters (ZeRO-3) across data-parallel ranks. This eliminates redundant memory without sacrificing data parallelism's simplicity.
- **Relevant chapters:** Distributed training, Memory optimization, Large-scale training, Data parallelism

## Abstract
Large deep learning models offer significant accuracy gains, but training billions to trillions of parameters is challenging. Existing solutions such as data and model parallelisms exhibit fundamental limitations to fit these models into limited device memory, while obtaining computation, communication and development efficiency. We develop a novel solution, Zero Redundancy Optimizer (ZeRO), to optimize memory, vastly improving training speed while increasing the model size that can be efficiently trained. ZeRO eliminates memory redundancies in data- and model-parallel training while retaining low communication volume and high computational granularity, allowing us to scale the model size proportional to the number of devices with sustained high efficiency. Our analysis on memory requirements and communication volume demonstrates: ZeRO has the potential to scale beyond 1 Trillion parameters using today's hardware. We implement and evaluate ZeRO: it trains large models of over 100B parameter with super-linear speedup on 400 GPUs, achieving throughput of 15 Petaflops. This represents an 8x increase in model size and 10x increase in achievable performance over state-of-the-art. In terms of usability, ZeRO can train large models of up to 13B parameters (e.g., larger than Megatron GPT 8.3B and T5 11B) without requiring model parallelism which is harder for scientists to apply. Last but not the least, researchers have used the system breakthroughs of ZeRO to create the world's largest language model (Turing-NLG, 17B parameters) with record breaking accuracy.

## Key Contributions
- Introduced ZeRO (Zero Redundancy Optimizer) with three progressive stages: ZeRO-1 (shard optimizer states), ZeRO-2 (shard gradients), ZeRO-3 (shard parameters), each reducing per-GPU memory proportionally to the number of GPUs
- Achieved super-linear speedup on 400 GPUs for 100B+ parameter models, demonstrating that memory savings translate directly to throughput gains
- Proved theoretically that ZeRO can scale to trillion-parameter models on existing hardware by eliminating all memory redundancy across data-parallel processes
- Made large model training accessible without model parallelism -- up to 13B parameters using only data parallelism with ZeRO, simplifying the engineering burden
- Enabled the training of Turing-NLG (17B parameters), the largest language model at the time, demonstrating practical impact

## Why This Paper Matters
ZeRO democratized large model training. Before ZeRO, fitting a large model required complex model parallelism; after ZeRO, researchers could train significantly larger models using the familiar data-parallel paradigm. The DeepSpeed library implementing ZeRO became standard infrastructure for training LLMs, and ZeRO's sharding approach directly influenced PyTorch FSDP (Fully Sharded Data Parallel). Today, virtually every large-scale training run uses some form of optimizer/parameter sharding descended from this work.
