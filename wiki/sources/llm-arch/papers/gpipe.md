<!-- scope: pipeline parallelism via micro-batch splitting
     deps: [[megatron-lm]]
     see-also: [[zero]], [[ultra-scale-playbook]]
-->

# GPipe: Easy Scaling with Micro-Batch Pipeline Parallelism
- **Core Insight:** Splitting a model across accelerators by layer and pipelining micro-batches nearly eliminates pipeline bubbles, enabling efficient training of very large models.
- **Guideline:** Use micro-batch pipeline parallelism alongside tensor parallelism to scale beyond single-device memory; tune the number of micro-batches to minimize bubble overhead.
- **Authors:** Yanping Huang, Yonglong Cheng, Ankur Bapna, Orhan Firat, Dehao Chen, Mia Xu Chen, HyoukJoong Lee, Jiquan Ngiam, Quoc V. Le, Yonghui Wu, Zhifeng Chen
- **Year:** 2019
- **URL:** https://arxiv.org/abs/1811.06965
- **Relevant chapters:** Distributed training, pipeline parallelism, model parallelism, scaling large models

## Abstract
Scaling up deep neural network capacity has been known as an effective approach to improving the quality of learning for several different machine learning tasks. In many cases, increasing model capacity beyond the memory limit of a single accelerator has required developing special algorithms or infrastructure. These solutions are often architecture-specific and do not transfer to other tasks. To address the need for efficient and task-independent model parallelism, we introduce GPipe, a pipeline parallelism library that allows scaling any network that can be expressed as a sequence of layers. By pipelining different sub-sequences of layers on separate accelerators, GPipe provides the flexibility of scaling a variety of different networks to gigantic sizes efficiently. Furthermore, GPipe utilizes a novel batch-splitting pipelining algorithm, resulting in almost linear speedup when a model is partitioned across multiple accelerators. We demonstrate the advantages of GPipe by training large-scale neural networks on two different tasks with distinct network architectures: (1) Image Classification: We train a 557-million-parameter AmoebaNet model and attain a top-1 accuracy of 84.4% on ImageNet-2012, (2) Multilingual Neural Machine Translation: We train a single 6-billion-parameter, 128-layer Transformer on a corpus spanning over 100 languages and achieve better quality than all bilingual models.

## Key Contributions
- Introduced micro-batch pipeline parallelism: splitting a mini-batch into M micro-batches that flow through pipeline stages, reducing bubble time to O(1/M) of total compute
- Demonstrated task-agnostic model parallelism -- works for any network expressible as a sequence of layers, unlike tensor parallelism which requires architecture-specific partitioning
- Re-materialization (activation checkpointing) to reduce peak memory from O(N) to O(N/K) where K is the number of pipeline stages, trading compute for memory
- Trained a 6B-parameter, 128-layer Transformer for multilingual NMT -- among the largest Transformers at the time
- Achieved near-linear scaling across accelerators by increasing the number of micro-batches relative to pipeline stages

## Key Figures/Tables to Study
- **Figure 2** (Pipeline schedule): Shows how M micro-batches flow through K pipeline stages. The bubble fraction shrinks as M/K grows. This is the core algorithmic contribution.
- **Figure 3** (Bubble overhead vs. M/K ratio): Demonstrates that with M >= 4K micro-batches, bubble overhead drops below 6%.
- **Table 1** (AmoebaNet scaling results): Shows near-linear throughput scaling from 1 to 8 accelerators.
- **Figure 4** (Multilingual NMT quality): 128-layer Transformer trained via GPipe outperforms all bilingual baselines.

## Architecture Details
- **Pipeline partitioning:** Model is split into K consecutive groups of layers, each placed on a separate accelerator
- **Micro-batch splitting:** Each mini-batch is divided into M equal micro-batches; forward passes pipeline through stages sequentially
- **Bubble time:** Fraction of idle time = (K-1) / (M + K - 1); with M >> K this approaches zero
- **Gradient accumulation:** Gradients are accumulated across micro-batches before a single synchronous weight update
- **Re-materialization:** During the backward pass, activations are recomputed from the latest checkpoint rather than stored, reducing memory from O(L) to O(L/K) per device
- **Comparison with data parallelism:** GPipe is orthogonal to data parallelism -- you can use both (DP across nodes, PP within a node)
- **Comparison with tensor parallelism (Megatron-LM):** TP splits individual layers across devices (requires all-reduce per layer); PP splits entire layers across devices (requires point-to-point communication between stages). Modern systems combine both.
- **Key limitation:** Requires the model to be expressible as a sequential pipeline -- skip connections within a stage are fine, but cross-stage skip connections complicate scheduling
- **Bubble reduction techniques (later work):** 1F1B scheduling (PipeDream), interleaved stages (Megatron-LM v2), and zero-bubble PP further reduce idle time
