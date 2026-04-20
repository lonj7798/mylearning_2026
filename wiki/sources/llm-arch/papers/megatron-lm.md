# Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism
- **Authors:** Mohammad Shoeybi, Mostofa Patwary, Raul Puri, Patrick LeGresley, Jared Casper, Bryan Catanzaro
- **Year:** 2019
- **URL:** https://arxiv.org/abs/1909.08053
- **Core Insight:** Tensor parallelism -- split matrix multiplications across GPUs within a single layer.
- **Guideline:** For models too large to fit on a single GPU, split individual weight matrices (attention heads, FFN columns) across GPUs using tensor parallelism. This requires only a few communication operations per layer and composes with pipeline and data parallelism.
- **Relevant chapters:** Distributed training, Model parallelism, Large-scale training, GPU systems

## Abstract
Recent work in language modeling demonstrates that training large transformer models advances the state of the art in Natural Language Processing applications. However, very large models can be quite difficult to train due to memory constraints. In this work, we present our techniques for training very large transformer models and implement a simple, efficient intra-layer model parallel approach that enables training transformer models with billions of parameters. Our approach does not require a new compiler or library changes, is orthogonal and complimentary to pipeline model parallelism, and can be fully implemented with the insertion of a few communication operations in native PyTorch. We illustrate this approach by converging transformer based models up to 8.3 billion parameters using 512 GPUs. We sustain 15.1 PetaFLOPs across the entire application with 76% scaling efficiency when compared to a strong single GPU baseline that sustains 39 TeraFLOPs, which is 30% of peak FLOPs. To demonstrate that large language models can further advance the state of the art (SOTA), we train an 8.3 billion parameter transformer language model similar to GPT-2 and a 3.9 billion parameter model similar to BERT. We show that careful attention to the placement of layer normalization in BERT-like models is critical to achieving increased performance as the model size grows. Using the GPT-2 model we achieve SOTA results on the WikiText103 (10.8 compared to SOTA perplexity of 15.8) and LAMBADA (66.5% compared to SOTA accuracy of 63.2%) datasets. Our BERT model achieves SOTA results on the RACE dataset (90.9% compared to SOTA accuracy of 89.4%).

## Key Contributions
- Introduced tensor parallelism for Transformers: splitting attention heads and FFN weight matrices across GPUs within a single layer, requiring only all-reduce communication operations
- Achieved 76% scaling efficiency on 512 GPUs, demonstrating practical large-scale training without custom compilers or library modifications
- Trained models up to 8.3 billion parameters (large for 2019), establishing new SOTA on WikiText103, LAMBADA, and RACE benchmarks
- Showed that the approach is orthogonal to pipeline parallelism and data parallelism, allowing all three to be combined for even larger models
- Discovered that layer normalization placement is critical for scaling BERT-like models, foreshadowing the pre-norm vs. post-norm debate

## Why This Paper Matters
Megatron-LM provided the tensor parallelism blueprint that every large model training effort uses today. The insight that individual matrix multiplications can be split across GPUs -- with only a few communication operations per layer -- made it possible to scale beyond single-GPU memory limits without architectural compromises. Modern training frameworks (Megatron-DeepSpeed, FSDP, etc.) all build on this approach. Without tensor parallelism, training models with hundreds of billions of parameters would be impractical.
