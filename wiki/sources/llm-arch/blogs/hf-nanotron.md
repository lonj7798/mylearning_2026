<!-- scope: distributed LLM training framework
     deps: [[ch-04]]
     see-also: [[flash-attention-explained]], [[raschka-llm-architecture-comparison]]
-->

# Nanotron Research -- HuggingFace Distributed Training Organization

- **Core Insight:** Nanotron is the reference implementation for the Ultra-Scale Playbook experiments.
- **Guideline:** Use nanotron for reproducing distributed training experiments.

- **URL:** https://huggingface.co/nanotron
- **Type:** framework / research organization
- **Relevant chapters:** distributed training, model parallelism, GPU acceleration, training infrastructure, scaling

## Content

### Organization Overview

**Nanotron Research** is a community-driven organization on Hugging Face focused on large-scale distributed AI model training. Their tagline: "Make GPUs go brrrrr."

**Focus Areas:**
- Large-scale distributed AI model training
- Model parallelization (tensor, pipeline, data, context parallelism)
- Low-level GPU acceleration
- Training infrastructure and optimization

**Followers:** 865

### Team Members (8)

1. Thomas Wolf (thomwolf) -- co-founder/CTO of Hugging Face
2. Nouamane Tazi (nouamanetazi)
3. Loubna Ben Allal (loubnabnl)
4. Ferdinand Mom (3outeille)
5. Nathan Habib (SaylorTwift)
6. Leandro von Werra (lvwerra)
7. Julien Chaumond (julien-c) -- co-founder of Hugging Face
8. Frere Thibaud (tfrere)

### Core Libraries

**1. Nanotron**
- GitHub: https://github.com/huggingface/nanotron
- The core distributed training framework for LLMs
- Supports tensor parallelism, pipeline parallelism, data parallelism, context parallelism
- Built on PyTorch, designed for multi-node GPU clusters
- Used internally at Hugging Face for training large models

**2. Picotron**
- GitHub: https://github.com/huggingface/picotron
- Complementary training optimization library
- Lightweight companion to Nanotron for smaller-scale experiments

### Key Resource: The Ultra-Scale Playbook

- **Title:** The Ultra-Scale Playbook: Training LLMs on GPU Clusters
- **Authors:** Nouamane Tazi, Ferdinand Mom, Haojun Zhao, Phuc Nguyen, Mohamed Mekkouri, Leandro von Werra, Thomas Wolf
- **Published:** July 30, 2025
- **Format:** Interactive HF Space + PDF book (PRO subscription required for PDF)
- **Views:** 3,790+
- **Content:** Comprehensive guide to distributed/parallelization techniques for training LLMs at scale

### Interactive Tools

**Predict Memory Space**
- Calculate and visualize memory usage for model training
- Helps estimate GPU memory requirements before launching training runs
- 107 views

### Published Models (14)

| Model | Date | Notes |
|-------|------|-------|
| nanotron/llama3-8b-infini-attention | Aug 2024 | Infini-attention variant of Llama 3 8B (22 downloads, 5 likes) |
| nanotron/minicpm-nanotron | Apr 2024 | MiniCPM ported to nanotron format |
| nanotron/doremi-llama-2.5b-optimized-weights | Feb 2024 | DoReMi domain reweighting, optimized |
| nanotron/doremi-llama-2.5b-reference | Feb 2024 | DoReMi domain reweighting, reference |
| + 10 additional experimental/benchmark models | Various | Various experimental checkpoints |

### Published Datasets (15)

| Dataset | Items | Date | Description |
|---------|-------|------|-------------|
| nanotron/book | 100 | Jul 2025 | Ultra-Scale Playbook data |
| nanotron/ultrascale-playbook-data | 303 | Mar 2025 | Benchmark data for playbook |
| nanotron/picotron_bench | 740 | Dec 2024 | Picotron benchmarks |
| nanotron/minipile_100_samples | 100 | Jul 2024 | Small evaluation dataset |
| nanotron/llama3-1024-passkey-retrieval-eval | 12.6k | Jul 2024 | Long-context evaluation |
| nanotron/llama3-16k-passkey-retrieval-finetuning | 77.3k | Jun 2024 | Long-context fine-tuning data |
| + Additional passkey retrieval & needle-in-haystack datasets | Various | 2024 | Context-length evaluation at 16k, 32k |

## Why This Is Useful

Nanotron is one of the most practical open-source resources for understanding how LLMs are actually trained at scale. While most LLM architecture courses cover the model design (attention, FFN, normalization), they often skip the distributed training infrastructure that makes large models possible. Nanotron fills this gap:

1. **The Ultra-Scale Playbook** is arguably the best single resource on distributed training parallelism strategies (tensor, pipeline, data, context, expert parallelism). It covers the "how do you actually train a 70B model across 256 GPUs" question.

2. **The nanotron library itself** is clean, well-documented training code that shows how model parallelism is implemented in practice -- essential for anyone who wants to go beyond understanding architecture on paper.

3. **The Predict Memory tool** solves a practical problem every training engineer faces: estimating GPU memory requirements before committing compute.

4. **The infini-attention and DoReMi models** demonstrate cutting-edge architectural innovations (infinite attention for long context, domain reweighting for data mixing) implemented in a production training framework.

For an LLM architecture curriculum, nanotron bridges the gap between "I understand how a transformer works" and "I can train one at scale."
