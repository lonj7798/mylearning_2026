<!-- scope: LongEmbed benchmark as a synthetic-plus-real reference for long-context embedding evaluation
     see-also: [[longalign]], [[babilong]]
-->

# LongEmbed
- **Core Insight:** Long-context embedding models need their own benchmark design, and synthetic long retrieval tasks are a key part of that evaluation mix.
- **Guideline:** For long embedding evaluation, include synthetic tasks with dispersed evidence plus real long-document retrieval tasks; short embedding benchmarks are not enough.
- **Authors:** Dawei Zhu, Liang Wang, Nan Yang, Yifan Song, Wenhao Wu, Furu Wei, Sujian Li
- **Year:** 2024
- **URL:** https://arxiv.org/abs/2404.12096
- **Relevant topics:** long-context embeddings, synthetic retrieval tasks, context extension

## Abstract
LongEmbed studies how to extend embedding models to long contexts and introduces a benchmark containing both synthetic and real-world long retrieval tasks. It matters here because it provides a clean synthetic-data reference for long embedding evaluation, not just long generation or chat.

## Key Contributions
- Introduces synthetic and real long-context retrieval tasks for embeddings.
- Shows that training-free context-extension tricks can help embedding models substantially.
- Highlights the advantage of RoPE-style approaches for long embedding windows.

## Connections
- Embedding-side counterpart to long-context generation/eval work such as [[longalign]] and [[babilong]].

