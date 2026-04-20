# Scaling Data-Constrained Language Models
- **Authors:** Niklas Muennighoff, Alexander M. Rush, Boaz Barak, Teven Le Scao, Aleksandra Piktus, Nouamane Tazi, Sampo Pyysalo, Thomas Wolf, Colin Raffel
- **Year:** 2023
- **URL:** https://arxiv.org/abs/2305.16264
- **Core Insight:** When data is limited, repeating data has diminishing returns; unique tokens matter more than epochs.
- **Guideline:** When training data is constrained, up to 4 epochs of repetition has negligible impact on loss, but beyond that returns decay rapidly. Prioritize collecting unique data over repeating existing data, and consider augmenting with code or relaxing quality filters.
- **Relevant chapters:** Scaling laws, Data efficiency, Pretraining, Compute-optimal training

## Abstract
The current trend of scaling language models involves increasing both parameter count and training dataset size. Extrapolating this trend suggests that training dataset size may soon be limited by the amount of text data available on the internet. Motivated by this limit, we investigate scaling language models in data-constrained regimes. Specifically, we run a large set of experiments varying the extent of data repetition and compute budget, ranging up to 900 billion training tokens and 9 billion parameter models. We find that with constrained data for a fixed compute budget, training with up to 4 epochs of repeated data yields negligible changes to loss compared to having unique data. However, with more repetition, the value of adding compute eventually decays to zero. We propose and empirically validate a scaling law for compute optimality that accounts for the decreasing value of repeated tokens and excess parameters. Finally, we experiment with approaches mitigating data scarcity, including augmenting the training dataset with code data or removing commonly used filters. Models and datasets from our 400 training runs are freely available at https://github.com/huggingface/datablations.

## Key Contributions
- Established empirically that up to 4 epochs of data repetition causes negligible loss degradation, but additional repetition has sharply diminishing returns
- Proposed a modified scaling law that accounts for the decreasing marginal value of repeated tokens, extending Chinchilla-style compute-optimal analysis to data-constrained regimes
- Ran 400 training runs systematically varying data repetition and compute budget (up to 900B tokens, 9B parameters), providing comprehensive empirical coverage
- Showed that augmenting training data with code and relaxing quality filters can partially mitigate data scarcity
- Quantified a looming practical problem: at current scaling trends, high-quality text data on the internet will be exhausted, requiring new strategies

## Why This Paper Matters
This paper addresses one of the most pressing practical constraints in LLM development: we are running out of unique training data. The Chinchilla scaling laws assumed unlimited data; this paper extends those laws to the realistic regime where data is finite. The finding that repetition has sharply diminishing returns after ~4 epochs has directly influenced training strategies at frontier labs, motivating investments in synthetic data generation, multimodal data, and more efficient data utilization. It is essential reading for understanding why the next generation of scaling may look fundamentally different.
