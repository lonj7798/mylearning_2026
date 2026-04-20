<!-- scope: Chinchilla scaling — compute-optimal model sizing
     deps: [[scaling-laws-kaplan]]
     see-also: [[gpt-3]], [[scaling-data-constrained]]
-->

# Training Compute-Optimal Large Language Models
- **Core Insight:** Most large models are undertrained on data; scale training tokens proportionally with parameters for compute-optimal performance.
- **Guideline:** For a fixed compute budget, allocate equally to model size and data; doubling parameters means doubling tokens.
- **Authors:** Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, Elena Buchatskaya, Trevor Cai, Eliza Rutherford, Diego de Las Casas, Lisa Anne Hendricks, Johannes Welbl, Aidan Clark, Tom Hennigan, Eric Noland, Katie Millican, George van den Driessche, Bogdan Damoc, Aurelia Guy, Simon Osindero, Karen Simonyan, Erich Elsen, Jack W. Rae, Oriol Vinyals, Laurent Sifre
- **Year:** 2022
- **URL:** https://arxiv.org/abs/2203.15556
- **Relevant chapters:** Scaling laws, compute-optimal training, data requirements, model sizing

## Abstract
We investigate the optimal model size and number of tokens for training a transformer language model under a given compute budget. We find that current large language models are significantly undertrained, a consequence of the recent focus on scaling language models whilst keeping the amount of training data constant. By training over 400 language models ranging from 70 million to over 16 billion parameters on 5 to 500 billion tokens, we find that for compute-optimal training, the model size and the number of training tokens should be scaled equally: for every doubling of model size the number of training tokens should also be doubled. We test this hypothesis by training a predicted compute-optimal model, Chinchilla, that uses the same compute budget as Gopher but with 70B parameters and 4x more data. Chinchilla uniformly and significantly outperforms Gopher (280B), GPT-3 (175B), Jurassic-1 (178B), and Megatron-Turing NLG (530B) on a large range of downstream evaluation tasks. This also means that Chinchilla uses substantially less compute for fine-tuning and inference, greatly facilitating downstream usage. As a highlight, Chinchilla reaches a state-of-the-art average accuracy of 67.5% on the MMLU benchmark, greater than a 7% improvement over Gopher.

## Key Contributions
- Demonstrated that most existing large language models are significantly undertrained relative to their parameter count, fundamentally challenging the "bigger model = better" assumption
- Established the equal-scaling law: for compute-optimal training, model size and training tokens should be scaled equally (doubling parameters requires doubling data)
- Trained Chinchilla (70B parameters, 1.4T tokens) which outperformed Gopher (280B), GPT-3 (175B), and Megatron-Turing NLG (530B) despite being 4x smaller than Gopher
- Used three complementary approaches to derive scaling laws (IsoFLOP profiles, IsoLoss contours, parametric fitting of the loss function), lending robustness to the conclusions
- Shifted industry practice toward training on far more data relative to model size, influencing the design of LLaMA, Mistral, and most subsequent open-weight models

## Key Figures/Tables to Study
- **Figure 1** (IsoFLOP curves): Shows optimal model size for each compute budget. The minimum of each curve gives the compute-optimal parameter count. Study how the optimal size shifts right with more compute.
- **Figure 3** (Optimal model size and tokens vs. compute): The core result -- both axes scale equally with compute. This directly contradicts the Kaplan et al. (2020) scaling law that favored scaling parameters over data.
- **Table 3** (Chinchilla vs. Gopher vs. GPT-3 vs. others): Head-to-head downstream comparison. Notice Chinchilla wins everywhere despite being much smaller.
- **Table A3** (IsoFLOP optimal model sizes): Detailed numerical results for the compute-optimal frontier. Use this as a lookup table for budgeting your own training runs.
- **Figure 4** (MMLU accuracy vs. compute): Shows Chinchilla reaching 67.5% MMLU, establishing it as the MMLU SOTA at the time.

## Architecture Details
- **Chinchilla model size:** 70B parameters
- **Chinchilla training tokens:** 1.4 trillion
- **Gopher comparison:** 280B parameters trained on 300B tokens (same compute budget)
- **Scaling law:** N_opt proportional to C^0.5, D_opt proportional to C^0.5 (equal scaling of parameters and data with compute)
- **Models trained for analysis:** 400+ models from 70M to 16B parameters on 5B to 500B tokens
- **Architecture family:** Decoder-only Transformer (same as Gopher/Chinchilla family)
- **Three estimation approaches:** (1) Fix compute, vary model size (IsoFLOP); (2) Fix loss, find minimum compute (IsoLoss); (3) Parametric fit of L(N,D) = E + A/N^alpha + B/D^beta
- **Key prediction:** For a given compute budget C, the optimal model size N and optimal token count D satisfy N ~ C^a and D ~ C^b where a ~ b ~ 0.5
- **Training hardware:** TPUv3/v4 pods (DeepMind infrastructure)
- **MMLU accuracy:** 67.5% (Chinchilla) vs. 60.0% (Gopher) vs. 43.9% (GPT-3)
- **Practical implication:** A 70B model trained on 1.4T tokens is preferable to a 280B model trained on 300B tokens, and is cheaper to serve at inference time
