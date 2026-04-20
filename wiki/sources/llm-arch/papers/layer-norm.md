# Layer Normalization
- **Authors:** Jimmy Lei Ba, Jamie Ryan Kiros, Geoffrey E. Hinton
- **Year:** 2016
- **URL:** https://arxiv.org/abs/1607.06450
- **Core Insight:** Normalize across features instead of batch dimension; works for variable-length sequences.
- **Guideline:** For sequence models and Transformers, use layer normalization (normalizing across the feature dimension of each sample) rather than batch normalization, because it is independent of batch size and handles variable-length sequences naturally.
- **Relevant chapters:** Normalization techniques, Transformer components, Training stability, RNN training

## Abstract
Training state-of-the-art, deep neural networks is computationally expensive. One way to reduce the training time is to normalize the activities of the neurons. A recently introduced technique called batch normalization uses the distribution of the summed input to a neuron over a mini-batch of training cases to compute a mean and variance which are then used to normalize the summed input to that neuron on each training case. This significantly reduces the training time in feed-forward neural networks. However, the effect of batch normalization is dependent on the mini-batch size and it is not obvious how to apply it to recurrent neural networks. In this paper, we transpose batch normalization into layer normalization by computing the mean and variance used for normalization from all of the summed inputs to the neurons in a layer on a single training case. Like batch normalization, we also give each neuron its own adaptive bias and gain which are applied after the normalization but before the non-linearity. Unlike batch normalization, layer normalization performs exactly the same computation at training and test times. It is also straightforward to apply to recurrent neural networks by computing the normalization statistics separately at each time step. Layer normalization is very effective at stabilizing the hidden state dynamics in recurrent networks. Empirically, we show that layer normalization can substantially reduce the training time compared with previously published techniques.

## Key Contributions
- Proposed layer normalization: computing normalization statistics (mean and variance) across all features within a single training example, rather than across the batch dimension
- Showed that layer normalization is naturally applicable to recurrent neural networks and variable-length sequences, unlike batch normalization
- Demonstrated that layer normalization produces identical computation at training and test time, eliminating the train/test discrepancy inherent in batch normalization
- Introduced learnable per-neuron gain and bias parameters that allow the network to recover expressive power after normalization
- Proved that layer normalization stabilizes hidden state dynamics in recurrent networks, enabling faster convergence

## Why This Paper Matters
Layer normalization became a fundamental component of the Transformer architecture. Every Transformer block applies layer normalization, and the choice of where to place it (pre-norm vs. post-norm) significantly affects training stability. Without layer normalization, training large Transformers would require far more careful hyperparameter tuning and would be much less stable. This paper provided the key normalization technique that made modern LLM training practical.
