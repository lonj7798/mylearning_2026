<!-- scope: multi-token prediction as a training objective for LLMs
     deps: [[attention-is-all-you-need]], [[pitfalls-next-token]]
     see-also: [[speculative-decoding]], [[deepseek-v3]]
-->

# Better & Faster Large Language Models via Multi-token Prediction
- **Core Insight:** Training LLMs to predict multiple future tokens simultaneously (rather than just the next token) improves sample efficiency, downstream performance, and enables self-speculative inference speedups.
- **Guideline:** Add multi-token prediction heads during pre-training for stronger representations; at inference time, use the extra heads for self-speculative decoding (3x speedup with no draft model needed).
- **Authors:** Fabian Gloeckle, Badr Youbi Idrissi, Baptiste Roziere, David Lopez-Paz, Gabriel Synnaeve
- **Year:** 2024
- **URL:** https://arxiv.org/abs/2404.19737
- **Relevant chapters:** Training objectives, speculative decoding, multi-token prediction, DeepSeek-V3 architecture

## Abstract
Large language models such as GPT and Llama are trained with a next-token prediction loss. In this work, we suggest that training language models to predict multiple future tokens at once results in higher sample efficiency. More specifically, at each position in the training corpus, we ask the model to predict the following n tokens using n independent output heads, operating on top of a shared model trunk. Considering multi-token prediction as an auxiliary training task, we measure improved downstream capabilities with no overhead in training time, and with a modest 2% overhead in number of parameters. Separately, we leverage the auxiliary prediction heads for self-speculative decoding and observe a 3x speed-up with greedy decoding and a 2.5x speed-up with nucleus sampling with no quality degradation.

## Key Contributions
- Demonstrated that multi-token prediction (n = 4 heads) during pre-training consistently improves performance on code and natural language benchmarks, especially at larger model scales (13B parameters)
- The improvement comes from better internal representations in the shared trunk, not just the extra prediction heads -- the main next-token head itself becomes more accurate
- Multi-token prediction acts as an implicit regularizer: it forces the model to encode information about longer-range future context in its hidden representations
- Self-speculative decoding: at inference time, the extra heads serve as draft predictors for speculative decoding, achieving 3x speedup without needing a separate draft model
- Training overhead is minimal: n-1 extra linear heads add ~2% parameters; forward pass through the shared trunk is unchanged; only the loss computation and backward pass through the heads add cost

## Key Figures/Tables to Study
- **Figure 1** (Architecture diagram): Shows shared Transformer trunk with n independent output heads. Each head predicts token at position t+k (k = 1, ..., n). Understand that heads share the same trunk representation.
- **Figure 2** (Scaling behavior): Multi-token prediction gains increase with model size. At 13B, 4-token prediction significantly outperforms next-token on code benchmarks. This scaling trend is the key result.
- **Figure 3** (Self-speculative decoding speedup): Shows 3x speedup with greedy decoding, 2.5x with nucleus sampling, with no quality loss.
- **Table 1** (Benchmark results): Multi-token prediction improves HumanEval, MBPP, and natural language tasks. Gains are more pronounced on code.
- **Figure 5** (Byte-level experiments): Multi-token prediction is especially effective for byte-level models where individual tokens carry less information.

## Architecture Details
- **Training objective:** At each position t, predict tokens at t+1, t+2, ..., t+n simultaneously using n independent output heads on top of a shared Transformer trunk
- **Loss:** Sum of cross-entropy losses across all n heads, with equal weighting: L = sum_{k=1}^{n} L_k where L_k = -log P(x_{t+k} | x_{<=t})
- **Output heads:** Each head is an independent linear layer (unembedding) mapping from the trunk's hidden dimension to vocabulary size. Heads do NOT share parameters with each other.
- **Memory optimization:** To avoid materializing n vocabulary-sized logit tensors simultaneously, the paper uses a sequential head computation with gradient accumulation
- **Optimal n:** n = 4 works best in practice; n = 2 gives smaller gains; n = 8 shows diminishing returns. The sweet spot balances representation quality against dilution of the primary next-token signal.
- **Why it works (intuition):** Predicting token t+4 from position t requires the model to build richer, more forward-looking representations in the shared trunk. This implicit planning signal improves the quality of all predictions, including the primary next-token prediction.
- **Self-speculative decoding:** At inference, heads 2..n propose draft tokens in parallel; head 1 (the primary head) verifies them. Accepted tokens skip autoregressive steps. This is "self-speculative" because no external draft model is needed.
- **Adopted by DeepSeek-V3:** DeepSeek-V3 uses multi-token prediction (MTP) as part of its architecture, validating this approach at frontier scale. DeepSeek uses n = 2 (predict next 2 tokens).
- **Relationship to pitfalls-next-token:** This paper provides a practical solution to some of the teacher-forcing limitations identified in [[pitfalls-next-token]] -- multi-token prediction forces the model to reason about future states rather than relying on ground-truth context shortcuts.
