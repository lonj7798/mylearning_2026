<!-- scope: multi-token prediction as a training objective with inference acceleration implications
     deps: transformer-inference-loop
     see-also: medusa, eagle, speculative-decoding
-->

# Better & Faster Large Language Models via Multi-token Prediction
- **Core Insight:** Training a model to predict multiple future tokens can improve sample efficiency and provide future-token heads that accelerate inference.
- **Guideline:** Treat multi-token prediction as both a training objective and a possible speculation mechanism; benchmark acceptance and verification costs before assuming decode speedup.
- **Authors:** Fabian Gloeckle, Badr Youbi Idrissi, Baptiste Roziere, David Lopez-Paz, Gabriel Synnaeve
- **Year:** 2024
- **URL:** https://arxiv.org/abs/2404.19737
- **Relevant topics:** multi-token prediction, auxiliary heads, future-token loss, inference speedup, coding benchmarks

## Abstract
The paper proposes training language models to predict the next `n` tokens at each position using `n` independent output heads on top of a shared trunk. The auxiliary objective improves downstream performance, especially for code and algorithmic tasks, and creates a path to faster inference because models trained with four-token prediction can generate up to 3x faster in the reported setup.

## Key Contributions
- Introduces multi-token prediction as an auxiliary training objective for LLMs.
- Uses multiple independent future-token heads while sharing the main model trunk.
- Reports stronger gains for larger models and generative coding benchmarks.
- Links the objective to better induction heads and algorithmic reasoning.
- Demonstrates inference acceleration potential from predicting multiple tokens per step.

## Key Figures/Tables to Study
- Architecture figure: shared trunk with multiple future-token heads.
- HumanEval/MBPP tables: show benchmark gains from the objective.
- Inference speedup results: key for separating training quality and serving acceleration.
- Algorithmic task analysis: supports the induction-head explanation.

## Technical Details
At each sequence position, the model predicts token `t+1`, `t+2`, ..., `t+n` through separate heads. The loss is an auxiliary objective during training, and the additional heads can be used at inference to propose future tokens.

For serving, multi-token heads resemble Medusa-style proposals or model-native speculative heads. The course should distinguish two claims: the training objective can improve model quality, and the heads can accelerate generation if their proposed tokens are accepted often enough by the target verification logic.

Recent MoE model reports and serving frameworks also use "MTP" to mean model-provided speculative proposals, but implementations differ from this paper.

## Connections
- [[medusa]] is an inference-acceleration framework based on extra future-token heads.
- [[eagle]] drafts using feature prediction instead of direct future-token heads.
- [[speculative-decoding]] provides the verification framework needed to preserve target behavior.
