<!-- scope: multi-head speculative decoding framework attached to the target LLM
     deps: fast-inference-from-transformers-via-speculative-decoding
     see-also: eagle, multi-token-prediction-inference
-->

# Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads
- **Core Insight:** A target LLM can grow lightweight extra heads that propose multiple future tokens, avoiding the deployment burden of a separate draft model.
- **Guideline:** Consider Medusa-style heads when you can fine-tune or augment the target model and want speculative speedups without maintaining an assistant checkpoint.
- **Authors:** Tianle Cai, Yuhong Li, Zhengyang Geng, Hongwu Peng, Jason D. Lee, Deming Chen, Tri Dao
- **Year:** 2024
- **URL:** https://arxiv.org/abs/2401.10774
- **Relevant topics:** multi-token heads, tree attention, speculative decoding, single-model acceleration, LLM serving

## Abstract
Medusa augments an LLM with multiple decoding heads that predict several future tokens from the current hidden state. The heads generate candidate continuations arranged as a tree. A tree-based attention mechanism allows the original model to verify multiple candidates in parallel, accepting a valid prefix while preserving target-model behavior under the verification process.

## Key Contributions
- Removes the need for a separately deployed draft model.
- Adds multiple lightweight decoding heads trained to predict future token offsets.
- Uses tree attention to verify several candidate branches in one target-model pass.
- Offers Medusa-1 and Medusa-2 training modes for different access/quality settings.
- Reports practical speedups with modest additional parameters.

## Key Figures/Tables to Study
- Medusa architecture figure: shows the base model plus future-token heads.
- Tree attention diagram: central to understanding multi-branch verification.
- Training objective section: compare head-only training and joint fine-tuning.
- Latency/speedup tables: inspect speedup versus acceptance length.

## Technical Details
At each decode step, the base LLM computes hidden states. Medusa heads predict tokens at future offsets, producing a pool of candidates. These candidates form a draft tree rather than a single linear draft. The verification pass masks attention so each branch is evaluated consistently with its prefix.

The method trades extra training and a modest parameter increase for simpler deployment. It is most attractive when an operator owns the model weights and can add heads, and less attractive when only black-box target access or arbitrary off-the-shelf models are available.

## Connections
- [[fast-inference-from-transformers-via-speculative-decoding]] supplies the general verification idea.
- [[eagle]] uses feature-level drafting instead of independent output heads.
- [[multi-token-prediction-inference]] trains future-token heads as a core objective rather than only an inference add-on.
- [[eagle-2]] shares the tree-verification concern but dynamically changes candidate branching.
- [[self-speculative-decoding]] is another single-model path, but it drafts by skipping layers.
