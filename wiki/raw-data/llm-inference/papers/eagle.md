<!-- scope: feature-level speculative sampling framework for LLM acceleration
     deps: fast-inference-from-transformers-via-speculative-decoding
     see-also: eagle-2, medusa, self-speculative-decoding
-->

# EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty
- **Core Insight:** Drafting in feature space can be easier and more accurate than directly drafting discrete future tokens, if token uncertainty is handled explicitly.
- **Guideline:** Use EAGLE-style feature predictors when high acceptance rate matters and you can train a lightweight drafter tied to the target model's hidden representations.
- **Authors:** Yuhui Li, Fangyun Wei, Chao Zhang, Hongyang Zhang
- **Year:** 2024
- **URL:** https://arxiv.org/abs/2401.15077
- **Relevant topics:** speculative sampling, feature extrapolation, draft model, tree verification, lossless acceleration

## Abstract
EAGLE argues that speculative decoding should account for uncertainty in next-feature prediction, not just next-token prediction. It drafts autoregressively at the second-to-top-layer feature level and incorporates the token from one step ahead to reduce uncertainty. The target LLM verifies draft candidates in parallel, preserving the target distribution while improving speed through higher-quality drafts.

## Key Contributions
- Introduces Extrapolation Algorithm for Greater Language-model Efficiency.
- Drafts target-model features instead of directly relying on a separate token-level LM.
- Uses one-step-ahead token information to handle feature uncertainty.
- Builds draft trees for parallel verification by the target model.
- Reports strong speedups across model families while preserving generation quality.

## Key Figures/Tables to Study
- Feature-level drafting diagram: shows why the drafter operates near the target model's top layers.
- Uncertainty analysis: explains the one-step-ahead token design.
- Draft tree construction section: connects to verification cost and acceptance length.
- Speedup/acceptance tables: compare against vanilla speculative decoding and Medusa.

## Technical Details
The EAGLE drafter predicts the next hidden feature representation conditioned on current context and token information. The target model's remaining layers or head can turn those features into token candidates. Candidate paths are verified by the original model, so accepted outputs follow the target distribution.

The systems tradeoff is that EAGLE requires training and storing a feature-level draft module. In return, the draft model can be more aligned with the target model than an arbitrary smaller assistant model, increasing the expected number of accepted tokens per target verification pass.

## Connections
- [[eagle-2]] improves EAGLE with context-aware dynamic draft trees.
- [[medusa]] also avoids a separate general-purpose assistant but uses future-token heads.
- [[hf-assisted-generation]] is the library-level assistant-model interface; EAGLE is a stronger specialized proposer.
- [[multi-token-prediction-inference]] is another way to make the target model expose future-token proposals.
- [[eagle-2]] should be read immediately after this page because it changes the tree policy, not the core drafter premise.
