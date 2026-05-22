<!-- scope: exact parallel decoding method without auxiliary draft model
     deps: transformer-inference-loop
     see-also: self-speculative-decoding, speculative-decoding
-->

# Break the Sequential Dependency of LLM Inference Using Lookahead Decoding
- **Core Insight:** Some future n-grams can be found and verified in parallel from the LLM itself, reducing sequential dependency without a separate draft model.
- **Guideline:** Use lookahead decoding when auxiliary models or extra heads are undesirable and exactness is required, but validate speedups on the target workload.
- **Authors:** Yichao Fu, Peter Bailis, Ion Stoica, Hao Zhang
- **Year:** 2024
- **URL:** https://arxiv.org/abs/2402.02057
- **Relevant topics:** lookahead decoding, exact decoding, Jacobi iteration, n-gram verification, no draft model

## Abstract
Lookahead decoding is an exact parallel decoding algorithm for LLM inference that does not require an auxiliary draft model or datastore. It uses a lookahead branch to generate candidate n-grams and a verification branch to accept valid tokens. The method exploits parallelism available inside a transformer forward pass while preserving the output of the original model.

## Key Contributions
- Proposes an exact acceleration method without a separate draft model.
- Uses a lookahead mechanism to generate and maintain candidate n-grams.
- Verifies candidates with the original model to preserve correctness.
- Provides an open-source implementation from the LMSYS/hao-ai-lab line.
- Reports speedups on LLaMA-style models and chat workloads.

## Key Figures/Tables to Study
- Lookahead branch and verification branch diagram: core mental model.
- Algorithm pseudocode: tracks candidate n-gram pool and verification.
- Speedup tables: inspect dependence on model, sequence, and hyperparameters.
- GitHub demo: useful for connecting paper algorithm to serving behavior.

## Technical Details
The method maintains candidate continuations produced from parallel lookahead computation. The verification branch checks whether the original model would produce those tokens. Accepted n-grams advance the output by multiple tokens; failed candidates are discarded or refreshed.

Unlike assistant-model speculative decoding, no separate model has to be trained, loaded, or synchronized. The cost is that the candidate generation strategy is more constrained and may deliver variable acceptance depending on repetition and task structure.

## Connections
- [[self-speculative-decoding]] also avoids an external assistant but drafts by skipping layers.
- [[prompt-lookup-decoding]] is a simpler n-gram reuse method for prompts with repeated text.
- [[fast-inference-from-transformers-via-speculative-decoding]] is the standard assistant-model comparison point.
- [[speculative-decoding]] provides the broader draft/verify vocabulary used to classify it.
- [[hf-assisted-generation]] is useful contrast because it exposes the assistant-model implementation path.
