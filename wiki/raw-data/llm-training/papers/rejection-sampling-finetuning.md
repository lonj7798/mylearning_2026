<!-- scope: rejection-sampling fine-tuning as iterative post-training data improvement
     see-also: [[best-of-n]], [[west-of-n]], [[iterative-sft-rl]]
-->

# Rejection-Sampling Fine-Tuning
- **Core Insight:** Sampling multiple candidates from the current policy and retaining only the highest-scoring ones is a simple but powerful way to turn inference compute into better supervised data.
- **Guideline:** Before full RL, run best-of-N generation with a reward model or verifier and fine-tune on the accepted samples; this is often the highest-leverage first iteration.
- **Authors:** Commonly surfaced in Llama 2 and later open post-training recipes
- **Year:** 2023
- **URL:** https://arxiv.org/abs/2307.09288
- **Relevant topics:** best-of-N, rejection sampling, iterative post-training, RSFT

## Abstract
Rejection-sampling fine-tuning is not a standalone algorithm paper so much as a recurring post-training recipe: generate multiple responses per prompt, score them with a reward model or rule-based verifier, keep the best responses, and fine-tune the model on that filtered set. It became a standard bridge between SFT and RLHF/RLVR.

## Key Contributions
- Turned policy sampling into a self-improving supervised-data loop.
- Often delivers large quality gains before expensive online RL.
- Provides cleaner data than raw teacher distillation because selection is prompt-specific.

## Technical Details
- Sample `N` candidates per prompt.
- Score with RM, judge model, or rule-based verifier.
- Keep top candidates, discard low scorers.
- Fine-tune the policy on the accepted set and optionally repeat.

## Connections
- Related to [[best-of-n]] and [[west-of-n]].
- Frequently appears inside iterative stacks summarized in [[iterative-sft-rl]].

