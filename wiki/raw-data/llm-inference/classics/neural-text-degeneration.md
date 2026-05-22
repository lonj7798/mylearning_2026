<!-- scope: decoding failure modes and nucleus sampling for open-ended generation
     deps: [[language-models-are-unsupervised-multitask-learners]], [[beam-search]]
     see-also: [[hf-generation-strategies]]
-->

# The Curious Case of Neural Text Degeneration
- **Core Insight:** High-likelihood decoding methods such as greedy and beam search often produce repetitive, bland, or looping text for open-ended generation; sampling from a dynamic nucleus better matches human text statistics.
- **Guideline:** Use deterministic search for constrained tasks, but prefer calibrated sampling methods such as nucleus sampling for open-ended continuation.
- **Authors:** Ari Holtzman, Jan Buys, Li Du, Maxwell Forbes, Yejin Choi
- **Year:** 2019
- **URL:** https://arxiv.org/abs/1904.09751
- **Relevant topics:** nucleus sampling, top-p, repetition, beam search, open-ended generation

## Abstract
The paper investigates why neural language models produce degenerate text under maximum-likelihood-style decoding. It shows that human text does not usually follow the highest-probability path at every step and proposes nucleus sampling, which samples from the smallest token set whose cumulative probability exceeds threshold `p`.

## Key Contributions
- Diagnosed repetition and dullness as decoding-time failures, not only model-training failures.
- Showed that beam search is poorly matched to open-ended generation despite strong likelihood.
- Introduced nucleus sampling (`top_p`) as an adaptive alternative to fixed top-k sampling.
- Compared decoding methods using diversity, likelihood, and human judgments.
- Made decoding strategy a core part of LLM output quality.

## Key Figures/Tables to Study
- **Figure 1:** Examples of degeneration under beam search and greedy decoding.
- **Probability mass analysis:** The unreliable tail motivates truncating the distribution before sampling.
- **Human evaluation tables:** Quality/diversity comparisons across decoding strategies.
- **Nucleus sampling definition:** The smallest vocabulary subset with cumulative probability >= `p`.

## Technical Details
For each generation step, sort tokens by probability. Nucleus sampling chooses the smallest set `V_p` such that:

```text
sum_{x in V_p} P(x | context) >= p
```

Then it renormalizes probabilities inside `V_p` and samples. Unlike top-k, the candidate set grows when uncertainty is high and shrinks when the model is confident.

The paper's key inference lesson is that the argmax sequence is not necessarily the best text. Search can over-optimize local model likelihood and enter repetitive attractors. Sampling injects diversity while truncation removes low-quality tail tokens.

## Connections
- [[beam-search]]: explains why beam search remains useful for translation/summarization but risky for open-ended chat or story generation.
- [[hf-generation-strategies]]: maps nucleus sampling to `do_sample=True, top_p=...`.
- [[language-models-are-unsupervised-multitask-learners]]: GPT-2 was a key model family in the degeneration analysis.
- [[openai-streaming-and-token-usage]]: decoding policy affects output length and therefore streamed token usage.
