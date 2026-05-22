<!-- scope: beam search as sequence-level approximate decoding
     deps: []
     see-also: [[neural-text-degeneration]], [[hf-generation-strategies]]
-->

# Beam Search Strategies for Neural Machine Translation
- **Core Insight:** Beam search keeps the top-scoring partial sequences at each step, approximating sequence-level maximization without enumerating the full output tree.
- **Guideline:** Use beam search for constrained sequence tasks where likelihood aligns with quality; avoid assuming larger beams improve open-ended LLM generations.
- **Authors:** Markus Freitag, Yaser Al-Onaizan
- **Year:** 2017
- **URL:** https://arxiv.org/abs/1702.01806
- **Relevant topics:** beam search, decoding, sequence scoring, length normalization, translation

## Abstract
Beam search is a classic decoding algorithm for sequence models. Freitag and Al-Onaizan study strategies for making neural machine translation beam search faster by varying candidate sizes over time. The paper is useful as a compact primary source for beam-search mechanics and tradeoffs before LLM-specific decoding concerns.

## Key Contributions
- Formalized practical beam search for neural machine translation decoders.
- Explored flexible candidate selection to speed decoding without losing translation quality.
- Highlighted the latency/quality tradeoff controlled by beam size.
- Reinforced the distinction between model probability and search procedure.
- Provides a baseline for comparing later LLM sampling methods.

## Key Figures/Tables to Study
- **Beam-search algorithm description:** Stepwise expansion, scoring, pruning.
- **Speed/quality tables:** Show that candidate pruning can preserve BLEU while reducing work.
- **Beam size comparisons:** Larger beams improve search but increase compute and may change length bias.
- **Stopping criteria discussion:** Completed hypotheses must be compared against unfinished ones carefully.

## Technical Details
At each step, beam search:

```text
1. Expands each active hypothesis with candidate next tokens.
2. Adds token log probabilities to cumulative sequence scores.
3. Keeps only the best B partial hypotheses.
4. Moves completed hypotheses aside when EOS is generated.
```

The beam width `B` controls branching. `B=1` is greedy search. Larger `B` explores more alternatives but multiplies decode compute and memory for active hypotheses.

LLM inference often uses sampling instead of beam search for open-ended text because high-probability beams can collapse into repetitive generic continuations. For translation, summarization, and constrained code-like formats, beam search can still be reasonable.

## Connections
- [[neural-text-degeneration]]: shows why high-likelihood beam outputs can degenerate in open-ended generation.
- [[hf-generation-strategies]]: `num_beams` and `early_stopping` expose beam-search behavior in Transformers.
- [[batching-for-inference]]: multiple beams behave like multiple active sequences, increasing KV-cache and scheduler load.
- [[openai-streaming-and-token-usage]]: beam search is usually hidden in hosted APIs but affects latency when exposed.
