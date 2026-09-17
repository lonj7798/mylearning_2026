---
chapter: ch-00
course: llm-training
phase: read
excerpt_of: primary source, OpenAI PDF (no library card as of 2026-09-15)
source_url: https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf
created_at: "2026-09-15"
---

# Excerpt: Language Models are Unsupervised Multitask Learners (GPT-2)

**Paper:** Radford, Wu, Child, Luan, Amodei, Sutskever (OpenAI), 2019. Source type: official technical report (not on arXiv; the PDF carries no date).

## Framing (§1)
- Supervised single-task systems are "brittle and sensitive to slight changes in the data distribution" and are "better characterized as narrow experts rather than competent generalists".
- The authors suspect that single-task training on single-domain datasets is a major contributor to the lack of generalization, and propose measuring performance on a wide range of domains and tasks.

## Zero-shot results (Abstract)
- Conditioned on a document plus questions, the model reaches 55 F1 on CoQA, matching or exceeding 3 of 4 baseline systems without using the 127,000+ training examples.
- Capacity is essential to zero-shot task transfer; increasing it improves performance log-linearly across tasks.
- GPT-2 (1.5B) is state of the art on 7 of 8 tested language-modeling datasets zero-shot and still underfits WebText.

## Generalization vs memorization (§4, Table 6)
- Bloom filters of 8-grams from WebText training text (lower-cased alphanumeric words, single-space delimiter); false-positive rate upper bounded by 1/10^8.
- Test sets of common language-modeling datasets share 1-6% of 8-grams with WebText train, 3.2% on average; many share more with their own training splits, 5.9% on average; 1BW test shares 13.19% with its own training set (Table 6).
- LAMBADA: average overlap 1.2%; excluding all examples with any overlap shifts perplexity from 8.6 to 8.7 and accuracy from 63.2% to 62.9%.
- CoQA: about 15% of news-domain documents are in WebText, where the model scores about 3 F1 higher; overall gain about 0.5-1.0 F1; no CoQA questions or answers are in WebText because CoQA was released after the WebText link cutoff.
- Winograd Schema Challenge: 10 schemata with any 8-gram overlap, 2 of them spurious.
- Recommendation: n-gram overlap-based de-duplication as a verification step when creating training and test splits.

## Verification
- Read on 2026-09-15 against the OpenAI PDF text (Abstract, §1, §4, Table 6).
