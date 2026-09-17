---
chapter: ch-08b
course: llm-training
phase: read
excerpt_of: primary source, OpenAI PDF (no library card as of 2026-09-15)
source_url: https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf
created_at: "2026-09-15"
---

# Excerpt: Language Models are Unsupervised Multitask Learners (GPT-2)

**Paper:** Radford, Wu, Child, Luan, Amodei, Sutskever (OpenAI), 2019. Source type: official technical report (not on arXiv; the PDF carries no date; GPT-3 cites it as 2019).

## Framing (§1, §2)
- Supervised single-task systems are described as "narrow experts rather than competent generalists" (§1).
- The two largest multitask NLP efforts cited trained on 10 and 17 (dataset, objective) pairs; from a meta-learning view each pair is one training example (§1).
- Eq. 1: p(x) = Π_i p(s_n | s_1, ..., s_{n−1}) (§2).
- A general system should model p(output | input, task); language can express task, input, and output as one sequence, e.g. (translate to french, english text, french text) (§2).
- Because the supervised objective equals the unsupervised objective evaluated on a subset of the sequence, the global minimum of the unsupervised objective is also the global minimum of the supervised objective; preliminary experiments found learning "much slower than in explicitly supervised approaches" (§2).
- Speculation: a model with sufficient capacity "will begin to learn to infer and perform the tasks demonstrated in natural language sequences in order to better predict them" (§2).

## Data (§2.1)
- Rejected approach: Trinh & Le's Common Crawl subsample selected for similarity to the Winograd Schema Challenge; the authors "want to avoid making assumptions about the tasks to be performed ahead of time".
- WebText: outbound Reddit links with at least 3 karma; 45 million links; Dragnet and Newspaper extractors.
- Preliminary version used for all results: no links after Dec 2017; after de-duplication and heuristic cleaning, slightly over 8 million documents, 40 GB of text; all Wikipedia documents removed.
- Table 1: naturally occurring English-French translation demonstrations found in WebText.

## Input representation and model (§2.2, §2.3, Table 2)
- Byte-level BPE; merges across character categories blocked, with an exception for spaces; code-point BPE would need a base vocabulary over 130,000.
- Vocabulary 50,257; context 1024 tokens (up from 512); "a larger batchsize of 512 is used" (unit not stated).
- Pre-layer normalization, extra layer norm after the final self-attention block, residual weights scaled by 1/√N at initialization (N = number of residual layers).
- Table 2: 117M / 12 layers / d_model 768; 345M / 24 / 1024; 762M / 36 / 1280; 1542M / 48 / 1600.
- Learning rate of each model manually tuned for best perplexity on a 5% held-out WebText sample; values not printed. All models still underfit WebText (§3).
- Not reported anywhere in the report: optimizer, schedule, tokens or steps trained, compute.

## Zero-shot results (§3)
- Language modeling: state of the art on 7 of 8 datasets; de-tokenizers give 2.5 to 5 perplexity gains (§3.1, Table 3).
- CBT: 93.3% common nouns, 89.1% named entities (validation set; test book The Jungle Book is in WebText) (§3.2).
- LAMBADA: perplexity 99.8 → 8.6; accuracy 19% → 52.66%, 63.24% with a stop-word filter (§3.3).
- Winograd Schema Challenge: 70.70%, 273 examples (§3.4).
- CoQA: greedy decoding conditioned on document, conversation history, and "A:" gives 55 F1 on the development set, matching or exceeding 3 of 4 baselines trained on 127,000+ question-answer pairs (§3.5).
- Summarization with "TL;DR:", top-k sampling k = 2, first 3 generated sentences (§3.6). Table 4 R-AVG: Bottom-Up Sum 32.75; Lede-3 31.55; Seq2Seq + Attn 23.99; GPT-2 TL;DR: 21.40; Random-3 20.98; GPT-2 no hint 15.03. Text: removing the hint drops the aggregate by 6.4 points.
- Translation (§3.7): WMT-14 En→Fr 5 BLEU; Fr→En 11.5 BLEU (best unsupervised MT 33.5). Non-English pages were deliberately removed; a byte-level language detector found 10 MB of French, about 500x smaller than monolingual French corpora in prior unsupervised MT work.
- Natural Questions (§3.8): 4.1% exact match; the smallest model does not exceed a 1.0% most-common-answer baseline; GPT-2 answers 5.3 times more questions correctly; 63.1% accuracy on the 1% most confident questions.
- Abstract and Fig. 1: zero-shot task performance improves in a log-linear fashion with capacity.

## Generalization vs memorization (§4, Table 6)
- Bloom filters of WebText 8-grams; test sets of common LM datasets share 1-6% of 8-grams with WebText train (average 3.2%) and 5.9% on average with their own training splits.
- LAMBADA: excluding examples with any overlap changes perplexity 8.6 → 8.7 and accuracy 63.2% → 62.9%.
- Train and test WebText perplexity improve together with model size (Fig. 4).

## Discussion (§6)
- Zero-shot performance "is still far from use-able" for practical applications; on many practical tasks performance is likely no better than random.

## Verification
- Read on 2026-09-15 against the OpenAI PDF text (Abstract, §1-§4, §6-§7, Tables 1-6). No library card exists; this excerpt is the citable extract for ch-08b.
