---
chapter: ch-12a
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/long-tail-knowledge.md (planned card; not present on 2026-09-15)
source_url: https://arxiv.org/abs/2211.08411
created_at: "2026-09-15"
---

# Excerpt: Large Language Models Struggle to Learn Long-Tail Knowledge

**Authors:** Nikhil Kandpal, Haikang Deng, Adam Roberts, Eric Wallace, Colin Raffel (UNC Chapel Hill, Google Research, UC Berkeley)
**Version read:** arXiv:2211.08411v2 (27 Jul 2023); v1 November 2022; ICML 2023.
**Status:** no library card existed for this slug on 2026-09-15; values read in the v2 PDF text.

## Method (§2)
- Entity linking with DBpedia Spotlight over The Pile (825GB), ROOTS English (490GB), C4 (305GB), OpenWebText (39GB), and Wikipedia (Dec 2018); 2.1TB took about 3 weeks on a 128-core machine (§2.1).
- Relevant document = document containing both the salient question entity and the answer entity (§2.2). QA data: TriviaQA and Natural Questions, train and validation splits.
- Human check on 300 TriviaQA pairs: 33% of sampled relevant documents contain enough information to answer, 27% some relevant information; about 60% precision (§2.3).
- Models: GPT-Neo family (125M-20B, the Pile), BLOOM (560M-176B, ROOTS), GPT-3 (counts estimated, appendix only). 4-shot prompts, greedy decoding, exact match (§3).

## Results
- Correlation (§3.1, Fig. 1): BLOOM-176B TriviaQA accuracy rises from 25% to above 55% as relevant documents increase from 10^1 to 10^4. BLOOM-176B has over 4× the accuracy of BLOOM-560M on questions with more than 10^5 relevant documents. Similar trends on Natural Questions.
- Document counts correlate across corpora: Spearman 0.87-0.97 between ROOTS, Pile, C4, OpenWebText, and Wikipedia (Table 1).
- Question-entity-only or answer-entity-only counts stop correlating with accuracy when co-occurrence is below 5 (Fig. 13).
- Human accuracy on Natural Questions with background text is highest for questions with few relevant documents, the opposite of models (Fig. 7).
- Causal test (§3.2, Fig. 5): a 4.8B LM trained one epoch on C4, then a counterfactual LM trained without all relevant documents for 100 sampled TriviaQA questions per log-spaced bin (about 30% of C4 removed). Accuracy drops are large for questions with many relevant documents and small for questions with few.
- Scaling (§4.2, Fig. 6): for Natural Questions items with fewer than 100 relevant documents, BLOOM accuracy is log-linear in parameters (R² = 0.98); matching a strong supervised baseline or human accuracy would require over 10^18 parameters by extrapolation. (The introduction gives "one quadrillion parameters" as its example.)
- Data scaling (§4.1): the authors expect moderate dataset scaling (for example 5×) to give small gains, because sources are correlated.
- Retrieval (§4.3): oracle gold paragraphs remove the low-count accuracy drop (Fig. 7); BM25 top-3 Wikipedia paragraphs improve accuracy in all count ranges, especially rare ones; BM25 recall has a mild dependence on count (Fig. 8, Fig. 9).

## Limits
Entity-linking errors; closed-book factoid QA only; GPT-3 counts are estimates.

## How ch-12a uses it
§7 (long-tail frequency dependence), Generalization lens (c) (stratify by relevant-document count), Common mistakes.
