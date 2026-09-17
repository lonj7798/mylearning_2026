---
chapter: ch-23
course: llm-training
phase: read
excerpt_of: primary source arXiv:2211.08411v2 (no library card as of 2026-09-15; ch-12a holds a longer excerpt)
source_url: https://arxiv.org/abs/2211.08411
created_at: "2026-09-15"
---

# Excerpt: Large Language Models Struggle to Learn Long-Tail Knowledge

**Paper:** Nikhil Kandpal, Haikang Deng, Adam Roberts, Eric Wallace, Colin Raffel. arXiv v1 2022-11; v2 2023-07-27 read (ICML 2023, PMLR 202). Source type: paper.

## Measurement (§2)
- A relevant document for a QA pair contains both the salient question entity and the answer entity, found by entity linking the pretraining corpus (The Pile, ROOTS, C4, OpenWebText, Wikipedia).
- QA sets: TriviaQA and Natural Questions; accuracy is reported per log-scaled bin of relevant-document count (§3).

## Results
- BLOOM-176B TriviaQA accuracy rises from 25% to above 55% as relevant documents increase from 10¹ to 10⁴ (§1, §3.1).
- BLOOM-176B has over 4× the accuracy of BLOOM-560M on TriviaQA questions with more than 10⁵ relevant documents (§3.1).
- Counterfactual: a 4.8B model retrained on C4 with the relevant documents of sampled questions removed (about 30% of C4) loses accuracy on those questions, and the loss is larger for questions that had more relevant documents (§3.2, Fig. 5).
- Relevant-document counts are highly correlated across corpora (Spearman 0.87–0.97, Table 1, §4.1).
- For Natural Questions items with fewer than 100 relevant documents, a log-linear fit of BLOOM accuracy against parameters extrapolates to over 10¹⁸ parameters to match a strong supervised baseline or human performance (§4.2, Fig. 6).
- Oracle retrieval (gold paragraph) and BM25 retrieval raise accuracy most on rare questions; BM25 recall still depends mildly on document count (§4.3, Figs. 7–9).

## Verification
- Read on 2026-09-15 against arXiv:2211.08411v2 PDF text (Abstract, §1, §3, §4).
