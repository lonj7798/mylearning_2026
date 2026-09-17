---
chapter: ch-12a
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/physics-of-lm-3-2-knowledge-manipulation.md (planned card; not present on 2026-09-15)
source_url: https://arxiv.org/abs/2309.14402
created_at: "2026-09-15"
---

# Excerpt: Physics of Language Models: Part 3.2, Knowledge Manipulation

**Authors:** Zeyuan Allen-Zhu (Meta AI / FAIR Labs), Yuanzhi Li (Mohamed bin Zayed University of AI)
**Version read:** arXiv:2309.14402v2 (16 Jul 2024); v1 September 2023. V2 adds Llama/Mistral experiments and larger data; the authors state conclusions are unchanged.
**Status:** no library card existed for this slug on 2026-09-15; values read in the v2 PDF text.

## Tasks (Abstract, §1)
Retrieval ("What is A's attribute X?"), classification ("Is A's attribute X even or odd?"), comparison ("Is A greater than B in attribute X?"), inverse search ("Which person's attribute X equals T?").

## Setup (§2, §4)
- bioS data from Part 3.1 with N = 100k (appendix: N = 2M or 5M). Manipulation experiments start from a model pretrained on bioS multi5+permute, which extracts birth dates at nearly 100% and majors at 98% (§4).
- LoRA fine-tuning on the manipulation task for P_train individuals (|P_train| = 2.5k to 50k); OOD test on P_test (50k) (§4).
- "Hints" = chain-of-thought: the attribute is written out before the answer; during training, hints are included with probability 0.5 (§4).

## Results
- Result 3 (Fig. 4): without CoT, even/odd birth month needs 10,000 training samples to reach 75% test accuracy; ranking months needs 50,000 samples for 85%; ranking 100 majors "barely outperforms random" even with 2.5 million samples.
- Fig. 4 row "major classify %5 (10k)", BIO-pretrained model: chance 20.0; trained without hints 23.6; trained with hints and tested with hints 86.4; trained with hints and tested without hints 24.1; hint accuracy 84.5.
- Result 4: including CoT examples in training does not improve test accuracy when hints are removed. Footnote 14 (worked example): hint accuracy 91.0% predicts 91.0% + (1 − 91.0%) × 50% = 95.5% test accuracy with hint; observed 94.2% (birth month classify %2).
- Result 5: a model first QA-fine-tuned for extraction does not manipulate knowledge better than the BIO-pretrained model.
- Result 6 (Fig. 5): GPT-4 compares birth dates of celebrities born 1900-1950 at 71.1%, and 1900-1910 at 52.3%.
- Result 7 (Fig. 6): inverse search accuracy is near zero on P_test for all 16 non-reversed bioS datasets, for both QA fine-tuning and BIO+QA mixed training (full-name accuracy 0.0 in every non-reversed row of the fine-tuning block). Accuracy rises only when the name appears after attributes in the pretraining data (multi5+reverse1/2/6 datasets), which the authors note is no longer inverse search. Holds for Llama and for GPT2/Llama/Mistral on 50x larger data and 5.5x larger models (Fig. 14).
- Authors' interpretation (§5): the limitation "is due to its left-to-right autoregressive training design. If the model learns 'A equals B' it cannot infer 'B equals A' unless it is also in the training data."
- Result 8 (Fig. 7): GPT-4 predicts the next sentence of Pride and Prejudice with 65.9% accuracy and the preceding sentence with 0.8%.
- Result 9: to support inverse search, use RAG, add reversed knowledge to training data (for example via a rewrite prompt), or add line numbers; GPT-4 finds a preceding Bible verse via CoT on verse numbers (Fig. 8).

## Limits
GPT-4 experiments are illustrative because its training data is unknown (§4, §5); controlled results are on synthetic biographies.

## How ch-12a uses it
§6 (manipulation, CoT, inverse search), §9 (reversed data), Generalization lens (b), Common mistakes.
