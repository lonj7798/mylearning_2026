---
chapter: ch-12a
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/physics-of-lm-3-1-knowledge-storage.md (planned card; not present on 2026-09-15)
source_url: https://arxiv.org/abs/2309.14316
created_at: "2026-09-15"
---

# Excerpt: Physics of Language Models: Part 3.1, Knowledge Storage and Extraction

**Authors:** Zeyuan Allen-Zhu (Meta / FAIR Labs), Yuanzhi Li (Mohamed bin Zayed University of AI)
**Version read:** arXiv:2309.14316v3 (16 Jul 2024); v1 September 2023. V3 adds Llama experiments.
**Status:** no library card existed for this slug on 2026-09-15; values read in the v3 PDF text.

## Data (§2)
- bioS: N = 100,000 synthetic people, six attributes (birth date, birth city, university, major, company name, company city), six sentences, each from about 50 templates. "bioS single" = one biography per person with fixed sentence order.
- Augmentations (§2, §4.2): multiM (M differently templated entries per person), permute (shuffle sentences), fullname (replace pronouns with the full name). bioR: biographies written by Llama.
- QA: six questions per person; exact-match generation accuracy. Fine-tune on QAs for half of the people (P_train), test on the other half (P_test), whose QAs are never trained.

## Models and training (§2.1, App. C, App. D)
- GPT2 with rotary embedding: 12 layers, 12 heads, 768 dims (124M) for bioS; 12-20-1280 (302M) for bioR; 12-32-2048 (682M) for the Fig. 2 negative result. Llama architecture also used.
- Pretraining and mixed training (App. C): 512-token windows of concatenated biographies; AdamW, weight decay 0.1, ε 1e-6, LR 0.001, 1,000-step linear warmup, cosine decay to 0.0001; batch 96; 80,000 steps (bioS), 150,000 steps (bioR), 200,000 steps (682M). Next-token accuracy on attribute tokens well above 99% (Remark C.1).
- Mixed training: QA_r = 0.8, a 2:8 BIO-to-QA token ratio (App. C).
- QA fine-tuning (App. D): full fine-tuning with LR {0.001, 0.0003, 0.0001}, weight decay {0.01, 0.001}, no warmup, cosine to 10%, batch 48, 50,000 steps, best test accuracy reported; LoRA with weight decay 0.01, LR 0.0003.

## Results (mean QA accuracy on P_test, GPT2, bioS, Fig. 3: BIO pretrain + QA fine-tune | mixed training)
| Pretraining data | Fine-tune | Mixed |
|---|---|---|
| bioS single | 9.7 | 86.6 |
| single + fullname | 48.9 | 85.9 |
| single + permute1 | 4.4 | 82.5 |
| single + permute5 | 70.0 | 93.7 |
| multi2 | 41.1 | 89.2 |
| multi5 | 41.0 | 91.8 |
| multi5 + fullname | 82.4 | 92.0 |
| multi5 + permute | 96.6 | 95.5 |
| multi5 + permute + fullname | 96.2 | 95.7 |

- Result 1 (§3): mixed training on bioS single gives 86.6% OOD accuracy, bioR single 77.7%; the model first fits QAs, then aligns them with biographies.
- Result 2 (§4.1, Fig. 2): despite 99+% first-token accuracy, bioS single gives near-zero test accuracy for all fine-tuning parameters; holds for the 682M model with 1,350 exposures per person. For bioS single, birth date (always the first attribute) reaches 33.5% (Fig. 3). Footnote: follow-up work with 1B models and N = 20M confirms similar results.
- Result 3 (§4.2): five diverse entries with shuffling raise accuracy from 9.7% to 96.6%; more augmentation gives higher accuracy. English-to-French translation raises accuracy to "about 40%" (footnote).
- Result 4 (§5.1, Fig. 5): P-probing; in bioS single, company-name probe accuracy is about 2% until the token right before the attribute, where it reaches 100%; in multi5 + permute, all six attributes are predictable at nearly 100% from the first position after the name.
- Result 5 (§5.2, Fig. 7): QA fine-tune accuracy tracks Q-probing accuracy (linear probe on the hidden state at the end of the name alone).
- Result 6 (§6, Fig. 8): adding 100,000 augmented "celebrity" people to pretraining raises QA accuracy for unaugmented "minority" people from 4.4% to 86.8% (bioS) and from 10.0% to 76.3% (bioR); minority QAs are not used in fine-tuning. Replacing celebrity data with WikiBook does not help (Remark 6.1).
- Result 7 (§7, Fig. 9): a GPT2-based bidirectional model with masked LM does not store knowledge for extraction unless the attribute is a single word or independent words.

## Recommendations stated by the authors (Abstract, §1, §8)
Rewrite pretraining data with small auxiliary models to provide knowledge augmentation, especially for rare but critical data; add more instruction-style (QA) data into pretraining.

## Limits
Synthetic biographies, six attributes, models up to 682M in this paper.

## How ch-12a uses it
§5 (storage vs extraction), §9 (data design), Recipe, Generalization lens (a), Common mistakes.
