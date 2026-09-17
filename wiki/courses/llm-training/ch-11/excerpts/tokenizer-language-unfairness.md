---
chapter: ch-11
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/tokenizer-language-unfairness.md (planned card; not present on 2026-09-15)
source_url: https://arxiv.org/abs/2305.15425
created_at: "2026-09-15"
---

# Excerpt: Language Model Tokenizers Introduce Unfairness Between Languages

**Authors:** Aleksandar Petrov, Emanuele La Malfa, Philip H.S. Torr, Adel Bibi (University of Oxford).
**Version read:** arXiv:2305.15425 (PDF served 2026-09-15; NeurIPS 2023); v1 May 2023.
**Status:** no library card existed for this slug on 2026-09-15; values read in the PDF text at the stated locus.

## Definition (§3)
For a sentence s_A in language A and its translation s_B, the premium of A relative to B under tokenizer t is |t(s_A)| / |t(s_B)|. Parity holds when the ratio is close to 1.

## Data (§4)
FLORES-200: the same 2,000 Wikipedia sentences human-translated into 200 languages. Languages with more than 10% of characters mapped to UNK are reported as "—".

## Table 1 (premium relative to English; English-centric tokenizers)
| Language | GPT-2 / RoBERTa | ChatGPT / GPT-4 (cl100k_base) | FlanT5 |
|---|---|---|---|
| Portuguese | 1.94 | 1.48 | 2.21 |
| German | 2.14 | 1.58 | 1.37 |
| Italian | 2.01 | 1.64 | 2.18 |
| Chinese (Simplified) | 3.21 | 1.91 | — |
| Japanese | 3.00 | 2.30 | — |
| Bulgarian | 5.51 | 2.64 | — |
| Standard Arabic | 4.40 | 3.04 | — |
| Burmese | 16.89 | 11.70 | — |
| Shan | 18.76 | 15.05 | — |

## Other results
- Multilingual tokenizers (Table 4, §4.3): XLM-R, NLLB, mT5, M2M100 (SentencePiece with upsampling of rare languages) and BLOOM (byte-level BPE) are closer to parity than English-centric tokenizers, but "all five models have languages with premiums of more than 2.5". Examples: Shan 4.43 (XLM-R), 1.94 (NLLB), 12.06 (BLOOM).
- Byte-level and character-level (Table 5, §4.4): ByT5 (UTF-8 bytes) premiums range from 0.87 (Yue Chinese) to 3.94 (Shan); CANINE (UTF-32 codepoints) gives Shan a premium of 4.58 relative to Yue Chinese. Latin characters need 1 byte, Cyrillic, Greek, Arabic 2 bytes, CJK 3 bytes.
- Tokenizers trained for other target languages (Table 2, §4.2): CamemBERT, GottBERT, PhoBERT give English the lowest premium among non-target languages (1.20, 1.35, 1.20).
- Latency (§5.2, Fig. 2): RoBERTa processing time is linear in tokenized length; Shan takes almost twice the time of English.
- Context (§5.3): for Burmese and Dzongkha a fixed context holds "less than a tenth of the content" that it holds in English.
- Vocabulary share (§6, Fig. 3): encoding the English FLORES-200 corpus with one third of the cl100k_base vocabulary makes English sequences about 10% longer; a 10-fold reduction makes them about 30% longer.
- Proposal (§6): train monolingual tokenizers, then merge them starting from the 256 byte tokens and repeatedly add the most frequent token of the language with the highest premium. The paper does not train a model with such a tokenizer.

## Limits
Premiums are tokenizer properties measured on one parallel corpus with English-centric named entities (§6); the paper does not measure downstream task accuracy.

## How ch-11 uses it
§3 (fertility disparity and effective context), worked example, Generalization lens, Common mistakes.
