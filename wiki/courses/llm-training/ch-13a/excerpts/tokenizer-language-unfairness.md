---
chapter: ch-13a
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/tokenizer-language-unfairness.md (planned card; not present on 2026-09-15)
source_url: https://arxiv.org/abs/2305.15425
created_at: "2026-09-15"
---

# Excerpt: Language Model Tokenizers Introduce Unfairness Between Languages

**Authors:** Aleksandar Petrov, Emanuele La Malfa, Philip H.S. Torr, Adel Bibi (University of Oxford)
**Version read:** arXiv:2305.15425 as served on 2026-09-15 (NeurIPS 2023 text; arXiv v1 May 2023); body and appendices.
**Status:** no library card existed for this slug on 2026-09-15; every value below was read at the stated locus.

## Definition (§3)
- Tokenizer parity for language A with respect to B at parallel sentences s_A and s_B holds when |t(s_A)| / |t(s_B)| ≈ 1. The ratio |t(s_A)| / |t(s_B)| is the **premium** for A relative to B; t(s) is the tokenization and |t(s)| its length.
- Measured on FLORES-200: the same 2,000 Wikipedia sentences human-translated into 200 languages (§4). Languages with more than 10% of characters mapped to UNK are not reported ("—").

## English-centric tokenizers (§4.1, Table 1): premium relative to English
| Language | GPT-2 / RoBERTa | ChatGPT / GPT-4 (cl100k_base) | FlanT5 |
|---|---|---|---|
| Portuguese | 1.94 | 1.48 | 2.21 |
| Spanish | 1.99 | 1.55 | 2.23 |
| German | 2.14 | 1.58 | 1.37 |
| French | 2.00 | 1.60 | 1.60 |
| Italian | 2.01 | 1.64 | 2.18 |
| Chinese (Simplified) | 3.21 | 1.91 | — |
| Japanese | 3.00 | 2.30 | — |
| Vietnamese | 4.54 | 2.45 | — |
| Bulgarian | 5.51 | 2.64 | — |
| Standard Arabic | 4.40 | 3.04 | — |
| Burmese | 16.89 | 11.70 | — |
| Odia | 13.38 | 12.48 | — |
| Dzongkha | 16.36 | 12.33 | — |
| Shan | 18.76 | 15.05 | — |

- §1 rounds the cl100k_base values as Italian "about 1.6", Bulgarian 2.6, Arabic 3, Shan "as high as 15 times".
- Mechanism example (§4.1): a Shan word for "you" is one consonant and three diacritics, four Unicode codepoints, 9 cl100k_base tokens; English "you" is one token. cl100k_base uses three tokens for more than 65% of kanji characters (§2).
- FlanT5 has more than 10% UNK for 42% of languages (§4.1).

## Tokenizers for other target languages (§4.2, Tables 2-3)
- English is the language closest to parity for CamemBERT (1.20), GottBERT (1.35), and PhoBERT (1.20), ahead of Dutch (1.73) and Luxembourgish (1.75) for German, and Catalan (1.59) for French. MuRIL (16 Indian languages and English) is most token-efficient for English (Table 3: English 1.00, Hindi 1.16, Urdu 1.26, Kashmiri 1.75).
- Tokenizers for Arabic, Chinese, and Japanese have lower premiums for languages that share their script (§4.2).

## Multilingual and byte-level tokenizers (§4.3-4.4, Tables 4-5)
- Table 4, premium relative to English (XLM-R / NLLB / mT5 / M2M100 / BLOOM): Yue Chinese 0.93 / 1.05 / 0.95 / 1.03 / 0.93; Indonesian 0.94 / 0.93 / 1.08 / 0.98 / 0.96; Chinese (Simp.) 0.97 / 1.11 / 0.92 / 1.05 / 0.95; Japanese 1.11 / 1.01 / 0.90 / 1.20 / 1.81; Bulgarian 1.16 / 1.31 / 1.28 / 1.23 / 2.49; Std. Arabic 1.18 / 1.40 / 1.35 / 1.29 / 1.14; Italian 1.19 / 1.25 / 1.34 / 1.25 / 1.62; Uyghur 1.41 / 1.40 / 2.57 / 3.00 / 3.67; Central Kanuri 2.60 / 2.54 / 2.43 / 2.49 / 2.10; Kabiyè 2.98 / 1.56 / 2.83 / 2.71 / 3.34; Shan 4.43 / 1.94 / 3.28 / 4.63 / 12.06; Dzongkha — / 1.48 / 4.24 / — / 7.36; Std. Tibetan — / 1.44 / 3.68 / — / 6.66; Santali — / 2.49 / — / — / 12.71. (Indonesian mT5 is printed "1,08".)
- "All five models have languages with premiums of more than 2.5"; all are closer to parity than the English-centric tokenizers of Table 1 (§4.3).
- Table 5, premium relative to English (CANINE UTF-32 / ByT5 UTF-8): Yue Chinese 0.31 / 0.87; Chinese (Trad.) 0.32 / 0.89; Chinese (Simp.) 0.34 / 0.93; Standard Arabic 0.88 / 1.60; Bulgarian 1.04 / 1.89; Standard Tibetan 1.13 / 3.31; Italian 1.18 / 1.19; Burmese 1.24 / 3.51; Dzongkha 1.25 / 3.64; Tok Pisin 1.28 / 1.28; Tumbuka 1.30 / 1.32; Shan 1.42 / 3.94; Japanese 0.44 / 1.27.
- CANINE: Shan premium 4.58 relative to Yue Chinese. ByT5 range: 0.87 (Yue Chinese) to 3.94 (Shan) (§4.4).
- Two sources of disparity: different numbers of characters for the same content, and UTF-8 bytes per codepoint (1 for ASCII, 2 for other Latin, Greek, Cyrillic, Arabic, Hebrew, 3 for CJK) (§4.4).

## Consequences (§5)
- Cost: per-token pricing makes German or Italian about 50% more expensive than English with cl100k_base, and Dzongkha, Odia, Santali, Shan more than 12 times (§5.1).
- Latency: RoBERTa processing time is approximately linear in tokenized length; Shan takes almost twice the English time (§5.2, Fig. 2).
- Context: with a fixed context, "one can process less than a tenth of the content in languages like Burmese and Dzongkha than they can in English" (§5.3).

## Proposals (§6)
- Train models with a multilingually fair subword tokenizer; a separate billing tokenizer does not fix latency or context (§6).
- Build a fair tokenizer by training monolingual tokenizers and merging, starting from the 256 byte tokens and repeatedly adding the most frequent token of the language with the highest premium (§6).
- Fig. 3: with one third of the cl100k_base vocabulary, English FLORES-200 sequences become about 10% longer; a 10-fold vocabulary reduction gives about 30% longer English sequences.
- Parallel-corpus caveat: FLORES-200 contains English-centric names and institutions, which may favor English; different translations of one sentence have different lengths (§6).

## How ch-13a uses it
§4.1 (premium definition and tables), §4.2 (context and compute consequences), §4.3 (English sequence length with a smaller vocabulary), §5.2 (FLORES-200 caveat), figure panel B, Common mistakes.
