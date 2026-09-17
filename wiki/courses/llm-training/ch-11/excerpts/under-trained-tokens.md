---
chapter: ch-11
course: llm-training
phase: read
excerpt_of: none (no library card planned on 2026-09-15)
source_url: https://arxiv.org/abs/2405.05417
created_at: "2026-09-15"
---

# Excerpt: Fishing for Magikarp: Automatically Detecting Under-trained Tokens in Large Language Models

**Authors:** Sander Land, Max Bartolo (Cohere).
**Version read:** arXiv:2405.05417 (PDF served 2026-09-15); v1 May 2024; EMNLP 2024.
**Status:** no library card existed; values read in the PDF text at the stated locus.

## Definitions (§1, §2.1)
- Under-trained tokens: tokens in the vocabulary that are "nearly or entirely absent during model training". "Untrained" is reserved for tokens with clear evidence of zero occurrences.
- Partial UTF-8 sequences: tokens that are not complete UTF-8 characters. Unreachable tokens: tokens that are never produced when their decoded string is re-tokenized. Special tokens: manually defined control tokens.

## Indicators (§2.2)
- Reference set t_ref of known unused tokens (for example `<unused_token123>` or embedding rows above the tokenizer vocabulary size). u_ref = mean of E_out rows in t_ref.
- Tied embeddings: cosine distance C(E_out, u_ref)_i = 1 − (A_i · x)/(‖A_i‖ ‖x‖).
- Untied embeddings: the norm of E_in rows. Input rows of tokens that never appear change only through weight decay (toward zero) or stay at the initial value.
- Mechanism stated by the authors: unseen output rows "experience similar updates in training, 'moving away' from the mean output vector", which lets the model produce "highly negative logits for tokens that are never the correct prediction".

## Verification (§2.3-2.4)
The top 2% of candidates by indicator are tested with repetition prompts; a token is confirmed when its maximal output probability is below 1%. On OLMo v1.7 7B, testing all tokens confirmed 191 of 49,575; testing the top 2% confirmed 175 of 993. Indicators correlate with first-epoch token counts across ten orders of magnitude (Fig. 2).

## Table 1 (confirmed / tested)
| Model | Vocabulary | Tied | Confirmed |
|---|---|---|---|
| GPT-2 XL | 50,257 | yes | 67/999 |
| GPT-J 6B | 50,400 | no | 200/999 |
| GPT-NeoX 20B | 50,277 | no | 10/993 |
| OLMo v1.7 7B | 50,280 | no | 178/993 |
| Llama2 7B | 32,000 | no | 20/639 |
| Mistral 7B v0.3 | 32,000 | no | 53/637 |
| Rakuten 7B | 48,000 | no | 66/957 |
| Qwen1.5 32B | 151,646 | no | 2450/2966 |
| Llama3 8B | 128,256 | no | 556/2540 |
| Command R (35B) | 255,029 | yes | 306/5012 |
| Gemma 2B | 256,000 | yes | 3161/5117 |
| StarCoder2 15B | 49,152 | no | 128/968 |
Confirmed tokens are typically 5-50% of tested candidates, "corresponding to 0.1–1% of the total vocabulary size" (§3).

## Findings (§3.2, §5)
- GPT-NeoX 20B, whose tokenizer was trained on the Pile used for model training, has few under-trained tokens; GPT-J and Phi-2, which reuse the GPT-2 tokenizer on different data, have more.
- Rakuten 7B (extended Japanese vocabulary with continued pre-training): under-trained fragments among added tokens, "proportional to the extended vocabulary".
- Qwen and Llama 3 extend cl100k; added tokens include under-trained entries.
- StarCoder2: single documents (one Java file, one base-64 file) define several long tokens.
- "The most salient factors ... aside from simply having a large vocabulary, appears to be whether the tokenizer was trained on similar data as the model."
- Recommendations (§5): identical pre-processing across tokenizer training data, model training data, and inference (carriage returns, special tokens in plain text); check unreachable tokens by encoding and decoding the vocabulary; check for under-trained tokens after small test runs.

## How ch-11 uses it
§4 (under-trained tokens), Negative samples (push-down on unseen output rows), Common mistakes.
