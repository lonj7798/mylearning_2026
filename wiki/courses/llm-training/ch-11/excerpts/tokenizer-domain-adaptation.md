---
chapter: ch-11
course: llm-training
phase: read
excerpt_of: none (no library card planned on 2026-09-15)
source_url: https://arxiv.org/abs/2402.01035
created_at: "2026-09-15"
---

# Excerpt: Getting the most out of your tokenizer for pre-training and domain adaptation

**Authors:** Gautier Dagan (University of Edinburgh), Gabriel Synnaeve, Baptiste Rozière (Meta AI).
**Version read:** arXiv:2402.01035v2 (7 Feb 2024); v1 Feb 2024.
**Status:** no library card existed; values read in the v2 PDF text at the stated locus.

## Compression metric (§2.1)
Normalized Sequence Length NSL = Σ length(T_λ(D_i)) / Σ length(T_β(D_i)) over N examples, with the Llama tokenizer as T_β. NSL 0.75 means 25% fewer tokens than Llama.

## Tokenizer training data (§2.3, Fig. 2)
With 10B training characters and varying code and multilingual shares, "training on more code improves code compression, training on multilingual data improves multilingual compression, and training on an even mix of all three subset leads to the best global average compression". NSL on a subset varies by 5-6 percentage points across mixes.

## Selected compression values (Table 1, NSL vs Llama; Avg / Code / English / Multilingual)
GPT-2 50k 1.13 / 1.19 / 0.86 / 1.33; GPT-4 100k 0.85 / 0.75 / 0.84 / 0.95; InCoder 50k 1.03 / 0.74 / 1.02 / 1.31; "Identity" pre-tokenization 32k 0.92 / 0.69 / 0.89 / 1.16.

## Changing the tokenizer of a pre-trained model (§3)
- Base models: NL 1.5B (GPT-2 XL architecture, Llama tokenizer, 1T tokens of a Llama-2-like mix) and Code 1.5B (NL 1.5B + 500B code tokens) (§3).
- Evaluation: HumanEval and MBPP, n = 200 samples, temperature 0.6, top-p 0.95, token healing (§3).
- New embeddings initialized with Fast Vocabulary Transfer (FVT) (§3).
- Tokens needed (§3.1, Fig. 6): models whose tokenizer was changed (GPT-4 NL, Punct NL) differ from the unchanged Llama NL after 5B tokens, and the difference "almost disappears and even inverts (on Pass@1) after 50B tokens".
- FVT ablation (Table 4, 32k GPT-4 tokenizer, 1.5B): with FVT HumanEval Pass@1 20.5%, Pass@100 65.3%, MBPP Pass@1 27.2%, Pass@100 70.8%; without FVT 18.4%, 58.6%, 25.0%, 69.2%. Extending the Llama tokenizer to 80k ("Merged") 20.8%, 67.5%, 27.6%, 70.9%: "only small gains" over a distinct tokenizer.
- Vocabulary size (Table 3, 1.5B): 32k / 64k / 128k / 256k give HumanEval Pass@1 20.5 / 20.6 / 20.8 / 20.5%; Pearson r = −0.13, p = 0.87 between size and Pass@1.
- 7B (Table 5, Llama 2 7B + 500B code tokens, 80k tokenizers): Code Llama 7B 32.1 / 84.2 / 41.9 / 81.8; Punct 30.4 / 85.1 / 41.4 / 81.9; GPT-4 30.8 / 86.2 / 42.6 / 83.4; Merged 31.2 / 86.3 / 42.0 / 80.9 (HumanEval Pass@1, Pass@100, MBPP Pass@1, Pass@100).

## Limits
Target-domain metrics only (code generation). The paper does not report English or multilingual task performance after the tokenizer change, so retention of non-code ability is not measured.

## How ch-11 uses it
§5 (extending or changing a tokenizer), Recipe rows, Generalization lens.
