---
chapter: ch-32b
course: llm-training
phase: read
excerpt_of: arXiv:2405.14591v1 (no library card at the time of writing; chapter-local verified extract)
source_url: https://arxiv.org/abs/2405.14591
created_at: "2026-09-15"
---

# Excerpt: Base of RoPE Bounds Context Length

- **Authors:** Xin Men, Mingyu Xu, Bingning Wang, Qingyu Zhang, Hongyu Lin, Xianpei Han, Weipeng Chen (Baichuan Inc., ISCAS)
- **Year:** 2024 (arXiv v1 2024-05-23)
- **Source type:** paper
- **Used in:** ch-32b §2.5, Common mistakes

## Claim (abstract)
"LLMs may obtain a superficial long-context ability based on the OOD theory." The authors "derive that the
base of RoPE bounds context length: there is an absolute lower bound for the base value to obtain certain
context length capability."

## Derivation (§4.2–4.3)
- Theorem 1: with i.i.d. query and key components (standard deviation σ) and a similar key k* = q + ε,
  (1/2σ²)(E[qᵀR_{m,θ}k*] − E[qᵀR_{m,θ}k]) = Σ_{i=0}^{d/2−1} cos(mθ_i) (Eq. 9).
- B_{m,θ} = Σ_{i=0}^{d/2−1} cos(mθ_i) "measures the ability to give more attention to similar tokens than random
  tokens"; it decreases as m grows and can fall below zero for a small base.
- Obtainable context length: L_θ = sup{L | B_{m,θ} ≥ 0, ∀m ∈ [0, …, L]} (Eq. 10). Lower bound on the base for a
  target length L: base_L = inf{base | B_{m,θ} ≥ 0, ∀m ∈ [0, …, L]} (Eq. 11), solved numerically.
- "this boundary is not very strict because the stacking of layers in LLMs allows the model to extract
  information beyond the single layers' range" (§4.3 note).

## Table 2 (lower bound of RoPE base per context length)
| Context | 1k | 2k | 4k | 8k | 16k | 32k | 64k | 128k | 256k | 512k | 1M |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Lower bound | 4.3e3 | 1.6e4 | 2.7e4 | 8.4e4 | 3.1e5 | 6.4e5 | 2.1e6 | 7.8e6 | 3.6e7 | 6.4e7 | 5.1e8 |

The head dimension d used for Table 2 is not stated next to the table.

## Experiments (§3, §5)
- Following Liu et al. (2024b), Llama2-7B fine-tuned to 32K with base 500 keeps low perplexity up to 128K "but
  the model could not retrieve related information for context length as short as 1k" (§3, Fig. 3).
- Fine-tuning setup: Llama2-7B and Baichuan2-7B (base 1e4), fixed LR 2e-5, global batch 128, 1000 steps (§5.1).
- A Llama-like 2B model pretrained from scratch for 1T tokens with base 100 and 4,096-token context "was
  capable of retrieving information from only the most recent approximately 500 tokens" (§5.3, Fig. 7).
  Fine-tuning it at 32K with base 1e4 raised effective length but it stayed "significantly below 32k"; base
  1e6 gave a longer effective length (§5.3).
- Summary (Table 3): with a base below the bound, "The model can keep perplexity low, but can't retrieve useful
  information from long context."

## Verification
- Checked on 2026-09-15 against https://arxiv.org/abs/2405.14591 (v1 PDF): abstract, §3, §4.2–4.3, Table 2,
  Table 3, §5.1, §5.3–5.4.
- Not reported in the checked sections: the exact head dimension for Table 2; short-context benchmark effects.
