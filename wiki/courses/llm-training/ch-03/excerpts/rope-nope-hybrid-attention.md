---
chapter: ch-03
course: llm-training
phase: read
excerpt_of: primary source arXiv:2501.18795v2 (no library card as of 2026-09-15)
source_url: https://arxiv.org/abs/2501.18795
created_at: "2026-09-15"
---

# Excerpt: Rope to Nope and Back Again: A New Hybrid Attention Strategy (QK-Norm results only)

**Paper:** Bowen Yang, Bharat Venkitesh, Dwarak Talupuru, Hangyu Lin, David Cairuz, Phil Blunsom, Acyr Locatelli
(Cohere). arXiv v1 2025-01; v2 2025-10-22 read; NeurIPS 2025. Source type: paper. ch-03 uses only §2 and App. B.

## Setting (§2.1, Table 1)
- Three 8B variants (32 layers, d_model 4096, 32 heads, 8 KV heads, vocabulary 256000) pretrained on 750B tokens:
  batch 4M tokens, AdamW, peak LR 7e-3, linear warmup 2000 steps, cosine to 3.5e-4 over 179,000 steps. SFT mixes
  short (8192) and long (65536) data at 3:1, batch 0.5M tokens.
- RoPE variant: θ = 10,000 in pretraining, 2 million in SFT. QK-Norm variant: LayerNorm on queries and keys before the
  RoPE rotation, all else identical. NoPE variant: no positional embedding and no QK-Norm.

## Results (§2.2, Table 2; App. B, Table 9)

| Model | Val loss | MMLU | HellaSwag | CommonsenseQA | ARC-E | ARC-C | Needles 65k |
|---|---|---|---|---|---|---|---|
| RoPE | 1.52 | 48.55 | 73.74 | 68.30 | 81.05 | 39.13 | 9.82 |
| QK-Norm | 1.53 | 48.21 | 73.68 | 68.23 | 80.54 | 38.98 | 7.93 |
| NoPE | 1.58 | 47.61 | 72.16 | 66.42 | 76.94 | 37.12 | 9.03 |

- "For long context evaluations, QK-Norm performs the worst among the three variants, despite its decent performance
  in other capabilities" (§2.2.1). All evaluations are on the SFT models (Table 2 caption).
- Attention mass: the QK-Norm variant assigns the least attention to the needle, "markedly lower attention mass on
  the 'Begin' segment and substantially higher attention mass on the 'Context' segment" (§2.2.2).
- Authors' explanation (Interpretation): "the normalization operation mitigates magnitude information from the dot
  product of Query and Key vectors which tends to result in attention logits being closer in terms of magnitude and
  flatter in terms of distribution" (§2.2.2).
- Table 9, aggregated attention entropy at 8k / 32k / 128k: RoPE 6.02 / 6.95 / 7.62; QK-Norm 10.71 / 12.46 / 14.14.
- The proposed hybrid architecture removes QK-Norm "due to its poorly shaped attention" (§3).

## Verification
- Read on 2026-09-15 against arXiv:2501.18795v2 PDF text (§1–§2, §3 opening, App. B).
- Not reported: seeds or variance for Table 2; the Needles score scale is not defined in the parts read.
