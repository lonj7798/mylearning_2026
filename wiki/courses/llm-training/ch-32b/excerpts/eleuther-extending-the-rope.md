---
chapter: ch-32b
course: llm-training
phase: read
excerpt_of: EleutherAI blog, "Extending the RoPE" (no library card at the time of writing; chapter-local verified extract)
source_url: https://blog.eleuther.ai/yarn/
created_at: "2026-09-15"
---

# Excerpt: Extending the RoPE (EleutherAI blog)

- **Authors:** Honglu Fan, Bowen Peng, Jeffrey Quesnelle (three of the four YaRN authors)
- **Date:** 2023-11-13
- **Source type:** practitioner write-up by the method authors; experiments are referred to the YaRN preprint ([[yarn]])
- **Used in:** ch-32b §2.1–2.3, §3

## Origin of the community methods (as stated in the post)
- Position Interpolation: "kaiokendev [2] and Chen et al. [3] found that it would be more efficient ... to
  finetune the original model" with g(m) = m/s, h(θ_d) = θ_d, s = L′/L.
- "NTK-aware" interpolation "was proposed in public as a reddit post". Change: b′ = b · s^{|D|/(|D|−2)},
  g(m) = m, h(θ_d) = b′^{−2d/|D|}; it "spread[s] out the interpolation pressure across multiple dimensions by
  scaling high frequencies less and low frequencies more".
- "NTK-by-parts": "A fix addressing this issue of 'NTK-aware' was first posted in public as a GitHub pull
  request." Ramp γ(r) with α, β; "we found that α=1, β=32 is ideal for Llama family models."
- Dynamic scaling "was first proposed in a reddit post": s = l′/L if l′/L > 1, else 1.

## Stated limits and trade-offs
- PI "normally requires finetuning on about 1-10 billion tokens" and "after finetuning on longer sequences, the
  perplexity slightly increases for short sequences compared with the original pretrained model."
- PI vs NTK-aware: without fine-tuning NTK-aware "shows better (lower) perplexity than PI on longer sequences";
  it "performs worse than PI after finetuning on longer context data."
- Wavelength λ_d = 2π b′^{2d/|D|}; when λ_d > L the positions up to L "only occupy a portion of the whole period
  window."
- Fixed scale factor: "the model may experience a flat reduction of performance at lengths less than L or an
  abrupt degradation on longer sequences starting at L′ + 1 tokens."
- YaRN temperature: √(1/t) = 0.1 ln(s) + 1 for Llama 2; for the YaRN 128K Mistral-7B the authors "determined
  a = 0.07, b = 1.0" in √(1/t) = a ln(s) + b.
- "dynamic NTK" "works exceptionally well on models pretrained on L without any finetuning (L′ = L)."

## Verification
- Checked on 2026-09-15 against https://blog.eleuther.ai/yarn/ (page text as served on that date).
- Reliability note: the Reddit post and GitHub pull request themselves were not opened for this excerpt; their
  content is known here only through this post and the YaRN paper, so claims about them are **anecdotal**.
- Not reported by the source: short-context benchmark scores (the post shows perplexity charts only).
