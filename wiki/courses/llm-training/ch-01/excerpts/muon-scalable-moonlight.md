---
chapter: ch-01
course: llm-training
phase: read
excerpt_of: primary source, arXiv:2502.16982 (no library card as of 2026-09-15)
source_url: https://arxiv.org/abs/2502.16982
created_at: "2026-09-15"
---

# Excerpt: Muon is Scalable for LLM Training (Moonlight)

**Authors:** Jingyuan Liu, Jianlin Su, Xingcheng Yao, Zhejun Jiang, Guokun Lai, Yulun Du, et al. (Moonshot AI, UCLA)
**Version read:** arXiv:2502.16982v1 (2025-02-24). Source type: official technical report (Moonshot AI trained and released Moonlight).

## Update rules (§2.1-§2.2)
- Muon (Eq. 1): M_t = μM_{t−1} + ∇L_t(W_{t−1}); O_t = Newton-Schulz(M_t); W_t = W_{t−1} − η_t O_t. Newton-Schulz approximates (M_t M_tᵀ)^{−1/2} M_t = UVᵀ for M_t = UΣVᵀ, which orthogonalizes the momentum. Coefficients a = 3.4445, b = −4.7750, c = 2.0315; N = 5 iterations.
- With weight decay (Eq. 3): W_t = W_{t−1} − η_t (O_t + λW_{t−1}). Reason: without it, "both the weight and the layer output's RMS keep growing to a large scale, exceeding the high-precision range of bf16". Fig. 2: 800M model, 100B tokens (about 5× optimal): vanilla Muon converges faster at first; Muon with weight decay reaches lower validation loss than vanilla Muon and AdamW.
- Update-RMS matching (Eq. 4): W_t = W_{t−1} − η_t (0.2 · O_t · √max(A, B) + λW_{t−1}) for a matrix of shape [A, B]. Lemma 1: the Muon update RMS of a full-rank [A, B] matrix is √(1/max(A, B)). "AdamW's update RMS is usually around 0.2 to 0.4"; with the 0.2 scale, Muon "can directly reuse the learning rate and weight decay tuned for AdamW". AdamW still handles non-matrix parameters (RMSNorm, LM head, embeddings).

## Scaling claim (Abstract, §3.2, Fig. 1a)
"Muon achieves ∼ 2× computational efficiency compared to AdamW with compute optimal training"; Muon "only requires about 52% training FLOPs to match the performance of AdamW under compute-optimal" settings.

## Moonlight pretraining (§3.3)
2.24B activated / 15.29B total non-embedding parameters (DeepSeek-V3-Small architecture), 5.7T tokens, context 8K, weight decay 0.1 for all stages; LR warmup to 4.2e-4 over 2k steps, cosine to 4.2e-5 until 5.2T tokens, then an increase to 1e-4 over 100 steps followed by linear decay to 0 over the last 500B tokens.

## Table 4 (checkpoints at 1.2T tokens, before cooldown; same architecture and data)
| Benchmark | Moonlight-A (AdamW) | Moonlight (Muon) |
|---|---|---|
| MMLU | 60.2 | 60.4 |
| MMLU-pro | 26.8 | 28.1 |
| BBH | 45.3 | 43.2 |
| TriviaQA | 57.4 | 58.1 |
| HumanEval | 29.3 | 37.2 |
| MBPP | 49.2 | 52.9 |
| GSM8K | 43.8 | 45.0 |
| MATH | 16.1 | 19.8 |
| CMMLU | 58.2 | 58.8 |

## SFT and optimizer mismatch (§3.5)
- Table 6 (1.2T checkpoints, 2 epochs of tulu-3-sft-mixture, LR 5e-5 linearly to 0), pretrain/SFT optimizer: Muon/Muon MMLU 55.7, GSM8K 68.0; AdamW/Muon 55.3, 62.1; Muon/AdamW 50.2, 64.9; AdamW/AdamW 52.0, 64.6.
- Table 7 (Qwen2.5-7B base, tulu-3-sft-mixture packed to 8k, cosine 2e-5 → 2e-6): Adam-SFT vs Muon-SFT MMLU 71.4 vs 70.8; HumanEval 79.3 vs 77.4; MBPP 71.9 vs 71.6; GSM8K 89.8 vs 85.8.
- "when the SFT optimizer differs from the pretraining optimizer, SFT with Muon does not show a significant advantage over AdamW" (§3.5); the mismatch is listed as an open problem (§4).

## Verification
- Checked on 2026-09-15 against arXiv:2502.16982v1 (Abstract, §2.1-§2.2, §3.3, §3.5, §4, Tables 4, 6, 7).
