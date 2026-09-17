---
chapter: ch-08a
course: llm-training
phase: read
excerpt_of: primary source arXiv:2401.02954v1 (no library card as of 2026-09-15)
source_url: https://arxiv.org/abs/2401.02954
created_at: "2026-09-15"
---

# Excerpt: DeepSeek LLM: Scaling Open-Source Language Models with Longtermism

**Report:** DeepSeek-AI (Xiao Bi, Deli Chen, Guanting Chen, Shanhuang Chen, Damai Dai, Chengqi Deng, et al.). arXiv v1 2024-01-05 read. Source type: official technical report. This excerpt covers pre-training settings and §3 scaling laws only.

## Pre-training settings (§2.2–2.3, Table 2)
| Model | Layers | d_model | Heads / KV heads | Context | Batch (sequences) | Peak LR | Tokens |
|---|---|---|---|---|---|---|---|
| 7B | 30 | 4096 | 32 / 32 | 4096 | 2304 | 4.2e-4 | 2.0T |
| 67B | 95 | 8192 | 64 / 8 | 4096 | 4608 | 3.2e-4 | 2.0T |
- Table 2 caption: hyperparameters chosen "based on our findings in Section 3".
- AdamW β1 = 0.9, β2 = 0.95, weight decay 0.1; init std 0.006; gradient clipping 1.0; 2000 warmup steps; multi-step schedule: LR falls to 31.6% of max after 80% of training tokens and to 10% after 90% (§2.3).
- For a 1.6B model on 100B tokens, final performance with the multi-step schedule is "essentially consistent" with cosine (Fig. 1a); the multi-step schedule allows reuse of the first stage when the token budget changes (§2.3).

## Scaling laws for hyperparameters (§3.1, Eq. 1)
- Grid search at 1e17 FLOPs shows loss is stable over a wide range of batch sizes and learning rates (Fig. 2a).
- Near-optimal = generalization error within 0.25% of the minimum; fitted over budgets 1e17 to 2e19 FLOPs:
  η_opt = 0.3118 · C^−0.1250, B_opt = 0.2920 · C^0.3271 (Eq. 1).
- Validated at 1e20 FLOPs for a 2.94B FLOPs/token model (Fig. 2b). The report notes that optimal parameters also vary slightly with the model/data split at a fixed budget (§3.1).

## Model-scale representation (§3.2, Eq. 2, Table 3)
- 6N₁ = 72·n_layer·d_model² (non-embedding parameters); 6N₂ = 72·n_layer·d_model² + 6·n_vocab·d_model (all parameters); M = 72·n_layer·d_model² + 12·n_layer·d_model·l_seq (non-embedding FLOPs per token, including attention); C = M·D (Eq. 2).
- Table 3 (n_vocab = 102400, l_seq = 4096): for n_layer = 8, d_model = 512: N₁ = 25.2M, N₂ = 77.6M, M = 352M, 6N₁/M = 0.43, 6N₂/M = 1.32; for n_layer = 80, d_model = 8192: 6N₁/M = 0.92, 6N₂/M = 0.94. Differences reach up to 50% for small models (§3.2).

## Optimal model/data allocation (§3.2, Eq. 4, Fig. 4–5)
- IsoFLOP profiles at 8 budgets from 1e17 to 3e20 FLOPs, about 10 allocations per budget, hyperparameters from Eq. 1, loss on a 100M-token validation set (metric: bits-per-byte) (§3.2, Fig. 4).
- M_opt = 0.1715 · C^0.5243, D_opt = 5.8316 · C^0.4757 (Eq. 4).
- The fitted loss curve predicts DeepSeek LLM 7B and 67B, "models with 1000× compute budget" (§3.2, Fig. 5).

## Data quality and allocation (§3.3, Table 4)
| Dataset | a (N_opt or M_opt ∝ C^a) | b (D_opt ∝ C^b) |
|---|---|---|
| OpenAI (OpenWebText2), Kaplan et al. | 0.73 | 0.27 |
| Chinchilla (MassiveText) | 0.49 | 0.51 |
| DeepSeek early in-house data | 0.450 | 0.550 |
| DeepSeek current in-house data | 0.524 | 0.476 |
| DeepSeek on OpenWebText2 | 0.578 | 0.422 |
- The authors rank quality as OpenWebText2 > current in-house > early in-house, and state that higher quality shifts more of added compute to model scale; the explanation offered ("logical clarity and less predictive difficulty") is labeled an intuitive speculation (§3.3).

## Verification
- Read on 2026-09-15 against arXiv:2401.02954v1 PDF text (Abstract, §2.2–§3.3).
- Not reported by the source: the units of C and B in Eq. 1 are not printed; with C = M·D for the 7B model (M = 42.3B FLOPs/token derived from Eq. 2, D = 2.0T), Eq. 1 gives B ≈ 9.2M tokens and η ≈ 4.2e-4, matching Table 2 (2304 × 4096 = 9.4M tokens) if C is in FLOPs and B in tokens (derived).
