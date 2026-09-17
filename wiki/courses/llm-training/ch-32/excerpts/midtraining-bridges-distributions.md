---
chapter: ch-32
course: llm-training
phase: read
excerpt_of: primary source arXiv:2510.14865v2 (planned library card papers/midtraining-bridges-distributions.md; not present on 2026-09-15)
source_url: https://arxiv.org/abs/2510.14865
created_at: "2026-09-15"
---

# Excerpt: Midtraining Bridges Pretraining and Posttraining Distributions

**Authors:** Emmy Liu, Graham Neubig, Chenyan Xiong (Carnegie Mellon University). arXiv v1 2025-10-16; read at v2 (2026-02-02). Source type: paper.

## Definition (§2.1, §2.2)

- Midtraining: "any intermediate phase between training stages", with data "more specialized than general pretraining data ... while maintaining a mixture with general pretraining data"; |D_pre| > |D_mid| > |D_target| (§2.1).
- Continued pretraining is "the limiting case where the mixture weight on general pretraining data is zero" (§2.2).

## Forgetting bound (§2.3, Eq. 1–5)

Post-training runs K gradient steps θ_{k+1} = θ_k − η∇J_T(θ_k). Forgetting is ΔP(K) = J_P(θ_K) − J_P(θ_0). If J_P is L_P-smooth, J_T is L_T-smooth and η ≤ 1/L_T:

```
ΔP(K) ≤ −η Σ_{t=0}^{K−1} ⟨∇J_P(θ_t), ∇J_T(θ_t)⟩  +  L_P η (J_T(θ_0) − J_T*)
```

Midtraining changes only θ_0 = θ_0(t, w) (start time t, mixture weight w), so it can lower the second term by starting post-training closer to the target (§2.3).

## Setup (§3.1, Table 1)

- Pythia-family models 70M–1B pretrained on C4 for 128B tokens (about 61k steps), cosine LR, max 3e-4, AdamW (§3.1).
- Midtraining mixes: Starcoder 196B, Math 12B (MAmmoTH + OpenMathInstruct), FLAN 3.5B, KnowledgeQA 9.6B, DCLM 51B; control continues C4 for the same tokens (Table 1, §3.1).
- SFT targets: GSM8K, SciQ, CodeSearchNet-Python (PyCode), LIMA; forgetting = C4 validation loss after SFT; 5 seeds after hyperparameter search (§3, §4).
- Proximity advantage: PA(M → T) = prox(M, T) − prox(C4, T), token-unigram similarity under the model tokenizer (Eq. 6). Figure 2 example similarities: PyCode vs C4 0.54, vs StarCoder 0.70; GSM8K vs C4 0.61, vs Math Combined 0.65.

## Results

Table 2 (1B, SFT and C4 validation loss after SFT; Δ vs C4-only baseline):

| SFT target | Midtrain mix | SFT loss | C4 loss |
|---|---|---|---|
| PyCode | Starcoder (20%) | 1.888 (−0.286) | 3.070 (−0.005) |
| PyCode | DCLM (20%) | 2.175 (+0.001) | 3.106 (+0.031) |
| GSM8K | Math (12%) | 0.851 (−0.091) | 3.018 (−0.116) |
| GSM8K | Starcoder (20%) | 0.927 (−0.015) | 3.158 (+0.024) |
| SciQ | Starcoder (20%) | 1.873 (+0.015) | 3.245 (+0.044) |
| LIMA | FLAN (5%) | 3.316 (−0.001) | 2.900 (+0.000) |

- Finding 1: benefits are domain-specific; mismatched mixes and FLAN give little benefit (§4).
- Finding 2: gains correlate with proximity advantage across sizes (Figure 3 panel labels: 70m r = 0.521, 160m r = 0.591, 410m r = 0.466, 1b r = 0.622) (§5.1).
- Finding 3 (Table 3): midtraining beats continued pretraining on 100% specialized data. 160M PyCode: pretrain-only SFT 2.314 / C4 5.254; Starcoder (20%) 2.134 / 5.079; continued pretraining (Starcoder) 2.219 / 5.369. 160M GSM8K: 1.163 / 5.308; Math (12%) 1.114 / 5.230; continued pretraining (Math) 1.159 / 5.326 (§5.2).
- Finding 4: timing and mixture weight interact. For code on 70M/160M, 80% weight is best when introduced early but worse than 10% when introduced at 105B tokens; the sequence 10% @ 42B, 20% @ 63B, 30% @ 84B degrades, so later, heavier mixing does not compensate (§6, Figure 4).
- Finding 5: midtrained models change less in late layers during SFT (linear CKA, 70M) (§7, Figure 6).
- Limits stated by the authors: scale (up to 1B) and extension to RL post-training remain to be tested (§9). All metrics are validation losses, not benchmark accuracy.

## Used in

ch-32 §4, Generalization lens.
