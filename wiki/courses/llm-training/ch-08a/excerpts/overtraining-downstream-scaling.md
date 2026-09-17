---
chapter: ch-08a
course: llm-training
phase: read
excerpt_of: primary source arXiv:2403.08540v2 (no library card as of 2026-09-15)
source_url: https://arxiv.org/abs/2403.08540
created_at: "2026-09-15"
---

# Excerpt: Language models scale reliably with over-training and on downstream tasks

**Paper:** Samir Yitzhak Gadre, Georgios Smyrnis, Vaishaal Shankar, Suchin Gururangan, Mitchell Wortsman, Rulin Shao, et al. (Columbia; Toyota Research Institute; UT Austin; Apple; University of Washington; and others). arXiv v1 2024-03; v2 2024-06-14 read. Source type: paper.

## Definitions (§2.1)
- Token multiplier M = D/N (training tokens per parameter). The Figure 1 caption prints "M = N/D"; §2.1 defines M = D/N.
- Over-training: M > M*, where M* is the compute-optimal multiplier for a training distribution (§2.1).
- Loss decomposition L(C) = E + L′(C), with E the irreducible loss and L′ the reducible loss, often modeled as L′ = λ·C^−η (Eq. 1).

## Scaling law for over-training (§2.2, Eq. 3–4)
- Observation: for fixed M, reducible loss versus compute gives parallel lines in log-log space for M from 20 to 640; the slope η stays about constant while the intercept shifts with M (Fig. 2).
- Starting from L(N, D) = E + A·N^−α + B·D^−β (Eq. 3) and assuming α = β, substituting C = 6ND and M = D/N gives
  L(C, M) = E + (a·M^η + b·M^−η)·C^−η, with η = α/2, a = A·(1/6)^−η, b = B·(1/6)^−η (Eq. 4, App. B).

## Scaling law for downstream error (§2.3, Eq. 5–6)
- Err(L) = ε − k·exp(−γL), with Err the average top-1 error, L the validation loss, and ε, k, γ fitted (Eq. 5); equivalently Err(PP) = ε − k·PP^−γ with PP = exp(L) (Eq. 6).
- Three-step method: fit (C, M) ↦ L, fit L ↦ Err, chain to (C, M) ↦ Err (§2.3).

## Testbed (§3.2–3.4)
- A 435-model grid search at M = 20 on the OpenLM data mix selects base configurations N ∈ {0.011B, 0.079B, 0.154B, 0.411B}; learning rate fixed at 3e-3 in sweeps (qk-LayerNorm reduces LR sensitivity) (§3.2).
- Training sets C4 (138B tokens), RedPajama (1.15T), RefinedWeb (600B); M ∈ {5, 10, 20, 40, 80, 160, 320, 640}; 1.4B models at M = 20 and the largest M without repeating tokens; 6.9B models at M = 20; 104 models in total (§3.2).
- Default fits (Table 1): Eq. 4 from 5 runs totalling 2.4e19 FLOPs (~100 A100 hours); Eq. 5 adds a 1.4B M = 20 run, 2.7e20 FLOPs (~1,000 A100 hours).
- Downstream suite: 17 of 46 LLM-Foundry tasks on which at least one 0.154B model is 10 points above chance (§3.4). Validation loss on C4 eval (§3.4).
- Training details: sequence length 2,048, GPT-NeoX-20B tokenizer, AdamW with β2 = 0.95, independent weight decay 1e-4, z-loss 1e-4, linear warmup and cosine decay to 3e-5, no weight tying (App. C).

## Results (§4, Table 2)
- C4 eval loss of the 1.4B, 900B-token RedPajama run (M = 640) predicted within 0.7% relative error from fits using 300× less compute; the 6.9B, M = 20 run also within 0.7% (§4).
- Average 17-task top-1 error predicted within 0.05% relative error for 6.9B on 138B RedPajama tokens and 3.6% for 1.4B on 900B tokens, using 20× less compute (§4).
- Removing the 1.4B model from the Eq. 5 fit raises the 6.9B error from 0.05% to 10.64% (§4).
- Individual tasks at 6.9B (Table 2), relative error by training set C4 / RedPajama / RefinedWeb: ARC-Easy 28.96% / 5.21% / 26.06%; LAMBADA 15.01% / 14.39% / 16.55%; OpenBookQA 16.80% / 8.44% / 1.92%; HellaSwag 79.58% / 25.73% / 81.96%; 17-task average 0.14% / 0.05% / 2.94%.

## Limits (§4, §6, App. F)
- At M = 5 scaling is unreliable; multipliers from 10 to 80 give points roughly on the compute-optimal frontier, so the optimal multiplier "may lie in a range rather than take a single value" (§4, Fig. 9).
- Models trained on C4 and evaluated on code have high relative error; German evaluation remains reliable (§4, Fig. 10).
- Stated limitations include: hyperparameter sweeps are needed; post-training is not considered ("Quantifying to what degree over-training the base model provides benefits after post-training is an open area of research"); per-task prediction is left to future work (§6).

## Verification
- Read on 2026-09-15 against arXiv:2403.08540v2 PDF text (Abstract, §1–§4, §6, App. C, App. F excerpts).
- Source-internal inconsistency: M defined as D/N in §2.1 and printed as N/D in the Figure 1 caption.
