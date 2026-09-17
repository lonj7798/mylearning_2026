---
chapter: ch-31a
course: llm-training
phase: read
excerpt_of: primary source (no library card at wiki/raw-data/llm-training/papers/redi-reinforcement-distillation.md on 2026-09-15)
source_url: https://arxiv.org/abs/2505.24850
source_version: arXiv v2 (2025-12-14); v1 2025-05-30
created_at: "2026-09-15"
---

# Excerpt: Harnessing Negative Signals: Reinforcement Distillation from Teacher Data for LLM Reasoning (Xu, Peng, Long, Xu, Chu, Qi)

Facts used by [[read]], read in the arXiv v2 PDF on 2026-09-15.

## Data (§3.1; App. C.2)
- Source: OpenR1-Math-Raw (cn_k12 subset excluded). A response is correct only if both the Llama judge label and the Math-Verify label are "True"; otherwise it is incorrect.
- D_SFT: 78k (x, y_w) positive traces (Stage 1). D_Pref: 53k (x, y_w, y_l) triplets, one incorrect trace paired with the correct trace of the same problem (Stage 2). Total 131k.
- D_Pref filters: instances with queries over 800 tokens, or chosen or rejected responses over 19,000 tokens, removed (App. C.2).

## Objectives (§2.2)
- Stage 1 SFT: L_SFT = −E[log π_θ(y_w | x)] (Eq. 1).
- Symmetric reference-free objective (β → 0 limit of SimPO-style losses): L_symm = E[−log π_θ(y_w|x)/|y_w| + log π_θ(y_l|x)/|y_l|] (Eq. 4).
- REDI: L_REDI = E[−log π_θ(y_w|x)/|y_w| + α·log π_θ(y_l|x)/|y_l|], α ∈ [0, 1]; α = 0 is SFT on positives, α = 1 is Eq. 4 (Eq. 5).
- Collapse (§2.2): "Collapse manifests as a rapid decrease in the likelihood of both positive (y_w) and negative (y_l) responses, accompanied by declining task accuracy."

## Training (§3.2; App. C.3-C.4)
- Base: Qwen2.5-Math-1.5B. Stage 1: AdamW (β1 0.9, β2 0.999, ε 1e−8, weight decay 1e−4), peak LR 5e−5, 10% linear warm-up then linear decay, batch 128, 3 epochs (Qwen-SFT-1.5B-3ep, used for ablations) or 5 epochs (Qwen-SFT-1.5B-5ep, used for the final model), max length 32,768.
- Stage 2: 1 epoch over D_Pref, batch 32, same optimizer and schedule shape. Explored α ∈ {0.2, 0.5, 0.8, 1.0}; best α = 0.8 with LR 1e−6. DPO best β = 0.001, LR 2e−7; SimPO best β = 2, γ = 1, LR 5e−7.
- Final-model compute: about 136 A100-80GB GPU hours for SFT 5 epochs + REDI 1 epoch (App. C).

## Results (pass@1 over 16 samples; temperature 0.6, top-p 0.95)
- Table 2 (all from Qwen-SFT-1.5B-3ep, average of MATH-500, AIME24, AMC23, Minerva, OlympiadBench): start 41.9; DPO 47.2; SimPO 47.2; symmetric REDI (α = 1.0, LR 2e−7) 47.7; REDI (α = 0.8, LR 1e−6) 48.3.
- Table 1: Qwen-SFT-1.5B-5ep 45.8; Qwen-REDI-1.5B 49.5 (MATH-500 83.1); DeepSeek-R1-Distill-Qwen-1.5B 48.6 (MATH-500 83.2).
- DPO stability (§4.2, Fig. 3; matched initial step sizes): β = 0.001 peaks at about 80.9% MATH-500 then collapses with a sharp drop in chosen and rejected log-probabilities; β = 0.01 and 0.1 stay stable with peaks of about 80.3% and 78.3%.
- Symmetric vs asymmetric (§4.3, Fig. 6): α = 1.0 with LR 1e−6 learns fast and collapses; α = 0.8 with LR 1e−6 reaches a high peak without collapse. Lower α (0.5, 0.2) "tended to degrade peak performance" (§4.4).
- Other models (Table 4): Llama-3.2-3B SFT 28.0 → REDI 32.9; Qwen2.5-Math-7B 62.5 → 65.1.
- Out-of-domain (Table 5): Qwen-SFT-1.5B-3ep vs Qwen-REDI-1.5B: GPQA 28.3% → 35.9%; OlympiadBench 37.5% → 43.4%; HumanEval pass@1 7.1% → 19.9%. The text names the final Qwen-REDI-1.5B (initialized from the 5-epoch SFT model, App. C.4), but the OlympiadBench value 43.4 equals the 3-epoch-initialized REDI row of Table 2, while Table 1 gives 45.2 for Qwen-REDI-1.5B. Which checkpoint was compared is therefore unclear; if it is the final model, the comparison also includes two extra SFT epochs.
- Table 2 reports "the best checkpoint for each configuration".
- Diversity (§4.7; App. D.3, Tables 6-7): REDI "maintains or improves pass@16 while enhancing pass@1".
