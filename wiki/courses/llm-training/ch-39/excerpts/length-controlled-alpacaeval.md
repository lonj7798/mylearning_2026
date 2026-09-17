---
chapter: ch-39
course: llm-training
phase: read
excerpt_of: "primary source (no library card at the time of writing)"
source_url: https://arxiv.org/abs/2404.04475
created_at: "2026-09-15"
verified_against: "arXiv:2404.04475v2 (10 Mar 2025), cached plain text"
---

# Length-Controlled AlpacaEval: A Simple Way to Debias Automatic Evaluators

Yann Dubois, Balázs Galambosi, Percy Liang, Tatsunori B. Hashimoto.

## Claims the chapter uses
- **The regression (§3, Eq. 1).**
  `q(y = 1 | z_m, z_b, m, b, x) = logistic( θ_m − θ_b + φ_{m,b} · tanh( (len(z_m) − len(z_b)) / std(len(z_m) − len(z_b)) ) + (ψ_m − ψ_b) γ_x )`,
  with a model term, a length term and an instruction-difficulty term. The length feature is standardised
  and passed through `tanh`; the fitted model has `3M + N` parameters for M models and N = 805 instructions.
- **The length-controlled win rate (§3, Eq. 2).** Setting the length term to zero:
  `winrate_LC(m, b) = 100 · E_x logistic(θ_m − θ_b + (ψ_m − ψ_b) γ_x)`, the counterfactual win rate if the
  evaluated model's outputs had the same length as the baseline's.
- **Correlation with human preference (Abstract, Fig. 1).** Length control raises the Spearman correlation
  with the LMSYS Chatbot Arena from 0.94 to 0.98.
- **Gameability test (§4).** Prompting models to "Answer with as much detail as possible" or to be "as
  concise as possible" moves the raw AlpacaEval win rate of gpt4_1106_preview from 22.9% to 64.3%; under
  length control the same prompts move it only from 41.9% to 51.6%.

## Conditions and limits
- Length control removes a length-mediated effect from an automatic judge's preference; it does not
  measure whether the extra length was useful, and it does not control other spurious features.
