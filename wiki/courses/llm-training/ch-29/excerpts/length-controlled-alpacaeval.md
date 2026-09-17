---
chapter: ch-29
course: llm-training
phase: read
excerpt_of: https://arxiv.org/abs/2404.04475 (primary text, arXiv v2; no library card existed on 2026-09-15)
source_url: https://arxiv.org/abs/2404.04475
created_at: "2026-09-15"
---

# Excerpt: Length-controlled AlpacaEval — a regression that removes the length term

**Artifact:** Dubois, Galambosi, Liang, Hashimoto, "Length-Controlled AlpacaEval: A Simple Way to Debias Automatic Evaluators" (arXiv v1 2024-04; v2 2025-03). Stanford University.

## Problem (Abstract, §2)

- AlpacaEval asks an LLM auto-annotator to compare a model's output with a baseline output on 805 instructions; the win rate is the expected probability that the annotator prefers the model (§2).
- The annotator prefers longer outputs. Prompting the baseline model (gpt4_1106_preview) to be verbose or concise moves its AlpacaEval win rate from 22.9% to 64.3% (§4.1, Figure 3).

## Regression (§3, Eq. 1–2)

```
q(y = 1 | z_m, z_b, m, b, x) = logistic( θ_m − θ_b
                                       + φ_{m,b} · tanh( (len(z_m) − len(z_b)) / std(len(z_m) − len(z_b)) )
                                       + (ψ_m − ψ_b) γ_x )                                   (Eq. 1)

winrate_LC(m, b) = 100 · E_x[ logistic( θ_m − θ_b + (ψ_m − ψ_b) γ_x ) ]                     (Eq. 2)
```

- y = 1 when the annotator prefers the model output; m = evaluated model, b = baseline; z_m, z_b = their outputs; x = instruction.
- θ = model term; φ_{m,b} = length coefficient; ψ and γ_x = model-specific and instruction-specific difficulty terms.
- The model is fitted as a GLM with logit link on annotator preferences; the LC win rate sets the length difference to zero (§3). With a constant baseline, the GLM has 3M + N parameters for M models and N instructions (§3).
- Properties: the baseline against itself gets 0.5 (identity), and swapping m and b gives 1 − q (symmetry), because tanh is odd (§3).

## Results (§4)

- Gameability: under the same verbosity prompts, the LC win rate of gpt4_1106_preview moves only from 41.9% to 51.6%; the normalized standard deviation across the three verbosity prompts falls from 25% to 10% (§4.1).
- Spearman correlation with LMSYS Chatbot Arena rises from 0.94 to 0.98 (Abstract, §4.2, Figure 1).
- A white-box attack: truncating GPT-4 outputs to a few characters, except outputs that are much better and about the baseline's length, raises the win rate from 3.7 (AlpacaEval 2.0) to 25.9 (LC without regularization); a regularization term on φ lowers it to 12.2 (§4.3).

## Use in ch-29

- The lab's judge-based metric, when used, is the LC win rate, reported with the raw win rate and the mean output length of each arm.
- A synthetic pool built with Evol-Instruct "deepening" and "reasoning steps" operators can lengthen responses. A raw win-rate gain that disappears under length control is a length effect, not a quality effect.
- Judge metrics are a supplement. The held-out suite in ch-29 §7 uses programmatic scoring (IFEval, MMLU-Pro, GSM8K, HumanEval, BBH, knowledge probe) for the go/no-go decision.

## Connections

- [[judge-llm-bias]] — position, verbosity, and self-enhancement bias of LLM judges.
- [[deita]] — reports AlpacaEval and MT-Bench gains that do not move together (§3.3).
