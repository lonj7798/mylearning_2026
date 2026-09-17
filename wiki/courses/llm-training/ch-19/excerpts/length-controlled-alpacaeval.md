---
chapter: ch-19
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/length-controlled-alpacaeval.md (library card not present on 2026-09-15; quotations below are taken from the primary source)
source_url: https://arxiv.org/abs/2404.04475
primary_version: arXiv:2404.04475v2 (2025-03); v1 2024-04
created_at: "2026-09-15"
---

# Excerpt: Length-Controlled AlpacaEval: A Simple Way to Debias Automatic Evaluators

Authors: Yann Dubois, Balázs Galambosi, Percy Liang, Tatsunori B. Hashimoto (Stanford University; independent researcher). Checked against the v2 PDF on 2026-09-15. Used by ch-19 `read.md` §10.

## AlpacaEval setup (§2)
> "AlpacaEval is an LLM-based automated evaluation metric – it operates on a fixed set of 805 instructions chosen to be representative of user interactions on the Alpaca web demo. For each instruction, both a baseline model b (currently GPT-4 turbo) and the evaluated model m produce responses. A GPT-4 turbo-based evaluator then compares the responses head-to-head and outputs the probability of preferring the evaluated model."

The win rate is `winrate(m, b) = 100 · E_x[f(z_m, z_b, x)]`, where `f` is the auto-annotator's preference probability (§2).

## Regression (§3, Eq. 1-2)
```
q(y = 1 | z_m, z_b, m, b, x) :=
  logistic( θ_m − θ_b  +  φ_{m,b} · tanh( (len(z_m) − len(z_b)) / std(len(z_m) − len(z_b)) )  +  (ψ_m − ψ_b) γ_x )

winrate_LC(m, b) = 100 · E_x[ logistic( θ_m − θ_b + (ψ_m − ψ_b) γ_x ) ]
```
- Terms: model identity (θ), length difference (φ), instruction difficulty (γ_x with per-model weight ψ).
- "we zero out the 'length of output' term to obtain counterfactual estimates of AlpacaEval win rate."
- The GLM has 3M + N parameters for M models and N instructions; 5-fold cross-validation with L2 regularization; γ_x is fitted once jointly and reused; weak regularization on φ_{m,b} is added against truncation attacks.

## Results (§4)
- Length gameability (§4.1): prompting with "Answer with as much detail as possible." or "Be as concise as possible while still providing all the necessary information to answer the question." moves the baseline gpt4_1106_preview "from 22.9% to 64.3%"; with length control "only fluctuate from 41.9% to 51.6%". "the normalized standard deviation across the three verbosity prompts decreases from 25% to 10%".
- Chatbot Arena (§4.2): "controlling for length increased the Spearman correlation with Chat Arena from 0.94 to 0.98"; computed on 38 models for AlpacaEval and AlpacaEval-LC and 34 for MT-bench; bootstrap p-value 0.07 versus AlpacaEval and 0.06 versus MT-bench.
- Leaderboard shifts (§4.2): "the biggest rank losses are in open-source models that have gone through the RLHF process"; the authors call this "consistent with the hypothesis that existing open models had exploited the length bias of AlpacaEval."
- Truncation attack (§4.3): GPT-4 outputs post-processed by truncation go from 3.7 (AlpacaEval 2.0) to 25.9 (LC without regularization) and 12.2 (LC with regularization).

Table 1 (§4.4):

| Metric | Chatbot Arena correlation (↑) | Gameability (↓) | Adversarial win rate gain (↓) |
|---|---|---|---|
| Win rate | 0.94 | 26% | 0.0 |
| Length-controlled | 0.98 | 10% | 8.5 |
| Length-normalized | 0.96 | 15% | 3.6 |
| Length-balanced | 0.95 | 15% | 40.8 |

## Stated limit (§2)
> "Although Chatbot Arena likely still contains biases (e.g. internet users may focus on surface features rather than 'hard to measure' capabilities such as factuality), it represents the largest publicly available human evaluation process today."
