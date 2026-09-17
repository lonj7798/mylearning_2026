---
chapter: ch-45
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/self-correct-rl.md (card not re-verified; see Corrections)
source_url: https://arxiv.org/abs/2409.12917
version: arXiv v2, 2024-10-04
verified: 2026-09-15 (rewritten from the primary text; replaces the 2026-04 excerpt)
---

# Excerpt: SCoRe — Training Language Models to Self-Correct via Reinforcement Learning

**Used by:** [[read]] §1, §5, Negative samples, Recipe

## Metrics (§3)

Accuracy@t1 and Accuracy@t2 are first- and second-attempt accuracy; Δ(t1, t2) is their difference; Δ^{i→c} is the share of problems fixed at the second attempt and Δ^{c→i} the share of correct answers broken at the second attempt.

## The problem, measured (Table 1, Gemini 1.5 Flash on MATH500)

| Method | Acc.@t1 | Acc.@t2 | Δ(t1,t2) | Δ^{i→c} | Δ^{c→i} |
|---|---|---|---|---|---|
| Base model | 52.6% | 41.4% | −11.2% | 4.6% | 15.8% |
| STaR (D_STaR) | 55.4% | 41.2% | −14.2% | 5.4% | 19.6% |
| STaR (D⁺_STaR) | 53.6% | 54.0% | 0.4% | 2.6% | 2.2% |
| Pair-SFT (D_SFT) | 52.4% | 54.2% | 1.8% | 5.4% | 3.6% |
| Pair-SFT (D⁺_SFT) | 55.0% | 55.0% | 0% | 0% | 0% |

The two failure modes the authors name for SFT on self-generated correction traces: collapse to non-correcting behavior, and inability of offline methods to be robust to first-attempt distribution shift (§4).

## Method (§5)

Stage I (Eq. 3): maximize the second-attempt reward while a KL term with coefficient β2 keeps the first-attempt distribution near the base model; the default KL on both turns stays with a smaller weight β1.

Stage II (Eq. 4): maximize the sum of both attempts' rewards with KL β1, and add to the second attempt the shaping bonus

```
b̂(y2 | y1, y*) = α · ( r̂(y2, y*) − r̂(y1, y*) ),   α a positive constant, ideally larger than 1.0
```

All experiments use the instantaneous reward only, which is equivalent to a discount factor γ = 0 (App. A). With γ = 0.8 and α = 1.0, standard multi-turn RL still collapses to non-correcting behavior (App. A, Fig. 9).

Turn 2 prompt (App. C): "There might be an error in the solution above because of lack of understanding of the question. Please correct the error, if any, and rewrite the solution." It does not reveal whether the first attempt was correct.

Hyperparameters (App. B, Table 5): MATH — Gemini 1.5 Flash, Adam, learning rate 5e−6, 3,000 steps, batch 512, sampling temperature 1.0, α = 10, β1 = 0.01, β2 = 0.1. MBPP — Gemini 1.0 Pro, learning rate 1e−5, 1,500 steps, batch 128, temperature 1.0, α = 10, β1 = 0.01, β2 = 0.25. Checkpoints are selected by the highest training reward (§6).

## Results as printed

MATH500 (Table 2): SCoRe Acc.@t1 60.0%, Acc.@t2 64.4%, Δ = 4.4%, Δ^{i→c} 5.8%, Δ^{c→i} 1.4%. Relative to the base model, Δ improves by 15.6 points and Acc.@t2 by 23.0 points (§6.1).

Code (Table 3, trained on MBPP, evaluated on HumanEval): base MBPP-R 47.3%, Acc.@t1 53.7%, Acc.@t2 56.7%, Δ = 3.0%; SCoRe MBPP-R 60.6%, Acc.@t1 52.4%, Acc.@t2 64.6%, Δ = 12.2%. The abstract states the gain as 9.1% on HumanEval; §6.1 writes "9% higher than the base model".

Ablations on MATH (Table 4): full SCoRe Δ = 4.4%; without multi-turn training −2.4%; without Stage I 2.2%; without reward shaping 2.6%; with STaR instead of REINFORCE in Stage II 2.2%.

Inference-time scaling (§6.2): with a budget of 32 solutions per problem, parallel sampling alone gains 7.4% while splitting the budget into parallel samples plus one self-correction round gains 10.5%.

Evaluation note (§6): the MATH training split is augmented with 4,500 problems from the MATH test set and results are reported on the remaining 500 problems.

## Corrections to the library card `papers/self-correct-rl.md`

1. "9.1 pts on MBPP" → the 9.1-point self-correction gain is on **HumanEval**; MBPP is the training set and MBPP-R is a separate offline repair task (47.3% → 60.6%).
2. "α = 2.0" → α = 10 for both MATH and MBPP (App. B Table 5).
3. "Stage II loss: [r(y1) + α(r(y2) − r(y1))]·Σ∇log π" → the bonus is added to the second attempt's reward only, and with γ = 0 it enters the gradient of the second-attempt tokens (§5.2, App. A).
4. "Base model: Gemini 1.0 Pro; also reproduced on Gemma-2-9B" → MATH uses Gemini 1.5 Flash and code uses Gemini 1.0 Pro (§6); no Gemma run appears in v2.
5. "Optimizer: AdamW, lr 1e-6, batch 256" → Adam with the per-task values in Table 5 above.
6. "Figure 4 (reward-shaping bonus coefficient)" → v2 has no such sweep; the shaping ablation is Table 4.
