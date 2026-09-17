---
chapter: ch-43a
course: llm-training
phase: read
excerpt_of: primary source (no library card at wiki/raw-data/llm-training/papers/negative-sample-reinforcement.md on 2026-09-15)
source_url: https://arxiv.org/abs/2506.01347
source_version: arXiv v2 (2025-10-25); v1 2025-06-02; NeurIPS 2025
created_at: "2026-09-15"
---

# Excerpt: The Surprising Effectiveness of Negative Reinforcement in LLM Reasoning (Zhu, Xia, Wei, Chen, Chen, Meng; UVA and Princeton)

Facts used by [[read]], read in the arXiv v2 PDF on 2026-09-15.

## Decomposition (§2.2, Eqs. 2-4)
- RLVR with r ∈ {−1, +1} splits into `L_RLVR = L_PSR + L_NSR`, where PSR (positive sample reinforcement) sums −π_θ(y|x) over correct responses and NSR (negative sample reinforcement) sums +π_θ(y|x) over incorrect ones. Both are on-policy: responses are sampled from the model during training.
- §C: in RLVR the batch-mean reward stays in [−1, 1], so mean subtraction preserves the sign of each sample's reward; this is what makes the split well defined (it does not hold for reward-model scores).

## Token-level gradients (§4.2, Eqs. 7-8; App. A)
With `π_v = π_θ(v | x, y_<t)` and loss `L = −R · (1/T) Σ_t π_θ(y_t | x, y_<t)`:
- PSR: `−∂L_PSR/∂z_v ∝ π_v(1 − π_v)` for the sampled token, `−π_{y_t} π_v` for every other token.
- NSR: `−∂L_NSR/∂z_v ∝ −π_v(1 − π_v)` for the sampled token, `+π_{y_t} π_v` for every other token.
- Stated properties of NSR: the update on a sampled token is scaled by (1 − π_{y_t}), so confident tokens inside a wrong answer receive small updates; unsampled tokens gain logit in proportion to their current probability ("prior-guided probability redistribution"); updates stop once the model no longer produces the wrong answer.
- Comparison with unlikelihood (App. B, Eqs. 10-11): for `L = −log(1 − π_{y_t})` the sampled-token gradient is −π_v and other tokens receive `π_{y_t}/(1 − π_{y_t}) · π_v`, so "confident predictions are penalized more aggressively".
- Comparison with an entropy bonus (App. B): the entropy gradient is `−π_v(log π_v − Σ_{v'} π_{v'} log π_{v'})`, which pushes down any token more probable than average and pushes up any token less probable than average, "conflict[ing] with the model's prior knowledge".
- §4.3: clipping constrains magnitude but not direction; the GRPO advantage rescales but keeps the reward's sign; so the analysis carries to PPO and GRPO.

## Setup (§3.1; App. D.1)
- Models Qwen2.5-Math-7B, Qwen3-4B (non-thinking mode), Llama-3.1-8B-Instruct; training set MATH (7,500 problems) in verl; prompt batch 1,024, 8 rollouts per prompt, temperature 1.0, mini-batch 256, learning rate 1e-6; maximum context 4,096 (Qwen2.5-Math-7B, Llama-3.1-8B-Instruct) and 32,768 (Qwen3-4B).
- Evaluation: unbiased pass@k estimator with 256 samples per problem (temperature 0.6, top-p 0.95) for the 7B and 8B models and 64 for Qwen3-4B; k up to 256.

## Results (Table 1, Qwen2.5-Math-7B)
| Method | MATH pass@1 | MATH pass@256 | AIME 2025 pass@1 | AIME 2025 pass@256 | AMC23 pass@1 | AMC23 pass@256 |
|---|---|---|---|---|---|---|
| Base model | 63.2 | 96.9 | 6.1 | 46.7 | 41.0 | 100.0 |
| PPO | 76.6 | 96.3 | 8.5 | 43.3 | 62.0 | 97.5 |
| GRPO | 76.3 | 95.5 | 10.3 | 50.0 | 61.7 | 97.5 |
| REINFORCE | 74.8 | 92.0 | 9.2 | 50.0 | 61.8 | 92.5 |
| PSR | 74.1 | 91.2 | 11.6 | 43.3 | 62.6 | 92.5 |
| NSR | 75.7 | 96.9 | 10.0 | 53.3 | 60.9 | 100.0 |
| W-REINFORCE | 76.6 | 96.7 | 10.6 | 56.7 | 62.0 | 97.5 |

- W-REINFORCE (Eq. 9) multiplies the PSR term by λ and keeps the NSR term at weight 1; λ = 0.1 in the main runs. λ ablation on MATH pass@256 (Table 4): λ = 0 gives 96.9, λ = 0.05 gives 97.1, λ = 0.1 gives 96.7, λ = 0.2 gives 95.9, λ = 1 (plain REINFORCE) gives 92.0; performance is "relatively stable when λ ≤ 0.2".
- Entropy on a held-out test set (Figure 5b): NSR stays close to the base model's entropy, PSR drops rapidly, PPO and GRPO fall in between. Correct-sample ratio and fully-solved-prompt ratio rise for NSR but stay below PSR's (Figures 5c-d).
- Qwen3-4B non-thinking (§3.2): NSR reaches MATH pass@1 94.0 and pass@64 98.0, GRPO 93.9 and 98.2, against the model's own thinking mode at 94.5 and 97.8; PSR does not improve the base model.
- Llama-3.1-8B-Instruct (Figure 4): every method, including NSR, ends below the base model's pass@256; NSR degrades it least.

## Limitation stated by the authors (App. F)
- "Performance degradation with extended NSR training": training with NSR over hundreds of gradient steps gives "a noticeable decline in performance", while W-REINFORCE does not show this; the authors note GRPO shows comparable degradation under extended updates and suggest NSR as a warm-up phase.
- Results are limited to sparse binary verifiable rewards; behaviour under dense or subjective rewards is stated as an open question.
