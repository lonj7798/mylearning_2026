---
chapter: ch-33
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/model-reports/tulu-3.md
source_url: https://arxiv.org/abs/2411.15124
created_at: "2026-04-23"
revised: "2026-09-15 (generality revision; rewritten from arXiv:2411.15124v5 §6, Tables 21–23, App. B.4)"
---

# Excerpt: Tülu 3 RLVR — objective, data, PPO settings, and final results

**Primary source:** arXiv:2411.15124v5 §6 (Eq. 7–8), Table 21, Table 22, Table 23, §6.2–6.4, App. B.4.
**Library cards:** [[tulu-3]], [[rlvr-tulu3]]. The previous version of this excerpt, and the [[rlvr-tulu3]] card, stated a binary {0, 1} reward, a unit-test code verifier with a 5-second timeout, "10M episodes", "LR ~1e-6, β_KL ~0.04", and gains of "+5–10pp GSM8K, +~4pp IFEval" against a Llama 3.1 8B Instruct baseline of 84.7/41.5/80.5. None of these appear in the paper; the paper values are below.

## Objective (§6, Eq. 7–8)

max over π_θ of E_{y∼π_θ(x)} [ v(x, y) − β · KL[π_θ(y|x) ‖ π_ref(y|x)] ], with v(x, y) = α if the answer is correct, 0 otherwise.

- v: verification function on the prompt–completion pair.
- α = 10: "We set α = 10 based on pilot experiments and did not tune it further."
- β: KL penalty coefficient; π_ref: the starting DPO checkpoint.
- The optimizer is PPO (Schulman et al., 2017).

## Data (Table 22)

| Prompt set | Count | Verification |
|---|---:|---|
| GSM8K train | 7,473 | exact match against the extracted final number (8-shot CoT prompt added) |
| MATH train | 7,500 | exact match after "flex" answer extraction (3-shot CoT prompt added) |
| IF verifiable | 14,973 | prompt-specific constraint verifiers (Tülu 2 SFT instructions + IFEval taxonomy constraints) |
| Total | 29,946 | |

The paper focuses on "two domains (mathematics, exact instruction following)" and leaves more complex verifiers to future work; code execution feedback is cited as related work only (§6.1, footnote 17).

## Implementation details (§6.2)

1. Value model initialized from a general reward model (best in Fig. 21).
2. Dropout disabled in RM and RL training, so rollout and learning log-probabilities match.
3. Prompts reshuffled between epochs; the ablations ran about 100,000 / 7,473 ≈ 13 epochs over GSM8K.
4. Responses without an end-of-sequence token get −10.
5. Advantages are whitened (mean subtracted, divided by standard deviation).

## Table 21 — PPO hyperparameters (RLVR column)

| Hyperparameter | Value |
|---|---|
| γ; GAE λ | 1.0; 0.95 |
| Mini-batches N_mb; PPO update iterations K | 1; 4 |
| Clip ε; value coefficient c1; gradient norm threshold | 0.2; 0.1; 1.0 |
| LR schedule; learning rate | linear; 3 × 10⁻⁷ (1 × 10⁻⁷ for 70B) |
| Generation temperature; max token length; max prompt length | 1.0; 2,048; 2,048 |
| No-EOS penalty reward | −10.0 |
| Effective batch size | 224 (640 for 70B) |
| Response length | 2,048 (1,024 for GSM8K only) |
| Total episodes | 100,000 |
| β sweep; warmup sweep | [0.1, 0.05, 0.03, 0.01]; [0.0, 0.1] |

Caption: "The final 8B RLVR model used β = 0.05 and ω = 0.0; the final 70B RLVR model used β = 0.07 and ω = 0.07." §6.4 text for 70B instead gives "0.1 warmup ratio, 2048 response length, 400,000 episodes, 640 effective batch size, and β = 0.7". The open-instruct 70B command uses β 0.07 and warmup 0.1 ([[open-instruct-allenai-recipes]] L375–383). The open-instruct 8B command sets `total_episodes` 10,000,000 as a planned horizon (L329).

## Selection and results (§6.3–6.4, Table 23)

- Checkpoints evaluated every 100 steps (40 for 70B); the 8B checkpoint with the best MATH and IFEval was released. "Some 8B runs were able to achieve GSM8k scores of up to 89.4% and IFEval scores of up to 84.8% (although such models tended to perform worse in other metrics, dragging down their overall average)."
- "For all of these models we took an earlier than final checkpoint from the run" (§6.3).

| Benchmark (8B) | Llama 3.1 8B Instruct | Tülu 3 DPO | Tülu 3 RLVR |
|---|---:|---:|---:|
| Avg. | 62.2 | 64.4 | 64.8 |
| MMLU | 71.2 | 68.7 | 68.2 |
| TruthfulQA | 55.1 | 56.1 | 55.0 |
| BBH | 62.8 | 65.8 | 66.0 |
| MATH | 42.5 | 42.0 | 43.7 |
| GSM8K | 83.4 | 84.3 | 87.6 |
| HumanEval+ | 82.9 | 78.6 | 79.2 |
| IFEval | 80.6 | 81.1 | 82.4 |
| AlpacaEval 2 | 24.2 | 33.5 | 34.5 |
| Safety (6-task avg.) | 75.2 | 87.2 | 85.5 |

At 70B: GSM8K 93.5 → 93.5, MATH 62.3 → 63.0, IFEval 82.6 → 83.2, safety 89.0 → 88.3 (Table 23). Table 6 prints different BBH (68.7 → 69.0) and average (64.7 → 65.1) values for the same 8B checkpoints.

## Over-optimization (§6.2.1, App. B.4)

"More KL divergence typically results in lower average scores" (Fig. 21). At β = 0.01 on constraint prompts, the prompt "Measure the length of the given object in cm. Pen. In your response, the letter e should appear 14 times." received fourteen comma-separated letters "e" and no measurement (Fig. 28). The β = 0.1 model answered "The length of a typical pen is approximately 15 centimeters." (Fig. 29).

## Used in

ch-33 §4.1–4.3, Negative samples and negative feedback, Recipe.
