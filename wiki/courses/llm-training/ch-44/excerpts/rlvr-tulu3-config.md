---
chapter: ch-44
course: llm-training
phase: read
excerpt_of: arXiv:2411.15124v5 (Tülu 3: Pushing Frontiers in Open Language Model Post-Training), §6 and Tables 21-23
source_url: https://arxiv.org/abs/2411.15124
created_at: "2026-04-23"
revised: "2026-09-15 (generality revision; rewritten from the primary source because the library cards [[rlvr-tulu3]] and [[tulu-3]] predate the 2026-09 audits and state RLVR domains and gains the paper does not support)"
---

# Excerpt: Tülu 3 RLVR, read from the paper

Used by [[read]] §5, the Recipe table, and the negatives section. Checked on 2026-09-15 against arXiv:2411.15124v5.

## Objective and reward (§6, Eq. 7-8)
```
max_πθ  E_{y ~ πθ(x)} [ R_RLVR(x, y) ] = [ v(x, y) − β KL[ πθ(y|x) ‖ π_ref(y|x) ] ]

v(x, y) = α   if correct
        = 0   otherwise
```
- "We set α = 10 based on pilot experiments and did not tune it further" (§6). The reward is not 0/1; a correct answer is worth 10.
- The optimizer is PPO, applied after preference finetuning (§6).

## Domains and prompts (§6.1, Table 22)
| Prompt set | Count | Verification |
|---|---|---|
| GSM8K train | 7,473 | 8-shot CoT prompt, extract the final number, compare with the label |
| MATH train | 7,500 | 3-shot CoT prompt, extract the answer, "flex" MATH evaluation logic |
| IF verifiable | 14,973 | one verification function per constraint template from the IFEval taxonomy |
| Total | 29,946 | — |

There is no code verifier in the Tülu 3 RLVR stage. Code execution feedback appears once, in footnote 17, as related work that the paper leaves to future work.

## Implementation details (§6.2)
1. The value model is initialized from a general reward model trained on UltraFeedback; Figure 21 reports this beats initializing from the DPO policy on GSM8K and on the average score.
2. Dropout is set to 0 so that rollout-phase and learning-phase log-probabilities match.
3. Training runs multiple epochs over the prompt set (about 13 epochs in the GSM8K-only ablation, 100,000 / 7,473); prompts are shuffled between epochs; checkpoints are inspected every 40-100 steps.
4. **Non-EOS penalty**: a sampled response that does not end with an EOS token receives −10.
5. Advantages are whitened (mean subtracted, divided by the standard deviation).
- Adding reward-model scores on top of verifiable rewards performed worse on GSM8K and was noisier on the average score (§6.2.1, Fig. 22).
- Lower β produces larger KL and, in these runs, lower average scores; Appendix B.4 shows over-optimized outputs from high-KL IFEval runs (§6.2.1, Fig. 21).

## Hyperparameters (Table 21, RLVR column, and its caption)
| Setting | Value |
|---|---|
| γ; GAE λ | 1.0; 0.95 |
| Mini-batches N_mb; clip ε; value coefficient c1; grad-norm clip | 1; 0.2; 0.1; 1.0 |
| LR (schedule) | 3e-7 linear (1e-7 for 70B) |
| Effective batch size | 224 (640 for 70B) |
| PPO update iterations K | 4 |
| Response length | 2,048 (1,024 for GSM8K only) |
| Max prompt length; generation temperature | 2,048; 1.0 |
| Total episodes | 100,000 |
| KL coefficient β (swept) | [0.1, 0.05, 0.03, 0.01] |
| Warm-up ratio ω (swept) | [0.0, 0.1] |
| Penalty for a response without EOS | −10.0 |
| Final 8B run | β = 0.05, ω = 0.0 (Table 21 caption) |
| Final 70B run | β = 0.07, ω = 0.07 (Table 21 caption); §6.4 text writes β = 0.7, 0.1 warmup ratio, 2,048 response length, 400,000 episodes, 640 effective batch — a conflict inside the paper |
| Compute | final 8B RL run 65 hours on 8 GPUs; 70B 60 hours on 48 GPUs; 405B 46 hours on 256 GPUs (§6.3) |

The value 10,000,000 episodes in the library card [[tulu-3]] is not in the paper.

## Results (Table 23)
| Benchmark | Llama 3.1 8B Instruct | Tülu 3 8B DPO | Tülu 3 8B RLVR | Llama 3.1 70B Instruct | Tülu 3 70B DPO | Tülu 3 70B RLVR |
|---|---|---|---|---|---|---|
| GSM8K (8-shot CoT) | 83.4 | 84.3 | 87.6 | 93.7 | 93.5 | 93.5 |
| MATH (4-shot CoT, flex) | 42.5 | 42.0 | 43.7 | 56.4 | 62.3 | 63.0 |
| IFEval (strict) | 80.6 | 81.1 | 82.4 | 88.0 | 82.6 | 83.2 |
| Average of 11 evaluations | 62.2 | 64.4 | 64.8 | 73.4 | 75.9 | 76.0 |

- §6.4: single 8B runs reached GSM8K 89.4 and IFEval 84.8, but scored worse elsewhere and were not selected. At 70B the report describes modest IFEval and MATH gains and no GSM8K gain, with GSM8K near saturation, and notes that the 70B run keeps KL below 1 throughout.
