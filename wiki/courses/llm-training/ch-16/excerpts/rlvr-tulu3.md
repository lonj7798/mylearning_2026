---
chapter: ch-16
course: llm-training
phase: read
artifact: "Tülu 3: Pushing Frontiers in Open Language Model Post-Training — RLVR stage"
source_url: https://arxiv.org/abs/2411.15124
version: arXiv v5 (2025-04-14)
verified_on: "2026-09-15"
supersedes: the 2026-04 version of this file, which described a pass-rate-filtered pool derived from the 939K SFT mix
---

# Excerpt: Tülu 3 RLVR — the prompt set as actually reported

Used by [[read]] §1, §4 and the Recipe table.

## Objective (§6, Eq. 7–8)

```
max_{π_θ}  E_{y∼π_θ(x)} [ R_RLVR(x, y) ] = [ v(x, y) − β KL[π_θ(y|x) ‖ π_ref(y|x)] ]
v(x, y) = α if correct, 0 otherwise
```

`v`: the verifiable reward function; `α = 10`, set from pilot experiments and not tuned further
(§6). Optimization is PPO, run after preference finetuning.

## The prompt set (§6.1, Table 22)

| Prompt dataset | Count | Verification |
|---|---|---|
| GSM8K Train | 7,473 | exact match against the extracted answer |
| MATH Train | 7,500 | exact match against the extracted answer |
| IF verifiable | 14,973 | prompt-specific verifiers |
| **Total** | **29,946** | |

Construction, as described: the GSM8K and MATH training splits are used with their standard 8-shot
and 3-shot chain-of-thought prompts; the IF set is built by sampling instructions from the Tülu 2
SFT mix and combining them with constraints from the IFEval taxonomy, one verification function per
constraint template. The report states the combination "results in a mixture of roughly 30,000
prompts with ground truth labels".

**Corrections to the earlier chapter text.** The set is not derived by filtering the 939,344-prompt
SFT mix, no rollout pass-rate band is applied, no "filtered beats unfiltered" ablation is reported,
and code is not part of the RLVR stage (the verifier list in §6.1 is math, MATH-style math, and
instruction-following constraints only).

## Training settings (Table 21 and its caption; §6.4)

Shared: γ = 1.0, GAE λ = 0.95, mini-batches 1, clip ε 0.2, value coefficient 0.1, gradient-norm
threshold 1.0, linear LR schedule, generation temperature 1.0, max prompt length 2,048, −10 penalty
for responses without an EOS token, advantage whitening, dropout disabled.

| Setting | 8B RLVR | 70B RLVR |
|---|---|---|
| Learning rate | 3×10⁻⁷ | 1×10⁻⁷ |
| Effective batch size | 224 | 640 |
| PPO update iterations K | 4 | 4 |
| Response length | 2,048 (1,024 for GSM8K-only runs) | 2,048 |
| Total episodes | 100,000 | 400,000 |
| KL coefficient β | 0.05 | 0.07 (Table 21 caption) / 0.7 (§6.4 body) |
| Warm-up ratio ω | 0.0 | 0.07 (caption) / 0.1 (body) |

The value model is initialized from a general reward model; §6.2 reports this initialization gives
the highest GSM8K test score and the highest average against initialization from the anchored DPO
model (Figure 21). β was swept over [0.1, 0.05, 0.03, 0.01] in the ablations (§6.2).

## Episodes and epochs

§6.2 states "In our RLVR ablation experiments, we train for roughly 100,000/7,473 ≈ 13 epochs" for
the GSM8K-only pool. For the combined 29,946-prompt set at 100,000 episodes the corresponding
number is ≈ 3.3 epochs (derived, not printed in the report).

## Results and over-optimization

Table 23 (8B): MATH 42.0 → 43.7, GSM8K 84.3 → 87.6, IFEval 81.1 → 82.4 against the DPO starting
point, average 64.4 → 64.8. At 70B: MATH 62.3 → 63.0, IFEval 82.6 → 83.2, GSM8K unchanged at 93.5,
which the report attributes to saturation. §6.2 notes that a larger KL budget does not necessarily
improve verifiable rewards, and Appendix B.4 shows over-optimized outputs from higher-KL IFEval
runs. The "no reward hacking by construction" framing in the library card is not supported by these
two passages.

## 405B (§8.1)

"Given the model's saturation of GSM8K from SFT and DPO training alone, we removed the GSM8K data,
and we additionally found that the IFEval data did not help much in initial RLVR runs. As such, for
Tülu 3 405B RLVR we only used the MATH train set." MATH improved by over 5 points within 25 RLVR
steps; training stopped at 75 steps for compute reasons.
