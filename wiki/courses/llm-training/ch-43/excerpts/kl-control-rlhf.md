---
chapter: ch-43
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/kl-control-rlhf.md
source_url: https://arxiv.org/abs/2203.02155
primary_version: "InstructGPT arXiv:2203.02155v1; Korbak et al. arXiv:2205.11275v2"
created_at: "2026-04-23"
revised: "2026-09-15 (generality revision; each claim re-read in the primary papers)"
---

# Excerpt: KL-controlled RLHF — the objective, where the penalty is applied, and what it does not fix

The library card [[kl-control-rlhf]] covers a lineage (Jaques 2019, Stiennon 2020, Ouyang 2022, Korbak 2022)
and has not been re-verified against every paper in it. The facts below were read in the primary texts named.

## InstructGPT objective (arXiv:2203.02155v1, Eq. 2)
```
objective(φ) = E_{(x,y)~D_π^RL}[ r_θ(x, y) − β·log( π_φ^RL(y|x) / π^SFT(y|x) ) ]
               + γ·E_{x~D_pretrain}[ log π_φ^RL(x) ]
```
- `r_θ` is the reward-model score, `π^SFT` the frozen reference, `β` the KL reward coefficient, and `γ` the
  pretraining-loss coefficient. For the models labelled "PPO", `γ = 0`; "PPO-ptx" keeps `γ > 0`.
- The penalty enters the reward, per token, before PPO's advantage estimation. "These models are also used to
  compute the KL reward, in the same way as Stiennon et al. (2020), with β = 0.02" (App. C.4). All RL models
  in the paper are trained for 256k episodes, batch 512, minibatch 64, one inner epoch.
- App. E.7: the human Likert score as a function of the KL reward coefficient is poor at both 0 and 2; "The
  optimal value is around 0.01 and 0.02."
- App. E.6: with the pretraining term switched off (`γ = 0`), sweeping the KL coefficient up to 2.0 — 100× the
  default — did not remove the regressions on public NLP datasets, while a large coefficient lowered the
  validation reward. The authors conclude that the pretraining data mix, not a larger KL penalty, is what
  keeps the pretrained capabilities.

## Korbak et al. 2022 (arXiv:2205.11275v2)
The KL-regularized objective has the closed-form optimum `π_KL-RL(x) = (1/Z)·π_0(x)·exp(r(x)/β)` (Eq. 5), and
maximizing the objective is equivalent to minimizing `D_KL(π_θ, π_KL-RL)` (Eq. 7). `π_0` is the prior, `β` the
temperature of the tilt, `Z` the normalizer. The paper's stated motivation is that reward maximization without
the penalty has a degenerate optimum — a distribution concentrated on the highest-reward sequences — which it
calls distribution collapse (§1-§2).

## Removed in the 2026-09 revision
The previous version of this excerpt stated that adding the KL to the loss "breaks the advantage-based policy
gradient and empirically trains worse", that production stacks run β in 0.01-0.1, and that all frameworks
default to k3. None of these is supported by the primary texts: the framework defaults are in
[[entropy-logging-patterns]] (verl and TRL GRPO default to no reference KL at all; OpenRLHF and TRL PPO put
k1 in the reward), and [[grpo]] gives the reason DeepSeekMath states for the loss placement — "avoiding
complicating the calculation of Â" (§4.1.1).
