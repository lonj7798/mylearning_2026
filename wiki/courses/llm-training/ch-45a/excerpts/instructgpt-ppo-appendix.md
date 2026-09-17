---
chapter: ch-45a
course: llm-training
phase: read
excerpt_of: "verbatim settings from arXiv:2203.02155 App. C.3, C.4, E.7, E.9, E.11; the library card [[rlhf-instructgpt]] prints a PPO learning rate and an epochs-per-rollout value that this appendix does not contain"
source_url: https://arxiv.org/abs/2203.02155
primary_version: arXiv:2203.02155v1 (4 Mar 2022)
created_at: "2026-09-15"
---

# Excerpt: InstructGPT PPO settings (App. C.3, C.4, E.7, E.9)

Ouyang et al. (OpenAI), 2022. ch-45a uses this appendix as the oldest row of the RL-stage ledger.

## C.3 — the model PPO starts from
> "We initialize the RLHF models from a pretrained GPT-3 model and apply supervised fine-tuning for
> 2 epochs on the demonstration dataset. We also mix in 10% pretraining data during fine-tuning... We use a
> batch size of 32 for 1.3B and 6B models and 8 for the 175B model... The resultant LR's for the 1.3B, 6B, and
> 175B models are 5e-6, 1.04e-5 and 2.45e-6, respectively."

These are the PPO *initialization* models. The separate SFT baselines use LR 9.65e-6 (1.3B, 6B) and 5.03e-6
(175B) for 16 epochs (App. C.1), so the two learning rates must not be merged into one "InstructGPT SFT LR" row.

## C.4 — PPO
> "These models are also used to compute the KL reward, in the same way as Stiennon et al. (2020), with
> β = 0.02 (see Equation 2). We train all the RL models for 256k episodes. These episodes include about 31k
> unique prompts, after filtering out prompts with PII and deduplication based on common prefixes. The batch
> size for each iteration is 512, with a minibatch size of 64. In other words, each batch is randomly split into 8
> minibatches and is trained on for only a single inner epoch (Schulman et al., 2017). A constant learning rate
> is applied with a warmup over the first 10 iterations, starting with one tenth of the peak learning rate.
> Exponential moving averages of the weights are applied, with a decay rate of 0.992. No discount is applied
> when estimating the generalized advantage (Schulman et al., 2016). The PPO clip ratio is set to 0.2, and the
> sampling temperature is 1 for rollouts."

> "A fixed learning rate of 9e-6 for the value function is used for 1.3B and the 6B policies and 5e-6 for the
> 175B policy."

> "We use 8 times more pretraining examples than the number of the RL training episodes... We multiply the
> pretraining gradients by a coefficient, γ = 27.8."

## E.7, E.9, E.11 — what was swept
- KL reward coefficient: "Both 0 and 2 for KL reward coefficient result in poor performance. The optimal value
  is around 0.01 and 0.02" (E.7, Fig. 36).
- Policy learning rate: "we scan the learning rate in log-linear space, from 2.55e-6 to 2.55e-5... All runs with
  learning rate greater than 8.05e-6 diverged, for PPO models without pretraining data mix... we picked the
  checkpoints with the highest likert scores, as our final models" (E.9).
- Batch size: 64, 128, 256, 512, 1024 compared on the 1.3B model; 512 best by human evaluation. Minibatch
  8/16/32/64 compared; 32 "optimal and is slightly better than 64", but "our final models used a minibatch size
  of 64, since it has better GPU utilization" (E.11).
- Pretraining ratio: 4 raised the pretraining log-probability loss during training; 32 gave better Likert scores at
  several times the training time; 8 chosen as a middle ground (E.11).
- Episodes: "Using the 1.3B model, we did not find it helpful to train more than 256k episodes, for PPO with
  pretraining data mix" (E.11).

## Values this appendix does not contain
- A policy learning rate for the released PPO or PPO-ptx models: only the sweep range (2.55e-6 to 2.55e-5) and
  the selection rule are printed. The value 1.41e-5 appears nowhere in arXiv:2203.02155 (searched the full text).
- "4 epochs per rollout": C.4 states one inner epoch over 8 minibatches.
- A maximum response length. App. C.1 prints "language models and RL policies have a context length of 2k
  tokens", which bounds prompt plus response together; no separate generation cap is given.

## Verification
- Read on 2026-09-15 from the cached primary text of arXiv:2203.02155 (scratchpad `sources/rlhf-instructgpt.txt`),
  App. C.1, C.3, C.4, E.7, E.9, E.11.
- Not reported: policy LR of the released models; response-length cap; GAE λ (only "no discount" is stated).
