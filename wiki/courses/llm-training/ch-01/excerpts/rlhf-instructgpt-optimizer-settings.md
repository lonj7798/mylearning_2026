---
chapter: ch-01
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/rlhf-instructgpt.md (existing card without a Verification section on 2026-09-15; its hyperparameter table is not used by ch-01)
source_url: https://arxiv.org/abs/2203.02155
created_at: "2026-09-15"
---

# Excerpt: InstructGPT optimizer settings, epochs, learning-rate sensitivity, and PPO-ptx

**Paper:** Long Ouyang, Jeff Wu, Xu Jiang, Diogo Almeida, Carroll L. Wainwright, Pamela Mishkin, et al. (OpenAI), "Training language models to follow instructions with human feedback". Read arXiv:2203.02155v1 (2022-03-04). Source type: paper by the organization that trained the models.

## Precision and optimizer (App. C)
"All models use fp16 weights and activations, with fp32 master copies of weights." "All models are trained with the Adam optimizer, with β1 = 0.9 and β2 = 0.95." Weight decay, ε, and gradient clipping are not printed in App. C.

## SFT (§3.5, App. C.1)
- "We trained for 16 epochs, using a cosine learning rate decay, and residual dropout of 0.2." "we find that our SFT models overfit on validation loss after 1 epoch; however, we find that training for more epochs helps both the RM score and human preference ratings, despite this overfitting." Final SFT models were selected by RM score on the validation set (§3.5).
- 1.3B and 6B: LR 9.65e-6, batch 32; 175B: LR 5.03e-6, batch 8; cosine to 10% with no warmup; LR chosen by geometric search over 7 LRs (1.3B, 6B) and 5 LRs (175B); epochs tuned by geometric search (App. C.1).

## Reward model (App. C.2)
6B RM, one epoch, LR 9e-6, cosine to 10%, batch 64 prompts. "changes of up to 50% in the learning rate resulted in similar performance. Training was quite sensitive to the number of epochs: multiple epochs quickly overfit the model to the training data with obvious deterioration in the validation loss."

## PPO and PPO-ptx (§3.5 Eq. 2, App. C.3-C.4)
- objective(φ) = E_{(x,y)∼D_{π_φ^RL}}[ r_θ(x, y) − β log(π_φ^RL(y|x)/π^SFT(y|x)) ] + γ E_{x∼D_pretrain}[ log π_φ^RL(x) ]; "For 'PPO' models, γ is set to 0."
- β = 0.02; 256k episodes over about 31k unique prompts; batch 512, minibatch 64, one inner epoch; constant LR with warmup over the first 10 iterations from one tenth of peak; EMA of weights with decay 0.992; PPO clip ratio 0.2; sampling temperature 1 (App. C.4).
- ptx: "We use 8 times more pretraining examples than the number of the RL training episodes." PPO and pretraining gradients are computed in consecutive steps and accumulated; pretraining gradients are multiplied by γ = 27.8 (App. C.4).

## Learning-rate sensitivity of PPO (App. E.9)
"For both 1.3B and 6B models, we scan the learning rate in log-linear space, from 2.55e-6 to 2.55e-5, for both PPO with and without the pretraining data mix. All runs with learning rate greater than 8.05e-6 diverged, for PPO models without pretraining data mix. For the 175B models, we did similar experiments with two learning rates of 2.55e-6 and 3.74e-06 ... PPO with pretraining data mix appears to be less sensitive to change of the learning rate." Final checkpoints were picked by highest Likert score; a single final PPO LR is not printed.

## Alignment tax and the ptx fix (§1, §4.2, App. E.6)
- "During RLHF fine-tuning, we observe performance regressions compared to GPT-3 on certain public NLP datasets, notably SQuAD ..., DROP ..., HellaSwag ..., and WMT 2015 French to English translation ... This is an example of an 'alignment tax'" (§1).
- App. E.6 (1.3B): "By setting pretraining loss coefficient to greater or equal 20, the regression on these tasks can be recovered"; "a single value of 27.8 seems to work well across model sizes, from 1.3B to 175B". With γ = 0, "even by increasing the KL reward coefficient to 2.0, which is 100 times of the default value, the regressions still cannot be fixed."
- PPO-ptx "still lags behind GPT-3 on DROP, SQuADv2, and translation" (§4.2).

## Verification
- Checked on 2026-09-15 against arXiv:2203.02155v1 (§1, §3.5, §4.2, App. C, App. E.6, App. E.9).
