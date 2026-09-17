---
chapter: ch-38
course: llm-training
phase: read
excerpt_of: (no library card at time of writing; primary source; follow-up referenced in [[costa-huang-ppo-details]])
source_url: https://arxiv.org/abs/2403.17031
revised_at: "2026-09-15"
---

# Excerpt: The N+ Implementation Details of RLHF with PPO (Huang et al. 2024)

Authors: Shengyi Huang, Michael Noukhovitch, Arian Hosseini, Kashif Rasul, Weixun Wang, Lewis Tunstall. Checked against arXiv:2403.17031v1 (2024-03-24) on 2026-09-15. Setting: Pythia 1B, 2.8B, 6.9B, TL;DR summarization, four random seeds. Used in read.md §3, §4, §9 and Recipe.

## Reward (Eq. 3)
- R(x, y) = r_ϕ(x, y) − β D_KL(π_θ(y|x) ‖ π^SFT(y|x)).

## Details used
- Detail 7: dropout disabled, because with dropout "the log probabilities of tokens will not be reproducible, making calculating the KL penalty unreliable while also causing the ratios of the PPO to be not 1s during the first epoch".
- Detail 22: value model initialized from the RM; the warm start "can greatly improve initial gradients to the policy and reduce drift / alignment tax over training (Noukhovitch et al., 2023)". No ablation in this paper.
- Detail 23: EOS trick; completions without EOS receive a constant −1 score; it also "encourages the model to generate concise completions".
- Detail 24: optional reward whitening; "reward whitening makes the model's completions get a lower preference rate, and the completions are shorter"; length-controlled comparisons similar (Figures 11-12).
- Detail 25: advantage whitening.

## Table 7 (PPO hyperparameters, default)
Episodes 1,000,000 (about 8.56 epochs); AdamW eps 1e-5, lr 3e-6; linear scheduler; batch 512; β 0.05; γ 1.0; λ 0.95; 1 minibatch; K = 4 PPO update iterations per epoch; ε 0.2; value clip 0.2; c_1 0.1; value loss clipping True; sampling temperature 0.7. Detail 20: setup "closely follows Stiennon et al. (2020), except for a modified learning rate".

## Results (§7.1)
- GPT-3.5 prefers the best 6.9B model's summaries "nearly 80% of the time" over references.
- 1B over-optimization: KL "around 50 and 85 for two runs"; higher RLHF reward, but "less than 20% of time GPT prefers them over reference summaries".
- PPO responses are longer than SFT; controlling for length, PPO outperforms SFT across summary lengths.
