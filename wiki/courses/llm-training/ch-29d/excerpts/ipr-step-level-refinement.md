---
chapter: ch-29d
course: llm-training
phase: read
excerpt_of: primary source (no library card at wiki/raw-data/llm-training/**/ipr-step-level-refinement.md on 2026-09-15)
source_url: https://arxiv.org/abs/2406.11176
source_version: arXiv:2406.11176v2 (2024-09-24); v1 2024-06
created_at: "2026-09-15"
---

# Excerpt: Watch Every Step! LLM Agent Learning via Iterative Step-Level Process Refinement (IPR; Xiong, Song, Zhao, Wu, Wang, Wang, Li, Peng, Li; Peking University, Huawei)

Facts used by [[read]] §5 and the Recipe table. Read in the arXiv v2 PDF on 2026-09-15. Code: github.com/WeiminXiong/IPR.

## Step reward by Monte Carlo rollouts (§3.2)
- r_s(s_t, a_t) = E_{e_m ∼ π_s(e_{t:m} | e_{t−1})}[r_o(u, e_m)] (Eq. 3), estimated as (1/N) Σ_i r_o(u, e^(i)) over N rollouts from step t, and as r_o(u, e_n) at the last step (Eq. 5). The scorer π_s is the SFT agent with frozen parameters.

## Contrastive step pairs (§3.3)
- The agent receives the first t−1 steps of an expert trajectory and generates its own continuation from step t (Eq. 6).
- A mistake at step t is declared when the agent action's step reward is lower than the expert action's by more than τ and the agent continuation's outcome reward is lower than the expert's. The pair is (e_{t−1}, e^w_{t:n}, e^l_{t:m}); a trajectory-level pair set D_t is built from outcome rewards.
- Loss: L = L_o-DPO + L_s-DPO + L_SFT (Eq. 10). L_o-DPO is DPO on whole trajectories (Eq. 7); L_s-DPO is DPO on continuations conditioned on the shared prefix e_{t−1} (Eq. 8); L_SFT = −E log π_θ(e^w_n | u) on the winning trajectories (Eq. 9). The SFT term is added because DPO "only optimizes the relative differences between chosen and rejected data" and "the space of correct actions is significantly narrower than that of incorrect ones" (§3.3).
- The updated agent becomes the next base agent; the loop repeats up to an iteration limit.

## Settings (§4.1)
- Expert trajectories: GPT-4 in ReAct format, filtered to correct outcomes. Datasets (Table 1): WebShop 1,624 train / 200 test; ALFWorld 2,851 / 274 (140 seen, 134 unseen); InterCodeSQL 1,500 / 200.
- Llama-2-7B; 3 epochs; batch 48; AdamW; cosine schedule. MC scorer temperature 1 and N = 5. Pair generation temperature 0. τ = 0.5 (ALFWorld), 0.01 (WebShop), 0.1 (InterCodeSQL). LR searched from 1e-5 to 5e-5 and DPO β from 0.1 to 0.5. Iteration cap 4. 8 × A100 80G.

## Results (Table 2, average reward; WebShop / InterCodeSQL / ALFWorld seen / unseen / average)
- Llama-2-7B + SFT 60.2 / 54.9 / 60.0 / 67.2 / 60.6
- + PPO 64.2 / 52.4 / 22.1 / 29.1 / 42.0
- + RFT 63.6 / 56.3 / 62.9 / 66.4 / 62.3
- + ETO 67.4 / 57.2 / 68.6 / 72.4 / 66.4
- + Step-PPO 64.0 / 60.2 / 65.7 / 69.4 / 64.8
- + IPR 71.3 / 61.3 / 70.3 / 74.7 / 69.4
- ETO and IPR rows report the best iteration. The "5.8%, 7.2%, 2.5%, 3.2%" margins over ETO in §4.2 are relative; the absolute differences are 3.9, 4.1, 1.7, and 2.3 points (derived).

## Ablations (Table 4)
- Without o-DPO: 70.2 / 59.3 / 72.4 (WebShop / InterCodeSQL / ALFWorld unseen). Without s-DPO: 66.4 / 58.0 / 70.2. Without SFT: 61.8 / 31.7 / 64.9.
- Iterations 1-5 on WebShop: 63.6, 63.7, 68.2, 71.3, 68.1; the authors attribute the decline after iteration 4 to overfitting to the training set.
- Step-reward accuracy on WebShop against a heuristic page-level scoring rule reaches up to 82% (§5.3, Figure 3, τ = 0.35).
- A learned step reward model (MSE regression on 70k MC-labeled actions) scores between ETO and the MC method: Llama-2-7B 67.4 (no step reward) / 68.9 (reward model) / 71.3 (MC) (Table 5).
- Cost: "less than three times" the ETO training duration (App. C).

## Limits stated
Iterative preference learning on self-generated samples can overfit with limited training data; the step reward model was trained on one task only (Limitations).
