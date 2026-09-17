---
chapter: ch-40
course: llm-training
phase: read
excerpt_of: arXiv:2506.10947v2 (Spurious Rewards: Rethinking Training Signals in RLVR); library card [[spurious-rewards-rlvr]]
source_url: https://arxiv.org/abs/2506.10947
created_at: "2026-09-15"
revised: "2026-09-15 (generality revision; written from the primary source because the library card has no Verification section)"
---

# Excerpt: Spurious rewards and model-family dependence

Used by [[read]] §9, the Generalization lens, and the Common-mistakes table. Authors: Rulin Shao, Shuyue Stella Li, Rui Xin, Scott Geng, Yiping Wang, Sewoong Oh, et al. (UW, Ai2, UC Berkeley). Text read from arXiv v2 (25 Feb 2026) on 2026-09-15.

## Setup (§2.1, App. D)
GRPO on DeepScaleR data; 16 rollouts per prompt; rollout batch 64 prompts; mini-batch 128 rollouts; constant LR 5e-7; temperature 1.0; 300 steps; 8 GPUs. Reward functions replace the ground-truth verifier in decreasing order of information: ground truth, majority vote over 64 rollouts, one-shot label, format reward (`\boxed{}` present), incorrect label (an incorrect rollout treated as ground truth), and random reward (1 with probability 0.5).

## Main results
- Qwen2.5-Math-7B, MATH-500: random rewards give +21.4 absolute points against +29.1 for ground-truth rewards (Abstract, §2).
- Qwen2.5-7B also gains from most signals; Llama3.1-8B-Instruct gains only from informative signals; OLMo2-7B gains only from ground truth, with several signals producing losses (Fig. 1, §3).
- Code reasoning (reasoning written as Python without execution) is the behaviour amplified in Qwen2.5-Math: frequency 65% before RL, over 90% after training with any spurious reward. Answers containing code score 60.9% against 28.0% without, on Qwen2.5-Math-7B (§5, Table 1).
- Forcing the first sentence to be "Let's solve this using Python" changes MATH-500 by +24.2 (Qwen2.5-Math-1.5B) and +15.0 (Qwen2.5-Math-7B) but −28.6 (Llama3.2-3B-Instruct), −21.6 (Llama3.1-8B-Instruct), −1.2 (OLMo2-7B), −2.8 (OLMo2-7B-SFT) (Table 2).

## Mechanism (§4, App. B)
With random rewards the expected advantage is zero, so an unclipped objective has zero expected gradient. The clip term introduces a bias that favours tokens with high probability under the sampling policy: a token at probability 0.85 can rise to 1.02 only in principle (probabilities cannot exceed 1), so it is never upper-clipped and keeps a non-negative gradient, while a rarely sampled token at 0.02 is clipped at 0.024. Removing clipping in three ways — disabling the clip, setting the mini-batch equal to the rollout batch, or reducing the rollout size to a single update per rollout — removes the consistent gains from random rewards (Fig. 4).

## Statement the chapter uses
"A range of recent research on RLVR drew conclusions on Qwen2.5-Math-7B-centric experiments… we suggest that future RLVR research should be confirmed on other models and using spurious rewards as dummy baselines" (§1, §3). The authors re-ran test-time training and one-shot RL and found the same pattern: gains on Qwen, often none on other families (§3, App. E Fig. 15).

## Limits
One RL algorithm (GRPO), mathematics only, 300 steps, no seeds reported. The clipping-bias account is the authors' explanation for the random-reward case; it does not explain gains from informative rewards.
