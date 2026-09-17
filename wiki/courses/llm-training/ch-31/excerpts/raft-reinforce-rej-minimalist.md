---
chapter: ch-31
course: llm-training
phase: read
excerpt_of: arXiv:2504.11343v2 (no library card at the time of writing; chapter-local verified extract)
source_url: https://arxiv.org/abs/2504.11343
created_at: "2026-09-15"
---

# Excerpt: A Minimalist Approach to LLM Reasoning: from Rejection Sampling to Reinforce

- **Authors:** Wei Xiong, Jiarui Yao, Yuhui Xu, Bo Pang, Lei Wang, Doyen Sahoo, et al. (Salesforce AI Research; University of Illinois Urbana-Champaign)
- **Year:** 2025 (arXiv v1 2025-04; v2 2025-06-12; preprint)
- **Source type:** paper
- **Code:** https://github.com/RLHFlow/Minimal-RL
- **Used in:** ch-31 §2, §4, §7, Negative samples and negative feedback, Recipe

## What the paper does

The paper compares RAFT (rejection-sampling fine-tuning on positive samples only), vanilla REINFORCE, GRPO, PPO, and iterative DPO on math reasoning, and ablates which parts of GRPO matter (Abstract, §1). It reports that RAFT is close to GRPO, that GRPO's main advantage over REINFORCE comes from discarding prompts whose responses are all incorrect, and it proposes Reinforce-Rej, which filters both all-correct and all-incorrect prompts (Abstract).

## Objectives with loci

- **Reward.** r(x, a) ∈ {−1, 1} from a verifier (Math-Verify) (§3).
- **RAFT (Eq. 1).** For each prompt, sample n responses, keep responses with the highest reward (typically r = 1), and maximize L_RAFT(θ) = Σ_{(x,a)∈D} log π_θ(a | x) (§3).
- **REINFORCE (Eq. 5).** Token-level loss (1/|D|) Σ (1/|a|) Σ_t min(s_t(θ), clip(s_t(θ), 1 − ε, 1 + ε)) · r(x, a), with s_t(θ) = π_θ(a_t | x, a_<t) / π_θold(a_t | x, a_<t) (§3).
- **GRPO.** Replaces r(x, a) with A_t(x, a_i) = (r_i − mean(r_1…r_n)) / std(r_1…r_n) (§3).
- **RAFT++ (Eq. 6).** The REINFORCE form with the indicator I[r(x, a) = argmax_i r(x, a_i)] in place of the reward, so only highest-reward responses are trained, with importance ratio and clipping (§3).
- **Interpretation stated by the authors.** REINFORCE with ±1 reward "can be viewed as fine-tuning on the positive samples and unlearning on the negative samples", and unlearning on coarse negatives is less stable (§5).

## Setup (§4)

verl framework; Numina-Math prompts (~860k problems); Qwen2.5-Math-7B-base and LLaMA-3.2-3B-instruct; AdamW, learning rate 1 × 10⁻⁶; 1,024 prompts per iteration; n = 4 responses per prompt for RAFT and GRPO; mini-batch 512; maximum 4,096 generated tokens. The Table 1 caption states that batch size, mini-batch size, and actor learning rate were tuned per algorithm; the tuned values are not printed in v2. Evaluation: average@16 at temperature 1.0 on MATH500, Minerva Math, and OlympiadBench; AIME 2024 was excluded because its 30 problems gave noisy trends (§4).

## Results (Table 1, average column)

| Algorithm | Qwen2.5-Math-7B-base | LLaMA-3.2-3B-instruct |
|---|---|---|
| Base | 23.6 | 13.1 |
| RAFT | 52.3 | 25.9 |
| RAFT++ | 56.1 | 27.6 |
| Iterative DPO | 48.8 | not reported |
| Reinforce | 53.9 | 24.2 |
| GRPO | 56.3 | 28.4 |
| PPO | 52.5 | 26.9 |
| Reinforce-Rej | 56.4 | 28.5 |

## Findings with loci

- **Importance sampling without clipping** underperformed vanilla RAFT (§5, Fig. 2).
- **Early speed and plateau.** RAFT++ learns faster early, shows a turning point around iteration 100, and is surpassed by GRPO later (§5, Fig. 2).
- **Entropy.** RAFT++ policy entropy declines faster than GRPO's on both models; its improvement slows once entropy is low; KL from the initial policy rises faster early (§5.1, Fig. 3). The authors attribute the plateau to reduced exploration (Interpretation).
- **Clip higher.** Asymmetric clipping with ε_low = 0.2 and ε_high = 0.28 stabilized entropy for RAFT++ on LLaMA and improved later-stage reward (§5.1, Fig. 2 right).
- **GRPO ablation (LLaMA-3.2-3B-instruct, Fig. 4).** "Remove all wrong" gave the largest reward gain over vanilla REINFORCE; "Remove all correct" "does not help much" in the authors' words; mean-zero normalization alone raised KL without improving reward; standard-deviation normalization added little over removing both kinds of prompts (§5.1).
- **Conclusion (§6).** The authors advocate more selective mechanisms for incorporating negative samples rather than raw negative feedback.

## Verification

- Checked on 2026-09-15 against: https://arxiv.org/abs/2504.11343 (v2, 2025-06-12), full text.
- Not reported by the source: per-algorithm tuned hyperparameters for Table 1; seeds or confidence intervals; out-of-domain or non-math evaluations; pass@k at large k.
