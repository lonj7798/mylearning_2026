---
chapter: ch-41
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/generative-reward-models.md (arXiv:2410.12832v1)
source_url: https://arxiv.org/abs/2410.12832
created_at: "2026-04-23"
revised: "2026-09-15 (generality revision; rewritten to match the verified card and primary source)"
---

# Excerpt: Generative Reward Models (Mahan et al.)

Used by [[read]] §3.5, the negatives section, the Recipe, and the Generalization lens. Source card: [[generative-reward-models]].

The earlier version of this excerpt described a reward `log P("A is better" | x, y_A, y_B, rubric)`, a 3–10 pp gain from critiques, calibrated uncertainty, and rubric steerability. The paper prints none of these. It trains `−log π(I | x, y1, y2)` for an answer-indicator token, uses no rubric input, evaluates pairwise judgments only, and does not measure calibration (card Verification).

## Method (§4)
- GenRM: next-token loss on the indicator token "A" or "B" (Eq. 7).
- STaR-SFT: sample rationale and verdict; keep chains with the correct verdict; SFT on rationale and verdict (Eq. 8).
- STaR-Rationalizer: rationales from a post-rationalization prompt given the correct answer (§4, §5.2).
- STaR-DPO: DPO with chosen = rationale reaching the correct verdict, rejected = rationale reaching the wrong verdict (Eq. 9).
- Prompt based on the MT-Bench judge prompt with ties removed (App. A.1).

## Results (Llama-3.1-8B-Instruct)
- UltraFeedback-trained (§5.1): BT RM, PairRM, GenRM around 73–74% in-distribution; STaR-DPO 73.9%; STaR-SFT 67.4%. RewardBench: STaR-DPO 81.9%, GenRM 78.9%, base-model prior 77.8%; Safety: STaR-DPO 91.0% vs PairRM 81.8%.
- UltraInteract-trained (§5.2): explicit RMs about 94% in-distribution; on RewardBench Reasoning the BT RM falls below random, best GenRM 70.8%, LLM-as-a-judge 76.6%, STaR-DPO 87.2%.
- Rationale source (§5.3, Table 1): Llama 3.1 70B rationales slightly raise UltraFeedback accuracy and lower RewardBench accuracy.
- Majority voting over 32 samples adds 1.6–4.9 points depending on training and evaluation set (§5.4).

## Limits
One 8B model; accuracy on preference benchmarks only; use of the judge as an RL reward and reward hacking of generative RMs are future work (§7). No ablation separates the use of negatives from the change of loss between STaR-SFT and STaR-DPO.
