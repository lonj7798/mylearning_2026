---
chapter: ch-49
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/generative-reward-models.md
source_url: https://arxiv.org/abs/2410.12832
created_at: "2026-04-23"
revised_at: "2026-09-15"
---

# Excerpt: Generative Reward Models (GenRM / CoT-GenRM)

**Authors:** Dakota Mahan, Duy Van Phung, Rafael Rafailov, Chase Blagden, Nathan Lile, Louis Castricato, Jan-Philipp Fränken, Chelsea Finn, Alon Albalak (SynthLabs; Stanford University)
**Year:** 2024 (arXiv v1 2024-10)
**Source type:** paper
**Checked on:** 2026-09-15 against the library card, which was re-verified on 2026-09-14 against arXiv v1.

> Rewritten in the 2026-09 revision. The previous version of this excerpt carried an author list from a
> different paper and several claims the source does not make.

## Corrections to the previous version of this excerpt

1. Authors "Lifan Yuan, Ganqu Cui, … Maosong Sun" → the author list above (title page).
2. "reward = log P('A is better' | x, y_A, y_B, rubric)" → the paper trains `−log π(I | x, y1, y2)` where `I` is an answer-indicator token, and reads preference probabilities from output likelihoods or majority votes. No rubric input and no such formula is printed (§3, §4, Eq. 7).
3. "Fig. 4 calibration plot; GenRMs are calibrated where BT RMs are overconfident" → Fig. 4 compares UltraInteract-trained models on RewardBench Reasoning versus non-Reasoning. No calibration curve and no calibration claim appears in the paper.
4. "CoT improves accuracy 3–10 pp on RewardBench"; "GenRM ensembles give calibrated uncertainty" → not supported by the source; removed.
5. Pointwise 1–10 scoring → the paper evaluates pairwise judgments only.

## Method (§4)

- **GenRM**: loss `−log π(I | x, y1, y2)` with `I` the answer-indicator token ("A" or "B"); a classifier trained by next-token prediction (Eq. 7).
- **CoT-GenRM**: the model writes a rationale `r`, then the indicator. Three ways to obtain rationales:
  - **STaR-SFT** (Eq. 8): sample `(r, I)` from the current model, keep chains whose verdict matches the label, SFT on `−log π(I | x, y1, y2, r) − log π(r | x, y1, y2)`.
  - **STaR-Rationalizer**: rationales produced by a post-rationalization model that is told the correct answer, then trained with Eq. 8.
  - **STaR-DPO** (Eq. 9): DPO with chosen `(r_w, I_w)` a rationale ending in the correct verdict and rejected `(r_l, I_l)` a rationale ending in the wrong verdict.
- Prompts are based on the MT-Bench judge prompt with ties removed; the listed factors are helpfulness, relevance, accuracy, depth, creativity, and level of detail, and the prompt instructs the judge not to let length or position influence the verdict (App. A.1, Fig. 6).
- All models start from Llama-3.1-8B-Instruct; training sets are UltraFeedback (61k pairs) and UltraInteract; generative scores in Figs. 2 and 4 are majority votes over 32 samples (§5).

## Results

- **UltraFeedback-trained (§5.1):** zero-shot judge without reasoning 52.25% on UltraFeedback, rising to 67.75% with CoT and self-consistency; Bradley–Terry RM, PairRM, and GenRM about 73–74% in-distribution; STaR-DPO 73.9%, STaR-SFT 67.4%. On RewardBench, STaR-DPO 81.9% against GenRM 78.9%; Safety subset STaR-DPO 91.0% against PairRM 81.8%.
- **UltraInteract-trained (§5.2):** in-distribution STaR-DPO 90.2% against base 68.8%, with explicit reward models about 94%. On RewardBench Reasoning, Bradley–Terry falls **below random**, best GenRM 70.8%, zero-shot judge 76.6%, STaR-DPO 87.2%. On RewardBench non-Reasoning: judge 78.0%, STaR-DPO 75.0%.
- **Majority voting at 32 (§5.4):** +1.6% on UltraFeedback and +3.8% on RewardBench for UltraFeedback-trained models; +4.6% and +4.9% for UltraInteract-trained models.
- **Rationale source (§5.3, Table 1):** rationales from a stronger model do not help. GPT-4-bootstrapped rationales start lower and end higher; Llama 3.1 70B rationales raise in-distribution accuracy slightly and lower RewardBench accuracy. STaR-Rationalizer matches STaR-DPO in-distribution but falls below the base model on RewardBench, with Maj@1 dropping from 71.73 to 67.62 over three iterations (Table 3). The authors' hypothesis is that post-rationalized rationales are off-policy for the base model (Interpretation).

## Relevance to ch-49

The result ch-49 uses is the in-distribution versus out-of-distribution split: a Bradley–Terry head can reach about 94% on the training distribution and below random on RewardBench Reasoning, while a judge trained to reason reaches 87.2% there. In-distribution accuracy is not evidence of a judge that generalizes.

## Not tested by the paper (§7)

Using the trained judge as a reward inside PPO or online preference optimization, and reward hacking of generative reward models; both are listed as future work. Batch size, sequence length, compute, and the UltraInteract pair count are not reported.

## Connections

[[rewardbench]] (the out-of-distribution evaluation), [[direct-judgement-preference]] and [[self-taught-evaluators]] (concurrent judge-training methods), [[judge-llm-bias]] (source of the judge prompt and of the zero-shot judge failures it cites).
