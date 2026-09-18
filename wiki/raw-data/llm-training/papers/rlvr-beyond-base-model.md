<!-- scope: pass@k study of whether RLVR expands the reasoning boundary beyond the base model
     deps: [[grpo]], [[rloo]], [[deepseek-r1]]
     see-also: [[spurious-rewards-rlvr]], [[echo-chamber-rl-post-training]], [[prorl]], [[likelihood-displacement]]
-->

# Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model?
- **Core Insight:** Measured with pass@k at large k, RLVR-trained models beat their base models at k=1 but are overtaken by the base models as k grows; on AIME24 at k=1024 the base model solves 13.3% of problems the RLVR model fails, while the RLVR model solves 0.0% that the base model fails (§4.1, Table 2).
- **Guideline:** When judging whether an RLVR run added reasoning ability rather than sampling efficiency, report pass@k at large k (the paper uses k=128 to k=1024) alongside pass@1, because pass@1 can rise while pass@256 falls on the same training run (§4.4, Figure 1 right).
- **Authors:** Yang Yue, Zhiqi Chen, Rui Lu, Andrew Zhao, Zhaokai Wang, Yang Yue, Shiji Song, Gao Huang (LeapLab, Tsinghua University; Shanghai Jiao Tong University)
- **Year:** 2025 (arXiv v1 2025-04; v5 2025-11-24)
- **URL:** https://arxiv.org/abs/2504.13837
- **Source type:** paper
- **Relevant topics:** RLVR evaluation, pass@k, reasoning boundary, sampling efficiency, entropy, distillation vs RL

## Abstract
The paper tests the claim that RLVR gives language models reasoning abilities their base models do not
have. Across math, code generation, and visual reasoning, and across several model families and six RL
algorithms, the authors measure pass@k over a wide range of k. RLVR models score higher at small k;
base models score equal or higher at large k. Accuracy-histogram, solvable-set, and perplexity analyses
indicate that the reasoning paths the RLVR models produce are already inside the base models' sampling
distribution. Distillation, in contrast, raises the pass@k curve above the base model at all measured k.
The authors conclude that current RLVR improves sampling efficiency within the base model's support and
rarely adds new reasoning patterns (Interpretation, stated by the authors in §1 and §5).

## Key Contributions
- Uses pass@k at large k as the measure of the reasoning-capacity boundary, with an unbiased estimator (§2.2, §A.2).
- Reports the crossover pattern — RLVR ahead at small k, base ahead at large k — on math, code, and visual reasoning (§3.1–§3.3).
- Shows with accuracy histograms, solvable-set comparison, and perplexity that RLVR outputs lie inside the base distribution (§4.1, Figures 5–6, Table 2).
- Defines the Sampling Efficiency Gap ∆SE = base pass@256 − RL pass@1 and compares six RL algorithms under it (§4.3, Figure 8).
- Separates distillation from RLVR: the distilled model's pass@k curve is above the base model's throughout (§4.2, Figure 7).

## Key Figures/Tables to Study
- **Figure 1 (right):** on Omni-MATH-Train, pass@1 rises from 26.1 to 42.5 across GRPO steps while pass@256 falls (§4.4).
- **Figure 2:** pass@k curves, base vs RLVR, on six math benchmarks.
- **Table 2:** solvable/unsolvable split on AIME24 (k=1024) and MATH500 (k=128).
- **Figure 7:** base / Instruct / RLVR / distilled pass@k on Minerva.
- **Figure 8 (top):** ∆SE per RL algorithm; Tables 3 and 4 give the point values.

## Technical Details

### Experimental setup (§3, Table 1)
- Math: SimpleRLZoo GRPO models over Qwen2.5-7B/14B/32B-Base and LLaMA-3.1-8B, trained on GSM8K and the MATH training set with correctness reward only and no format reward (§3.1). Also evaluated: Oat-Zero-7B and DAPO-32B.
- Math benchmarks: GSM8K, MATH500, Minerva, Olympiad, AIME24, AMC23 (§3.1).
- Code: CodeR1-Zero-Qwen2.5-7B, trained on 12K LeetCode and TACO samples over 832 steps from Qwen2.5-7B-Instruct-1M; and DeepCoder-14B built on DeepSeek-R1-Distill-Qwen-14B, both at 32k response length. Benchmarks: LiveCodeBench v5 (279 problems, 2024-08 to 2025-01), HumanEval+, MBPP+ (§3.2).
- Visual: Qwen2.5-VL-7B trained with EasyR1 on Geometry3K; evaluated on MathVista-TestMini and MathVision-TestMini with multiple-choice questions removed (§3.3).

### Crossover result
- On Minerva with a 32B model, the base model exceeds the RL model by about 9% at k=128 (§3.1).
- For the Oat-Zero and DAPO models the RL model starts nearly 30% above the base model at small k and is eventually surpassed (§3.1, Figure 11).
- Coding and visual reasoning show the same crossover (§3.2, §3.3, Figures 3, 4, 12).
- Manual CoT inspection on problems with average accuracy below 5%: for visual reasoning, 7 of 8 problems have at least one correct CoT for both the original and the RL model (§3.3).

### Evidence that RLVR stays inside the base distribution (§4.1)
- Accuracy histogram on Minerva for Qwen2.5-7B: RLVR raises the frequency of per-problem accuracy near 1.0, lowers the 0.1–0.2 bins, and raises the frequency at accuracy exactly 0 (Figure 5).
- Solvable-set split (Table 2): AIME24 k=1024 — both solve 63.3%, base only 13.3%, RL only 0.0%, neither 23.3%. MATH500 k=128 — both 92.4%, base only 3.6%, RL only 1.0%, neither 3.0%.
- Perplexity: with Qwen2.5-7B-Base as the scoring model, the distribution of PPL over RL-model responses matches the lower part of the distribution over base-model responses (Figure 6); PPL_base(Y_RL) decreases over RL training (§C.4, Figure 15).

### Algorithm comparison (§4.3)
- Six algorithms re-implemented in VeRL: PPO, GRPO, Reinforce++, RLOO, ReMax, DAPO. KL term removed.
- Training: AdamW, constant learning rate 1e-6, prompt batch 256, 8 responses per prompt, PPO mini-batch 256, max rollout length 8,192 tokens, sampling temperature 1.0 (§4.3).
- Data: Omni-MATH-Rule split into 2,000 training and 821 in-domain test problems; MATH500 as out-of-domain (§4.3).
- ∆SE on the in-domain test set ranges from 42.6 (RLOO, best) to 43.9 (GRPO) and stays above 40 points for every algorithm (§4.3).

### Ablations (§4.4–§4.6)
- Rollouts per prompt raised from 8 to 32: pass@k improves slightly but the base model still overtakes the RL model (Figure 16).
- KL penalty with coefficient 0.001: similar pass@1 to GRPO without KL, lower pass@128 (Figure 16).
- Entropy: raising the RLVR model's sampling temperature to match the base model's entropy at T=0.6 improves its own pass@k but does not reach the base model's curve, so reduced entropy is a contributing but not sufficient explanation (§4.5, Figure 18).
- Near-frontier check: Magistral-Medium-2506 (pure RL from Mistral-Medium-3-2505, size undisclosed) at 40k context solves about 7 more AIME24 problems and 8 more AIME25 problems than the base at k=1, with the gap narrowing as k grows (§4.6, Figure 9).

## Findings relevant to generality
- The study measures coverage of solvable problems within the same task families; it does not measure transfer to domains absent from RL training. Out-of-domain here means MATH500 relative to Omni-MATH training data (§4.3).
- Across training steps, in-domain pass@1 and pass@256 move in opposite directions, so a single-k metric can report improvement while coverage falls (§4.4).

## Findings relevant to distillation
- DeepSeek-R1-Distill-Qwen-7B (DeepSeek-R1 distilled into Qwen2.5-Math-7B) is compared with the base Qwen2.5-Math-7B, the RL-trained Qwen2.5-Math-7B-Oat-Zero, and Qwen2.5-Math-7B-Instruct on Minerva (§4.2).
- The distilled model's pass@k curve is above the base model's across the plotted range (k up to 128), which the authors read as new reasoning patterns transferred from the teacher rather than reweighting of existing ones (§4.2, Figure 7).
- The authors suggest running RL from a distilled starting model, since distillation supplies a better prior (§5).

## Connections
- [[spurious-rewards-rlvr]] and [[echo-chamber-rl-post-training]] report related evidence that RLVR gains can come from reweighting base-model behavior.
- [[prorl]] argues that prolonged RL can raise pass@k above the base model, disputing the strongest reading of this paper.
- [[deepseek-r1]] is the zero-RL result this paper re-examines, and supplies the distilled checkpoints used in §4.2.
- [[grpo]] and [[rloo]] are two of the six algorithms compared under ∆SE in §4.3.
- [[rlvr-tulu3]] reports overoptimization at low KL from a different direction: average scores fall as KL grows.

## Verification
- Checked on 2026-09-18 against: https://arxiv.org/abs/2504.13837 (arXiv v5, 2025-11-24).
- Corrections to the previous card version:
  - Author list omitted the sixth author — "Yang Yue, Zhiqi Chen, Rui Lu, Andrew Zhao, Zhaokai Wang, Shiji Song, Gao Huang" → the paper lists eight authors, including a second Yang Yue between Zhaokai Wang and Shiji Song (title page).
  - "**Figure 2:** pass@k curves where base models overtake RL-trained models at large `k`" was the only figure cited for the main result; the distillation claim had no locus → distillation is §4.2 / Figure 7, with the specific models DeepSeek-R1-Distill-Qwen-7B, Qwen2.5-Math-7B (base), Qwen2.5-Math-7B-Oat-Zero, Qwen2.5-Math-7B-Instruct on Minerva.
  - Card gave no models, benchmarks, or numbers; all quantitative rows above were added from §3.1–§4.6.
  - Source type field was missing; added as `paper`.
- Removed as unsupported by the source:
  - "Figure 1: conceptual search-tree picture of RL narrowing the distribution toward rewarded paths" was stated without the accompanying finding; Figure 1 (left) is the search-tree illustration and Figure 1 (right) is the pass@1 vs pass@256 divergence — both are now stated with their content.
  - "Different RL algorithms perform similarly under this lens and remain far from optimal" without a number → replaced with the ∆SE range 42.6–43.9 (§4.3).
- Not reported by the source: pass@k for RLVR models trained with reward models rather than verifiers; results for DeepSeek-R1-Zero (self-hosting throughput made pass@k evaluation impractical, §4.6); the parameter count of Magistral-Medium.
