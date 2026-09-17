<!-- scope: Yue et al. (2025): pass@k at large k as a probe of the reasoning boundary of RLVR-trained models; coverage falls as pass@1 rises; algorithm comparison and the sampling-efficiency gap
     deps: [[grpo]], [[dr-grpo]]
     see-also: [[negative-sample-reinforcement]], [[spurious-rewards-rlvr]], [[rls-razor]]
-->

# Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model?
- **Core Insight:** For Qwen2.5-7B trained with GRPO on Omni-MATH, pass@1 rises from 9.9 (base) to 26.1, 33.6 and 42.5 at steps 150, 300 and 450 while pass@256 falls from 67.2 to 66.3, 65.3 and 64.3, and the out-of-domain MATH500 pass@256 falls from 96.2 to 95.4 (Table 4); the authors read this as RLVR raising the probability of solution paths the base model could already sample while narrowing coverage (Abstract, §4.1).
- **Guideline:** When judging whether an RLVR stage added capability or sharpened existing behaviour, report pass@k up to a large k (the paper uses k = 256, and notes k = 128 or 1024 is within practical cost) alongside pass@1, because six RLVR algorithms improved pass@1 and none improved pass@256 over the base model in the paper's comparison (§C.5, Table 3).
- **Authors:** Yang Yue, Zhiqi Chen, Rui Lu, Andrew Zhao, Zhaokai Wang, Yang Yue, Shiji Song, Gao Huang (LeapLab, Tsinghua University; Shanghai Jiao Tong University)
- **Year:** 2025 (arXiv v1 2025-04; v5 2025-11-24)
- **URL:** https://arxiv.org/abs/2504.13837
- **Source type:** paper
- **Relevant topics:** RLVR evaluation, pass@k, coverage, reasoning boundary, sampling efficiency, distillation versus RL

## Abstract
The paper probes the reasoning-capability boundary of RLVR-trained LLMs across model families, RL algorithms, and math, coding and visual-reasoning benchmarks, using pass@k at large k. RLVR improves sampling efficiency toward correct paths, but the authors find that current training rarely elicits fundamentally new reasoning patterns: RLVR models beat base models at small k, base models reach higher pass@k at large k, and the boundary often narrows as training progresses. Coverage and perplexity analysis indicates the RLVR model's reasoning paths are already inside the base model's sampling distribution. Treating the base model as an upper bound, six RLVR algorithms perform similarly and remain far from optimal. Distillation, by contrast, introduces new reasoning patterns from the teacher.

## Key Contributions
- Large-k pass@k as the evaluation of reasoning boundary rather than average-case accuracy (§3, §4.1).
- The crossing pattern: RL above base at k = 1, base above RL at large k, across Qwen2.5 7B/14B/32B, LLaMA-3.1-8B, and released zero-RL models (§4.1, Figure 2).
- The sampling-efficiency gap Δ_SE, defined as the RL model's pass@1 minus the base model's pass@k with k = 256 as a proxy for the boundary (§4.4, Figure 8).
- Comparison of six RLVR algorithms at matched evaluation (Table 3) and of training steps (Table 4).
- Distillation as the contrasting case that does expand the boundary (§5).

## Key Figures/Tables to Study
- **Figure 2:** pass@k curves of base and zero-RL models on GSM8K, MATH500, Minerva, Olympiad, AIME24, AMC23.
- **Figure 1 (right) / Table 4:** pass@1 and pass@256 at GRPO steps 150, 300, 450.
- **Figure 8 / Table 3:** pass@1 and pass@256 for GRPO, PPO, ReMax, RLOO, Reinforce++, DAPO.
- **Figure 16:** ablation on KL loss and rollout number n.

## Technical Details
- **Models and data (§4.1).** Qwen2.5 7B/14B/32B base variants and LLaMA-3.1-8B; zero-RL models from SimpleRLZoo, trained with GRPO on GSM8K and the MATH training set with a correctness-only reward and no format reward; also Oat-Zero-7B and DAPO-32B.
- **Table 3 (Qwen2.5-7B, pass@1 / pass@256).** Omni-MATH-Train: base 9.9 / 67.2; GRPO 26.1 / 66.3; PPO 27.2 / 65.8; ReMax 24.4 / 65.5; RLOO 28.6 / 66.4; Reinforce++ 28.2 / 67.7; DAPO 31.4 / 66.1. MATH500: base 34.5 / 96.2; GRPO 74.4 / 97.2; PPO 75.2 / 97.2; RLOO 75.0 / 97.4; Reinforce++ 75.4 / 96.8; DAPO 75.6 / 96.4.
- **Table 4 (GRPO training steps, pass@1 / pass@256).** Omni-MATH-Train: base 9.9 / 67.2; step150 26.1 / 66.3; step300 33.6 / 65.3; step450 42.5 / 64.3. Omni-MATH-Test: base 10.2 / 69.1; step150 25.1 / 68.3; step300 27.1 / 66.6; step450 28.3 / 63.9. MATH500: base 34.5 / 96.2; step150 74.4 / 97.2; step300 75.4 / 96.0; step450 76.3 / 95.4.
- **Magnitude of the crossing (§4.1).** On Minerva with a 32B model, the base model exceeds the RL model by about 9% at k = 128. For Oat-Zero and DAPO models the RL model starts nearly 30% above the base model at small k and is eventually overtaken (Figure 11).
- **Algorithm notes (§C.5).** DAPO has the highest pass@1 on all three datasets but needs roughly 3–6× more samples per batch because of dynamic sampling, and drops at k = 256. RLOO and Reinforce++ hold up across the full k range. ReMax is lowest at both ends.
- **KL and rollout ablation (Figure 16).** Increasing rollouts per prompt from n = 8 to n = 32 at unchanged prompt batch size reached a higher pass@128 despite lower pass@1 after only 220 steps; a KL loss of 0.001 is plotted as a third setting.
- **Practical cost (§3).** The authors state that evaluating with k = 128 or 1024 is within practical resource limits.

## Findings relevant to generality
- **What narrows.** The measured narrowing is coverage: the set of problems solvable within k attempts shrinks as pass@1 grows (§4.1, Table 4). This is a within-domain measurement; the out-of-domain MATH500 pass@256 also declines slightly across steps (Table 4).
- **Attribution.** Coverage and perplexity analysis is used to argue that RLVR reasoning paths already lie in the base model's distribution (Abstract, §4.3, Interpretation).
- **Contrast.** Distillation from a stronger teacher is reported to introduce new reasoning patterns and expand the boundary (§5), which makes distillation and RLVR different tools for generality.
- **Limits.** The evidence is from math, code and visual reasoning with binary verifiers, mostly Qwen2.5 base models and zero-RL training; instruction following, safety and long-horizon agentic tasks are not evaluated.

## Connections
- [[negative-sample-reinforcement]] — decomposes which part of the RLVR signal costs coverage.
- [[spurious-rewards-rlvr]] — the model-family dependence of zero-RL gains on the same Qwen2.5 base models.
- [[dr-grpo]], [[grpo]] — the objectives compared here.
- [[rls-razor]] — the other narrowing axis: forgetting of prior capability measured by KL.

## Verification
- Checked on 2026-09-15 against the cached primary text of https://arxiv.org/abs/2504.13837 (arXiv v5, 2025-11-24): Abstract, §3, §4.1, §4.4, §5, App. C.5 Tables 3–4, Figure 16 caption.
- Corrections to the previous card version (wiki/raw-data/llm-training/papers/rlvr-beyond-base-model.md, which carries no Verification section): that card states the claims qualitatively and gives no numbers or loci; the numbers above are added from the primary text. The author list there omits the second "Yang Yue" and the year is given as 2025 without a version.
- Not reported by the source: pass@k after RLVR on non-reasoning capabilities (instruction following, safety), and pass@k for models trained with a KL coefficient in the main comparison.
