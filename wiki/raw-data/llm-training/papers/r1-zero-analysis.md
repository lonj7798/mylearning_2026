<!-- scope: follow-up analyses of DeepSeek-R1-Zero emergent reasoning
     deps: [[deepseek-r1]], [[grpo]]
     see-also: [[entropy-mechanism-llm-rl]], [[let-verify]]
-->

# Analyses of R1-Zero — Where Does the "Aha Moment" Come From?
- **Core Insight:** The emergent long-chain reasoning in R1-Zero is not produced by the GRPO objective alone — reproduction work shows it requires (1) a base model already pretrained on reasoning-heavy data, (2) verifiable rewards with high-variance outcome signal, (3) long context, and (4) no PRM; remove any one and the "aha moment" disappears.
- **Guideline:** If you want R1-Zero behavior: start from a math-pretrained base, use GRPO + outcome-only rewards, keep context ≥ 32K, and do not use a PRM. Expect emergent backtracking/reflection around 4k training steps.
- **Authors:** multiple reproductions (Hu et al. "R1-Zero Analysis" + Open-Reasoner-Zero + TinyZero + Liu et al. Dr.GRPO)
- **Year:** 2025
- **URLs:**
  - Dr.GRPO: https://arxiv.org/abs/2503.20783 (Liu et al., "Understanding R1-Zero-Like Training: A Critical Perspective")
  - Open-Reasoner-Zero: https://github.com/Open-Reasoner-Zero/Open-Reasoner-Zero (Hu et al. 2025)
  - TinyZero: https://github.com/Jiayi-Pan/TinyZero (Pan et al. 2025)
- **Relevant topics:** emergent reasoning, GRPO bias, outcome rewards, verifiable RL, reproduction studies

## Abstract (synthesized)
Following DeepSeek-R1-Zero's demonstration that pure-RL training from a base model can elicit long chain-of-thought and self-reflection without any SFT warmup, a series of 2025 papers dissect the ingredients. Liu et al.'s "Dr.GRPO" paper isolates two biases in GRPO — response-length bias and per-prompt difficulty bias — that inflate reported gains and proposes a bias-corrected variant. Open-Reasoner-Zero and TinyZero reproduce R1-Zero at smaller scales (7B and 1.5B respectively) and identify the minimum-viable ingredient list: verifiable reward, long context, outcome-only advantage.

## Key Contributions
- **Dr.GRPO (Liu 2025):** identifies two biases in the standard GRPO advantage: (i) length bias from per-token mean aggregation that rewards longer correct responses and longer wrong responses asymmetrically; (ii) difficulty bias from per-prompt std normalization that inflates gradients on easy prompts. Proposes removing the std normalization and switching aggregation to batch-mean divided by `(B · max_completion_length)`.
- **Open-Reasoner-Zero (ORZ, 2025):** reproduces R1-Zero emergence on Qwen2.5-7B-Base with GRPO + rule-based verifier + 32K context. Confirms: (1) emergence happens; (2) it disappears if the base is not reasoning-pretrained; (3) asymmetric DAPO-style clipping ε_high=0.28 is stabilizing.
- **TinyZero (Pan 2025):** minimum-viable R1-Zero on Qwen2.5-Math-1.5B with a budget of ~$30; confirms the recipe scales down.
- **Shared finding:** no PRM needed; outcome-only verifier is sufficient and in fact dominates.

## Key Figures/Tables to Study
- **Dr.GRPO Figure 2:** length-bias curve — standard GRPO vs Dr.GRPO on AIME; Dr.GRPO's response-length curve is flat, GRPO's drifts up monotonically.
- **Dr.GRPO Table 1:** the bias-corrected loss aggregation vs standard GRPO, controlled across seeds.
- **ORZ Figure 3 (training dynamics):** entropy, response length, and accuracy time-series over 4K steps — the three converge to the emergence transition.
- **TinyZero plot of pass@1 vs step:** clean "aha moment" jump around step 1500 on 1.5B.

## Technical Details
- **Bias-corrected GRPO (Dr.GRPO) loss:**
  `L = −(1/(B · L_max)) · Σ_{i,t} mask_{i,t} · min(r_{i,t}·A_i, clip(r_{i,t})·A_i)` where `A_i = (R_i − μ_group)` with **no std normalization**.
- **ORZ recipe:**
  - Base: Qwen2.5-7B-Base.
  - Rollout: 8 samples/prompt, T=1.0.
  - Reward: 1 if verifier passes, else 0.
  - GRPO: ε_low=0.2, ε_high=0.28, β (KL)=0.
  - Context: 32K tokens; batch 128 prompts × 8 rollouts = 1024 seqs.
- **TinyZero recipe:** same structure at 1.5B scale, verifier-only, 3K training steps.
- **Reproduction caveats:** none of the papers reproduced R1-Zero from a *non-math-pretrained* base; the reasoning prior is necessary.

## Connections
- Corrects the GRPO loss of [[grpo]] and verl's [[verl-grpo]] — Dr.GRPO is now a supported `loss_type` in TRL.
- Reinforces the [[let-verify]] vs [[deepseek-r1]] tradeoff: outcome-only rewards beat PRM when the outcome is verifiable.
- Related to entropy-mechanism analyses ([[entropy-mechanism-llm-rl]]): emergence coincides with an entropy plateau, not entropy collapse.
- The "asymmetric clipping" finding ties into [[verl-ppo-loss]]'s `clip_ratio_high` support.
- Contradicts [[self-rewarding-lm]] saturation — verifiable RL scales further than judge-based RL because the reward source isn't drifting.
