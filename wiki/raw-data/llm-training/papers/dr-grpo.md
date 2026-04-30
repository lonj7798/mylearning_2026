<!-- scope: Dr. GRPO — unbiased GRPO that drops length and std normalization to fix length inflation
     deps: [[grpo]]
     see-also: [[ppo]], [[rloo]]
-->

# Understanding R1-Zero-Like Training: A Critical Perspective (Dr. GRPO)
- **Core Insight:** GRPO's 1/|o_i| length normalization creates a short-wins / long-loses-slowly gradient asymmetry, and std-normalization down-weights easy or hard questions; both can be removed without hurting accuracy, eliminating the response-length inflation seen in R1-Zero-style training.
- **Guideline:** For reasoning RL, swap in Dr. GRPO: drop the 1/|o_i| token average (use a fixed generation budget constant instead) and drop the std denominator in the advantage.
- **Authors:** Zichen Liu, Changyu Chen, Wenjun Li, Penghui Qi, Tianyu Pang, Chao Du, Wee Sun Lee, Min Lin
- **Year:** 2025
- **URL:** https://arxiv.org/abs/2503.20783
- **Relevant topics:** bias correction, RL for reasoning, length bias, unbiased policy gradient

## Abstract
R1-Zero-like RL training has sparked interest in verifiable-reward RL for LLMs. We identify an optimization bias in Group Relative Policy Optimization (GRPO) that artificially increases response length — especially for incorrect outputs — and degrades token efficiency. Dr. GRPO is an unbiased variant that drops two normalization terms from GRPO's loss and advantage, matching or exceeding reasoning accuracy while avoiding length blow-up.

## Key Contributions
- Diagnoses the "long wrong answers" pathology of GRPO as a direct consequence of 1/|o_i| token averaging combined with negative advantages.
- Shows that std-normalization of advantages introduces a difficulty bias (easy or hard questions with low reward variance get over-weighted).
- Proposes Dr. GRPO: drop both normalizers → unbiased REINFORCE-style update with a group-mean baseline.
- Demonstrates equal or better accuracy with ~30% shorter completions on MATH, AIME, and AMC.

## Key Figures/Tables to Study
- **Figure 1:** Response length curves during training — GRPO shoots up, Dr. GRPO flat.
- **Section 3:** Bias derivation — shortest/clearest explanation in the literature.
- **Table 2:** Dr. GRPO vs GRPO on Qwen2.5-Math-7B.

## Technical Details

### GRPO advantage (biased)
`Â_{i,t}^GRPO = (r_i − mean(r)) / std(r)`
applied to the per-token loss `(1/|o_i|) Σ_t (...)`.

### Dr. GRPO advantage (unbiased)
`Ã_{i,t} = r_i − mean({r_1, …, r_G})`
- No std division.
- No per-response length normalization: use a fixed generation budget L_max for token averaging, not the realized |o_i|.

### Dr. GRPO loss
`J_Dr.GRPO(θ) = E[ (1/G) Σ_i (1/L_max) Σ_t min(ρ_{i,t} Ã_{i,t}, clip(ρ_{i,t}, 1-ε, 1+ε) Ã_{i,t}) − β D_KL(π_θ || π_ref) ]`
where ρ_{i,t} and the k3 KL estimator are identical to GRPO.

### Why the fix removes the length bias
- In GRPO, a long incorrect o_i gets a *smaller per-token negative gradient* (divided by large |o_i|) than a short incorrect one. → easier to keep being wrong + verbose.
- Dropping 1/|o_i| restores equal per-token penalty → wrong-and-long is penalized proportionally to its length.

### Hyperparameters
Same as [[grpo]] except:
- No `/ std(r)` in advantage.
- Token-average denominator = fixed L_max (e.g., 4096) rather than |o_i|.

## Connections
- Parent: [[grpo]].
- Philosophical cousin (simpler): [[rloo]] — also uses an unbiased group baseline.
- Enabled by: [[deepseek-r1]] open-source recipe that made length inflation visible.
