<!-- scope: ProRL (arXiv:2505.24864) — prolonged GRPO/DAPO-style RL with KL penalty and reference-policy resets on a 136K-problem, five-domain verifiable set; pass@k boundary analysis of Nemotron-Research-Reasoning-Qwen-1.5B
     deps: [[grpo]], [[kl-control-rlhf]]
     see-also: [[rlvr-beyond-base-model]], [[echo-chamber-rl-post-training]], [[entropy-mechanism-llm-rl]], [[spurious-rewards-rlvr]], [[deepseek-r1]]
-->

# ProRL: Prolonged Reinforcement Learning Expands Reasoning Boundaries in Large Language Models
- **Core Insight:** RL for more than 2k steps from DeepSeek-R1-Distill-Qwen-1.5B raised average pass@1 from 44.45 to 60.14 on six math benchmarks and Reasoning Gym reward from 4.24 to 59.06 (Tables 1, 3), and on some tasks the base model solved nothing at any k up to 256 while the RL model did (§4.3, Fig. 5); on tasks where the base model already had high pass@128, mostly math, pass@128 stayed flat or fell (§4.2).
- **Guideline:** When a long RL run from a distilled reasoning checkpoint stagnates on validation, the authors hard-reset the reference policy and optimizer and continued training with a KL penalty, and report continued pass@1 and pass@16 gains (§3.3, Fig. 1 left); no ablation isolates the reset, the KL term, or task diversity, so treat the recipe as a single study.
- **Authors:** Mingjie Liu, Shizhe Diao, Ximing Lu, Jian Hu, Xin Dong, Yejin Choi, et al. (NVIDIA)
- **Year:** 2025 (arXiv v1 2025-05; only version; preprint)
- **URL:** https://arxiv.org/abs/2505.24864
- **Source type:** paper
- **Relevant topics:** prolonged RL, GRPO, DAPO clip-higher, dynamic sampling, KL penalty, reference-policy reset, entropy collapse, pass@k reasoning boundary, multi-domain RLVR, OOD generalization

## Abstract
The paper addresses whether RL with verifiable rewards adds reasoning ability or only raises the probability of outputs the base model can already sample, and whether more RL compute keeps improving results. It introduces ProRL, an RL recipe with a KL divergence penalty, periodic reference-policy resets, and a diverse task suite. The resulting 1.5B model outperforms its starting checkpoint across a range of pass@k values, including tasks where the starting checkpoint fails at every sample budget. The authors report that the size of the boundary gain is correlated with the starting checkpoint's competence on a task and with training duration. Weights are released as nvidia/Nemotron-Research-Reasoning-Qwen-1.5B.

## Key Contributions
- A training recipe for long RL runs: GRPO with DAPO's decoupled clipping and dynamic sampling, a KL penalty to a reference policy, and hard resets of the reference policy and optimizer state (§2.2–2.3.1).
- Nemotron-Research-Reasoning-Qwen-1.5B, trained on 136K verifiable problems in math, code, STEM, logic puzzles, and instruction following (§3.1, Table 4).
- A pass@k analysis at up to 256 samples comparing the starting checkpoint, an intermediate checkpoint, and the final model, grouped into Diminish, Plateau, and Sustained regimes (§4.2, Fig. 4).
- Evidence that boundary gain is negatively correlated with the starting checkpoint's pass@128 and that low-gain tasks have lower Creativity Index against DOLMA (§4.1, Fig. 3).
- OOD tests: Reasoning Gym tasks not seen in training, and graph_color at graph sizes larger than the size-10 training graphs (§4.3, Figs. 5–6).

## Key Figures/Tables to Study
- Fig. 1 left and Fig. 2: pass@1 and pass@16 over training steps; entropy and response length across 8 runs with reset points.
- Fig. 3: base pass@128 against post-RL gain; Creativity Index by task category.
- Fig. 4: pass@k curves for the three regimes. Fig. 5: boxnet (OOD). Fig. 6: graph_color by graph size.
- Tables 1–3: pass@1 by domain against DeepScaleR-1.5B, DeepCoder-1.5B, and DeepSeek-R1-Distill-Qwen-7B.

## Technical Details
- **Objective.** GRPO clipped surrogate with group-normalized advantage A(τ) = (R_τ − mean(R_G)) / std(R_G) (Eqs. 1–2), decoupled clip(r, 1 − ε_low, 1 + ε_high) (Eq. 3), and L_KL-RL = L_GRPO − β·D_KL(π_θ‖π_ref) (Eq. 4). β: KL coefficient; π_ref: reference policy; R_G: rewards of the responses sampled for the same prompt.
- **Entropy collapse.** Higher rollout temperature delayed but did not prevent entropy decline; clip-higher and dynamic sampling slowed it; the authors state that the KL penalty gave "a stronger and more stable solution" (§2.2.1, §2.3.1). No numeric comparison is given.
- **Why keep KL.** Recent works removed the KL penalty; the authors observe that this view "often applies to base models prior to any supervised fine-tuning" and argue that from a checkpoint already producing coherent CoT, KL helps stability and entropy (§2.3.1). This is an argument, not an ablation.
- **Reset.** As training proceeds the KL term "may increasingly dominate the loss"; π_ref is hard-reset to a recent online snapshot and optimizer states are reinitialized, triggered when validation stagnates or degrades (§2.3.1, §3.3, App. E).
- **Rewards.** Math binary via DeepScaleR verifier or math-verify; code continuous as the fraction of test cases passed, 0 on compile or syntax error or >5 s total timeout; STEM binary; logic puzzles continuous via Reasoning Gym verifiers; instruction following continuous (Table 4, App. D.1–D.5).
- **Run history.** Run 1: four domains, 8k response limit (the starting checkpoint supports 128k). Run 2: reset, same setup. Run 3: instruction-following data added. Runs 4–5: penalty for responses that do not terminate correctly. Runs 6–7: 32 rollouts, two resets. Run 8: 16k context, 16 rollouts (App. E, Fig. 2).
- **Evaluation.** vLLM, temperature 0.6, top_p 0.95, max response 32k; pass@1 from 16 samples for math, code, STEM; average continuous reward for logic puzzles and IFEval (§3.4). pass@k analysis uses 256 samples and 18 of 96 Reasoning Gym tasks chosen at random (§4).

## Recipe ledger
All loci refer to arXiv:2505.24864v1.

| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| Nemotron-Research-Reasoning-Qwen-1.5B | 1.5B | RL | initial checkpoint | DeepSeek-R1-Distill-Qwen-1.5B | §2.3.1 | verified 2026-09-14 | no ablation reported |
| Nemotron-Research-Reasoning-Qwen-1.5B | 1.5B | RL | framework; algorithm | verl; GRPO + DAPO decoupled clip + dynamic sampling | §3.2 | verified 2026-09-14 | no ablation reported |
| Nemotron-Research-Reasoning-Qwen-1.5B | 1.5B | RL | clip ε_low / ε_high | 0.2 / 0.4 | §3.2 | verified 2026-09-14 | §2.3: "helps retain entropy"; no numbers |
| Nemotron-Research-Reasoning-Qwen-1.5B | 1.5B | RL | dynamic sampling | drop prompts with accuracy 1 or 0 | §3.2 | verified 2026-09-14 | no ablation reported |
| Nemotron-Research-Reasoning-Qwen-1.5B | 1.5B | RL | KL coefficient β; where applied | β not reported; KL term in the loss (Eq. 4) | §2.3.1 | not reported (body, App. D–F, HF model card checked) | no ablation reported |
| Nemotron-Research-Reasoning-Qwen-1.5B | 1.5B | RL | reference reset rule | hard reset of π_ref and optimizer when validation stagnates or degrades; 8 runs | §3.3, App. E, Fig. 2 | verified 2026-09-14 | no ablation reported |
| Nemotron-Research-Reasoning-Qwen-1.5B | 1.5B | RL | samples per prompt | 16 (Runs 1–5, 8); 32 (Runs 6–7) | §3.2, App. E | verified 2026-09-14 | no ablation reported |
| Nemotron-Research-Reasoning-Qwen-1.5B | 1.5B | RL | rollout temperature | 1.2 | §3.2 | verified 2026-09-14 | §2.2.1: delays entropy collapse; no numbers |
| Nemotron-Research-Reasoning-Qwen-1.5B | 1.5B | RL | context / max response | "context window limit of 8096" (§3.2); 8k cap most of training; 16k for final ~200 steps | §3.2, §3.3, App. E | verified 2026-09-14 | App. E Run 8: marginal AIME gain, larger gains elsewhere; no table |
| Nemotron-Research-Reasoning-Qwen-1.5B | 1.5B | RL | batch; mini-batch | 256; 64 (4 gradient updates per rollout step); unit not stated | §3.2 | verified 2026-09-14 | no ablation reported |
| Nemotron-Research-Reasoning-Qwen-1.5B | 1.5B | RL | optimizer; LR | AdamW; constant 2 × 10⁻⁶ | §3.2 | verified 2026-09-14 | no ablation reported |
| Nemotron-Research-Reasoning-Qwen-1.5B | 1.5B | RL | training steps | "more than 2k" | §1, Fig. 1 left | verified 2026-09-14 | Fig. 1 left: pass@1, pass@16 still rising |
| Nemotron-Research-Reasoning-Qwen-1.5B | 1.5B | RL | data (prompts) | 136K: math 40k, code 24k, STEM 25k, logic 37k, instruction following 10k | §3.1, Table 4 | verified 2026-09-14 | no ablation of the mix reported |
| Nemotron-Research-Reasoning-Qwen-1.5B | 1.5B | RL | compute | 4 nodes × 8 H100-80GB; about 16k GPU hours | §3.2 | verified 2026-09-14 | n/a |
| Nemotron-Research-Reasoning-Qwen-1.5B | 1.5B | eval-gate | checkpoint monitoring | validation blend: AIME2024, Codeforces, GPQA-diamond, IFEval, graph_color | App. E | verified 2026-09-14 | final checkpoint selection rule not reported |

## Findings relevant to generality, negative feedback, long context
- **Breadth, in-domain (Result, single study).** Gains over the starting checkpoint (pass@1 for math, code, GPQA; average verifier reward for IFEval and Reasoning Gym, §3.4): math 44.45 → 60.14, code 23.08 → 37.49, GPQA Diamond 15.86 → 41.78, IFEval 44.05 → 66.02, Reasoning Gym 4.24 → 59.06 (Tables 1–3). The generalist model exceeds DeepCoder-1.5B on the code average (37.49 vs 30.96, Table 2) and DeepScaleR-1.5B on the math average (60.14 vs 54.54, Table 1).
- **OOD (Result, single study).** Reasoning Gym tasks unseen in training: acre 5.99 → 58.57, boxnet 0.00 → 7.91, game_of_life_halting 3.49 → 52.29 (Table 3). On boxnet the starting checkpoint "exhibits no capability"; the final model exceeds the intermediate checkpoint at all k (§4.3, Fig. 5). On graph_color, trained on size-10 graphs, the final model is higher than both comparison models at every tested size (Fig. 6).
- **Narrowing (Result, single study).** In the Diminish regime, "particularly in the math domain", pass@1 improves while pass@128 "often declines"; these tasks have high starting pass@128 (§4.2, Fig. 4). Plateau tasks gain early with negligible further gain from prolonged training (§4.2).
- **Predictor of gain (Result, single study).** Base pass@128 is negatively correlated with post-RL boundary gain; low-gain math and code tasks have lower Creativity Index, which the authors interpret as more pretraining exposure (Interpretation, §4.1, Fig. 3).
- **Negative signals.** Dynamic sampling discards prompts where all samples fail or all succeed (§2.3). Runs 4–5 add a penalty for responses that fail to emit a terminating token, after repetition raised response length (App. E); the penalty value is not reported.
- **Long context.** After training mostly at 8k, extension to 16k for about 200 steps gave "measurable improvements"; the model "adapts quickly" (§3.3, App. E Run 8).
- **Stated limits.** Only 1.5B was tested; resets add complexity "and may lead to inconsistent results"; no guarantee for untrained domains (App. A).

## Connections
- [[rlvr-beyond-base-model]] — ref. [13]; the pass@k claim ProRL disputes, and whose math-domain pattern ProRL's Diminish regime reproduces (§4.2).
- [[echo-chamber-rl-post-training]] — ref. [15]; the prior-amplification view discussed in §5.
- [[grpo]] — source of the GRPO objective (ref. [16]) used in Eqs. 1–2.
- [[kl-control-rlhf]] — background on the KL-regularized objective used in Eq. 4.
- [[entropy-mechanism-llm-rl]] — separate study of entropy collapse, the failure ProRL's KL and clip settings target.
- [[deepseek-r1]] — produced DeepSeek-R1-Distill-Qwen-1.5B, ProRL's starting checkpoint.
- [[spurious-rewards-rlvr]] — not cited by ProRL; related because §2.3.1 motivates the KL penalty partly as protection against "overfitting to spurious reward signals".

## Verification
- Checked on 2026-09-14 against: https://arxiv.org/abs/2505.24864 (arXiv v1, 2025-05-30; only version), body and App. A–F; HF model card nvidia/Nemotron-Research-Reasoning-Qwen-1.5B.
- Corrections: "improvements persist across pass@k evaluations" without qualification → the paper also reports a Diminish regime with falling pass@128, mainly in math (§4.2); "Ablations on training duration / resets / KL control" → no such ablations; the paper compares starting, intermediate, and final checkpoints and describes 8 runs (§4, App. E); "Task-diversity analysis" → only the dataset table (Table 4) and a comparison with domain-specialized models (§3.4); reset rationale "stale anchor policy" → KL term dominating the loss, with optimizer reinitialization (§2.3.1); old recipe omitted DAPO clip-higher and dynamic sampling (§2.3) and the fact that the "base model" is the SFT-distilled DeepSeek-R1-Distill-Qwen-1.5B, not a pretrained-only model.
- Removed as unsupported: "optimize for sustained exploration and dynamic stabilization"; "diversify the task suite before writing off RL" as a tested recommendation; "[[deepseek-r1]] is the obvious empirical reference point for long-horizon RL optimism"; "Many negative conclusions about RL may be conclusions about short-horizon RL" as a finding (§1 states it as the authors' hypothesis about prior work: over-reliance on math, where models are often overtrained, and runs of "no more than hundreds of steps").
- Internal inconsistencies in the source: §1 gives gains of 14.7 / 13.9 / 54.8 / 25.1 / 18.1 (math / code / logic / STEM / IF) while §3 gives 15.7 / 14.4 / 54.8 / 25.9 / 22.0; the Tables 1–3 differences match §3. §3.4 states +4.6 over DeepScaleR on math, but Table 1 averages differ by 5.60 (derived: 60.14 − 54.54). "%" in these statements denotes absolute points.
- Not reported by the source: β, number of steps per run, intermediate checkpoint's step, rule for releasing the final checkpoint, termination-penalty value, ablations of KL, resets, temperature, or task mix. The HF model card also lists later checkpoints (v2, 3,000 steps, 2025-07; BroRL, 2025-11) not described in this paper; the paper's model is revision v1 (2,000 steps per the card).
