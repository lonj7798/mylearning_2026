<!-- scope: reasoning-trace synthesis — code-augmented MCTS plus a process preference model, iterated over four self-evolution rounds on 747k math problems
     deps: [[rstar]], [[math-shepherd]]
     see-also: [[omegaprm]], [[deepseek-r1]], [[openmathinstruct-2]]
-->

# rStar-Math: Small LLMs Can Master Math Reasoning with Self-Evolved Deep Thinking
- **Core Insight:** Four rounds of MCTS data generation and retraining, with no distillation from a stronger model after the bootstrap round, raise Qwen2.5-Math-7B on MATH from 58.8% to 90.0% and give 53.3% on AIME 2024 (8/15 problems) under deep thinking over 64 trajectories (Abstract; Table 1).
- **Guideline:** When step-level labels are unavailable and only final answers are known, generate each step as natural-language CoT plus executable Python, keep only steps whose code runs, and train the step scorer as a pairwise preference model over high- and low-Q-value siblings rather than regressing Q-values, because the pairwise model (PPM) scores 89.4 on MATH against 88.2 for the Q-value-regression PRM and 82.6 for an outcome reward model at equal data (§4.3 Table 8).
- **Authors:** Xinyu Guan, Li Lyna Zhang, Yifei Liu, Ning Shang, Youran Sun, Yi Zhu, et al. (Microsoft Research Asia)
- **Year:** 2025 (arXiv v1 2025-01)
- **URL:** https://arxiv.org/abs/2501.04519
- **Source type:** paper
- **Relevant topics:** reasoning, MCTS, process reward models, self-evolution, code-augmented CoT, small-model math

## Abstract
rStar-Math shows that small language models can match or exceed OpenAI o1 on math reasoning without distillation from a superior model. A math policy small model performs test-time MCTS search guided by a small process reward model. Three components make this trainable: a code-augmented chain-of-thought synthesis method in which extensive MCTS rollouts produce step-by-step verified trajectories used to train the policy; a process preference model (PPM) trained on step-level preference pairs instead of annotated step scores; and a self-evolution recipe in which policy and PPM are built from scratch and improved over four rounds. Across four rounds with millions of synthesized solutions for 747k math problems, MATH accuracy rises from 58.8% to 90.0% for Qwen2.5-Math-7B and from 41.4% to 86.4% for Phi3-mini-3.8B.

## Key Contributions
- Code-augmented CoT: each MCTS step is a natural-language comment plus Python; only steps whose concatenated code executes are kept as valid nodes (§3.2).
- Process preference model trained with a Bradley-Terry pairwise ranking loss over two highest-Q and two lowest-Q sibling steps, avoiding per-step score annotation (§3.3, Eq. 4).
- Four-round self-evolution in which policy and PPM alternate, with terminal-guided Q-annotation in rounds 1-2 and PPM-augmented annotation from round 3 (§3.2, §3.4.2).
- Coverage growth of the 747k problem set from 60.17% solved in round 1 to 90.25% in round 4 (§3.4.2, Table 2).
- Reported emergence of self-correction inside MCTS trajectories without any self-reflection training data (§5).

## Key Figures/Tables to Study
- **Figure 1** — the three parts: verified trajectory, preference-pair construction, four rounds.
- **Figure 2** — a worked code-augmented CoT example with per-step execution.
- **Table 2** — share of the 747k problems solved per round, split by GSM/MATH/Olympiad level.
- **Table 3** — pass@1 of the policy SLM alone in each round (System 1, no search).
- **Table 6** — System 2 accuracy per round, the round-by-round headline.
- **Table 7** — SFT data ablation: verified trajectories vs. GPT-4 distillation, random sampling, rejection sampling.
- **Table 8** — reward-model ablation: ORM vs. PQM vs. PPM.

## Technical Details
- **Problem set:** 747k math word problems with final-answer labels, primarily from NuminaMath (competition-level only) and MetaMath; augmented with GPT-4-synthesized problems seeded from the 7.5k MATH train set and the 3.6k AMC-AIME training split, keeping a synthesized problem only if at least 3 of 10 GPT-4 solutions agree (§3.4.1).
- **Search settings:** maximum tree depth 16; 16 MCTS rollouts per problem by default, with a second 16-rollout pass for problems where all trajectories are wrong; 8 candidate nodes per step; UCT exploration constant c = 2. Bootstrap round uses 8 rollouts and 5 candidates because the policy is 236B. Round 4 uses 16 candidate nodes and 2 tree expansions per problem with different seeds (App. A.1; §3.4.1).
- **Q-value annotation:** terminal-guided in rounds 1-2, q(s_i)^k = q(s_i)^{k−1} + q(s_d)^k with q(s_d) = 1 for a correct final answer and −1 otherwise; from round 3 the PPM supplies a non-zero initial q(s_i)^0, while terminal nodes are still scored by ground truth (§3.2, Eqs. 2-3).
- **SFT selection:** per problem, the top-2 correct trajectories by average Q-value (§3.4.1).
- **PPM data:** per step, the two highest-Q candidates as positive and the two lowest as negative; positives must reach a correct final answer and negatives an incorrect one; intermediate pairs share the same prefix, the final-answer step relaxes this (§3.3).
- **Coverage per round** of the 747k problems, all levels (§3.4.2, Table 2): round 1 (DeepSeek-Coder-V2-Instruct, 236B) 60.17%; round 2 (policy SLM-r1) 66.60%; round 3 (SLM-r2 + PPM-r2) 77.86%; round 4 (SLM-r3 + PPM-r3) 90.25%. Olympiad-level coverage moves 20.99% → 56.04% → 62.16% → 80.58%.
- **System 2 accuracy per round** for the 7B model (§4.3, Table 6), MATH / AIME 2024 / Olympiad Bench: base 58.8 / 0.0 / 21.8; round 1 75.2 / 10.0 / 35.7; round 2 86.6 / 43.3 / 59.4; round 3 87.0 / 46.7 / 61.6; round 4 89.4 / 50.0 / 65.3. GPT-4o on the same rows is 76.6 / 9.3 / 43.3.
- **Policy-only pass@1 per round** (§3.4.2, Table 3), MATH: base Qwen2.5-Math-7B 58.8; SLM-r1 69.6; SLM-r2 73.6; SLM-r3 75.8; SLM-r4 78.4. The bootstrap model DeepSeek-Coder-V2-Instruct scores 75.3.
- **Headline results with 64 trajectories** (Table 1): rStar-Math (Qwen-7B) MATH 90.0, AIME 2024 53.3, Olympiad Bench 65.6, College Math 60.5, Omni-Math 50.5. o1-preview is 85.5 / 44.6 / – / – / 52.5 and o1-mini 90.0 / 56.7 / 65.3 / 57.8 / 60.5.
- **Reward-model ablation** (§4.3, Table 8), same round-4 data: ORM with best-of-N 82.6 MATH / 26.7 AIME; PQM with MCTS 88.2 / 46.7; PPM with MCTS 89.4 / 50.0.
- **SFT-data ablation** (§4.3, Table 7), Qwen2.5-Math-7B fine-tuned on each dataset, MATH: MetaMath 55.2; NuminaMath-CoT 69.6; random sampling from policy SLM-r3 72.4; rejection sampling with the trained ORM over 32 trajectories 73.4; step-by-step verified 78.4.
- **Inference cost** (App. A.1, Table 9): average generated tokens per trajectory, MATH 5453, AIME 2024 15693, AMC 2023 14544, Olympiad Bench 7889, College Math 4503, GSM8K 3299, GaokaoEn 2023 6375.

## Recipe ledger

| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| rStar-Math policy SLM (Qwen2.5-Math-7B base) | 7B | distill-SFT (each round) | epochs; sequence length; global batch (sequences) | 2 epochs; 4096 tokens; batch 128 | arXiv:2501.04519 App. A.1 "Training Details" | verified (2026-09-18) | no ablation reported |
| rStar-Math policy SLM (Qwen models) | 1.5B / 7B | distill-SFT | optimizer; LR schedule; initial LR | AdamW; linear scheduler; 7e-6 | arXiv:2501.04519 App. A.1 | verified (2026-09-18) | no ablation reported |
| rStar-Math policy SLM (Phi3-mini-Instruct) | 3.8B | distill-SFT | LR schedule; initial LR | cosine scheduler; 5e-6 | arXiv:2501.04519 App. A.1 | verified (2026-09-18) | no ablation reported |
| rStar-Math PPM | 7B | reward-model | epochs; batch; initial LR; head | 1 epoch; batch 512; 7e-6; scalar head with tanh, outputs in [-1, 1] | arXiv:2501.04519 §3.4.1, App. A.1 | verified (2026-09-18) | §4.3 Table 8: PPM 89.4 MATH vs PQM 88.2, ORM 82.6, 1 run |
| rStar-Math data generation | 236B policy (bootstrap) | distill-SFT data | rollouts; hardware; wall clock | 8 MCTS rollouts; 10 nodes of 8×80GB H100; about two weeks | arXiv:2501.04519 App. A.1 "Self-evolution Inference Costs" | verified (2026-09-18) | no ablation reported |
| rStar-Math data generation | 7B policy (rounds 2-4) | distill-SFT data | rollouts; hardware; wall clock | 16 rollouts (64 in the final round); 15 nodes of 4×40GB A100; three days per round, one week for the final round | arXiv:2501.04519 App. A.1 | verified (2026-09-18) | no ablation reported |
| rStar-Math inference | 1.5B-7B | eval-gate | candidate nodes per step; trajectories | 32 candidate nodes per step; up to 64 trajectories | arXiv:2501.04519 App. A.1 "Inference Setting"; Table 1 | verified (2026-09-18) | §4.2 Fig. 3: accuracy saturates near 64 trajectories on MATH, AIME, Olympiad Bench |
| rStar-Math data filter | — | distill-SFT data | synthetic-problem noise filter | drop synthetic problems whose trajectories reach below 50% accuracy | arXiv:2501.04519 App. A.1 | verified (2026-09-18) | no ablation reported |

## Findings relevant to generality and distillation
- **Generality (author claim, §4.2):** the method sets new top scores on Olympiad Bench, College Math and Gaokao En 2023, benchmarks the training set was not tuned for, which the authors offer as evidence against over-optimization on MATH/GSM8K/AIME.
- **Distillation (§4.3, Table 7):** randomly sampled code-augmented CoT from the self-evolved policy already matches or beats GPT-4-distilled MetaMath and NuminaMath-CoT on MATH (72.4 vs 55.2 and 69.6), which the authors read as evidence that self-generated data can replace distillation after enough rounds. The first round still bootstraps from a 236B model's outputs (§3.4.2).
- **Reward model dominates at test time (§5):** across three policy sizes, final System 2 accuracy converges once the same 7B PPM is applied, and a 7B policy with the 7B PPM beats Qwen2.5-Math-72B-Instruct paired with a 72B ORM.
- **Self-reflection (§5):** the authors report MCTS trajectories in which the model abandons an incorrect SymPy formulation and recovers, with no self-reflection training data or prompt.
- **Stated limits (§5, §6):** the method covers word problems, not theorem proving or geometry requiring visual understanding; extension to code or commonsense reasoning would need a domain feedback mechanism such as test cases or mutual verification.

## Connections
- [[rstar]] — the predecessor; same MCTS backbone, but verification by a second small model rather than a trained PPM, and no training.
- [[math-shepherd]], [[prm800k]], [[lets-verify]] — process reward models trained with pointwise or human step labels, the comparison class for PPM.
- [[omegaprm]] — Monte-Carlo step annotation for PRM data without MCTS-UCB search.
- [[openmathinstruct]], [[mammoth]] — tool-integrated reasoning data with code in the trace.

## Verification
- Checked on 2026-09-18 against: https://arxiv.org/abs/2501.04519 (arXiv v1, 8 Jan 2025).
- Corrections to the previous card version:
  - "747K training trajectories" / "747K trajectory dataset" → 747k math *problems*, with millions of synthesized solutions generated over four rounds (Abstract; §3.4.1).
  - "Round-0 generator: Qwen2.5-Math-7B-Instruct (or DeepSeek-Math-7B)" → the bootstrap policy is DeepSeek-Coder-V2-Instruct (236B) (§3.4.2, Table 2).
  - "Q-value confidence: MCTS visit-count ≥ threshold" → selection is the top-2 correct trajectories by average Q-value (§3.4.1).
  - "step-preference pairs … with Q-gap > δ" → two highest-Q positives and two lowest-Q negatives per step, with a correctness constraint, no gap threshold (§3.3).
  - "MATH improves 58 → 78 → 85 → 88 → 90 across four rounds" → System 2 per round is 58.8 (base) → 75.2 → 86.6 → 87.0 → 89.4 (Table 6); the 90.0 figure is the 64-trajectory result in Table 1.
  - "58.5 Olympiad" → 65.6 on Olympiad Bench (Table 1).
  - "Beats o1-mini on MATH; matches o1-preview on several" → MATH ties o1-mini at 90.0 and exceeds o1-preview's 85.5 by 4.5; AIME 53.3 is below o1-mini's 56.7 (Abstract; Table 1).
  - "PPM ablation: replacing PPM with a scalar PRM loses 6 MATH points" → 89.4 vs 88.2 for PQM, a 1.2-point difference; the 6.8-point gap is against ORM with best-of-N (Table 8).
  - "on the order of 100K GPU-hours across four rounds" → the paper gives node counts and wall clock instead (App. A.1); the GPU-hour figure is not reported.
  - "~400–1200 tokens per trace", "4–10 reasoning steps" → average generated tokens per trajectory range 3299 (GSM8K) to 15693 (AIME 2024); maximum tree depth is 16 (Table 9; App. A.1).
- Removed as unsupported by the source: "71.0 USAMO problems attempted"; "authors argue pairwise training avoids the Goodhart-style issues of scalar reward regression observed in math-shepherd / prm800k" (the paper's stated reason is the imprecision of per-step scores, §3.3); "authors mitigate with temperature scheduling" (no temperature schedule is described); "a step is discarded if … its Q-value falls below round-specific threshold".
- Not reported by the source: total GPU-hours; the released dataset size in trajectories; per-round token counts.
