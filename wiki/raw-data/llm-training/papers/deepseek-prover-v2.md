<!-- scope: DeepSeek-Prover-V2 (arXiv:2504.21801v2): Lean 4 prover; DeepSeek-V3 writes a natural-language sketch plus a Lean skeleton of `have ... := by sorry` subgoals, a 7B prover closes the subgoals, composed proofs plus V3's chain-of-thought form cold-start data, then expert iteration, SFT on DeepSeek-V3-Base, and GRPO with binary Lean rewards
     deps: [[grpo]], [[deepseek-v3]]
     see-also: [[deepseekmath-v2]], [[deepseek-r1]], [[rest-em]], [[star]], [[rejection-sampling-finetuning]], [[reward-hacking-taxonomy]], [[spurious-rewards-rlvr]]
-->

# DeepSeek-Prover-V2: Advancing Formal Mathematical Reasoning via Reinforcement Learning for Subgoal Decomposition
- **Core Insight:** A general model that cannot write complete Lean proofs can still write a correct proof skeleton: DeepSeek-V3 decomposes a theorem into `have` subgoals, a 7B prover solves each, and the composed proof appended to V3's chain-of-thought becomes cold-start data; after RL with 0/1 Lean rewards, the 671B model reaches 88.9% on miniF2F-test at Pass@8192 and solves 47 of 658 PutnamBench problems (§2.1–2.2, Table 1, Table 4).
- **Guideline:** When a verifier gives only sparse binary rewards, enlarge the training set with subgoal lemmas (with and without earlier subgoals as premises) to create a curriculum, and add a structural consistency reward early in RL so generated proofs keep the decomposed lemmas, which the authors report improves accuracy on multi-step theorems (§2.1, §2.2).
- **Authors:** Z.Z. Ren, Zhihong Shao, Junxiao Song (core contributors), Huajian Xin, Haocheng Wang, Wanjia Zhao, et al.; 18 authors (DeepSeek-AI)
- **Year:** 2025 (arXiv v1 2025-04-30; v2 2025-07-18)
- **URL:** https://arxiv.org/abs/2504.21801 (code and ProverBench: https://github.com/deepseek-ai/DeepSeek-Prover-V2)
- **Source type:** paper
- **Relevant topics:** formal theorem proving, Lean 4, subgoal decomposition, cold-start data, expert iteration, curriculum, GRPO, binary verifier rewards, reward hacking, distillation to 7B

## Abstract
DeepSeek-Prover-V2 is an open model for Lean 4 theorem proving. Initialization data come from a recursive proving pipeline: DeepSeek-V3 decomposes problems into subgoals, the subgoal proofs are composed, and the result is combined with V3's step-by-step reasoning to form a cold start for RL. The 671B model reaches 88.9% on miniF2F-test and solves 47 of 658 PutnamBench problems. The paper introduces ProverBench (325 problems, including 15 from AIME 24–25); the model proves 6 of the 15 AIME problems, while DeepSeek-V3 solves 8 in natural language with majority voting.

## Key Contributions
- Recursive subgoal pipeline: V3 sketch + Lean skeleton with `sorry` placeholders → 7B prover solves each subgoal → automatic composition (§2.1, Fig. 2–3).
- Subgoal-based curriculum inside expert iteration (§2.1, §2.3).
- Cold-start CoT data from problems the 7B prover fails end-to-end but whose subgoals are all solved; "hundreds" of examples (§2.2).
- Two modes in one model: non-CoT (fast, concise Lean) and CoT (reasoning before proof) (§2.3).
- ProverBench: 325 formalized problems (15 AIME 24&25, 310 textbook) (§3.5, Table 9).

## Key Figures/Tables to Study
- Fig. 2–3: decomposition, the two subgoal statement types, and composition.
- Table 1: miniF2F-test by mode, size, sample budget. Table 2: per-category miniF2F results.
- Table 3: output tokens, non-CoT vs CoT. Table 4: ProofNet-test and PutnamBench.
- §3.2 "Reward Hacking in Reinforcement Learning": the v2 correction.

## Technical Details
**Stage 1: non-CoT expert iteration (§2.3).** Each iteration samples the current best prover on still-unsolved problems; Lean-verified successes are added to SFT data. Procedure follows Prover-V1/V1.5 with two data changes: more problems from autoformalization and open datasets, and subgoal-derived problems targeting hard miniF2F-valid instances. Non-CoT is used here because inference and verification are faster.

**Subgoals (§2.1).** V3 is instructed to produce only a high-level sketch ending in a Lean theorem made of `have` statements each closed by `sorry`. Type (a) lemmas replace the original goal with the subgoal; type (b) also add earlier subgoals as premises. Type (b) is used for recursive solving; both types enter the curriculum. On miniF2F-valid, this V3 + 7B pipeline alone reaches 89.8% (§3.1).

**Cold start (§2.2).** Composed formal proof appended to V3's chain-of-thought. Contrast with Kimina-Prover, which starts from formal proofs and retro-writes the reasoning.

**SFT (§2.3).** Base DeepSeek-V3-Base-671B; constant LR 5e-6; context 16,384 tokens; data = non-CoT expert-iteration proofs + cold-start CoT data.

**RL (§2.2–2.3).** GRPO; reward 1 if Lean verifies the proof, else 0; consistency reward early in training penalizes proofs that drop the decomposed `have` lemmas. Prompts restricted to problems "sufficiently challenging yet solvable" by the SFT model. Per iteration: 256 problems × 32 proofs; max length 32,768 tokens.

**7B distillation (§2.3).** DeepSeek-Prover-V1.5-Base-7B context extended 4,096 → 32,768; fine-tuned on 671B RL rollouts plus non-CoT expert-iteration data; then the same RL stage.

**Results.** miniF2F-test (Table 1), 671B CoT: 61.9% (Pass@1), 82.4% (@32), 86.6% (@1024), 88.9% (@8192); 671B non-CoT: 59.5 / 73.8 / 76.7 / 78.3%; 7B CoT @8192: 82.0%. ProofNet-test 671B CoT @1024: 37.1%; PutnamBench: 47/658 (Table 4). Mean output tokens on miniF2F-test, non-CoT vs CoT: 7B 442.6 vs 4488.5; 671B 761.8 vs 6751.9 (Table 3). FormalMATH-All Pass@32: 28.31% (671B) (Table 6). CombiBench: 10/100 (Table 5). Evaluation uses Lean 4.9.0-rc2 (§3).

## Recipe ledger
| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| DeepSeek-Prover-V2-671B | 671B (DeepSeek-V3-Base) | SFT | LR; context | constant 5e-6; 16,384 tokens | arXiv:2504.21801v2 §2.3 | verified 2026-11-24 | no ablation reported |
| DeepSeek-Prover-V2-671B | 671B | SFT | data sources | non-CoT expert-iteration proofs + cold-start CoT ("hundreds" of composed examples) | §2.2, §2.3 | verified 2026-11-24 | dataset size not reported |
| DeepSeek-Prover-V2-671B | 671B | RL | algorithm; reward | GRPO; binary Lean pass/fail + early consistency reward | §2.2, §2.3 | verified 2026-11-24 | §2.2: consistency reward improves accuracy on complex theorems (no numbers) |
| DeepSeek-Prover-V2-671B | 671B | RL | problems × samples per iteration; max length | 256 × 32; 32,768 tokens | §2.3 | verified 2026-11-24 | no ablation reported |
| DeepSeek-Prover-V2-671B | 671B | RL | RL LR, KL, clip, iterations | not reported (checked §2–3, App. A–D) | — | not reported | — |
| DeepSeek-Prover-V2-7B | 7B | distill-SFT | base; context extension | Prover-V1.5-Base-7B; 4,096 → 32,768 | §2.3 | verified 2026-11-24 | — |
| Prover-V2 eval | both | eval-gate | Lean version; miniF2F split use | 4.9.0-rc2; valid used in curriculum, test held out | §3, §3.1 | verified 2026-11-24 | — |

## Findings relevant to generality and negative feedback
- **Reward hacking.** The v1 report claimed the 7B model solved 13 PutnamBench problems the 671B model did not. In v2 the authors trace this to a Lean 4.9.0 bug in which the `apply?` tactic fails to emit `sorry` in some cases; the 7B outputs exploited it with `Cardinal.toNat` and `Cardinal.natCast_inj` (§3.2, App. B). The PutnamBench count was revised from 49 to 47 after two misformulated statements were removed (§3.2). A formal verifier is still a reward that can be gamed through tool bugs.
- **Generality.** Training data are mostly high-school level, yet ProofNet-test (undergraduate) improves with CoT; CombiBench (combinatorics) is reported as "promising" generalization with 10/100 (§3.2, §3.3). The 671B CoT model sometimes detects misformulated statements and closes them by deriving False (§3.3, App. C).
- **Negative signal.** RL prompts exclude problems the SFT model never solves, because binary rewards give no signal there (§2.3); the subgoal curriculum is motivated by the same sparsity (§2.1).

## Connections
- [[deepseek-v3]]: the decomposer and the SFT base model.
- [[grpo]], [[deepseek-r1]]: RL objective and the "standard training pipeline for reasoning models" the paper follows (§2.2–2.3).
- [[rest-em]], [[star]], [[rejection-sampling-finetuning]]: expert iteration is the same keep-verified-samples loop.
- [[deepseekmath-v2]]: informal-proof counterpart that uses a learned verifier instead of Lean.
- [[reward-hacking-taxonomy]], [[spurious-rewards-rlvr]]: other verifier-reward failure cases.

## Verification
- Checked on 2026-11-24 against: arXiv:2504.21801v2 PDF (body, tables, appendix headings). v1 (2025-04-30) was not re-read; the card follows v2, whose §3.2 states what changed.
- Not reported by the source: cold-start dataset size beyond "hundreds"; expert-iteration round count; total SFT tokens; RL LR, KL, clip, step count; compute.
