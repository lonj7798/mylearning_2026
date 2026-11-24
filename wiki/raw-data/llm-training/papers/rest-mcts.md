<!-- scope: ReST-MCTS* (Zhang et al., NeurIPS 2024): per-step process rewards inferred from PRM-guided tree search (MCTS*) given only final answers; mutual self-training of the policy on verified traces and of the value model (PRM variant) on inferred step values
     deps: [[math-shepherd]], [[rest-em]], [[lets-verify]]
     see-also: [[omegaprm]], [[prm800k]], [[star]], [[self-rewarding-lm]], [[rstar-math]], [[best-of-n]], [[glm-4]]
-->

# ReST-MCTS*: LLM Self-Training via Process Reward Guided Tree Search
- **Core Insight:** Given only the correct final answer, a search tree built by MCTS* supplies a per-step value target v_k that combines whether a step can still reach the correct answer and how many steps remain (m_k), so one search both selects correct traces for policy SFT and labels partial solutions for the next value model; over two iterations this beats ReST-EM and Self-Rewarding on all three backbones' averages (§3.1–3.2, Eqs. 1–2; §4.1, Table 2).
- **Guideline:** When self-training on problems with known answers, label each partial solution by search outcomes from the tree rather than by final-answer filtering alone, train the PRM as a regression on these values, and alternate PRM and policy updates; expect a first iteration that does not beat Self-Consistency at equal tokens and a gain after the second (§4.1, Fig. 2a).
- **Authors:** Dan Zhang, Sining Zhoubian, Ziniu Hu, Yisong Yue, Yuxiao Dong, Jie Tang (Tsinghua University KEG; California Institute of Technology)
- **Year:** 2024 (arXiv v1 2024-06-06; v3 2024-11-18; NeurIPS 2024)
- **URL:** https://arxiv.org/abs/2406.03816 (code: https://github.com/THUDM/ReST-MCTS)
- **Source type:** paper
- **Relevant topics:** process reward models, automatic step labels, MCTS for reasoning, self-training, value models, best-of-N vs tree search at equal token budget

## Abstract
Self-training methods usually keep generated responses with correct final answers, which admits traces with wrong or useless intermediate steps. ReST-MCTS* integrates process-reward guidance with a tree search (MCTS*) to collect higher-quality traces and per-step values for training both the policy and the reward model. It avoids per-step human annotation: given the oracle final answer, it infers process rewards by estimating the probability that a step leads to the correct answer. These values refine the process reward model and select traces for policy self-training. At equal search budget the tree-search policy is more accurate than Best-of-N and Tree-of-Thought, and iterating on its traces improves three LLMs over several iterations and outperforms ReST-EM and Self-Rewarding LM.

## Key Contributions
- Quality value v_k and weighted reward w_k for partial solutions, bounded in [0, 1] (§3.1, Theorem 1).
- MCTS* with UCB selection, value-model expansion, greedy MC rollout, weighted-average backup, and a self-critic termination signal (§3.1; App. C.1).
- Value-target inference from a pruned search tree using hard estimation and reasoning distance (§3.2).
- Mutual self-training of policy and value model (Algorithm 1).

## Key Figures/Tables to Study
- Fig. 1 (inference of process rewards and the self-training loop), Fig. 3 (worked example of value inference), Table 2 (self-training comparison), Table 3 (verifier comparison vs Math-Shepherd), Fig. 2 (accuracy vs completion tokens), Table 6 (running time).

## Technical Details
**Value definitions (§3.1).** Partial solution p_k = [s₁ … s_k]; reasoning distance m_k = minimum steps the policy needs from p_k to reach the correct answer (m_k = K − k on a K-step correct trace).
- `v_k = 0` if k = 0, else `max(v_{k−1} + w_{s_k}, 0)` (Eq. 1)
- `w_{s_k} = (1 − v_{k−1}) / (m_k + 1) · (1 − 2 r_{s_k})` (Eq. 2), where r_{s_k} ∈ [0, 1] is the Math-Shepherd-style step score. Theorem 1: w ≤ 1 − v_{k−1} and v_k ∈ [0, 1]; v_k → 1 only when r → 0 and m_k = 0.
- A single value model V_θ is trained to predict v_k directly, instead of separate models for r and m (§3.1).

**Value-target inference (§3.2).**
1. Prune unfinished branches of the search tree T_q; verify leaf answers against a* by string match or LLM judge.
2. For every node on at least one correct trace, compute m_k from the pruned tree; set r_{s_k} = 1 − r^HE_{s_k}, i.e. a step counts as correct if it can reach a correct answer in the pruned tree (hard estimation, Eq. 11).
3. Compute v and w with Eqs. 1–2 from the root; collect (Q, p, v) into D_V^i.
4. Correct full traces form D_G^i for policy SFT (Eq. 13).

**Initial value data (§4; App. B.1, E.1).** Science: 11,554 SciInstruct questions with correct step-by-step solutions; every prefix of a reference solution is a positive sample with v_k = k/K (Eq. 20); ChatGLM2 generates 3 next steps per prefix that are treated as incorrect, with v from Eqs. 21–22. Total 473.4k samples. Value models: ChatGLM3-6B and Mistral-7B with an added linear head, AdamW, MSE loss (Eq. 37), LR 1e-6. On a 14k-sample test set, accuracy within absolute tolerance 0.1 is 69.3% (App. E.1).

**MCTS* (App. C.1, E.2).** Selection by UCB(C) = v_C + ε·sqrt(ln n_parent / n_C) from the root each iteration (Eq. 33). If v of the selected node ≥ l = 0.9, return it. Else expand b samples; value each child with V_θ; self-critic may emit End-of-Inference. Greedy MC rollout of up to m steps from the best child; update v_C ← α v_C + (1 − α) v_max (Eq. 34). Defaults: m = 2, α = 0.5, T = 50 iterations, b = 3. ToT baseline: greedy DFS, max depth 10, threshold 0.9.

**Results.**
- Self-training, 2 iterations, zero-shot MATH / GPQA-Diamond / CEval-Hard / average (Table 2). LLaMA-3-8B-Instruct, iteration 2: ReST-EM 33.52 / 25.25 / 21.71 / 26.83; Self-Rewarding 33.89 / 26.26 / 23.03 / 27.73; ReST-MCTS* 34.28 / 27.78 / 25.00 / 29.02 (iteration 0 zero-shot average 24.78). SciGLM-6B, iteration 2 averages: 33.27 / 33.56 / 35.90 (iteration 0: 33.63). Mistral-7B: MetaMATH, iteration 2 averages: 24.16 / 25.22 / 26.50 (iteration 0: 22.33).
- Verifiers, Mistral-7B: MetaMATH, 256 outputs, GSM8K / MATH500 (Table 3): SC 83.9 / 35.1; ORM 86.2 / 36.4; SC+ORM 86.6 / 38.0; Math-Shepherd 87.1 / 37.3; SC+MS 86.3 / 38.3; SC+ReST-MCTS* value 87.5 / 39.0.
- Equal-token search on MATH: after iteration 2, best accuracy 48.5% vs 42.5% for Self-Consistency (§4.1, Fig. 2a). SciBench average (Table 4): GLM4 CoT 12.68, ToT 15.82, ReST-MCTS* 16.77; GPT-3.5-turbo 6.92 / 8.44 / 10.06; LLaMA2-13B-Chat 2.23 / 2.37 / 2.90.
- Running time per question (Table 6): CoT+SC 41 s, ORM+BoN 43 s, PRM+BoN 73 s, MCTS* 108 s; MATH accuracy 37.0 / 33.5 / 34.5 / 41.5.

## Recipe ledger
| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| Value model (ChatGLM3-6B, Mistral-7B) | 6B / 7B | reward-model | init data; loss; optimizer; LR | 473.4k samples from 11,554 SciInstruct questions (+ math data per App. B.1); MSE; AdamW; 1e-6 | arXiv:2406.03816v3 §4; App. E.1 | verified 2026-11-24 | App. E.1: 69.3% within ±0.1 |
| Value model | 6B / 7B | reward-model | batch size; epochs | not reported | checked §4, App. B, E | not reported | — |
| MCTS* | — | search | l; m; α; T; b | 0.9; 2; 0.5; 50; 3 | App. C.1, E.2 | verified 2026-11-24 | no ablation reported for these values |
| Policies (LLaMA-3-8B-Instruct, Mistral-7B: MetaMATH, SciGLM-6B) | 6–8B | SFT (self-training) | iterations; question set | 2; same D_G each iteration from MATH, GSM8K, TheoremQA train | §4.1 | verified 2026-11-24 | Table 2 |
| Policies | 6–8B | SFT (self-training) | LR, epochs, traces per question N | not reported | checked §4.1, App. E | not reported | — |

## Findings relevant to generality and negative feedback
- **Negative feedback.** Wrong steps enter the value model as low-v targets (Eqs. 21–22, Fig. 3 example with w = −0.25); the policy is trained only on verified-correct traces (Algorithm 1 line 9). Policy learning from negatives is not used.
- **Generality.** Self-training data is mostly math (§4.1); out-of-distribution averages over GPQA-Diamond and CEval-Hard rise with iterations for ReST-MCTS* on all three backbones (Table 2). Mistral-7B: MetaMATH zero-shot MATH falls from 29.34 at iteration 0 to 24.40 at iteration 2 under ReST-MCTS*, while its three-benchmark average rises 22.33 → 26.50 (Table 2). The authors list coding, agents, conversation and tasks without ground-truth answers as untested (App. H).
- **Small policies.** Gains for LLaMA2-13B-Chat on SciBench are small (2.23 → 2.90); the authors state that small policies have weak step-wise inference for complex tree search (§4.3).

## Connections
- [[math-shepherd]]: source of the step score r and hard estimation; used as the PRM baseline (Table 3).
- [[rest-em]], [[self-rewarding-lm]], [[star]]: outcome-filtered self-training baselines (Table 1, Table 2).
- [[omegaprm]]: concurrent automatic step labeling by binary-search MC; compare its MC(s) labels with v_k here.
- [[lets-verify]], [[prm800k]]: human step labels that this method replaces.
- [[rstar-math]]: later MCTS-based self-evolution with a process preference model.

## Verification
- Checked on 2026-11-24 against https://arxiv.org/abs/2406.03816 (v3, 18 Nov 2024). PDF read via text extraction; the Fig. 1 page extracts as overlapping text, so Eqs. 5–17 in the appendix overlay were not used; Tables 2–4 and 6 values were reassembled from column order and checked against reported averages.
- Source inconsistency: Table 2 lists LLaMA-3-8B-Instruct ReST-EM iteration-1 average as 17.22, but its three scores (30.84, 26.77, 21.05) average 26.22.
- Not reported by the source: policy SFT hyperparameters; value-model batch size and epochs; compute cost of building the search trees; the value of the UCB exploration constant ε. Note that the paper uses m both for reasoning distance (§3.1) and for the rollout step limit (App. E.2).
