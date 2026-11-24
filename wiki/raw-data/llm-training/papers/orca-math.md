<!-- scope: Orca-Math (Microsoft Research, 2024) — 200K multi-agent synthetic grade-school math problems with GPT-4-Turbo solutions; SFT then iterative KTO/DPO on student samples labeled by teacher agreement
     deps: [[orca]], [[orca-2]], [[kto]], [[dpo]]
     see-also: [[agentinstruct]], [[metamath]], [[wizardmath]], [[evol-instruct]], [[rejection-sampling-finetuning]], [[self-rewarding-lm]], [[spin]]
-->

# Orca-Math: Unlocking the potential of SLMs in Grade School Math
- **Core Insight:** After one epoch of SFT on 200K synthetic problems (79.91% GSM8K), building preference pairs from four student samples per problem — positive if GPT-4 judges the student's final answer equal to the teacher's, negative otherwise — and training with KTO for two iterations raises Mistral-7B to 86.87% pass@1 with greedy decoding, while extra SFT epochs on the same data give 81.50% and then 80.44% (§3.2, §5, Table 2).
- **Guideline:** When a strong teacher solution exists for every problem, use the student's own sampled solutions as both positives and negatives instead of teacher-only positives; removing student-generated positives cost 2.27 points with both DPO and KTO (§5.1, Table 3).
- **Authors:** Arindam Mitra, Hamed Khanpour, Corby Rosset, Ahmed Awadallah (Microsoft Research)
- **Year:** 2024 (arXiv v1 2024-02-16; only version)
- **URL:** https://arxiv.org/abs/2402.14830
- **Source type:** paper
- **Relevant topics:** synthetic math data, multi-agent data generation, teacher-feedback preference learning, KTO vs DPO, iterative self-improvement, small-model reasoning

## Abstract
The paper trains Orca-Math, a 7B model based on Mistral-7B, to 86.81% on GSM8K pass@1 without ensembling, verifiers, code execution or tools. Two elements are named: a 200K-problem synthetic dataset created with collaborating GPT-4-Turbo agents, and an iterative procedure in which the student practices on the problems, receives teacher feedback on its solutions, and learns from preference pairs that combine its own solutions with that feedback. SFT alone gives 81.50%; iterative preference learning gives 86.81%. The paper contrasts this with Phi-GSM, which uses top-48 verifier selection to go from 68.2 to 81.5.

## Key Contributions
- Agent-Instruct data generation: "Ask Me Anything" problem expansion, Suggester–Editor difficulty escalation, plus a filtered DMath subset (§2).
- Teacher-feedback preference construction: student samples labeled by agreement with the teacher's answer, with synthetic negatives borrowed from other questions when all four samples are correct (§3.2).
- Comparison of SFT, DPO and KTO in iterations 2 and 3; KTO is best in both (§5, Table 2).
- Two ablations: student-generated positives (Table 3) and synthetic negatives (Table 4).
- TF-IDF + unigram-Jaccard contamination check against the GSM8K test set (§5.3).

## Key Figures/Tables to Study
- Table 2: the full training-procedure comparison; the main result of the paper.
- Tables 3 and 4: what happens when student positives or all-positive questions are removed.
- §4 prompt: the GPT-4 grader used both for labeling preference data and for evaluation.

## Technical Details
**Seed set (§2).** 36,217 problems from the train and validation splits of 11 datasets collected through Lila (NumGLUE, AddSub, ALGES, ASDiv, DRAW, GSM8K, MATHQA, MultiArith, SingleOP, SingleEQ, SVAMP).

**Agents (§2).**
1. Ask Me Anything: convert the question to a statement containing its answer, then write one new question per number in the statement (3 few-shot demonstrations). Output: 120,445 problems with a narrative similar to the seed. Solutions by GPT-4-Turbo.
2. Suggester & Editor (AutoGen): the Suggester proposes ways to make a problem harder without writing it; the Editor writes the harder problem. Two rounds per problem; problems whose GPT-4-Turbo answer exceeds 1,800 characters are dropped. Output: 37,157 problems.
3. DMath: 6,216 of 7,943 DMath training problems, kept where the GPT-4-Turbo answer matches the gold answer.
- The paper does not give a component sum for "200K"; 36,217 + 120,445 + 37,157 + 6,216 = 200,035, which suggests the seed set is included (derived, not stated).

**Iteration 1 SFT (§3.1).** Mistral-7B; format `USER:\n{question}\n\nASSISTANT:\n{answer}`; loss on answer tokens only; no packing.

**Iteration 2 data (§3.2).** Four samples per problem from M1 (top_p 0.95, temperature 0.7). The GPT-4-based exact-match prompt compares each student answer with the teacher answer. q⁺ = teacher solution plus matching student solutions; q⁻ = non-matching student solutions. For about 80K questions q⁻ is empty; one negative is then sampled from each of 4 other questions with non-empty q⁻. Pairs are the Cartesian product q⁺ × q⁻.

**Iteration 3 data (§3.2).** Same procedure with samples from M2 (the KTO model from iteration 2).

**Evaluation (§4).** GPT-4 extracts the final short answer and compares it with the gold answer ("GPT4-based-Exact-Match"); greedy decoding; GSM8K test set of 1,319 problems (§5).

**Results (Table 2, GSM8K pass@1).** SFT (M1) 79.91; SFT→SFT 81.50; SFT→DPO 84.23; SFT→KTO (M2) 85.06; SFT→SFT→SFT 80.44; M2→DPO 84.91; M2→KTO 86.87.

**Other benchmarks (Table 5, M1 → Orca-Math).** AddSub 88.99 → 91.74; ASDiv 91.10 → 91.10; MultiArith 98.28 → 98.28; SingleOp 98.74 → 99.37; SinglEq 97.25 → 99.08; SVAMP structured 87.63 → 91.30.

## Recipe ledger
| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| Orca-Math (Mistral-7B) | 7B | SFT | LR; per-device batch; epochs; packing; hardware | constant 1e-6; 3; 1; no; 8 nodes × 8 A100 | arXiv:2402.14830v1 §3.1, §5 | verified 2026-11-24 | Table 2: 2 epochs 81.50, 3 epochs 80.44 |
| Orca-Math | 7B | SFT | loss masking | answer tokens only | §3.1 | verified 2026-11-24 | no ablation reported |
| Orca-Math | 7B | preference | student samples per problem; top_p; temperature | 4; 0.95; 0.7 | §3.2 | verified 2026-11-24 | no ablation reported |
| Orca-Math | 7B | preference | β; per-device batch; grad accumulation; epochs | 0.3; 3; 11; 1 | §5 | verified 2026-11-24 | no ablation reported |
| Orca-Math | 7B | preference | LR iteration 2; iteration 3 | constant 1e-6; constant 1e-7 | §5 | verified 2026-11-24 | no ablation reported |
| Orca-Math | 7B | preference | algorithm | KTO (iterations 2 and 3) | Table 2 | verified 2026-11-24 | Table 2: KTO 85.06 vs DPO 84.23 (iter 2); 86.87 vs 84.91 (iter 3) |
| Orca-Math | 7B | eval-gate | decoding; metric | greedy; GPT-4 exact-match | §4, Table 2 caption | verified 2026-11-24 | not applicable |

## Findings relevant to generality and negative feedback
- **Negative samples.** Negatives are the student's own wrong solutions, labeled by disagreement with the teacher's answer, not by a ground-truth checker (§3.2). Removing all-positive questions (and therefore their borrowed synthetic negatives) dropped DPO from 84.23 to 60.73, while KTO moved from 85.06 to 85.22 in iteration 2 and from 86.87 to 85.44 in iteration 3 (§5.1, Table 4). The paper states only that including these questions "is beneficial"; it gives no mechanism for the DPO drop.
- **Positives.** Restricting q⁺ to the teacher solution lowers DPO to 81.96 and KTO to 82.79 (−2.27 each) (Table 3).
- **Generality.** All evaluation is single-number word problems (GSM8K and six Lila test sets); no general-capability or forgetting measurement is reported.
- **Contamination.** With unigram Jaccard > 0.5 after TF-IDF top-10 retrieval, 8 GSM8K test questions overlap the training set (7 against seeds only); 0 for n ≥ 2 (§5.3).

## Connections
- [[orca]], [[orca-2]]: earlier teacher-explanation distillation from the same group; Orca 2 13B is listed at 65.73 on GSM8K (Table 1).
- [[agentinstruct]]: later generalization of the agent-based generation flow named here "Agent-Instruct".
- [[kto]], [[dpo]]: the two preference objectives compared.
- [[metamath]], [[wizardmath]]: Table 1 baselines built by rewriting or evolving GSM8K/MATH.
- [[rejection-sampling-finetuning]], [[spin]], [[self-rewarding-lm]]: other loops that reuse model samples as training signal.

## Verification
- Checked on 2026-11-24 against https://arxiv.org/abs/2402.14830 (v1 PDF, full text).
- Source inconsistency: abstract and §1 report 86.81%; Table 2 reports 86.87 for SFT→KTO→KTO. Both are carried with locations.
- Not reported: dataset release location in the paper; LR schedule beyond "constant"; sequence length; compute time; how many preference pairs each iteration produced.
