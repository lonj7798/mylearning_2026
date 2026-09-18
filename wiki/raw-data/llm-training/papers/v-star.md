<!-- scope: V-STaR — iterative self-training that trains a DPO verifier on the correct and incorrect solutions a STaR-style loop generates
     deps: [[star]], [[rest-em]]
     see-also: [[lets-verify]], [[prm800k]], [[math-shepherd]], [[rejection-sampling-finetuning]], [[training-verifiers-to-solve-math-word-problems]]
-->

# V-STaR: Training Verifiers for Self-Taught Reasoners
- **Core Insight:** Keeping the incorrect solutions a self-improvement loop generates and training a DPO verifier on correct/incorrect pairs gives 6 to 17 absolute points of test accuracy over STaR† and ORM-style verification on math reasoning, and 4 to 12 points on code generation, with LLaMA2 and CodeLLaMA 7B and 13B models (§4.3, Fig. 2).
- **Guideline:** When a task has an automatic correctness check and test-time compute is available for sampling several candidates, keep both correct and incorrect generations and train the verifier with DPO rather than the ORM objective, because with LoRA adapters the ORM verifier stopped improving beyond 4 candidates on GSM8K (§4.4, Fig. 5a). The negative gradient is applied to the verifier only; the generator is still fine-tuned on correct solutions alone (Alg. 1).
- **Authors:** Arian Hosseini, Xingdi Yuan, Nikolay Malkin, Aaron Courville, Alessandro Sordoni, Rishabh Agarwal
- **Year:** 2024 (arXiv v1 2024-02; COLM 2024)
- **URL:** https://arxiv.org/abs/2402.06457
- **Source type:** paper
- **Relevant topics:** self-training, verifiers, DPO, best-of-k, math reasoning, code generation

## Abstract
Self-improvement methods such as STaR iteratively fine-tune a language model on its own correct solutions and discard the incorrect ones, which for hard reasoning tasks is most of what was generated. V-STaR uses both classes: correct solutions augment the generator's training buffer, and correct/incorrect pairs train a verifier with DPO that judges whether a model-generated solution is correct. The verifier ranks candidate solutions at test time. Running the procedure for several iterations produces progressively better generators and verifiers, and yields a 4% to 17% test-accuracy improvement over existing self-improvement and verification approaches on common code-generation and math-reasoning benchmarks with LLaMA2 models.

## Key Contributions
- An iterative loop that maintains two buffers: D_GEN (original SFT data plus all correct generations so far) and D_VER (all generations, correct and incorrect, with their labels) (Alg. 1).
- Training the verifier with DPO on correct/incorrect pairs instead of the Cobbe et al. (2021) language-modeling-plus-classification ORM objective, reported as more effective under LoRA (§3.1, §4.4).
- An estimator for Best-of-k accuracy computed from N ≥ k stored samples, analogous to the Pass@k estimator, which removes the variance of repeated resampling (§4.2, Eq. 3).
- Comparison against SFT, STaR†, RFT, ORM verification, self-consistency, and a non-iterative V-STaR [1 Iter] baseline matched on total generation budget (§4.1, Table 1).

## Key Figures/Tables to Study
- **Figure 1 and Algorithm 1** — the two buffers and where correct and incorrect solutions are routed.
- **Table 1** — which data each baseline gives the generator and the verifier, and whether it iterates.
- **Figure 5** — Best-of-k curves for V-STaR, V-STaR [1 Iter], and the ORM-style verifier.
- **Figure 6 (right)** — the DPO verifier's Best-of-64 rises within 2k updates while its own Pass@1 degrades.
- **Figure 7** — per-iteration Best-of-64 on MBPP, separating generator gains from verifier gains.

## Technical Details
- **Loop (Alg. 1):** at iteration t the generator G^t is fine-tuned from the pretrained base G_base on D_GEN, not continued from the previous generator; k solutions per training query are sampled; correctness label z comes from final-answer match (math) or passing the test cases (code); correct solutions join D_GEN, all solutions join D_VER. The final verifier V^T is trained from G_SFT on preference pairs built from D_VER.
- **Settings (§4, §4.1):** T = 3 iterations, k = 16 completions per query per iteration. Models are LLaMA2 and CodeLLaMA 7B and 13B, fine-tuned with LoRA adapters. The reference policy G_SFT is trained on the original data for 2 epochs (GSM8K) and 3 epochs (MBPP). The MBPP first-iteration data comes from 3-shot pretrained CodeLLaMA (§B).
- **Preference pairs (§3.1):** correct solutions are the preferred completions and incorrect ones the dispreferred; m pairs per problem are taken from the Cartesian product of the correct and incorrect sets. The DPO objective (Eq. 2) uses the implicit reward r̂(x,y) = β log V(y|x) / G_SFT(y|x). At inference the verifier score is V(ŷ|x).
- **Test-time protocol (§4.1, §4.2):** 128 candidate solutions per test problem; Pass@1 for generator-only methods, Best-of-64 for verifier-based methods and self-consistency, computed with Eq. 3.
- **Results (§4.3, Fig. 2, Fig. 3):** gains of 6-17 points on math and 4-12 points on code over STaR† and Verification. A 7B V-STaR model surpasses base LLaMA2 70B (8-shot) on GSM8K and comes close to CodeLLaMA 34B zero-shot on HumanEval (Pass@1 55% on MBPP, 48% on HumanEval for CodeLLaMA 34B).
- **Iteration behaviour (§4.3, §4.8, Fig. 7):** gains are larger across verifier iterations than across generator iterations; a fourth MBPP iteration added 0.3%. Best-of-k saturates for k ≥ 16, and the gap between V-STaR and V-STaR [1 Iter] stays constant (§4.3).
- **Verifier in the training loop (§4.7):** sampling k = 64, keeping the top 8 by verifier score plus a matched number of incorrect samples (16 or fewer per query in total) gave MBPP Best-of-64 53.2 and Pass@1 46.34, which the authors report as no substantial gain over the simpler loop.
- **DPO verifier as a generator (§4.6, Fig. 6 right):** across three β values, Best-of-64 rises within about 2k updates while the model's own Pass@1 begins to degrade after a small number of updates.
- **Against self-consistency (§4.5, Fig. 6 left):** V-STaR outperforms majority voting for k ≤ 64 on GSM8K, with the gap narrowing at larger k; combining verifier scores with weighted reranking or weighted majority voting gave no gain.

## Recipe ledger

| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| V-STaR (LLaMA2 / CodeLLaMA) | 7B, 13B | distill-SFT (generator) | iterations T; samples per query per iteration k | 3; 16 | arXiv:2402.06457v2 §4.1 | verified 2026-09-18 | §4.3: a 4th MBPP iteration added 0.3% |
| V-STaR | 7B, 13B | distill-SFT (generator) | initialization each iteration | fine-tune pretrained G_base on D_GEN, not the previous generator | Alg. 1 | verified 2026-09-18 | no ablation reported |
| V-STaR | 7B, 13B | distill-SFT (generator) | adapter | LoRA | §4 Models | verified 2026-09-18 | chosen for compute; authors expect larger gains with full fine-tuning (§4.3) |
| V-STaR G_SFT (GSM8K) | 7B, 13B | SFT | epochs on original training data | 2 | §4 Models | verified 2026-09-18 | no ablation reported |
| V-STaR G_SFT (MBPP) | 7B, 13B | SFT | epochs on original training data | 3 | §4 Models | verified 2026-09-18 | no ablation reported |
| V-STaR verifier | 7B, 13B | preference | loss; pair construction | DPO on Cartesian product of correct × incorrect solutions per problem, reference policy G_SFT | §3.1, Eq. 2 | verified 2026-09-18 | §4.4, Fig. 5a: ORM-style verifier fails above 4 candidates on GSM8K |
| V-STaR verifier | 7B | preference | updates | about 2k | §4.6, Fig. 6 | verified 2026-09-18 | Best-of-64 rises within 2k updates while Pass@1 degrades |
| V-STaR | 7B, 13B | eval-gate | candidates per test problem; reported metric | 128 sampled; Pass@1 and Best-of-64 via Eq. 3 | §4.1, §4.2 | verified 2026-09-18 | §4.2: Eq. 3 replaces high-variance repeated resampling |

DPO β values, learning rates, batch sizes, LoRA rank, and sampling temperature are not reported in the paper or its appendices.

## Findings relevant to negative feedback
- The incorrect solutions are used as an explicit negative gradient, but only inside the verifier: the DPO objective raises the likelihood of correct solutions and lowers that of incorrect ones relative to G_SFT (§3.1, Eq. 2). The generator receives no negative term; it is fine-tuned on D_GEN, which contains only the original data and correct generations (Alg. 1).
- Labels are outcome-level: final-answer match for math, passing all test cases for code (§2.1). The paper does not report a false-negative rate for these checks.
- The paper reports a cost of the negative gradient on the verifier itself: its Pass@1 as a generator degrades after a small number of DPO updates, while its ranking ability improves (§4.6, Fig. 6 right). This is why the generator and verifier stay separate models.
- Iterating the loop makes the negatives harder, because later generators produce more plausible incorrect solutions; the paper attributes the advantage over V-STaR [1 Iter] at matched generation budget to this (§1, §4.3).

## Findings relevant to generality
- Out-of-domain transfer is measured by evaluating GSM8K-trained models on a 150-problem Level-1 subset of the MATH test set and MBPP-trained models on the full HumanEval test set; V-STaR stays ahead of the baselines on both, at lower absolute scores (§4, footnote 2, §4.3, Fig. 4).
- The MATH transfer subset is restricted to algebra, counting and probability, prealgebra, and number theory items whose final answer is a number and whose question contains no LaTeX (§4, footnote 2).

## Connections
- [[star]] — the greedy, correct-solutions-only predecessor; V-STaR's STaR† baseline is STaR with k samples per iteration.
- [[rest-em]] — the EM-style variant V-STaR cites as ReST_EM; it discards the incorrect solutions that V-STaR routes to D_VER.
- [[rejection-sampling-finetuning]] — the RFT baseline, a single non-iterative round.
- [[training-verifiers-to-solve-math-word-problems]] — Cobbe et al. (2021), the ORM verifier V-STaR compares against and outperforms under LoRA.
- [[lets-verify]], [[prm800k]], [[math-shepherd]] — step-level supervision, contrasted with V-STaR's outcome-level labels.

## Verification
- Checked on 2026-09-18 against: https://arxiv.org/abs/2402.06457 (arXiv v2, 14 Aug 2024; COLM 2024).
- Corrections to the previous card version:
  - "Authors: Eric Zelikman, Yuhuai Wu, Jesse Mu, Noah D. Goodman" → those are the STaR authors. V-STaR is by Arian Hosseini, Xingdi Yuan, Nikolay Malkin, Aaron Courville, Alessandro Sordoni, Rishabh Agarwal (Mila/Université de Montréal, Microsoft Research, University of Edinburgh, Google DeepMind) (title page).
  - "Verifier training signal: preference-style or classification-style supervision" → DPO on correct/incorrect pairs specifically, with the ORM classification objective as the compared baseline (§3.1, §4.4).
  - "The selected high-scoring traces are fed back into the reasoner for the next round" → the verifier is not in the loop in the main method; correct solutions enter D_GEN by their correctness label alone (Alg. 1). Putting the verifier in the loop was tried and gave no substantial gain (§4.7).
  - "EM-like self-improvement cycle" → V-STaR does not frame itself as EM; that framing belongs to ReST_EM (§2.1).
  - Year given without version → arXiv v1 2024-02, COLM 2024.
  - Related-card note: [[rest-em]] described V-STaR as a "value function over partial rationales"; the verifier is a DPO-trained solution-level scorer with no step-level or value-function component (§3.1).
- Removed as unsupported by the source: "verifier-guided selection is stronger than plain STaR-style self-training" stated without numbers (replaced with the §4.3 ranges); "connects self-training for reasoning to a broader process-supervision view later seen in PRM-style work" (the paper contrasts itself with process supervision in §5, it does not claim continuity); "a weak verifier can reinforce the wrong traces and create a bad feedback loop"; "if candidate generation collapses too early, the verifier has little real choice"; "much harder to apply where correctness is fuzzy" (the paper states the method needs correctness feedback, §6, without the comparative claim).
- Not reported by the source: DPO β values used for the released verifiers, learning rates, batch sizes, LoRA rank and target modules, sampling temperature for data generation, and total compute.
