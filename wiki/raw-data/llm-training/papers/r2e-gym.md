<!-- scope: R2E-Gym paper (arXiv:2504.07164): SWE-Gen procedural construction of executable SWE environments from commits (test collection or generation plus back-translated issues), SFT of Qwen-2.5-Coder agents on Sonnet-3.5-v2 trajectories, and analysis of execution-based, execution-free, and hybrid verifiers for test-time scaling.
     deps: [[swe-gym]], [[openhands-data]]
     see-also: [[r2e-gym-recipe]], [[deepswe]], [[swe-smith]], [[swe-rl]], [[humpback]]
-->

# R2E-Gym: Procedural Environments and Hybrid Verifiers for Scaling Open-Weights SWE Agents
- **Core Insight:** Qwen-2.5-Coder-32B fine-tuned on 3,321 successful Sonnet-3.5-v2 trajectories from R2E-Gym environments reaches 34.4% Pass@1 on SWE-Bench-Verified (Table 3), and a hybrid of a test-generating agent and a learned trajectory verifier reaches 51.0% Best@26, while each verifier alone plateaus at 43.7% and 42.8% (§4.2, Table 4).
- **Guideline:** When selecting one of several agent trajectories at test time, combine test execution with a learned verifier instead of either alone, because generated tests separate correct from incorrect patches poorly (for the majority of problems, under 20% of tests do, §4.2, Figure 5) and the learned verifier loses 5.2 points of Best@26 when it sees only the final patch, which shows reliance on trajectory text (Figure 7a).
- **Authors:** Naman Jain, Jaskirat Singh, Manish Shetty, Liang Zheng, Koushik Sen, Ion Stoica (UC Berkeley; Australian National University)
- **Year:** 2025 (arXiv v1 2025-04; marked "Preprint. Under review.")
- **URL:** https://arxiv.org/abs/2504.07164
- **Source type:** paper
- **Relevant topics:** executable SWE environments, synthetic task generation, back-translation, test generation, agent SFT from teacher trajectories, rejection sampling, outcome verifiers, test-time scaling, contamination control

## Abstract
The paper addresses two problems for open SWE agents: curating execution environments at scale and scaling test-time compute. It introduces R2E-Gym, an executable gym with more than 8.1K tasks. SWE-Gen builds these environments from commits by collecting or generating tests and back-translating code changes into problem statements, so human-written issues and unit tests are not required; training on it gives 34.4% Pass@1 on SWE-Bench-Verified with a 32B model. The paper then compares execution-based verifiers (tests) and execution-free verifiers (learned models): test-based verifiers have low distinguishability, and execution-free verifiers are biased toward stylistic features. Each saturates around 42-43%, and a hybrid of the two reaches 51%, which the authors describe as the best open-weight result and competitive with o1 and Sonnet with tools.

## Key Contributions
- SWE-Gen, a pipeline from commits to executable environments with Fail→Pass tests and back-translated issues; 8,135 problems, and a 4,578-problem subset (R2E-Gym-Subset) with no repository overlap with SWE-Bench (§2, Table 1).
- SFT agents at 7B, 14B, and 32B on R2E-Gym trajectories, compared with SWE-Gym at equal base model (§3, Table 3).
- An analysis of verifier failure modes: low test distinguishability, toxic tests, and execution-free reliance on agent thoughts (§4.2).
- A hybrid verifier (Eq. 2) and ablations of its parts (§4.3-§4.4, Figure 8).

## Key Figures/Tables to Study
- Table 1: executable-environment datasets and sizes. Table 3: resolve rates by size on SWE-Bench Lite and Verified.
- Figure 2: Pass@1 against number of SFT trajectories (100 to 3,200). Figure 3: thoughts ablation and real vs synthetic issues.
- Figure 4: Best@K against editing-agent and testing-agent rollouts. Table 4: comparison with proprietary and open systems.
- Figure 5: distinguishability and toxicity distributions. Figure 7: execution-free verifier ablation and attention analysis.

## Technical Details
**Commit and repository curation.** Python repositories with many commits are found with SEART GitHub search, and commits are filtered by rule-based and LLM-based heuristics (§2). Thresholds: at most 5 non-test files, 100 edited lines, 2000 characters of patch, 1 deleted entity, 3 added entities, 3 edited entities, and 10 statement-level changes; an LLM-as-judge filter follows (App. A). Build environments are found by a semi-manual search over dependency version combinations (App. A).

**Tests.** Existing tests give Fail→Pass (F2P) cases: tests that fail on the pre-commit code and pass on the commit (§2). Commits without tests get generated F2P tests from an Agentless-like method that uses the ground-truth patch as context (§2; App. A).

**Back-translation.** Problem statements are generated from the commit message, the non-test patch, test changes, old and new test executions, and failing assertions (App. A). The prompt tells the LLM to describe the failure and expected behavior without revealing the solution (App. A, Listing 3). The LLM used is not named. The authors state that including F2P tests makes statements precise, where naive back-translation of the diff produces generic statements (§2).

**Agent scaffold.** A ReAct agent built on OpenHands with four tools: file editor, search, execute bash, and submit; no internet access (§3; App. B).

**Verifiers (§4.1).** Execution-based (EB): a testing agent (Qwen-Coder-32B) writes a test script with M = 10 tests for the issue, and patches that do not have the top regression-test score get 0 (Eq. 1):
s_k^EB = TestScore_k if RS_k = max_j RS_j, else 0; TestScore_k = Σ_i Pass(P_k, Test_i).
P_k is candidate patch k of K, Test_i a generated test, RS_k the regression-test score of patch k. Execution-free (EF): Qwen2.5-Coder-14B reads the task, trajectory, and patch and outputs YES or NO; s^EF = P(YES)/(P(YES)+P(NO)) (§4.1).

**Hybrid (§4.3, Eq. 2).** s_k^H = Top_n(s_k^EF) + s_k^EB, where Top_n keeps s_k^EF for the n highest EF scores and sets −∞ otherwise; regression filtering is applied after top-n filtering. The value of n is not reported. EF scores lie in [0,1], so among the top-n patches they break ties between equal test counts.

**Metrics (§4.2; App. C.3).** Best@K is the share of problems where the verifier's top choice among K trajectories is correct. Distinguish(t) = 1[max_{p∈P_i} Pass(p,t) ≠ max_{p∈P_c} Pass(p,t)], with P_c the correct and P_i the incorrect patches; Toxic(t) = 1[max_{p∈P_i} Pass(p,t) > max_{p∈P_c} Pass(p,t)]. Example: if some correct patch passes t and no incorrect patch passes it, Distinguish(t) = 1 and Toxic(t) = 0.

**Test-time setup.** From R2E-Gym-32B on SWE-Bench-Verified: 1 trajectory at T = 0 and 25 at T = 0.8 and 0.9, giving Pass@26 = 64.4% (§4.2, Figure 14). "We sample 7 tests using our testing agent at temperature T = 0.8", with a fixed Django in-context example (§4.2).

**Recipe values.** SFT, testing-agent, and verifier hyperparameters are in [[r2e-gym-recipe]].

## Findings relevant to generality, negative feedback, long context, agentic training, distillation
- **Distillation (stage, prompts, teacher, quality control).** Distill-SFT of Qwen-2.5-Coder 7B/14B/32B on R2E-Gym-Subset tasks; teacher Sonnet-3.5-v2 in the paper's scaffold at temperature 0.2; only trajectories that pass R2E-Gym unit tests (synthetic and existing) are kept (§3; App. B). Pass@1 on Verified: 19.0 / 26.8 / 34.4 against 10.6 / 16.4 / 20.6 for SWE-Gym training (Table 3).
- **Agentic SFT data scaling.** With 100 to 3,200 trajectories, the 14B model begins to saturate at approximately 800, while 32B still improves (§3.1, Figure 2). Training with thoughts gives 34.4% against 30.4% without (Figure 3).
- **Synthetic vs real tasks.** With 400 trajectories each, synthetic problem statements give 27.8% Pass@1 and real GitHub issues 28.0% (§3.1, Figure 3).
- **Negative feedback.** Failed teacher trajectories are discarded for the editing agent (negative marginal value; App. B). The EF verifier is trained on 5,700 trajectories with equal positives and negatives, including on-policy trajectories from the trained 32B model; failures are the "NO" target (negative as content, App. C.2). The testing agent is trained on 2,203 Sonnet trajectories, "both positive and negative ... with minimal rejection sampling" (App. C.1).
- **Verifier limits.** For the majority of problems, under 20% of generated tests distinguish top-ranked correct from incorrect patches; many tests do not reproduce the bug (Pass→Pass) or fail the correct patch (Fail→Fail) (§4.2, Figures 5-6). Toxic tests reach up to 10% of tests for some problems (§4.2). The EF verifier scores 71.82% accuracy / 42.8% Best@26 with patch and trajectory, 68.01% / 37.6% with patch only, and 68.77% / 41.4% with thoughts removed (Figure 7a).
- **Hybrid ablations (§4.4, Figure 8).** Regression tests alone raise 42.9% to 47.4%; adding generated tests gives 51.0%. Agentless tests in the same framework give 48.8%. Top-n filtering raises 49.8% to 51.0%. Editing rollouts 16 → 21 give 47.6% → 48.4%; 5 more test rollouts give 49.3%. Hybrid Best@16 is 49.4% (Table 4).
- **Long context.** SFT uses a 20K max context due to compute limits; the authors name context parallelism as future work (App. B).
- **Generality.** Evaluation covers SWE-Bench Lite and Verified only; contamination control is repository-level exclusion (§2-§3).

## Connections
- [[r2e-gym-recipe]]: recipe ledger for the editing agent, testing agent, and EF verifier.
- [[deepswe]]: RL-only Qwen3-32B agent trained on 4.5K R2E-Gym-Subset problems with this paper's tools and hybrid test-time scaling.
- [[swe-gym]]: human-issue-based executable environments; the SFT baseline in Table 3 and Best@16 32.0% in Table 4.
- [[swe-smith]], [[swe-rebench]], [[skywork-swe]]: other pipelines that build executable SWE tasks at scale.
- [[swe-rl]]: SWE-RL, listed at 41.0% Best@500 in Table 4.
- [[humpback]], [[oss-instruct]]: back-translation references (Li et al. 2023; Wei et al. 2023) cited for SWE-Gen.
- [[training-verifiers-to-solve-math-word-problems]]: outcome-supervised verifiers (Cobbe et al. 2021), cited for the EF verifier.
- [[openhands-data]]: OpenHands, base of the agent scaffold.
- [[rejection-sampling-finetuning]]: filtering teacher samples by a correctness check before SFT.

## Verification
- Created on 2026-09-14 from https://arxiv.org/abs/2504.07164 (v1, 2025-04-09; the only version listed).
- Audit claims not found in the source:
  - "abstract: 8.7K tasks" → the abstract says "more than 8.1K tasks"; Table 1 gives 8,135.
  - Authors "(UC Berkeley)" → Jaskirat Singh and Liang Zheng are listed at Australian National University.
  - "a testing agent writes 10 tests per patch" → the testing agent writes one script of M = 10 tests for the issue, and each patch is scored against it (§4.1).
  - "fewer than 20% of generated tests actually distinguish" → the paper scopes this to "the majority of problems" (§4.2).
  - "the environments are also the RL substrate later used by DeepSWE" → not in this paper; stated in [[deepswe]] §2.1.
- Internal inconsistencies: R2E-Gym-Subset is 4,578 (§2, §3, Table 1) but "R2E-Gym-lite ... 4538" in App. B; the thoughts ablation is 34.4% in Figure 3 but "34.2%" in the §3.1 text; SWE-Gen yields "over 2.5 times" more problems than issue-based collection (§2) and "over 3 times" more executable environments (§1).
- Not reported by the source: the LLM used for back-translation and commit judging; n in Top_n; LR schedule shape beyond warmup ratio; number of testing-agent rollouts per problem in the final 51.0% configuration.
