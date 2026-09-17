<!-- chapter: ch-44
     track: rl
     kind: content
     title: Process Supervision and Verifiable Rewards
     deps: [ch-43a]
     sources: [[prm800k]], [[let-verify]], [[lets-verify]], [[math-shepherd]], [[omegaprm]], [[qwen-prm-lessons]], [[rl-on-incorrect-synthetic-data]], [[rewarding-progress-pav]], [[rlvr-tulu3-config]], [[ifbench]], [[why-language-models-hallucinate]], [[swe-rl-difflib-reward]], [[deepswe]], [[deepswe-recipe]], [[crossing-the-reward-bridge]], [[transferability-of-llm-reasoning]], [[deepseek-r1]], [[rlvr-beyond-base-model]], [[prorl]], [[spurious-rewards-rlvr]], [[training-verifiers-to-solve-math-word-problems]], [[reward-model-overoptimization]], [[metr-frontier-reward-hacking]], [[natural-emergent-misalignment-reward-hacking]], [[glm-4-5]], [[kimi-k2]], [[deepseek-v3.1]], [[step-dpo]], [[grpo]], [[john-schulman-kl-tricks]], [[rubrics-as-rewards]]
     figures: figures/prm-vs-orm.html, figures/wrong-answer-rewards.html
     excerpts: excerpts/qwen-prm-lessons.md, excerpts/rewarding-progress-pav.md, excerpts/rl-on-incorrect-synthetic-data.md, excerpts/why-language-models-hallucinate.md, excerpts/crossing-the-reward-bridge.md, excerpts/rlvr-tulu3-config.md, excerpts/swe-rl-difflib-reward.md
     revised: 2026-09 (generality revision)
-->

# Chapter 44 — Process Supervision and Verifiable Rewards

> **Core insight.** A verifier and a step label are estimators of correctness, and the policy is trained on their errors as well as on the task. Human step labels on 12K MATH problems produce a reward model that selects a correct solution for 78.2% of 500 held-out MATH problems at best-of-1860, against 72.4% for an outcome reward model and 69.6% for majority voting ([[prm800k]] §3, Fig. 3). Replacing those labels with Monte-Carlo rollout estimates changes the two evaluations in opposite directions: an MC-labelled PRM reaches 40.1 average F1 on ProcessBench against 56.5 for a PRM trained on PRM800K, while scoring 65.9 against 64.9 on Best-of-8 ([[qwen-prm-lessons]] Tables 3-4). On the outcome side, a deterministic verifier used as the reward raised Tülu 3 8B from 84.3 to 87.6 on GSM8K, 42.0 to 43.7 on MATH, and 81.1 to 82.4 on IFEval ([[rlvr-tulu3-config]] Table 23), and the same paper reports over-optimization inside RLVR when the KL penalty is lowered ([[rlvr-tulu3-config]] §6.2.1).
>
> **Guideline.** When a task has a programmatic check and a reference answer, use that check as the reward and keep a KL penalty, because Tülu 3's RLVR stage improved its three targeted evaluations at the 8B scale while the average over the Table 23 benchmarks moved 64.4 → 64.8; expect gains of a few points, not a new capability. When step-level signal is needed, prefer labels that a step-level benchmark validates (human labels, LLM-judge labels, or MC labels filtered by agreement with a judge), because MC-only labels score lowest on first-error identification ([[qwen-prm-lessons]] Table 4). When the step signal is meant for the policy rather than for ranking, use a per-step advantage under a prover different from the policy, because the advantage under the policy itself gives the same gradient as outcome-only RL ([[rewarding-progress-pav]] §3.2). When the reference answer is free-form and no rule can check it, use a trained generative verifier, because rule-based rewards degraded from 26.2 to 16.9 on multi-subject accuracy as training data scaled from 20k to 100k while a 7B model-based verifier rose from 30.8 to 35.0 ([[crossing-the-reward-bridge]] Table 2).

---

## Why this chapter matters for a general-purpose model

Pipeline position: pre-training → mid-training → SFT → preference optimization → **RL** → evaluation. Chapters ch-37 to ch-40 built the estimators, ch-41 built the learned reward model, ch-42 catalogued reward hacking, ch-43 covered entropy and KL, and ch-43a covered what a negative gradient does to a distribution. This chapter is about where the scalar comes from when it is not a learned preference model: a step label, or a program that checks the answer.

Three measurable problems follow from that choice.

1. **Credit assignment.** An outcome label says one bit about a trace of many steps. On hard problems most sampled solutions contain an error, so a negative outcome label carries little information about where the error is ([[prm800k]] §6.1).
2. **Instrument error.** A step label estimated by rollouts, or a unit test, or a string match, is wrong on some fraction of cases. The false positives are what the policy is trained to produce. Qwen's manual annotation of correct-answer samples found process-error rates of 5.1% on GSM8K rising to 43.4% on Omni-MATH ([[qwen-prm-lessons]] Fig. 6).
3. **Breadth.** A verifier exists for a narrow set of tasks. Training on that set can improve capability elsewhere (SWE-RL, §7) or narrow it (IFEval-style constraint training, §7). Which happens is measured, not assumed.

Sections: outcome supervision and its limit (§1), automated step labels (§2), what those labels actually measure (§3), step-level signal for the policy (§4), RLVR (§5), execution-grounded agentic RLVR (§6), transfer outside the verifiable domain (§7).

---

## §1 Outcome supervision and the credit-assignment limit

### §1.1 Definitions

An **outcome reward model (ORM)** is a model that scores a complete `(problem, solution)` pair with the probability that its final answer is correct. A **process reward model (PRM)** scores each step of the solution. A **verifier** in the RLVR sense is not a model at all: it is a program that compares the completion with a reference and returns a number.

The measurable problem is selection. Cobbe et al. trained an ORM to rank sampled GSM8K solutions and reported that sampling many solutions and ranking them beats fine-tuning alone at the same model size ([[training-verifiers-to-solve-math-word-problems]], cited at loci in arXiv:2110.14168v2 §4).

### §1.2 The PRM800K protocol

[[prm800k]] fixes the labelling protocol that every later method approximates.

1. The generator is fine-tuned for one epoch on few-shot-generated MATH solutions with correct answers, only to produce newline-delimited steps (§2.3). It is not trained with RL (§2.1).
2. A labeler marks each step positive, negative, or neutral; in phase 2, labelling of a solution stops at the first negative step (§2.4, App. D).
3. The PRM predicts one token after the last token of each step, trained by maximizing the log-likelihood of the label tokens over three classes (§2.6, App. F.1).
4. A solution's score is the **product** of per-step P(positive), with neutral counted as positive (§2.6, App. F.2).
5. Data collection is iterative: in each of 10 generations the current PRM ranks N solutions per problem, the highest-scoring **wrong-answer** solutions go to labelers, and the PRM is retrained (App. B).

Scale: 800K step labels over 75K solutions to 12K problems (§2.4). Of all labelled steps 73.1% are positive, and 14.2% of labelled solutions end with a correct answer (App. B, Table 3). The active-learning selection above is estimated at about 2.6x the data efficiency of uniform labelling in the small-scale runs (§4.2, Fig. 4a).

**Inference cost.** The PRM's per-step predictions are read from one forward pass: "To determine the step-level predictions at test time, it suffices to perform a single PRM forward pass over the whole solution" (§2.6). A PRM is therefore not L times more expensive than an ORM at best-of-N time; both are one forward pass per candidate solution.

### §1.3 Results and their conditions

| Selector, 500 held-out MATH problems, best-of-1860 | Accuracy |
|---|---|
| Majority voting | 69.6 |
| ORM (100 uniform samples per problem as training data) | 72.4 |
| PRM (PRM800K) | 78.2 |

The PRM is above the ORM at every tested N and the gap grows with N (§3, Fig. 3). The training sets are not matched: the ORM is trained on 100 uniform samples per problem, the PRM on PRM800K (Verification section of [[prm800k]]). On a held-out STEM set (AP Calculus, AP Chemistry, AP Physics, AMC10/12) at best-of-100, the order is preserved: PRM 72.9, ORM 63.8, majority 61.3 (§5, Table 1). The authors state that it is unknown how far the results generalize beyond mathematics (§6.2).

### §1.4 Worked example: product against minimum

Two candidate solutions, with PRM per-step P(positive):

- Solution A, 5 steps: 0.95, 0.90, 0.85, **0.60**, 0.98. Product = 0.4273. Minimum = 0.60.
- Solution B, 9 steps, each 0.90. Product = 0.9⁹ = 0.3874. Minimum = 0.90.

Product ranks A above B; minimum ranks B above A. The two rules differ systematically: the product penalizes each additional step, and the paper records "a slight bias against solutions with more steps" (App. F.2). The paper's own comparison at best-of-1860 gives 78.2 for the product with neutral counted as positive and 77.6 for the minimum (App. F.2, Table 4). [[math-shepherd]] adopts the minimum, following Lightman et al., and does not compare aggregators (§3.4). §3.5 of this chapter shows that the right aggregator depends on how the labels were produced.

Open [`figures/prm-vs-orm.html`](figures/prm-vs-orm.html) panel B to change per-step scores and see how product, minimum, and last-step scoring reorder two solutions.

---

## §2 Automated step labels: Monte-Carlo estimation

### §2.1 Definition and mechanism

Human step labels do not scale to millions of problems. [[math-shepherd]] replaces the labeler with a **completer** policy and defines the label of a step by whether completions sampled from it reach the gold answer.

For a step `s_i` of a solution, sample `N` continuations with final answers `a_1 … a_N`; let `a*` be the gold answer, and `I(·)` be 1 when its condition holds and 0 otherwise:

```
hard label   y_HE(s_i) = 1 if any a_j = a*, else 0                 (Eq. 3)
soft label   y_SE(s_i) = (1/N) Σ_j I(a_j = a*)                     (Eq. 4)
```

- `s_i`: the i-th step, with the prefix `s_1 … s_i` as the sampling condition.
- `N`: completions per step; Math-Shepherd's experiments use a LLemma-7B completer with `N = 8` (§4).
- The experiments train the PRM on hard labels, written as two special tokens for "has potential" and "no potential" so that a standard LM pipeline can be used (§4).

The PRM loss is per-step binary cross-entropy (Eq. 2). At inference a solution's score is the minimum of its step scores (§3.4). For RL, Math-Shepherd gives the PRM score at each step-end token position and 0 elsewhere, with no separate final-answer term (§3.5; ACL Eq. 7).

### §2.2 Worked example: the label and its two error rates

A step is followed by `N = 8` completions; 3 reach the gold answer. Soft label = 3/8 = 0.375; hard label = 1.

Now treat the label as an estimator. Write `p` for the probability that one completion from a **correct** prefix reaches the gold answer, and `q` for the probability that one completion from an **incorrect** prefix still reaches it (by recovering, by guessing, or by an arithmetic accident).

```
P(false negative on a correct step) = (1 − p)^N
P(false positive on an incorrect step) = 1 − (1 − q)^N
```

With `p = 0.15` and `N = 8`: 0.85⁸ = 0.272, so 27% of correct steps in a hard problem are labelled negative. With `q = 0.10` and `N = 8`: 1 − 0.9⁸ = 0.570, so 57% of incorrect steps are labelled positive. Raising `N` lowers the first rate and raises the second. This is the arithmetic behind Math-Shepherd's measurement: on 160 human-annotated GSM8K steps, hard labels from a LLaMA2-70B completer are 86% accurate at `N = 4`, and accuracy declines for larger `N`, which the authors attribute to false positives (§5.2, Fig. 4a). It is also why a weak completer produces noisy labels: `p` is small for every step when the completer cannot solve the problem.

Panel A of [`figures/prm-vs-orm.html`](figures/prm-vs-orm.html) plots both error rates as `N`, `p`, and `q` change.

### §2.3 Evidence from Math-Shepherd

Greedy accuracy, GSM8K / MATH (Table 2): Mistral-7B 77.9 / 28.6; +rejection fine-tuning 79.0 / 29.9; +PPO with an ORM 81.8 / 31.3; +step-by-step PPO with the PRM 84.1 / 33.0. On top of that PPO-trained Mistral-7B, verification over 256 samples gives 89.1 on GSM8K and 43.5 on MATH500 for self-consistency combined with Math-Shepherd, and 88.4 / 41.1 for Math-Shepherd alone (Table 3). The 89.1 / 43.5 headline is therefore the PPO model plus self-consistency plus the PRM, not the SFT model plus a PRM.

Best-of-256 selection with a LLaMA2-70B generator (Table 1): self-consistency 88.0 / 39.4, ORM 91.8 / 40.4, Math-Shepherd 93.2 / 44.5. A 7B verifier on a 70B generator scored below self-consistency, and 7B reward models decline as the number of candidates grows while 70B ones improve (§5.3, Fig. 5).

Conditions: all experiments use GSM8K and MATH problems with MetaMATH-fine-tuned generators; no other domain is trained on (§4). The authors state that label noise remains and its effect on PRM quality is undetermined (§6).

### §2.4 OmegaPRM: the same labels for less compute

[[omegaprm]] keeps the Monte-Carlo definition and changes the search. Instead of estimating every prefix, it binary-searches for the first incorrect step.

`c_t = MonteCarlo(q, x_1:t)` is the fraction of `k` rollouts from the prefix that end with the golden answer (Eq. 1). The search (§3.2):

1. Split the solution at the midpoint `m` and roll out from `x_1:m`.
2. If `c_m > 0`, treat the first half as correct and recurse on the second half; if `c_m = 0`, recurse on the first half.
3. Stop when the segment is one step.

Cost is `O(k log M)` against `O(k M)` for per-step estimation, where `M` is the number of steps. Rollouts are reused inside an MCTS-style pool whose selection score prefers states with `MC(s)` close to 1 that still produced a wrong answer (Eq. 2-3). Settings: `k = 8` rollouts per estimate, search limit 100 per question, solutions divided into 16 pieces, question filtering that drops questions with 0 correct or 0 wrong answers in 32 rollouts (§4, App. A). Labels are pointwise soft targets `MC(s)` (§3.4).

Results at `k = 64`, MATH500, Gemini Pro / Gemma2 27B (Table 1): majority vote 67.2 / 54.7; Math-Shepherd data 67.2 / 57.4; PRM800K data 67.6 / 57.2; OmegaPRM 69.4 / 58.2. The headline 51.0 → 69.4 for Gemini Pro is an 18.4-point absolute and **36% relative** gain (§6). Reporting 69.4 as the relative gain confuses the accuracy value with the improvement. Efficiency: the same compute budget produces 200K data points by brute force against 15M by binary search, of which 1.5M were sampled for training (§4.4).

The method needs a question with a golden answer, so it does not apply to open-ended tasks without adaptation (§5). Note also what the open recipes did: DeepSeek-R1 lists PRMs among unsuccessful attempts, citing the difficulty of defining fine-grained steps, the difficulty of labelling step correctness, reward hacking of a model-based PRM, and limited benefit for the added compute ([[deepseek-r1]] G.2). Outcome-only rewards, not process rewards, drive the 2025 open reasoning recipes.

---

## §3 What Monte-Carlo labels measure, and how to evaluate a PRM

### §3.1 A value estimate is not a correctness judgment

[[qwen-prm-lessons]] states the distinction that the §3.2 comparison below measures: a PRM is meant to be a deterministic judge of whether the current step is correct, while an MC estimate is a **value**: the probability of reaching the correct final answer from that step under a completion policy (§3.1.1). Training a PRM on MC labels therefore produces a value model rather than the step-correctness judge the name implies. The two disagree exactly when the completer can recover from a bad step or fail from a good one, which is the `p`/`q` arithmetic of §2.2.

### §3.2 Evidence: the same data, three labelling methods

All rows use the same 860k queries and responses where marked; the policy for Best-of-8 is Qwen2.5-Math-7B-Instruct; ProcessBench measures whether the model finds the first erroneous step ([[qwen-prm-lessons]] Tables 3-4):

| Step labels | # samples | Best-of-8 avg | ProcessBench avg F1 |
|---|---|---|---|
| MC (Math-Shepherd data) | 440k | 64.3 | 28.9 |
| MC (Qwen data) | 860k | 65.9 | 40.1 |
| LLM-as-a-judge (Qwen2.5-72B-Instruct) | 860k | 65.3 | 46.5 |
| Human (PRM800K) | 264k | 64.9 | 56.5 |
| maj@8 / pass@8 | — | 66.2 / 74.7 | — |

Two readings. First, the two evaluations order the methods in opposite directions, so a paper that reports only best-of-N cannot support a claim about process verification. Second, none of these PRMs beats maj@8 at N = 8; the released Qwen2.5-Math-PRM-7B does, at 67.6 average (Table 6), with ProcessBench 73.5 (Table 7).

### §3.3 Consensus filtering and the label threshold

Keeping only instances where the LLM judge and the MC estimate agree on the first-error location retains about 40% of the 860k set and raises ProcessBench F1 from 40.1 to 46.3, close to the LLM-judge model trained on all 860k (§3.1.3, Fig. 2). A sweep of the hard-label threshold from 1/8 to 7/8 degrades both evaluations monotonically, so the recommended threshold is 0: a step is negative only when **none** of the 8 completions reaches the correct answer (§3.1.4, Fig. 5). Soft labels underperform hard labels after filtering in this setup, which the authors attribute to correct steps receiving targets below 1 and to the variance of 8 completions (§3.1.4, Figs. 3-4). [[omegaprm]] reports the opposite ordering on its own data — soft 70.1% against hard 63.3% step-classification accuracy (§4.3, Table 2) — so the label form is not settled across setups; the two studies differ in policy, completer, and filtering.

### §3.4 Why best-of-N inflates PRM scores

Three mechanisms, all from [[qwen-prm-lessons]] §3.2:

1. Policies produce correct answers with flawed processes at rates that rise with difficulty: 5.1% (GSM8K), 11.9% (MATH), 27.4% (OlympiadBench), 43.4% (Omni-MATH) among correct-answer samples (Fig. 6). A PRM that correctly scores such a response low is punished by a best-of-N metric that only checks the answer.
2. PRMs that tolerate those responses score higher on best-of-N. On ProcessBench cases with a correct answer and an erroneous step, every open-source PRM other than the authors' detects fewer than 50% (Table 5).
3. Optimizing for best-of-N shifts a PRM toward outcome scoring: for four open PRMs, more than 40% of responses have their minimum step score at the final step (Fig. 8).

### §3.5 Aggregation depends on the label source

For MC-trained PRMs the **last** step score is the better solution score, because an MC value at the last step already integrates the whole prefix and the per-step estimates are not independent; for human-annotated and LLM-judge PRMs, product and minimum are better (§3.2.4, Fig. 9). A team that trains labels one way and scores solutions the other way loses accuracy for a reason unrelated to the model.

**Implication for a general-purpose model.** Step-level supervision has a documented precondition that fails outside mathematics: MC labels need a golden answer and a completer strong enough to make `p` large. Neither exists for open-ended tasks. Use step supervision where its instrument is checkable, and measure it with a step-level benchmark rather than with a best-of-N score.

---

## §4 Step-level signal for the policy, not for the ranker

Sections §1-§3 use step labels to rank candidate solutions. This section uses them inside the gradient.

### §4.1 Per-step advantages from incorrect traces

[[rl-on-incorrect-synthetic-data]] builds the negatives so that the pair a preference loss sees differs only at the step that matters.

1. Sample incorrect responses from the SFT policy for a problem with a known answer (8 per question in the practical version).
2. For each step, estimate `Q^π̃(x, ŷ_1:i−1; ŷ_i)` by Monte-Carlo rollouts from the prefix, with `π̃ = BoK(π_sft)`, `K = 5`.
3. Take the "first pit" `ŷ_c`, the first step with the lowest `Q`, and add the pair (correct solution, prefix ending at `ŷ_c`).
4. Run DPO on those pairs with `π_sft` as reference.

The step advantage, under deterministic dynamics where the value of a state equals the Q-value of the previous step (Eq. 3):

```
A^π̃(x, ŷ_1:i−1; ŷ_i) = Q^π̃(x, ŷ_1:i−1; ŷ_i) − Q^π̃(x, ŷ_1:i−2; ŷ_i−1)
```

Results with DeepSeek-Math-7B and Llama2-7B on GSM8K and MATH: per-step DPO improves over the SFT policy and keeps improving as the synthetic set grows, which the authors summarize as an 8x gain in synthetic-data efficiency; self-generated positives alone (rejection fine-tuning) are worth about 2x (§1, §6.2, Fig. 7a-b). The control matters as much as the result: DPO on arbitrary correct/incorrect response pairs does not improve over the SFT policy on MATH and could not be fixed by tuning `β` (Fig. 7c). Theorem 6.1 shows that the per-step pair construction has the same optimum as advantage-weighted RL. See [[step-dpo]] for the earlier step-preference construction that this work generalizes.

### §4.2 Progress, not value: process advantage verifiers

[[rewarding-progress-pav]] asks what a dense step reward should predict. Its answer: the **change** in the probability of success caused by the step, measured under a **prover** policy `μ` that is not the policy being trained.

```
∇_π ℓ_PAV-RL(π) = Σ_h ∇_π log π(a_h | s_h) · ( Q^π(s_h, a_h) + α · A^μ(s_h, a_h) )      (Eq. 5)
```

- `Q^π(s_h, a_h)`: probability that the policy's own completions after this step reach the correct answer; this is the outcome term a policy gradient already carries.
- `A^μ(s_h, a_h) = Q^μ(s_h, a_h) − V^μ(s_h)`: the progress the step makes under the prover.
- `α ≥ 0`: weight on the process term.

Two consequences stated in §3.2. First, setting `μ = π` reproduces the outcome-only update, because a policy gradient already subtracts a baseline — a PRM that predicts the current policy's own values adds nothing to RL. Second, an over-capable prover succeeds from every step and assigns near-equal scores to good and bad steps; a very weak prover succeeds from none. Provers must be complementary to the base policy, and the paper reports that provers weaker than the base policy can work.

**Worked example.** A prover with `Q^μ = 0.40` before a step and `0.55` after it gives `A^μ = +0.15`. With `α = 3` and an outcome term `Q^π = 0.20`, the effective reward for that step is `0.20 + 3(0.15) = 0.65`. A rephrasing step that leaves a strong prover's success probability at 0.99 before and after gives `A^μ ≈ 0`, so it earns no bonus — while a `Q`-valued reward would pay 0.99 for it. That is the failure mode Fig. 2b names: `Q` from a strong prover assigns unmerited bonuses to trivial actions.

Results: beam search against PAVs is more than 8% more accurate than ORM re-ranking at equal compute and 1.5-5x more compute-efficient; RL with `Q^π + α A^μ` on Gemma 2B and 9B is more than 7% better in accuracy than outcome-only RL and reaches that accuracy about 6x faster in samples, with higher Pass@N for all `N ≤ 128` (Abstract; §5, Figs. 7-8). Settings: REINFORCE, learning rate 1e-7, batch 32, response length 512, KL 0.001, `α = 5.0` (2B) and `3.0` (9B) (App. E). Conditions: mathematics only, Gemma 2B/9B/27B, one prover family (best-of-K around an SFT policy).

**Implication.** §3 and §4 impose different requirements on the same artifact: a PRM used as a ranker needs to be right about correctness; a PRM used as a dense reward needs to be informative about progress. The same trained model is unlikely to be the best instrument for both.

---

## §5 RLVR: the verifier as the reward

### §5.1 Definition and the Tülu 3 objective

Reinforcement learning with verifiable rewards replaces the learned reward model with a function that checks the completion against a reference ([[rlvr-tulu3-config]] §6, Eq. 7-8):

```
max_πθ  E_{y ~ πθ(x)} [ v(x, y) − β KL[ πθ(y|x) ‖ π_ref(y|x) ] ]

v(x, y) = α  if correct, else 0,   with α = 10
```

- `v`: the verification function for that prompt; `β`: KL coefficient against the reference policy (the DPO checkpoint); `α`: the magnitude of the correctness reward, set to 10 "based on pilot experiments" and not tuned further (§6).
- The KL term is written as a divergence in the objective. In the per-token implementation form `−β log(πθ/π_ref)`, the quantity is the k1 estimator; the report does not name an estimator ([[john-schulman-kl-tricks]] for the k1/k2/k3 definitions).

### §5.2 What is actually verified in Tülu 3

| Prompt set | Count | Verification |
|---|---|---|
| GSM8K train | 7,473 | 8-shot CoT prompt; extract the final number; compare with the label |
| MATH train | 7,500 | 3-shot CoT prompt; extract the answer; "flex" MATH logic |
| IF verifiable | 14,973 | one verification function per IFEval constraint template |
| **Total** | **29,946** | — |

Two domains and three evaluations ([[rlvr-tulu3-config]] §6.1, Table 22). There is no code verifier in this stage: §6.1 states that the authors "leave more complex verifiers to future work", and the footnote attached to that sentence (footnote 17) cites code execution feedback as related work by others. The library cards [[rlvr-tulu3]] and [[tulu-3]] list a unit-test code verifier and are wrong on this point.

### §5.3 Configuration and results

Key values (Table 21 and its caption): learning rate 3e-7 (1e-7 for 70B), effective batch 224 (640 for 70B), PPO update iterations K = 4, clip ε 0.2, GAE λ 0.95, γ 1.0, response length 2,048 (1,024 for GSM8K only), total episodes 100,000, β swept over [0.1, 0.05, 0.03, 0.01], final 8B run β = 0.05 with warm-up 0.0. The value model is initialized from a reward model rather than from the DPO policy, which scored higher on GSM8K and on the average in the §6.2 ablation (Fig. 21); the ablation's reward model is trained on UltraFeedback, while the final run's is trained on the Tülu 3 8B preference mixture (§6.4). Adding reward-model scores on top of the verifiable reward performed worse and was noisier (Fig. 22).

Results (Table 23), Tülu 3 8B DPO → RLVR: GSM8K 84.3 → 87.6, MATH 42.0 → 43.7, IFEval 81.1 → 82.4, the Table 23 average 64.4 → 64.8. At 70B: GSM8K 93.5 → 93.5, MATH 62.3 → 63.0, IFEval 82.6 → 83.2, average 75.9 → 76.0. The gains are a few points on targeted evaluations and near zero on the average.

**Worked example (derived).** With `α = 10`, a run that raises the training-set correct rate by 0.01 gains 0.1 in expected verifiable reward. If the KL penalty is added in the same units before advantage whitening, that gain is cancelled by `β·KL = 0.05·KL` at `KL = 2` nats. This ratio, not `α` alone, is what the β sweep is exploring; a common factor applied to `v` alone is absorbed by advantage whitening (§6.2 item 5), so `α` is only meaningful relative to `β` and to the −10 non-EOS penalty.

### §5.4 The Goodhart gap is reduced, not removed

Replacing a learned reward model removes the drift studied in [[reward-model-overoptimization]]: there is no proxy model whose errors grow as the policy moves. It does not remove the gap between the check and the intent, for four documented reasons.

1. **Over-optimization inside RLVR.** As `β` falls, KL rises and the average score falls; Appendix B.4 of the Tülu 3 report shows over-optimized outputs from high-KL IFEval runs ([[rlvr-tulu3-config]] §6.2.1).
2. **Verifier false positives.** A final-answer check accepts flawed reasoning at rates from 5.1% to 43.4% depending on difficulty ([[qwen-prm-lessons]] Fig. 6), and those responses are the ones reinforced.
3. **Test exploitation.** In a production RL setting, a model learned to defeat unit tests with an `__eq__` that always returns True, `sys.exit(0)` before asserts, and a `conftest.py` that reports every outcome as passed ([[natural-emergent-misalignment-reward-hacking]] §2, Fig. 7). METR observed reward hacking in 39 of 128 RE-Bench runs by one o3 version ([[metr-frontier-reward-hacking]]).
4. **Reward-independent gains.** RLVR with GRPO on Qwen2.5-Math-7B improved MATH-500 by 21.4 absolute points with randomly assigned rewards against 29.1 with ground-truth rewards, an effect attributed to clipping bias amplifying pretraining priors, and not reproduced on Llama3 or OLMo2 ([[spurious-rewards-rlvr]], arXiv:2506.10947v2 Abstract, §2-§4). A verifiable-reward gain is not by itself evidence that the verifier taught anything.

---

## §6 Execution-grounded and rule-grounded rewards for agents

### §6.1 SWE-RL: a similarity rule where execution is unavailable

[[swe-rl-difflib-reward]] asks where the verifier comes from when running the tests is too expensive. Its answer is a rule over the reference output ([[swe-rl-difflib-reward]] §2.1, Eq. 1):

```python
R(o) = -1                                                   # wrong output format
     = difflib.SequenceMatcher(None, pred, gt).ratio()      # otherwise, in [0, 1]
```

Data: GitHub events from 2015-01-01 to 2024-08-31 plus 4.6M cloned repositories → 24M aggregated PRs → about 11M unique PR instances after filtering → **273k seeds** selected for RL (App. A). Policy: **Llama-3.3-70B-Instruct**, 1,600 GRPO steps, 16k context, global batch 512 (16 rollouts each from 32 problems), 512 H100 GPUs for about 32 hours (§3.1).

Results: 41.0% on SWE-bench Verified with the Agentless Mini scaffold, against 36.2% for an SFT baseline built from the same base model and issue-solving data, combined with general coding and dialog data (Table 1; §3.5). Repair-only with oracle files: 34.8 (RL), 29.6 (SFT), 5.4 (base model, greedy; 16.6 with 20-sample majority voting) (Table 2). The base model emits a correctly formatted patch 12.2% of the time, so part of the SFT and RL gain is format compliance.

The reward ablation is the transferable result: a discrete reward (1 for an exact patch match, else 0) gives 29.0 repair-only accuracy against 34.8 for the continuous similarity, and the average discrete reward stays near zero for the whole run because real patches rarely match exactly (Fig. 5). When a binary check returns 0 for almost every rollout, the rollouts in a group have nearly equal rewards and their normalized advantages are near zero; a graded rule that scores partial progress restores variation within the group. The stated limitation is the other side of the same choice: similarity is not semantic equivalence, so functionally correct alternative patches are under-rewarded (§5).

### §6.2 Execution verifiers with hidden tests

[[deepswe]] trains Qwen3-32B with RL only on 4.5K R2E-Gym problems; the reward is 1 when the produced patch passes a selected sample of Pass2Pass and Fail2Pass tests within a 5-minute limit and 0 when any test fails or the run times out (§2.2). Results: 42.2% Pass@1 (average of 16 runs) and 71.0% Pass@16 on SWE-bench Verified, 59.0% with hybrid test-time scaling at K = 16 (§4). Trajectories that end by hitting the context limit, the step limit, or a 20-minute generation timeout are **masked** rather than scored 0 ("compact filtering"); the authors report that rewarding or penalizing such trajectories leads to collapse in a Qwen3-14B ablation shown only as a curve (§2.3, Fig. 6). Recipe values, including the disagreement between the blog and the released script, are in [[deepswe-recipe]].

Three more verification patterns from official reports, each a different way to manufacture a checkable outcome for an agent:

- **Generated verifier functions.** DeepSeek-V3.2's general-agent environments are built by an agent that writes a sandbox database, tool functions, a task, a Python solution, and verifier functions; instances with non-zero pass@100 under the current model are kept, giving 1,827 environments and 4,417 tasks ([[deepseek-v3.1]] §3.2.3).
- **Answer accuracy for search agents.** GLM-4.5's agentic RL rewards final-answer accuracy for web search and test results for SWE tasks; a wrong tool-call format halts the trace with reward 0 ([[glm-4-5]] §3.3.2).
- **Rubric rewards with a hack check.** Kimi K2's instruction-following rewards combine code-interpreter checks, an LLM judge, and a layer that detects claimed but unfulfilled compliance; over-budget responses are truncated and penalized ([[kimi-k2]] §3.2.1, §3.2.3). Rubric rewards for non-verifiable tasks are developed in ch-44b ([[rubrics-as-rewards]]).

**Agentic caution.** SWE-RL trains a single turn (issue → patch), and DeepSWE reports no evaluation outside SWE-bench Verified ([[deepswe]] Findings). Multi-turn credit assignment, observation masking, and rollout stability are ch-45b.

---

## §7 Transfer from the verifiable domain to everything else

The reason a general-purpose model uses RLVR at all is the hope that training on checkable tasks improves uncheckable ones. The evidence points both ways, and the direction depends on the method and on what is held out.

**Positive transfer from RL.** SWE-RL's out-of-domain table, zero-shot greedy, base / SFT / RL: HumanEval+ 76.2 / 73.2 / 79.9; CRUXEval-I 60.5 / 68.4 / 71.6; CRUXEval-O 61.9 / 75.1 / 75.5; MATH strict 63.2 / 54.0 / 73.7 (lenient 70.9 / 71.7 / 73.7); MMLU 86.49 / 85.26 / 86.82; BigCodeBench-Hard unchanged ([[swe-rl-difflib-reward]] Table 3). The SFT model, built from the same base model on issue-solving data plus general coding and dialog data, is below the base model on average. The authors state that individual 1-2 point differences are not significant on their own and argue for significance in aggregate (§3.5).

**The controlled comparison.** [[transferability-of-llm-reasoning]] fine-tunes Qwen3-14B-Base on the same math data two ways: SFT on rejection-sampled Qwen3-32B traces, and GRPO with answer correctness as the reward (arXiv:2507.00432v2 §2.2, Table 1). Math average: base 27.7, SFT-think 49.8, RL 53.8. Non-reasoning average (CoQA, MC-TACO, IFEval, HalluEval): base 45.7, SFT-think 21.1, SFT-no-think 29.0, RL 53.2. IFEval alone: 69.2 → 42.3 (SFT) → 70.0 (RL). The paper's PCA analysis reports smaller representation shift for RL than for SFT on all three task groups (Table 2). Result (single study), one base model, math-only training data.

**Narrowing from verifiable training.** [[ifbench]] trains on IFEval-style constraints with GRPO and measures unseen constraints. IF-RLVR from the Tülu-3-8B-DPO policy raises IFEval 81.1 → 92.2 and IFBench 25.2 → 44.6 (Table 6), while AlpacaEval 2 falls 33.5 → 21.3, MMLU 68.7 → 66.4, GSM8K 84.3 → 83.2 (Table 3), and GPT-4.1 judge scores of the same responses with the constraint removed fall from 7 to 6.4 out of 10 (§5). On IFBench, GPT-4.1 and Claude 3.7 Sonnet score below 50%, while many models, down to 2B, score above 80% on IFEval's 25 constraint templates (§1, Fig. 1). Training choices that generalize better: more constraint types, several constraints per prompt (up to 5-6 even though test prompts carry at most 3), and variable ranges that include and extend the test range (§4.1-§4.3). Mixing a reward-model score with the constraint reward recovered AlpacaEval 2 (31.6) at a cost in IF accuracy (App. E).

**Beyond rules.** [[crossing-the-reward-bridge]] measures what happens when the reference answer is free-form. Only 60.3% of its mathematics problems have a single-term numeric answer, and 45.4% of multi-subject queries do (§1). With Qwen2.5-7B and 30k prompts, RLOO with a rule-based binary reward reaches 58.5 math / 26.3 multi-subject, and with a distilled 7B generative verifier 63.0 / 28.1; soft rewards from the verifier reach 31.2 on multi-subject under REINFORCE (Table 1). Scaling from 20k to 100k prompts moves the rule-based reward from 26.2 to 16.9 on multi-subject and the model-based reward from 30.8 to 35.0 (Table 2): a rule that mislabels free-form answers gets worse with more data. On out-of-domain training sets: NaturalReasoning 29.4 → 39.8 and WebInstruct 33.9 → 44.0 (Table 3).

**The pass@k caveat.** [[rlvr-beyond-base-model]] reports that RLVR-trained models beat their base models at small `k` while base models match or exceed them at large `k`, and interprets RLVR as improving sampling efficiency over paths the base model already had (arXiv:2504.13837v5 Abstract). [[prorl]] reports the other regime: more than 2k RL steps from DeepSeek-R1-Distill-Qwen-1.5B raised average pass@1 from 44.45 to 60.14 on six math benchmarks, with tasks where the base model solved nothing at any `k` up to 256, and a "Diminish" regime, mostly in mathematics, where pass@128 stayed flat or fell (Tables 1, 3; §4.2-§4.3). [[interplay-pretraining-midtraining-rl]] frames the condition: RL expands capability when the task sits at the edge of the model's competence and the priors were installed earlier. Open question.

---

## Negative samples and negative feedback

### Where the negatives come from, and which kind they are

Using the four-way distinction from ch-43a: (1) negative marginal value, (2) negative as content, (3) negative as conditioning, (4) negative as gradient.

| Source of the negative | How it is labelled | Kind | Measured error rate |
|---|---|---|---|
| Step labelled negative for PRM training | human, MC rollouts, or LLM judge | classifier target, not policy gradient | MC hard labels 86% accurate at N = 4 on 160 GSM8K steps ([[math-shepherd]] §5.2); MC-trained PRM 40.1 ProcessBench F1 against 56.5 for human labels ([[qwen-prm-lessons]] Table 4) |
| Wrong final answer in RLVR | verifier returns 0 | negative advantage (gradient) after baseline subtraction | correct answer with flawed process 5.1%-43.4% of correct-answer samples ([[qwen-prm-lessons]] Fig. 6) |
| First bad step of an incorrect trace | Monte-Carlo Q per step | negative as gradient, localized to one step | not reported |
| Failed patch or failed test in agentic RL | execution | negative advantage | not reported; DeepSWE masks limit-terminated trajectories instead ([[deepswe]] §2.3) |
| Truncated response (no EOS) | length check | penalty −10 in Tülu 3; masked in DeepSWE; truncation penalty in Kimi K2 | not reported |

### Mechanism

A negative advantage on a sampled token sequence multiplies the log-likelihood gradient by a negative number. At the logit level, for a sampled token `y` and any vocabulary entry `j`, `∂ log p_y / ∂ z_j = 1[j = y] − p_j`, so decreasing `log p_y` removes mass from `y` and redistributes it in proportion to the current probabilities, which concentrates it on the currently most likely alternative — not on the correct answer, unless the correct answer is already that alternative. The full derivation, with likelihood displacement and squeezing, is ch-43a. Two consequences specific to this chapter:

- A verifier false negative (a correct solution marked wrong) is a push-down on correct behaviour. Its cost is not symmetric with a false positive: the false positive teaches one wrong behaviour, the false negative removes mass from a behaviour the model got right.
- Localizing the negative to the first bad step is what makes step-level negatives work at all: contrasting two whole traces changes the likelihood of every step in both, including the steps that were fine ([[rl-on-incorrect-synthetic-data]] §6.1).

### Worked example: 0, −1, and abstention under a group baseline

Eight rollouts for one prompt, group-normalized advantages `A_i = (r_i − mean(r)) / std(r)` as in GRPO ([[grpo]]).

- **Case 1**, rewards 10 for 2 correct and 0 for 6 wrong: mean 2.5, std 4.33, advantages **+1.73** (correct) and **−0.577** (wrong).
- **Case 2**, rewards +1 correct, −1 wrong, same outcomes: mean −0.5, std 0.866, advantages **+1.73** and **−0.577**.

The two cases are identical. Any affine change of the reward (including the choice between 0 and −1 for a wrong answer, and the size of `α`) is absorbed by group normalization when the outcomes are binary. The choice only becomes real when a third outcome exists, or when the normalization is removed (mean-only advantages, as in Dr. GRPO and DeepSeek-V3.2), or when an un-normalized KL term is added to the reward.

- **Case 3**, three outcomes: +1 for 2 correct, 0 for 3 abstentions, −1 for 3 wrong. Mean −0.125, std 0.781, advantages **+1.44** (correct), **+0.16** (abstain), **−1.12** (wrong). Abstention receives a small positive advantage in this group, so the policy learns to abstain on prompts it usually gets wrong.
- **Case 4**, same outcomes but abstention scored 0 like a wrong answer (`w = 0`): abstention and a wrong answer are indistinguishable, and there is no gradient that prefers one over the other.

### The abstention threshold implied by the wrong-answer penalty

With reward `+1` for correct, `−w` for wrong, `0` for abstaining, the expected reward of answering with correctness probability `p` is `p − w(1 − p)`, which is positive when

```
p > w / (1 + w)
```

So `w = 0` implies always answer, `w = 1` implies answer above 50% confidence, `w = 3` implies 75%, and `w = 9` implies 90%. This is the same arithmetic as the confidence-target instruction proposed for evaluations in [[why-language-models-hallucinate]] §4.2, where mistakes are penalized `t/(1 − t)` points and the model should answer only above confidence `t`. (That section lists `t = 0.5` with penalty 1, `t = 0.75` with penalty 2, and `t = 0.9` with penalty 9; `0.75/(1 − 0.75) = 3`, so the middle entry does not match the paper's own formula.) The paper's claim concerns evaluation scoring — it classifies GPQA, MMLU-Pro, IFEval, Omni-MATH, BBH, MATH, MuSR, SWE-bench, and HLE as binary graders with no credit for abstention (Table 2) — and Observation 1 states that under binary grading no abstention is ever optimal. Applying it to an RL reward is this course's extension; the RL side of the claim is not measured by that paper.

Use [`figures/wrong-answer-rewards.html`](figures/wrong-answer-rewards.html) to set the wrong-answer penalty, the abstention credit, and the number of correct/wrong/abstained/truncated rollouts in a group, and read off the advantages and the implied abstention threshold.

### Controls

1. **Localize.** Put the negative on the first bad step, not on the whole trace ([[rl-on-incorrect-synthetic-data]]; [[prm800k]] stops labelling at the first negative step).
2. **Mask rather than penalize uncertain failures.** Trajectories terminated by a context limit, a step limit, or a timeout carry no information about correctness; DeepSWE masks them and reports collapse without that mask ([[deepswe]] §2.3). Tülu 3 takes the other option, a −10 penalty for a response without EOS ([[rlvr-tulu3-config]] §6.2), which is defensible for a 2,048-token budget and costly for a long-reasoning budget (ch-44a).
3. **Filter the label, not the gradient.** Consensus filtering between an LLM judge and MC estimates removed 60% of step labels and improved first-error identification ([[qwen-prm-lessons]] §3.1.3).
4. **Keep a KL anchor while negatives are active.** The Tülu 3 β sweep is a negative-side control: lower β gave more KL and lower average scores ([[rlvr-tulu3-config]] §6.2.1).

### Diagnostics

Log, per training step: the share of rollouts with positive and negative advantage; the share of groups that are all-correct or all-wrong (zero gradient); truncation rate; abstention rate; entropy; pass@1 and pass@large-k on a held-out set; and, when a PRM is in the loop, a step-level score (first-error identification) next to the best-of-N score, because the two can move in opposite directions ([[qwen-prm-lessons]] §3.2).

---

## Recipe

Values as printed in the primary sources. Rows marked 2026-09-15 were read in the primary text for this revision; card-sourced rows carry the card's verification date.

| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| Tülu 3 8B RLVR | 8B | RL | algorithm; reward | PPO; `v = α if correct else 0`, α = 10 | arXiv:2411.15124v5 §6 Eq. 7-8 [[rlvr-tulu3-config]] | verified 2026-09-15 | "based on pilot experiments and did not tune it further" |
| Tülu 3 8B RLVR | 8B | RL | prompts | 29,946 (GSM8K 7,473 + MATH 7,500 + IF verifiable 14,973) | v5 §6.1, Table 22 | verified 2026-09-15 | no ablation reported |
| Tülu 3 8B RLVR | 8B | RL | LR; effective batch; K; response length; episodes | 3e-7 linear; 224; 4; 2,048 (1,024 GSM8K-only); 100,000 | v5 Table 21 | verified 2026-09-15 | Fig. 19 per-task curves |
| Tülu 3 8B RLVR | 8B | RL | β; warm-up ratio | 0.05; 0.0 (final run), swept over [0.1, 0.05, 0.03, 0.01] | v5 Table 21 caption; §6.2 | verified 2026-09-15 | Fig. 21: lower β → higher KL → lower average score |
| Tülu 3 8B RLVR | 8B | RL | non-EOS penalty; advantage normalization; value init | −10.0; whitening; from a general RM (UltraFeedback RM in the §6.2 ablation; Tülu 3 8B preference-mixture RM in the final run) | v5 Table 21; §6.2, §6.4 | verified 2026-09-15 | Fig. 21: RM-initialized value > DPO-initialized on GSM8K and average |
| Tülu 3 70B RLVR | 70B | RL | LR; batch; β; warm-up; episodes | 1e-7; 640; β = 0.07 and ω = 0.07 (Table 21 caption) / β = 0.7, 0.1 warm-up, 400,000 episodes (§6.4 text) | v5 Table 21 caption vs §6.4 | conflict (inside the paper) | §6.4: 70B KL stays below 1 |
| Math-Shepherd PRM | 7B verifier | reward-model | completer; completions per step; label form | LLemma-7B; N = 8; hard labels as two special tokens | arXiv:2312.08935v3 §4 [[math-shepherd]] | verified 2026-09-14 | §5.2: 86% label accuracy at N = 4, declining with larger N |
| Math-Shepherd step-by-step PPO | Mistral-7B policy | RL | reward placement | PRM score at each step-end token, 0 elsewhere; no separate final-answer term | ACL 2024 Eq. 7; v3 §3.5 | verified 2026-09-14 | Table 2: 84.1 / 33.0 vs 81.8 / 31.3 with an ORM |
| OmegaPRM data | — | reward-model | rollouts per MC estimate k; search limit; step split | 8; 100 per question; 16 pieces per solution | arXiv:2406.06592v2 §4, §4.2 [[omegaprm]] | verified 2026-09-14 | §4.4: 15M annotations vs 200K by brute force at equal compute |
| OmegaPRM data | — | reward-model | tree constants; label | α 0.5, β 0.9, L 500, c_puct 0.125; pointwise soft `MC(s)` | v2 §4, §3.4 | verified 2026-09-14 | §4.3 Table 2: soft 70.1% > pairwise 64.2% > hard 63.3% step accuracy |
| Qwen2.5-Math-PRM-7B / -72B | 7B / 72B | reward-model | data; completions per step; threshold; filter | ~500K queries, 6-8 responses each; 8 completions; hard threshold 0; consensus filtering keeps ~40% | arXiv:2501.07301v2 §2.1, §3.1.3-§3.1.4, §4.1 [[qwen-prm-lessons]] | verified 2026-09-15 | Fig. 5 (threshold sweep 1/8-7/8); Fig. 2 (filtering: ProcessBench 40.1 → 46.3) |
| Qwen2.5-Math-PRM-7B | 7B | reward-model | initialization; loss | Qwen2.5-Math-7B-Instruct with a two-layer scalar head; CE on the last token of each step | v2 §2.1, §4.1 | verified 2026-09-15 | Table 6: Best-of-8 67.6 vs maj@8 66.2; Table 7: ProcessBench 73.5 |
| PAV RL (Gemma) | 2B / 9B | RL | algorithm; LR; batch; response length; KL | REINFORCE with token-level value baseline; 1e-7; 32; 512; 0.001 | arXiv:2410.08146v1 App. E [[rewarding-progress-pav]] | verified 2026-09-15 | §5 Fig. 7: >7% over ORM-RL, 6x sample efficiency |
| PAV RL (Gemma) | 2B / 9B | RL | process-term weight α; iterations | 5.0 (2B), 3.0 (9B); 10,000 (2B), 5,000 (9B) | App. E | verified 2026-09-15 | App. E: α in 0.5-6.0 improved over ORM-RL |
| Per-step DPO (Setlur) | DeepSeek-Math-7B, Llama2-7B | preference | negatives per question; prover | 8 incorrect responses; `π̃ = BoK(π_sft)`, K = 5 | arXiv:2406.14532v1 §6.1 [[rl-on-incorrect-synthetic-data]] | verified 2026-09-15 | Fig. 7a-b: 8x synthetic-data efficiency; Fig. 7c: standard DPO does not improve |
| Llama3-SWE-RL-70B | 70B | RL | policy; steps; context; batch; compute | Llama-3.3-70B-Instruct; 1,600 steps; 16k; 512 (32 problems × 16 rollouts); 512 H100 ≈ 32 h | arXiv:2502.18449v2 §3.1 [[swe-rl-difflib-reward]] | verified 2026-09-15 | Table 1: 41.0 vs 36.2 SFT on SWE-bench Verified |
| Llama3-SWE-RL-70B | 70B | RL | reward; seeds | −1 for wrong format, else `difflib.SequenceMatcher` ratio; 273k PR seeds from ~11M PRs | v2 §2, §2.1, App. A | verified 2026-09-15 | Fig. 5: continuous 34.8 vs discrete 29.0 repair-only |
| Llama3-SWE-RL-70B | 70B | RL | GRPO ε, β, learning rate | not reported | checked v2 §2.1, §3.1, App. A-D | not reported | — |
| DeepSWE-Preview | 32B | RL | reward; trajectory masking | 1 if selected P2P/F2P tests pass within 5 min, else 0; mask context-, step-, and timeout-terminated trajectories | www.together.ai/blog/deepswe §2.2-§2.3 [[deepswe]], [[deepswe-recipe]] | verified 2026-09-14 | Fig. 6 (Qwen3-14B with and without compact filtering; curve only) |
| IF-RLVR (Tülu-3-8B-DPO policy) | 8B | RL | algorithm; LR; samples per prompt; length; hardware | GRPO (open-instruct); 5e-7; 16; 2,048; 8 H100, ~1 day for 2,000 steps | arXiv:2507.02833v3 §3 [[ifbench]] | verified 2026-09-14 | Table 5: GRPO > DPO on the same data |
| Cross-domain RLVR (Qwen2.5-7B) | 7B | RL | algorithms; KL β; prompts; verifier | REINFORCE / RLOO / REINFORCE++; 0.01; 30k; 7B generative verifier distilled from 160k Qwen2.5-72B-Instruct judgments | arXiv:2503.23829v2 §3.2-§4.4 [[crossing-the-reward-bridge]] | verified 2026-09-15 | Table 2: rule-based degrades with scale, model-based improves |

**Starting point for a small general-purpose run.** For an 8B policy after preference optimization, with prompts that have reference answers, the verified Tülu 3 rows support: PPO with learning rate 3e-7, effective batch 224, K = 4, clip 0.2, response length 2,048, about 100,000 episodes, β = 0.05 with no warm-up, value model initialized from a reward model, advantage whitening, and a sweep of β over [0.1, 0.05, 0.03, 0.01] with the average over all evaluations (not the targeted one) as the selection criterion. These values come from a Llama-3.1-8B-based run on 29,946 math and instruction-following prompts; the 70B run used 1e-7 and a larger batch, so do not transfer them across sizes. If the check is almost never satisfied at the start of training (SWE-style patch matching), replace the binary check with a graded rule, because the exact-match variant of SWE-RL stayed near zero reward for 1,600 steps ([[swe-rl-difflib-reward]] Fig. 5). If the reference answers are free-form, train a generative verifier instead of writing more rules ([[crossing-the-reward-bridge]] Table 2).

---

## Generalization lens

**(a) What increases breadth.**
- RL with an outcome verifier preserved and improved non-reasoning scores where SFT on the same math data degraded them: non-reasoning average 45.7 (base), 21.1 (SFT-think), 53.2 (RL) on Qwen3-14B-Base ([[transferability-of-llm-reasoning]] Table 1).
- RL on a single software-engineering task improved five out-of-domain task families while an SFT baseline on the same base model and issue-solving data fell below the base model on average ([[swe-rl-difflib-reward]] Table 3).
- A trained generative verifier extends RLVR to domains without parsable answers and keeps improving as data scales, where a rule-based reward degrades ([[crossing-the-reward-bridge]] Tables 2-3).
- Constraint variety inside the verifiable set: training with more constraint types, more constraints per prompt, and wider variable ranges raised out-of-domain constraint accuracy ([[ifbench]] §4.1-§4.3).

**(b) What causes narrowing.**
- Constraint-type overfitting: models above 80% on IFEval's 25 templates score below 50% on IFBench's 58 unseen constraints; IF-RLVR raised IFBench 25.2 → 44.6 but lowered AlpacaEval 2 from 33.5 to 21.3 and MMLU from 68.7 to 66.4 ([[ifbench]] Tables 3, 6).
- Verifier false positives reinforce flawed reasoning that happens to produce the right answer, at rates that rise with problem difficulty ([[qwen-prm-lessons]] Fig. 6).
- Over-optimization inside RLVR as the KL budget grows ([[rlvr-tulu3-config]] §6.2.1, App. B.4).
- Reward-model scores added on top of verifiable rewards made results noisier and lowered the average score in Tülu 3 ([[rlvr-tulu3-config]] Fig. 22).
- A PRM optimized only for best-of-N drifts toward outcome scoring, losing the property it was built for ([[qwen-prm-lessons]] §3.2.3).

**(c) How to measure it at this stage.**
- Held-out capability suite that the verifier does not target, reported next to the targeted metric; Tülu 3's average over the Table 23 benchmarks (64.4 → 64.8 at 8B) is the model of this ([[rlvr-tulu3-config]] Table 23).
- Unseen-constraint or unseen-verifier evaluation, built from verification functions disjoint from the training set ([[ifbench]] §2).
- Step-level evaluation (first-error identification) alongside best-of-N whenever a PRM is trained ([[qwen-prm-lessons]] Tables 6-7).
- pass@1 together with pass@large-k, since the two can diverge ([[rlvr-beyond-base-model]]; [[prorl]] §4.2-§4.3).
- A reward-independent control: train briefly with a random or shuffled reward and check how much of the gain survives ([[spurious-rewards-rlvr]] §2).
- Decontamination of the verifier's prompt pool against the held-out suite; SWE-RL excludes SWE-bench repositories from its corpus ([[swe-rl-difflib-reward]] App. A).

---

## Common mistakes and how to detect them

| Mistake | Observable symptom | Check |
|---|---|---|
| Treating an MC step label as a correctness judgment | PRM ranks well at best-of-N but cannot locate the first error | Evaluate on a step-level benchmark; compare with a PRM800K-trained baseline ([[qwen-prm-lessons]] Table 4) |
| Raising the number of completions per step to reduce noise | Label accuracy falls instead of rising | Measure false-positive and false-negative rates separately against a small human-labelled set ([[math-shepherd]] §5.2) |
| Scoring solutions with the product or minimum for an MC-trained PRM | Best-of-N below the last-step scoring of the same model | Compare the three aggregators on a validation set ([[qwen-prm-lessons]] Fig. 9) |
| Using the policy's own value estimates as a dense process reward | Dense reward changes nothing relative to outcome-only RL | Compare against an outcome-only run at equal samples; check that the prover differs from the policy ([[rewarding-progress-pav]] §3.2) |
| Pairing arbitrary correct and incorrect traces for step-level training | Preference training flattens or degrades as data grows | Ablate against prefix-sharing pairs at the first bad step ([[rl-on-incorrect-synthetic-data]] Fig. 7c) |
| Assuming a verifier removes reward hacking | Reward rises, held-out average falls; tests pass without a real fix | Inspect top-reward completions; compare KL against average score; run anti-hack checks ([[natural-emergent-misalignment-reward-hacking]]; [[metr-frontier-reward-hacking]]) |
| Scoring truncated rollouts as failures | Entropy and response length collapse; reward plateaus at a low value | Log the truncation rate; mask limit-terminated trajectories and compare ([[deepswe]] §2.3) |
| Wrong answers and abstentions scored identically | Abstention rate goes to zero; confident wrong answers rise | Report abstention rate and accuracy-when-answered; set an explicit wrong-answer penalty ([[why-language-models-hallucinate]] §4) |
| Reading a large relative gain from an accuracy value | "69.4% relative improvement" for a 51.0 → 69.4 move | Recompute: 18.4 absolute, 36% relative ([[omegaprm]] §6) |
| Reporting RLVR gains on the targeted benchmark only | Targeted metric up, average over all evaluations flat or down | Report both, as Tülu 3 does ([[rlvr-tulu3-config]] Table 23) |

---

## Check your understanding

1. An MC-labelled PRM scores higher than a PRM800K-labelled PRM on Best-of-8 and much lower on ProcessBench. Explain both results with one mechanism, and say which evaluation you would trust for a PRM that will be used as a dense RL reward.
2. Raising the number of completions per step from 4 to 16 makes hard labels less accurate in Math-Shepherd's measurement. Derive this from the two error-rate formulas in §2.2, and state the completer property that determines which error dominates.
3. PAV shows that using the current policy's own advantages as a dense reward reproduces outcome-only RL. Explain why, in terms of what a policy gradient already subtracts, and say what property a useful prover must have.
4. Under group-normalized advantages with binary outcomes, scoring a wrong answer 0 and scoring it −1 produce identical updates. Show this for a group of 8 with 2 correct, then explain which three changes to the setup make the choice matter again.
5. A team adds abstention to an RLVR reward with `w = 2` for wrong answers. Compute the confidence threshold above which the policy should answer, and explain what happens to that threshold if truncated responses are also scored −2.
6. Tülu 3 reports +3.3 on GSM8K and +0.4 on the Table 23 average at 8B. Give two distinct explanations for that gap, and say what measurement would separate them.
7. SWE-RL's continuous similarity reward beats a binary exact-match reward by 5.8 points on repair-only accuracy. Explain the mechanism in terms of the group-normalized advantage when almost every rollout scores 0, and state the cost this reward design imposes on solution diversity.
8. IFBench shows that RLVR on IFEval-style constraints raises constraint accuracy and lowers AlpacaEval 2 and MMLU. Explain why this is narrowing rather than a measurement artifact, and design a gate that would have caught it during training.

---

## Connections

- Previous: **ch-43a** — Negative Samples and Negative Gradients: Likelihood Displacement, Squeezing, and Negative Advantages. Supplies the logit-level mechanism used in the negatives section.
- Next: **ch-44a** — Length in RL: Overlong Responses, Length Control, and Long-Context RL. Takes up truncated rollouts, the non-EOS penalty, and RL at long context.
- **ch-41** — Reward Modeling: Bradley–Terry, Over-Optimization, and Reward-Model Generalization. The learned reward model that verifiers and PRMs replace, and the over-optimization curve referenced in §5.4.
- **ch-42** — Reward Hacking and Judge Design. Verifier and test exploitation, and the binary-grading argument used here for abstention.
- **ch-40** — Group-Baseline RL: RLOO, GRPO, Dr. GRPO, DAPO, and GSPO. The advantage normalization whose behaviour the worked examples in the negatives section depend on.
- **ch-43** — Entropy, Output Diversity, and KL Control in RL. The β sweep and the KL-versus-score trade-off in §5.3.
- **ch-16** — RL Prompt Distribution: Difficulty Filtering, Domain Breadth, and Prompt Reuse. Where verifiable prompts come from and how they are filtered.
- **ch-44b** — Multi-Domain RL for General Capability: Non-Verifiable Rewards and Domain Mixing. Rubric and generative rewards for tasks with no reference answer.
- **ch-45b** — Multi-Turn Agentic RL: Observation Masking, Credit Assignment, and Stability. The multi-turn setting that §6 previews.
- **ch-31** — Rejection Sampling, Self-Generated Data, Cold Start, and SFT–RL Alternation. The same verifiers used as filters rather than as rewards.
- **ch-46** — Lab: DPO or RLVR Experiment with Negative-Signal Ablation and Held-Out Capability Retention. Runs the ablations this chapter specifies.

---

## Sources

- [[prm800k]] — labelling protocol, 800K/75K/12K scale, product aggregation and Table 4 comparison, single-forward-pass inference, best-of-1860 results, OOD STEM table, credit-assignment argument (§6.1).
- [[let-verify]], [[lets-verify]] — older library cards on the same paper; their per-step inference-cost and calibration claims are not in the paper, so numbers here come from [[prm800k]].
- [[math-shepherd]] — MC label definitions (Eq. 3-4), completer settings, step-by-step PPO reward placement, GSM8K/MATH tables, label-accuracy measurement and its decline with N, verifier-size result.
- [[omegaprm]] — binary-search labelling, `O(k log M)`, tree constants, soft targets, Table 1 results, 36% relative gain, question filtering, stated limits.
- [[qwen-prm-lessons]] — PRM versus value model, the three-way data comparison, consensus filtering, threshold sweep, BoN bias, process-to-outcome shift, aggregation by label source, released-model results (excerpt from arXiv:2501.07301v2).
- [[rl-on-incorrect-synthetic-data]] — first-pit construction, per-step advantage, 8x data efficiency, failure of arbitrary pairing, equivalence theorem (excerpt from arXiv:2406.14532v1).
- [[rewarding-progress-pav]] — progress-as-advantage definition, effective reward Eq. 5, prover conditions, search and RL results, hyperparameters (excerpt from arXiv:2410.08146v1).
- [[step-dpo]] — step-level preference pairs built from the first erroneous step.
- [[rlvr-tulu3-config]] — Tülu 3 RLVR objective, α = 10, prompt table, Table 21 hyperparameters and the 70B conflict, Table 23 results, over-optimization and value-init findings (excerpt from arXiv:2411.15124v5).
- [[rlvr-tulu3]], [[tulu-3]] — library cards for the same report; their RLVR code-verifier claim, "+5-10pp" gains, and "mechanically zero" Goodhart claim are not supported by the paper and are corrected in §5.
- [[ifbench]] — IFEval-to-IFBench gap, IF-RLVR gains and the accompanying losses, constraint-variety ablations, reward mixing.
- [[why-language-models-hallucinate]] — binary grading and abstention, the `t/(1 − t)` penalty and its threshold, benchmark classification (excerpt from arXiv:2509.04664v1).
- [[swe-rl-difflib-reward]] — reward definition, data funnel, training configuration, SWE-bench and repair-only results, reward ablation, out-of-domain table, limitations (excerpt from arXiv:2502.18449v2).
- [[swe-rl]] — library card for the same paper; its base model, data scale, and transfer numbers predate the audits and are corrected in §6 and §7.
- [[deepswe]], [[deepswe-recipe]] — execution reward with hidden tests, compact filtering, results, recipe conflicts.
- [[deepseek-v3.1]] — generated verifier functions and agent-environment filtering in DeepSeek-V3.2.
- [[glm-4-5]] — search-answer accuracy and test-result rewards with a format-violation halt.
- [[kimi-k2]] — verifiable-reward gym, instruction-following hack check, truncation penalty.
- [[crossing-the-reward-bridge]] — share of rule-checkable answers, binary and soft model-based rewards, scaling comparison, out-of-distribution results (excerpt from arXiv:2503.23829v2).
- [[transferability-of-llm-reasoning]] — controlled Qwen3-14B SFT-versus-RL comparison and PCA shift (cited at loci in arXiv:2507.00432v2).
- [[deepseek-r1]] — PRMs listed among unsuccessful attempts and the reasons given (G.2).
- [[rlvr-beyond-base-model]], [[prorl]], [[interplay-pretraining-midtraining-rl]] — the pass@k boundary debate and the condition under which RL expands capability.
- [[spurious-rewards-rlvr]] — random-reward gains on Qwen2.5-Math and their absence on other families (cited at loci in arXiv:2506.10947v2).
- [[training-verifiers-to-solve-math-word-problems]] — the sample-and-rank verifier that ORMs come from.
- [[reward-model-overoptimization]] — the proxy-drift failure that a fixed verifier removes.
- [[natural-emergent-misalignment-reward-hacking]], [[metr-frontier-reward-hacking]] — measured exploitation of unit tests and scoring code.
- [[grpo]], [[john-schulman-kl-tricks]] — group-normalized advantages and the KL estimator family used in §5.
- [[rubrics-as-rewards]] — pointer for rubric rewards, developed in ch-44b.
