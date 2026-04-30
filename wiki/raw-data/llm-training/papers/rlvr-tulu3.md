<!-- scope: RL with verifiable rewards (RLVR) as a Tülu-3 post-training stage
     deps: [[prm800k]]
     see-also: [[deepseek-r1]], [[reward-model-overoptimization]]
-->

# Tülu 3: Pushing Frontiers in Open Language Model Post-Training (RLVR)
- **Core Insight:** When a task has a programmatic ground-truth check (exact-match math, unit-test-passing code, regex-matched instruction following), replace the learned reward model with a deterministic verifier — this collapses the proxy/gold gap that causes reward hacking in classical RLHF.
- **Guideline:** For every prompt, define a per-example verifier that returns binary {0,1}; use that as the reward in PPO/GRPO with a small KL penalty to the SFT reference; mix RLVR-capable prompts with RLHF-style prompts to get broad coverage while keeping the no-hacking guarantee on verifiable tasks.
- **Authors:** Nathan Lambert, Jacob Morrison, Valentina Pyatkin, Shengyi Huang, Hamish Ivison, et al. (Allen AI)
- **Year:** 2024
- **URL:** https://arxiv.org/abs/2411.15124
- **Relevant topics:** RLVR, verifiable rewards, math grader, IFEval-style constraint rewards, GSM8K exact match, PPO, DPO, Tülu 3 recipe

## Abstract
Tülu 3 is a family of open post-trained Llama-3.1 models (8B and 70B) whose final stage is Reinforcement Learning with Verifiable Rewards (RLVR): standard PPO where the reward signal is a programmatic verifier rather than a learned preference model. The paper releases data, code, RMs, verifiers, and evaluation harness. Tülu 3 8B/70B outperform the Llama-3.1 Instruct baselines and close a large part of the gap to top closed models on MATH, GSM8K, IFEval, and BBH.

## Key Contributions
- **Formal RLVR setup:** for a prompt `x` paired with a verifier `v: (x, y) → {0, 1}`, the reward is simply `r(x, y) = v(x, y)` — no RM.
- **Three verifier domains used in Tülu 3:**
  - *Math:* extract the final numeric/symbolic answer and compare to the reference using a tolerant grader (SymPy / normalized string match on MATH, exact integer match on GSM8K).
  - *Constrained instruction following:* IFEval-style constraints ("respond in JSON", "use exactly 3 bullet points") checked with regex / parsers.
  - *Code:* run model-generated code against unit tests in a sandbox; reward = 1 iff all tests pass.
- **Why it sidesteps reward hacking:** the verifier is a fixed, interpretable function. There is no proxy RM to drift; there is no OOD region where the reward spuriously rises. Goodhart's gap (see **[[reward-model-overoptimization]]**) is mechanically zero on verifiable prompts.
- **KL control still used:** standard KL-to-SFT penalty (per-token, added to reward) with a small β, otherwise the policy collapses to a degenerate high-reward mode like constant-answer-guessing.
- **Prompt curation:** only prompts with a verifier + a known reference answer enter the RLVR set; RLHF/DPO handles the rest.
- **Numbers (Tülu 3 8B vs Llama-3.1 8B Instruct):** GSM8K 87.6 vs 84.7, MATH 43.7 vs 41.5, IFEval 82.4 vs 80.5 — gains cleanly attributable to RLVR stages.

## Key Figures/Tables to Study
- **Fig. 1 / Table 1** (overall benchmark comparison) — headline Tülu 3 vs baselines.
- **RLVR ablation table** (with vs without RLVR stage) — isolates its contribution.
- **Pipeline diagram** showing SFT → DPO → RLVR ordering.
- **Verifier coverage table** — which task suites have verifiers vs which are RLHF-only.

## Technical Details
- **PPO settings:** group/batch ~128 prompts, 4 rollouts each, length up to 2k–4k tokens, lr ~1e-6, β_KL ~0.04, clip range 0.2.
- **Reward = 0/1 binary;** advantage estimation via GAE with λ = 0.95, γ = 1.0 (episodic).
- **Verifier cost:** code verifier requires sandbox (isolate-style, per-rollout timeout 5s); math verifier is pure Python and effectively free.
- **Mixing strategy:** when RLVR prompts are trained alongside RLHF/DPO prompts, a separate reward pipeline is used for each, but the same policy network is updated.
- **Failure mode to watch:** if the verifier has loopholes (string-match math graders that accept "42" inside prose), RLVR can hack those loopholes. Treat verifier engineering like unit-test engineering.

## Connections
- Real open-source recipe for the idea made famous by **[[deepseek-r1]]** (pure RL from a rule-based verifier).
- Complements process-reward-model training (**[[prm800k]]**, **[[math-shepherd]]**): RLVR uses only the outcome signal; PRMs give per-step signal — both can coexist.
- The no-RM property sidesteps all of **[[reward-model-overoptimization]]** on verifiable prompts.
- Canonical citation for "verifiable rewards" as a class.
