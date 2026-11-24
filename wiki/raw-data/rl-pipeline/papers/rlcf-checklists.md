<!-- scope: RLCF, "Checklists Are Better Than Reward Models For Aligning Language Models" (arXiv:2507.18624, CMU / Apple) — per-instruction checklists generated from candidate responses, item scores by a 72B judge (+ verification programs), weighted into a reward for DPO pair selection
     deps: [[rubrics-as-rewards]], [[dpo]]
     see-also: [[deepseek-grm]], [[ifbench]], [[judge-llm-bias]], [[multichallenge]], [[rl-design-guide]]
-->

# Checklists Are Better Than Reward Models For Aligning Language Models (RLCF)
- **Core Insight:** Replacing a scalar reward model with a per-instruction checklist (weighted yes/no-style requirements scored 0–100 by a judge) to choose DPO pairs improved Qwen2.5-7B-Instruct on every instruction-following benchmark reported — InFoBench 78.1 → 84.1, FollowBench HSR 71.4 → 75.3, IFEval prompt-loose 75.0 → 77.3 — where Skywork and ArmoRM reward-model DPO lowered IFEval strict (Tables 2–3); the checklist reward is a worse general RM (RewardBench Safety 71.4 vs 93.0 for Skywork-27B, Table 5).
- **Guideline:** When the reward must check whether a response satisfies the specific requirements of each instruction, generate the checklist from the instruction plus responses of several model sizes (candidate-based), score each item with a judge averaged over samples, and add a verification program only for items the generator is confident can be checked exactly, because candidate-based checklists beat direct ones on IFEval (77.3 vs 74.3) and FollowBench (75.3 vs 72.8) (Table 6); budget for grading cost, since 25 judge samples on WildChat took 92 hours vs 40 hours for 5 on one 8×H100 node (§5.7, Fig. 4).
- **Authors:** Vijay Viswanathan, Yanchao Sun, Shuang Ma, Xiang Kong, Meng Cao, Graham Neubig, Tongshuang Wu (CMU, Apple)
- **Year:** 2025 (arXiv:2507.18624; v2 read, 1 Dec 2025)
- **URL:** https://arxiv.org/abs/2507.18624
- **Source type:** paper
- **Relevant topics:** checklist/rubric rewards, LLM-judge scoring, verification programs, instruction following, preference-pair construction, grader cost

## Summary
For 130,000 WildChat instructions (WildChecklists), Qwen2.5-72B-Instruct writes a checklist with 0–100 importance weights. Responses from the policy are sampled (T = 1.3, top-p 0.9); a Qwen2.5-72B-Instruct judge scores each item 0–100, averaged over 25 samples; a generated verification program's score is averaged in when available; item scores are combined by importance weights. Pairs with the largest per-criterion differences (top 40%) train DPO. The method is offline preference optimization, not online RL, but the grader design is the same as an online rubric reward.

## Key Contributions
- Direct vs candidate-based checklist generation (§2, Table 1: candidate preferred overall 51.2% vs 40.6% automatic; 56.0% vs 38.0% manual on 50 rows).
- Two universal requirements added to every checklist to curb long preambles, weight 100/100 (§2).
- Comparison against reward-model and AI-judge baselines on five benchmarks (Tables 2–4).

## Key Figures/Tables to Study
- Table 1 (checklist quality), Tables 2–4 (main results), Table 5 (RewardBench), Table 6 (direct vs candidate), Fig. 4 (grading cost), App. A Fig. 5 (pair filtering).

## Technical Details
**DPO setup.** 2 epochs, batch 1,024, max length 2,048, cosine LR 3e-6 → 2e-6, OpenRLHF, about 3 hours on one 8×H100 node.

**Baselines (InFoBench overall / FollowBench HSR).** Skywork 82.0 / 69.5; ArmoRM 83.5 / 70.4; UltraFeedback 80.0 / 72.6; AI Judge 76.1 / 70.3; RLCF 84.1 / 75.3.

**Chat benchmarks (Table 4).** Arena-Hard vanilla 51.3 → 54.6 (Skywork 55.1); AlpacaEval LC 36.2 → 37.1 (Skywork 41.5).

**Pair filtering (App. A, Fig. 5).** Keeping 60–80% of pairs performs similarly; discarding 90% causes a large drop.

**Retention (Table 9).** GSM8K 83.2 → 82.2; TruthfulQA MC1 43.5 → 42.0.

**Other models (Table 10).** Llama 3.1 8B Instruct FollowBench HSR 68.0 → 72.3; OLMo 2 7B Instruct IFEval prompt-loose 78.1 → 79.6.

## Verification
- Read arXiv HTML v2 on 2026-11-24.
- The intro's "6.9% relative" InFoBench gain does not match Table 2 (78.1 → 84.1 is about 7.7%); the card uses the table.
- The paper's §5.7 claim of "55% less clock time" with 5 samples does not match 40 vs 92 hours (about 57%); the card gives the hours.
- Verification-program effect is only reported for base-model rows (Arena-Hard 19.4 with vs 23.1 without, Table 4), which favors not using programs there; no instruct-model breakdown was found.
