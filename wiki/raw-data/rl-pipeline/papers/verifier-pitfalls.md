<!-- scope: Rule- vs model-based verifiers in math RL (arXiv:2505.22203; v3 title "From Accuracy to Robustness: A Study of Rule- and Model-based Verifiers in Mathematical Reasoning") — rule false negatives, model-verifier hacking during RL, hybrid verifier, discriminative vs generative
     deps: [[math-verify]]
     see-also: [[master-rm-one-token]], [[general-reasoner-verifier]], [[deepseek-grm]], [[reward-hacking-taxonomy]], [[judge-llm-bias]], [[rl-design-guide]]
-->

# From Accuracy to Robustness: A Study of Rule- and Model-based Verifiers in Mathematical Reasoning (listed in the course outline as "Pitfalls of Rule- and Model-based Verifiers")
- **Core Insight:** Rule verifiers are precise (about 0.99–1.00) but miss correct answers: recall averages 86% (about 14% of correct responses marked wrong) and falls as the policy gets stronger (Skywork-OR1 responses: 0.78 / 0.86 / 0.88 for VERL / Qwen-Math / HF Math-Verify), while a trained generative verifier used as the RL reward was hacked after about 450 iterations by single-symbol and gibberish outputs, ending in training collapse (§1, §3.2, Table 5, §5.2).
- **Guideline:** When the rule verifier's recall on your policy's outputs is below its precision, use a hybrid grader — rules first, a model verifier only on rule-rejected responses — because on Qwen2.5-7B GRPO this raised the average from 55.0 to 57.3 (Table 2), and choose a discriminative verifier over a generative one for that fallback, because xVerify-3B-I kept training reward within 0.004 of the GPT-4o oracle with no late collapse, while the generative R1-Distill-Verifier-1.5B reached 35.0% attack success under adversarial prefixes (Table 3, §6.2).
- **Authors:** Yuzhen Huang, Weihao Zeng, Xingshan Zeng, Qi Zhu, Junxian He
- **Year:** 2025 (arXiv:2505.22203; v3 read)
- **URL:** https://arxiv.org/abs/2505.22203
- **Source type:** paper
- **Relevant topics:** rule-based verifiers, false negatives, model-based verifiers, reward hacking, hybrid grading, adversarial robustness

## Summary
The paper measures three rule verifiers on responses from many models across math datasets, then evaluates model-based verifiers on the responses the rules rejected, then uses each as an RL reward for Qwen2.5-7B (GRPO, DeepScaleR data) and tracks training reward against a GPT-4o oracle. It ends with an adversarial-pattern test of verifiers.

## Key Contributions
- Recall of rule verifiers by dataset and by generator strength (Table 5, Figs. 2–3).
- Static precision/recall of model verifiers on rule-rejected responses (Table 1).
- RL with a hybrid verifier and with model verifiers; detection of verifier hacking against an oracle (Table 2, §5).
- Attack-success rates of hacking patterns per verifier (Table 3).

## Key Figures/Tables to Study
- Table 5 (rule recall), Table 1 (model verifiers), Table 2 (RL with hybrid), Fig. of training vs oracle reward (§5), Table 3 (attacks).

## Technical Details
**Rule recall (Table 5; VERL / Qwen-Math / HF).** Math 0.92 / 0.95 / 0.95; DeepScaleR 0.86 / 0.94 / 0.94; ORZ-Math 0.89 / 0.94 / 0.95; Skywork-OR1 0.78 / 0.86 / 0.88. Long-CoT R1-Distill generators average 0.92 recall. On WebInstruct-Verified (general science) HF recall is 47% (§4.3). Named failure types: equivalent forms in context (0.5π vs 90°), unit conversion (3 hours vs 180 minutes).

**Model verifiers on HF-rejected responses (Table 1, precision / recall).** DS-R1-Distill-Qwen-7B 0.79 / 0.59; Qwen2.5-7B 0.84 / 0.51; R1-Distill-Verifier-1.5B 0.73 / 0.62; xVerify-3B-I 0.90 / 0.78; general-verifier 0.90 / 0.86; random 0.17 / 0.44.

**Hacking (§5.2, App. L).** Dominant exploits: "Single Symbol" (e.g., `{`) and "Gibberish". With the trained verifier, Skywork-OR1 performance fell from 58.7 to 55.5 (App. I).

**Attacks (Table 3, adversarial prefixes).** R1-Distill-Verifier-1.5B 35.0 (base 21.7); general-verifier 22.1; xVerify-0.5B-I 0.0; xVerify-3B-I 0.2. Scale does not remove vulnerability: Qwen2.5-Math-72B 12.0% average vs Qwen2.5-72B 1.2%.

**Practical guidance (§7).** (1) Use rules alone only when both precision and recall are high. (2) Otherwise use a hybrid. (3) Prefer discriminative verifiers for robustness in the RL loop.

## Verification
- Read arXiv HTML v3, first 100K of 132K characters, on 2026-11-24. Other columns of Table 3 could not be mapped from the extracted text and are omitted.
- The course outline uses the earlier title; the slug keeps it.
- Policy results are for one model (Qwen2.5-7B) and one data mix; no variance reported in the portion read.
