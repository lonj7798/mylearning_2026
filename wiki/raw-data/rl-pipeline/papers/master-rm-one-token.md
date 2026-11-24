<!-- scope: "One Token to Fool LLM-as-a-Judge" (arXiv:2507.08794, Tencent AI Lab / Princeton) — master-key responses (":", "Thought process:", "Solution") give false positives in reference-based LLM judges; Master-RM trained with truncated-response negatives
     deps: [[judge-llm-bias]], [[generative-reward-models]]
     see-also: [[verifier-pitfalls]], [[general-reasoner-verifier]], [[deepseek-grm]], [[reward-hacking-taxonomy]], [[lilianweng-reward-hacking]], [[rl-design-guide]]
-->

# One Token to Fool LLM-as-a-Judge (Master-RM)
- **Core Insight:** Reference-based LLM judges that decide whether a response matches a reference answer accept content-free responses — a blank space, ".", ":", "Thought process:", "Solution", or 解 — at overall average / worst-case false-positive rates of 66.8 / 90.9% for Qwen2.5-72B, 80.6 / 95.1% for LLaMA3-70B, 14.6 / 53.6% for GPT-4o and 12.4 / 57.7% for Claude-4 (Table 1); the issue was found when an RLVR policy collapsed to responses under 30 tokens such as "Solution." that the Qwen2.5-72B judge rewarded (§1, Figs. 2–3).
- **Guideline:** When an LLM judge grades RL rollouts against a reference answer, add to its training set negatives made of reasoning openers with no answer (the authors kept only the first sentence of 20K regenerated responses, labeled NO), and before an RL run probe the judge with the master-key strings, because the resulting Master-RM-7B had 0.1 / 2.9% average / worst FPR with Cohen's κ = 0.91 against GPT-4o (Tables 1–2); do not expect CoT prompting or majority voting to fix it, because for Qwen2.5-7B CoT raised average FPR from 12.6 to 40.4 (Table 3a, column order inferred).
- **Authors:** Yulai Zhao, Haolin Liu, Dian Yu, Haitao Mi, Dong Yu, S.Y. Kung, Meijia Chen (Princeton, UVA, Tencent AI Lab, Rutgers)
- **Year:** 2025 (arXiv:2507.08794; v3 read)
- **URL:** https://arxiv.org/abs/2507.08794
- **Source type:** paper
- **Relevant topics:** LLM-as-judge, reward hacking, false positives, adversarial robustness, reward-model training data

## Summary
The paper tests general LLM judges (GPT-4o, GPT-o1, Claude-4, Qwen2.5-72B, LLaMA3-70B) and specialized verifiers (General-Verifier, Omni-Judge) on math and multi-subject benchmarks where the candidate response is replaced by a "master key". All are affected. The fix is data augmentation: truncated reasoning openers labeled as incorrect, mixed into a 160K multi-subject verification set, then SFT.

## Key Contributions
- Taxonomy of master keys: non-word symbols, reasoning openers, multilingual openers (§2, §3.1).
- FPR measurements across judges and benchmarks (Table 1).
- Master-RM-7B / 32B with near-zero FPR and judge agreement comparable to GPT-4o (Tables 1–2).
- Negative result for inference-time fixes (CoT, voting) and the question-removal test (Table 3).

## Key Figures/Tables to Study
- Fig. 2–3 (RL collapse), Table 1 (FPR), Table 2 (agreement, VerifyBench), Table 3 (inference-time strategies), Fig. 4 (FPR vs model size).

## Technical Details
**Selected FPRs (Table 1, "Overall Avg | Worst").** GPT-o1 6.0 | 37.3; General-Verifier 9.7 | 87.0 (66.8% on MATH with a single blank space); Omni-Judge 14.3 | 54.1; Master-RM-7B 0.1 | 2.9; Master-RM-32B 0.8 | 5.4.

**Master-RM data (§3.2).** 20K instances sampled from a 160K multi-subject RLVR set; responses regenerated with CoT by GPT-4o-mini; only the first sentence kept and labeled NO; combined into 180K; SFT of Qwen2.5-7B-Instruct and Qwen2.5-32B-Instruct.

**Agreement (Table 2).** Cohen's κ with GPT-4o: Master-RM-7B 0.91; Master-RM-32B 0.89 (0.87 with humans). VerifyBench accuracy: Master-RM-32B 95.15 vs GPT-o1 95.70; VerifyBench-Hard 86.80 vs 88.80.

**Question removal (Table 3b).** Removing the question from the judge prompt cut Qwen2.5-72B FPR on MATH from 78.7 | 88.6 to 2.8 | 6.8, but the authors note general reasoning often needs the question.

**Scale (Fig. 4).** FPR is non-monotonic in Qwen2.5-Instruct size: lowest around 7–14B, higher at 32B and 72B.

## Verification
- Read arXiv HTML v3, first 100K of 155K characters, on 2026-11-24.
- The text's "up to 35% FPR in GPT-4o" for punctuation could not be mapped to a Table 1 cell and is not used.
- Table 3a column order was inferred from layout by the reader; treat the CoT numbers as approximate.
