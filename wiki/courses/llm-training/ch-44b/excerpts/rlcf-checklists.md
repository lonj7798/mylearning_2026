---
chapter: ch-44b
course: llm-training
phase: read
excerpt_of: arXiv:2507.18624v2 (no library card at the time of writing; chapter-local verified extract)
source_url: https://arxiv.org/abs/2507.18624
created_at: "2026-09-15"
---

# Excerpt: Checklists Are Better Than Reward Models For Aligning Language Models (RLCF)

- **Authors:** Vijay Viswanathan, Yanchao Sun, Shuang Ma, Xiang Kong, Meng Cao, Graham Neubig, Tongshuang Wu (CMU; Apple)
- **Year:** 2025 (arXiv v1 2025-07; v2 2025-12-01; NeurIPS 2025)
- **Source type:** paper (code and data released)
- **Used in:** [[read]] §2.3, Negative samples and negative feedback, Generalization lens

## Method (§2–§3)
1. Qwen2.5-72B-Instruct generates a checklist of yes/no requirements for each instruction, conditioned on candidate responses from Qwen2.5-0.5B/1.5B/3B/7B ("candidate-based" generation); each item carries an importance weight out of 100. 130,000 WildChat instructions become the WildChecklists dataset.
2. Two "universal requirements" are appended to every checklist (directness and context-appropriate tone, total weight 100/100) after the authors observed responses opening with long preambles, which they read as reward hacking.
3. Each response is graded per item by an LM judge (mean of 25 sampled scores in 0–100 from Qwen2.5-72B-Instruct) and, where the model is confident it can check the requirement exactly, by a generated verification program; the two scores are averaged.
4. Item scores are averaged with the importance weights. Only the 40% of response pairs with the largest difference on at least one criterion are kept, and those pairs train DPO (2 epochs, batch 1024, max length 2048, cosine LR 3e-6 → 2e-6, OpenRLHF, one 8×H100 node, about 3 hours per model).

The method is preference-based; the authors list policy-gradient use of checklist feedback as future work (§7 Limitations).

## Results on Qwen2.5-7B-Instruct
- IFEval and InFoBench (Table 2): baseline IFEval average 77.3, RLCF 78.6, DPO via Skywork-Reward-Gemma-2-27B 76.0, DPO via ArmoRM 76.0; InFoBench overall 78.1 baseline, 84.1 RLCF, 82.0 Skywork, 83.5 ArmoRM.
- FollowBench (Table 3): average hard satisfaction rate 71.4 baseline, 75.3 RLCF, 69.5 Skywork, 70.4 ArmoRM; CSL 3.05 → 3.30 (RLCF), 2.88 (Skywork).
- Arena-Hard and AlpacaEval (Table 4): Arena-Hard vanilla 51.3 baseline, 54.6 RLCF, 55.1 Skywork; AlpacaEval vanilla 33.5 baseline, 36.2 RLCF, 44.8 Skywork.
- RewardBench as a judge (Table 5): checklist-based reward 90.0 Chat, 80.7 Chat Hard, 71.4 Safety, 88.5 Reasoning; Skywork-27B 96.1 / 89.9 / 93.0 / 98.1.
- Non-target tasks after RLCF (Table 9): XSTest unsafe 83.0 → 81.0, XSTest overall 86.0 → 86.9, GSM8K 83.2 → 82.2, TruthfulQA MC1 43.5 → 42.0, MC2 60.4 → 59.0. The authors attribute this to training prompts that "focus primarily on daily assistance and writing".
- Judge cost (§5.7, Fig. 4): grading with 3, 5, 10, 25 samples took 32, 40, 72, 92 hours on one 8×H100 node; 5 samples retain most of the effect, and fewer samples hurt the FollowBench "content" and "situation" categories.

## Verification
- Checked on 2026-09-15 against https://arxiv.org/abs/2507.18624 (v2, 2025-12-01): Abstract, §1–§5.7, §7, Tables 1–5, 9.
- Not reported: results with an RL algorithm other than DPO; checklist quality on non-English prompts (non-English conversations are filtered out).
