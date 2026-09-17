---
chapter: ch-35
course: llm-training
phase: read
excerpt_of: arXiv:2504.21318v1 (the library card phi-4 mixes Phi-4 and Phi-4-reasoning and has no Verification section; chapter-local verified extract)
source_url: https://arxiv.org/abs/2504.21318
created_at: "2026-09-15"
---

# Excerpt: Phi-4-reasoning Technical Report

- **Authors:** Microsoft (Marah Abdin, Sahaj Agarwal, Ahmed Awadallah, Vidhisha Balachandran, Harkirat Behl, Lingjiao Chen, et al.)
- **Year:** 2025 (arXiv v1 2025-04-30)
- **Source type:** official technical report
- **Used in:** ch-35 §6.3, Negative samples and negative feedback, Recipe, Generalization lens

## Models (§1)
Phi-4-reasoning: 14B, SFT of Phi-4 only. Phi-4-reasoning-plus: Phi-4-reasoning plus outcome-based RL on math.

## Teachable prompts (§2.1)
- Seeds come from web sources, existing datasets, licensed collections, and synthetic problems; responses for SFT are generated with o3-mini (§2).
- "We specifically target seeds situated at the edge of Phi-4's current abilities." Where no verifiable answer exists, "plurality responses from a strong reference model" serve as proxy ground truth, and difficulty is the agreement rate of weaker models (for example Phi-4 or GPT-4o) with it; rubric-based LLM evaluators score the number and complexity of reasoning steps.
- Some seeds are rewritten into verifiable forms (for example proofs into questions with a numeric answer, Figure 3).
- Decontamination list includes AIME-2024, MATH, GPQA, LiveCodeBench, Codeforces, OmniMATH, SWE-Bench Verified, SimpleQA, ArenaHard, MT-Bench; AIME-2025 was released after data finalization (§2.2).

## SFT (§3)
- Over 1.4 million prompt-response pairs, 8.3 billion unique tokens; about 16K steps, global batch 32, context 32K; AdamW, LR 1e-5, linear warm-up over 450 steps, weight decay 1e-4. The final model was trained for 16B tokens on the chosen mixture (§3.2).
- LR grid over [1e-6, 2e-5]; 1e-5 gave the best balance (§3.1).
- "While we have a relatively long SFT stage with 2+ passes over reasoning data sources, we do not see any catastrophic forgetting compared to the base Phi-4 model on more general capabilities" (§3).
- Teacher choice (§3.2): o3-mini with medium reasoning effort had "similar effect to DeepSeek-R1 when used as teachers, but o3-mini medium was more token efficient"; high effort was a stronger teacher "consistently across tasks" and produced longer responses; context was extended to 32k to fit them.

## Phi-4-reasoning-plus RL (§4)
- Seed pool 72,401 math problems; 64 problems per iteration; the selected checkpoint was trained "for 90 steps, over only ∼6k examples (and 8 trajectories of responses per example)".
- GRPO in verl: global batch 64 on 32 H100, LR 5e-8 with cosine warm-up over 10 steps, G = 8, β = 0.001, entropy coefficient 0.001, 32k maximum length.
- Reward: length-aware accuracy in [0.5, 1.0] for correct answers and [−1.0, −0.5] for incorrect answers (cosine scaling with L_max 31,744, L_pos_control 25,600, L_neg_control 3,702); −0.5 for a missing end token; −1.0 for an invalid think block; 5-gram repetition penalty; R_final = (8/13)·R_acc_scaled + (1/13)·R_rep (§4.1).
- "Additional GRPO training for only 90 steps boosts AIME performance by more than 10% (Figure 7a)"; further steps gave no additional gains (§4.2).

## Generality and length (§1)
- Both models improve by 30 to 60 percentage points over Phi-4 on TSP, 3SAT, and BA-Calendar, which were not targeted in training.
- Phi-4-reasoning-plus is 22 points above Phi-4 on IFEval, 16 on FlenQA, 10 on ArenaHard (Table 2).
- Phi-4-reasoning-plus "uses approximately 1.5× more tokens than Phi-4-reasoning on average".
- Two runs of average-of-5 evaluation can differ by up to 5–10 percentage points on AIME.

## Verification
- Checked on 2026-09-15 against https://arxiv.org/abs/2504.21318 (v1): §1, §2.1–2.2, §3, §3.1–3.2, §4.1–4.2.
- Not reported by the source: the share of each domain in the 1.4M set; the pass-rate thresholds for "teachable" seeds.
