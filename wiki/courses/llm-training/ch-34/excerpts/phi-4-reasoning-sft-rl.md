---
chapter: ch-34
course: llm-training
phase: read
excerpt_of: "Phi-4 Technical Report (arXiv:2412.08905v1) and Phi-4-reasoning Technical Report (arXiv:2504.21318v1)"
source_url: https://arxiv.org/abs/2504.21318
created_at: "2026-04-23"
revised: "2026-09-15 (rewritten against both primary texts; the earlier version attributed the 90-step GRPO run to Phi-4-reasoning instead of Phi-4-reasoning-plus and simplified the reward to +1/−0.5)"
---

# Excerpt: Phi-4 data, overfitting checks, and post-training; Phi-4-reasoning SFT and Phi-4-reasoning-plus RL

The library card [[phi-4]] has not been verified and combines both reports. Values below were read in the primary texts.

## Phi-4 (arXiv:2412.08905v1, 2024-12-12)
- **Pretraining (§3):** 14B; about 10T tokens; peak LR 0.0003; weight decay 0.1; global batch 5760; default context 4,096.
- **Phi-3 phases as described here (§3.1):** phase 1 "largely of filtered web data"; phase 2 "primarily synthetic tokens
  and a much smaller allocation for ultra-filtered and reasoning-heavy web data".
- **Synthetic data (§2.2):** "a total of about 400B unweighted tokens" of synthetic data for pretraining and midtraining.
- **Observations (§3.1):** more epochs over synthetic data beat fresh web tokens on reasoning-heavy benchmarks; "Models
  trained only with synthetic data underperformed on the knowledge-heavy benchmarks and demonstrated increased
  hallucinations." Table 3 (13B ablation models, relative to phi-3-medium): synthetic only MMLU +0.8, MATH +4.9,
  HumanEval +12.1, TQA −14.8; synthetic + web rewrites TQA −7.7.
- **Data mixture (§3.2):** ablations at 1T tokens and 7B scale; "the only benchmark that shows a clear benefit from web
  data is TQA"; synthetic-heavy mixes were "marginally better" but knowledge-heavy web data was added "to improve
  knowledge benchmarks".
- **Pretraining benchmarks (§3, Table 2; change relative to phi-3-medium):** phi-4 (4k) MMLU +3.0, MMLU-pro +10.3, GSM8k
  +2.2, HumanEval +7.8, ARCC +1.1, MBPP +6.8, MATH +8.9, TQA −0.7; phi-4 (16k) +2.7, +8.9, +1.2, +9.0, +0.9, +9.6, +8.4, −1.5.
- **Midtraining (§3.3):** 4K → 16K; 30% newly curated long-context data and 70% recall tokens from pretraining; RoPE base
  250K; maximum LR divided by 10; 250B tokens.
- **SFT (§4.1):** LR 1e-6; "around 8B tokens"; math, coding, reasoning, conversation, identity, safety, and 40 languages.
- **DPO (§4.2-4.3):** round 1 with Pivotal Token Search (PTS) pairs; round 2 "judge-guided DPO" with about 850k pairs
  labeled by GPT-4o. PTS keeps questions with 0.2 ≤ p(success) ≤ 0.8 and turns a pivotal token into a pair of single
  tokens t_acc and t_rej that increase or decrease p(success | prefix) (§4.3).
- **Table 9 (SFT / DPO stage 1 / DPO stage 2 only / phi-4 = stage 1 + 2):** MMLU 82.8 / 84.8 / 84.2 / 84.8; GPQA 47.3 /
  53.6 / 52.4 / 56.1; MATH 77.1 / 80.5 / 77.6 / 80.4; HumanEval 79.5 / 81.6 / 81.5 / 82.6; MGSM 80.8 / 80.8 / 81.5 / 80.6;
  SimpleQA 3.7 / 2.9 / 2.9 / 3.0; DROP 82.8 / 86.1 / 71.8 / 75.5; MMLUPro 61.9 / 70.0 / 67.2 / 70.4; HumanEval+ 77.9 /
  81.9 / 81.4 / 82.8; ArenaHard 56.7 / 66.5 / 69.8 / 75.4; IFEval 66.2 / 63.0 / 63.0 / 63.0; PhiBench (internal) 48.2 /
  54.5 / 53.0 / 56.2. Each DPO stage includes 1-5% hallucination and safety data.
- **Refusal data (App. A.1):** SFT uses (question, correct answer) where base phi-4 was usually correct, (question,
  refusal) where it was usually wrong, and (bogus question, refusal); DPO uses correct > refusal where phi-4 sometimes
  answered correctly and refusal > wrong where it sometimes answered incorrectly.
- **Hallucination (§4.4, Fig. 6):** SimpleQA answers go from 90.0% incorrect (base) to 38.7% incorrect and 57.5% not
  attempted (SFT) to 15.8% incorrect, 81.1% not attempted, 3.0% correct (final). The authors state the simple-evals F1
  gives the base model a higher score than the final model.
- **Fresh evaluation (§1.1, App. C):** November 2024 AMC-10/12, 78 questions released on or after November 6, 2024,
  after all training data were collected; 10 generations per question at temperature 0.5 (App. C; the Fig. 1 caption instead says "100 runs"); phi-4 averages
  91.8 of 150 (Fig. 1 bar label). Footnote 8: all three final candidate models scored above 89; the final model was chosen "before measuring its
  score but after seeing the scores for the other two candidates".
- **Weaknesses (§8):** factual hallucination (for example invented biographies), weak strict instruction following,
  long answers to simple problems, tuned for single-turn queries.

## Phi-4-reasoning and Phi-4-reasoning-plus (arXiv:2504.21318v1, 2025-04-30)
- **Models (§1):** Phi-4-reasoning is SFT on Phi-4; Phi-4-reasoning-plus adds RL on "a small set of ∼6K high-quality
  math-focused problems".
- **Prompt selection (§2.1):** seeds "at the edge of Phi-4's current abilities"; where no ground truth exists, plurality
  answers of a strong reference model are the proxy, and difficulty is estimated from "the agreement rate of weaker
  model's (e.g., Phi-4 or GPT-4o) generations"; rubric-based LLM evaluators score reasoning complexity.
- **Decontamination (§2.2):** against the following list (GPQA appears twice; 23 distinct names): AIME-2024, MATH, GPQA, LiveCodeBench, Codeforces, OmniMATH, SWE-Bench Verified,
  SimpleQA, DROP, AGIEval, ARC-C/E, CommonsenseQA, GSM8k, HellaSwag, HumanEval, MBPP, OpenBookQA, PIQA, WinoGrande,
  ArenaHard, MT-Bench, PhiBench. AIME-2025 was released after the data were finalized.
- **SFT (§3, §3.1-3.2):** two placeholder tokens repurposed as `<think>` and `</think>`; RoPE base doubled; 32K maximum
  length. "over 1.4 million prompt-response pairs, totaling 8.3 billion unique tokens"; "roughly 16K steps, with a global
  batch size of 32 and a context length of 32K tokens"; AdamW, LR 1e-5, linear warmup over 450 steps, weight decay 1e-4;
  "2+ passes over reasoning data sources"; final model "trained for 16B tokens". LR grid [1e-6, 2e-5]; 1e-5 best.
  Mixture weights are epochs per data cluster, tuned per domain and combined additively. o3-mini high effort was a stronger
  teacher than medium; o3-mini medium had "similar effect to DeepSeek-R1" and was more token-efficient.
- **Transfer (§1, §3, Table 2):** 30 to 60 percentage-point gains over Phi-4 on TSP, 3SAT, and BA-Calendar, described
  as domains "not directly targeted during supervised fine-tuning or reinforcement learning". Table 2 (Phi-4 →
  Phi-4-reasoning → plus; reasoning models at temperature 0.8, Phi-4 at 0.0): FlenQA 3K 82.0 → 97.7 → 97.9; IFEval Strict
  62.3 → 83.4 → 84.9; ArenaHard 68.1 → 73.3 → 79.0; HumanEvalPlus 83.5 → 92.9 → 92.3; MMLUPro 71.5 → 74.3 → 76.0; Kitab
  no-context precision 19.3 → 23.2 → 27.6; Kitab with-context precision 88.5 → 93.8 → 93.6; Kitab no-context recall 8.2 →
  4.9 → 6.3; Kitab with-context recall 68.1 → 74.8 → 75.4; Toxigen toxic 72.6 → 86.7 → 77.3; Toxigen neutral 90.0 → 84.7 →
  90.5; PhiBench 2.21 58.2 → 70.6 → 74.2.
- **Variance (§1, §5.1, Table 1):** two average-of-5 runs "can differ significantly (by up to 5-10 percentage points on
  AIME)"; AIME 2025 reported as pass@1 over 50 runs. AIME 25: Phi-4-reasoning 63.1 (std 6.3), plus 78.0 (4.6); AIME 24:
  74.6 (5.1), 81.3 (1.8).
- **RL reward (§4.1):** correct answers: R ∈ [0.5, 1.0] decreasing with length via ρ+ and cosine scaling; incorrect
  answers: R ∈ [−1.0, −0.5], increasing toward −0.5 as length grows via ρ−; missing `<|im_end|>` → −0.5; invalid thinking
  block → −1.0; repetition penalty from 5-gram frequencies; R_final = (8/13)·R_acc + (1/13)·R_rep (footnote 3: maximum
  8/13 ≈ 0.62).
- **RL settings (§4, §4.2):** 72,401 math seed problems, 64 sampled per iteration; verl; global batch 64 on 32 H100; LR
  5e-8 with cosine warmup over 10 steps; G = 8; KL β = 0.001 (the printed objective writes the KL term as
  D_KL(π_θ ‖ π_θold)); entropy coefficient 0.001; 32K maximum length with outputs
  clipped at 31K. Checkpoint: "the model with the best observed AIME 2024 score, which is the model trained for 90 steps,
  over only ∼6k examples". GRPO adds more than 10% AIME; further steps did not add gains (Fig. 7a).

## Verification
- Read on 2026-09-15 in the cached full texts of arXiv:2412.08905v1 (§1.1, §2.2, §3-§4.5, §8, App. C, Tables 1, 3, 9,
  Fig. 1, 6) and arXiv:2504.21318v1 (§1-§5.1, Tables 1-2).
