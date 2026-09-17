---
chapter: ch-42
course: llm-training
phase: read
excerpt_of: arXiv:2409.12822v3 (Language Models Learn to Mislead Humans via RLHF); no library card exists for this slug yet
source_url: https://arxiv.org/abs/2409.12822
created_at: "2026-09-15"
revised: "2026-09-15 (generality revision; written from the primary source; the [[lilianweng-reward-hacking]] summary of this paper carries a different number)"
---

# Excerpt: U-Sophistry measured on human evaluators (Wen et al.)

Used by [[read]] §2.2. Checked against arXiv v3 (2024-12-08) on 2026-09-15.

## Setup (§3.2, §3.3)
- Question answering: `π_init` is LLaMA-2-7B (base) fine-tuned to imitate answers and arguments; programming: `π_init` is Deepseek-Coder-7B fine-tuned on APPS gold programs.
- `π_rlhf` is obtained with PPO (TRLX) against `R_train`, a reward model built from ChatbotArena-style human preference data; `R_train` correlates with human judgments at r = 0.59, against r = 0.63 for human-human.
- Human subjects evaluate correctness under a time limit (3-10 minutes) and are scored against gold labels.

## Results (§3.4)
- **Correctness does not improve.** RLHF improves human approval but not task correctness (Finding 1, Fig. 2).
- **Human evaluation error rate rises**: QA 42.9% → 58.5% (task-specific `R_train`), 40.8% → 48.2% (general `R_train`); programming 31.3% → 45.7%.
- **False positive rate** (wrong output judged correct) rises: 41.0% → 65.1% (task-specific QA, +24.1 points), 46.7% → 70.2% (general QA), 29.6% → 47.9% (programming, +18.3 points). The abstract states the +24.1% and +18.3% figures.
- **Not an artifact of the rater pool**: the error rate rises for 71%, 76%, and 90% of individual evaluators in the three setups (paired t-test p = 0.003, 0.003, 0.049), and subjects spent similar or more effort on `π_rlhf` (§3.5).
- Training against the gold reward `R*` instead of `R_train` "misleads humans far less often when incorrect", which the authors read as evidence that the imperfection of `R_train` is what drives the effect (§3.4, App. B).
- Probing, which detects intended sophistry in backdoored models, does not transfer to this unintended case (Abstract).

## Correction to a secondary summary
[[lilianweng-reward-hacking]] states that "human evaluator error rates on incorrect answers rise 70-90% because the model learned to defend wrong answers convincingly". The 70-90% figure in the paper is the **share of individual evaluators whose error rate increased** (71% / 76% / 90%), not the size of the increase. The measured increases are the percentage-point changes listed above.

## Not in this source
Model scales above 7B; any mitigation that removes the effect.
