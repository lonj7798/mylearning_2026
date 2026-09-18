<!-- scope: Kimi k1.5 technical report (arXiv:2501.12599): long-context RL (128k) with an online mirror-descent variant, RL prompt curation, length penalty, partial rollouts, CoT reward model for math, long2short methods
     deps: [[rlhf-instructgpt]], [[dpo]]
     see-also: [[kimi-k1-5-recipe]], [[kimi-k2]], [[deepseek-r1]], [[rloo]], [[replay-buffer-rlhf]]
-->

# Kimi k1.5: Scaling Reinforcement Learning with LLMs
- **Core Insight:** Kimi k1.5, trained with RL at a 128k context window using a variant of online policy mirror descent and without Monte Carlo tree search, value functions, or process reward models, reports 77.5 pass@1 on AIME 2024 and 96.2 EM on MATH-500 (Abstract, Table 2); its long2short RL variant reaches 60.8 pass@1 on AIME 2024 with 3,272 tokens per response on average (§3.4).
- **Guideline:** When a long-CoT model must serve under a limited token budget, compare long2short methods on accuracy per token, because in this report long2short RL (length penalty plus a reduced maximum rollout length) had the highest token efficiency among long2short RL, DPO, model merging, and shortest rejection sampling (§2.4, §3.4, Figure 7).
- **Authors:** Kimi Team (Moonshot AI). App. A lists contributors alphabetically by first name: Angang Du, Bofei Gao, Bowei Xing, Changjiu Jiang, Cheng Chen, Cheng Li, et al.
- **Year:** 2025 (arXiv v1 2025-01; v4 2025-06)
- **URL:** https://arxiv.org/abs/2501.12599
- **Source type:** official technical report
- **Relevant topics:** long-CoT RL, online policy mirror descent, RL prompt curation, reward hacking, length penalty, curriculum and prioritized sampling, CoT reward model, partial rollouts, long2short (merging, shortest rejection sampling, DPO, RL), multimodal RL

## Abstract
The report describes the training of Kimi k1.5, a multimodal LLM trained with reinforcement learning (RL). The authors identify two main ingredients: scaling the RL context length and improved policy optimization. Together they form an RL framework that does not use Monte Carlo tree search, value functions, or process reward models. The long-CoT model reports 77.5 on AIME, 96.2 on MATH 500, the 94th percentile on Codeforces, and 74.9 on MathVista, which the authors describe as matching OpenAI o1. The report also presents long2short methods that use long-CoT techniques to improve short-CoT models; the short-CoT model reports 60.8 on AIME, 94.6 on MATH-500, and 47.3 on LiveCodeBench, and the authors state it outperforms GPT-4o and Claude Sonnet 3.5 by up to +550% (Abstract).

## Key Contributions
- RL context scaled to 128k, with partial rollouts that continue unfinished trajectories from a replay buffer in the next iteration (§1, §2.6.2).
- A long-CoT RL formulation optimized with a variant of online policy mirror descent: a KL-regularized objective against the previous iterate, a mean-reward baseline, and no value network (§2.3.2).
- RL prompt curation rules: model-estimated difficulty, removal of question formats with easily guessed answers, and removal of easy-to-hack prompts (§2.1).
- A length reward, curriculum sampling, and prioritized sampling (§2.3.3, §2.3.4).
- Four long2short methods: weight-averaging merge, shortest rejection sampling, DPO, and long2short RL (§2.4).
- Infrastructure: synchronous iterative RL system, hybrid colocated Megatron/vLLM deployment, and a code sandbox (§2.6).

## Key Figures/Tables to Study
- Tables 2 and 3: long-CoT and short-CoT results against o1, GPT-4o, Claude 3.5 Sonnet, DeepSeek V3, and others.
- Figures 5 and 6: training accuracy and response length over RL iterations; accuracy against mean token length (Figure 5 caption: an internal long-CoT model smaller than k1.5).
- Figure 7: long2short accuracy against token length on MATH-500 and AIME 2024.
- Figure 8 (model size vs response length), Figure 9 (curriculum vs uniform sampling), Figure 10 (comparison with ReST).

## Technical Details
Recipe values with loci are in [[kimi-k1-5-recipe]]. This section summarizes mechanisms.

**Training stages.** Pretraining, vanilla SFT, long-CoT SFT, then RL (§2). Pretraining has three stages: vision-language pretraining, cooldown with curated and synthetic data, and long-context activation to 131,072 tokens (§2.5.1, App. B.4).

**RL prompt set.** Prompts come from STEM, competitions, and general reasoning, text and image-text, balanced with a domain tagging system (§2.1). Difficulty is the pass rate of an SFT model that answers each prompt ten times at a "relatively high" sampling temperature (§2.1). Multiple-choice, true/false, and proof-based questions are excluded because a correct final answer can be reached by incorrect reasoning (§2.1). A prompt is removed as easy-to-hack if a model without CoT guesses the correct answer within N = 8 attempts (§2.1).

**Long-CoT SFT.** A small warm-up dataset of verified long-CoT paths for text and image inputs is built by prompt engineering, targeting planning, evaluation, reflection, and exploration (§2.2). Its size is not reported.

**Reward.** r(x, y, y*) ∈ {0, 1}; code uses test cases, and free-form answers use a trained reward model (§2.3.1). For math, a value-head "Classic RM" (about 800k data points, InstructGPT-inspired) and a "Chain-of-Thought RM" (about 800k CoT-labeled examples, outputs step-by-step reasoning then a JSON correctness judgment) were built; in manual spot checks their accuracies were about 84.4 and 98.5, and RL used the CoT RM (§2.3.5). For coding, test cases are generated with CYaRon and filtered against ground-truth submissions; 323 of 1,000 sampled contest problems entered the training set (§2.3.5).

**Policy optimization.** At iteration i the objective is max_θ E[r(x, y, y*)] − τ KL(π_θ(x) ‖ π_θi(x)) (Eq. 2). θ: policy parameters; π_θi: the current model used as reference; τ > 0: regularization strength (value not reported). The closed-form optimum gives a squared-error surrogate; τ log Z is replaced by the mean sampled reward r̄. For each problem, k responses are sampled from π_θi and the gradient is (1/k) Σ_j [∇ log π_θ(y_j, z_j|x)(r_j − r̄) − (τ/2) ∇ (log π_θ/π_θi)²] (Eq. 3). z: the chain of thought; k: responses per problem (value not reported). The optimizer is reset at the start of each iteration (§2.3.2).

**Length reward.** With len(i) the length of response i among k samples, λ = 0.5 − (len(i) − min_len)/(max_len − min_len); correct responses receive λ and incorrect responses receive min(0, λ); all receive 0 if max_len = min_len (§2.3.3). It is added to the task reward with a weighting parameter (value not reported). Training runs without the length penalty first and then with a constant penalty; the switch point is not reported (§2.3.3).

**Sampling.** Curriculum sampling uses the difficulty labels that come with the data and moves from easy to hard; prioritized sampling draws problem i with probability proportional to 1 − s_i, where s_i is its tracked success rate during RL (§2.3.4).

**Partial rollouts.** Each rollout has a fixed output token budget; an unfinished trajectory is saved to the replay buffer and continued in the next iteration, and only the current segment needs on-policy computation (§2.6.2). Segments can be excluded from the loss, and detected repetitions are terminated early and can be penalized (§2.6.2).

## Recipe ledger
Full table (pretraining stages, vanilla SFT, RL prompt curation, RL, reward models, long2short, evaluation settings): [[kimi-k1-5-recipe]].

## Findings relevant to generality, negative feedback, long context, distillation
- **Generality.** From preliminary experiments, the authors state that prompt-set quality and diversity are critical and that diverse coverage, balanced difficulty, and accurate evaluability reduce reward hacking and overfitting to superficial patterns; no ablation numbers are given (§2.1). The long-CoT model trails o1 on LiveCodeBench (62.5 vs 67.2) and MMMU (70.0 vs 77.3) (Table 2). The short-CoT model reports MMLU 87.4, C-Eval 88.3, and IF-Eval 87.2; the IF-Eval number comes from an intermediate model (Table 3, App. C.1). Model merging is cited as useful for maintaining generalization ability (§2.4).
- **Negative feedback.** Against ReST, which fits the best sampled response without penalizing incorrect ones, the mirror-descent method shows better sample complexity on 11 evaluation sets and their total; the authors attribute this to negative gradients (§3.5, Figure 10). This is negative as gradient; no share of the gain is quantified. The authors also argue that value-based credit assignment would penalize erroneous but recoverable reasoning steps and so reduce exploration (Interpretation, §2.3.2). In long2short DPO, negatives are wrong longer responses and correct responses 1.5 times longer than the chosen shortest correct response (§2.4).
- **Long context.** On an internal model smaller than k1.5, accuracy and response length rise together over RL iterations, and harder benchmarks show steeper length growth (§3.3, Figure 5). Two model sizes trained on the same data: the smaller one reaches comparable accuracy with longer CoTs, while the larger one is more token efficient (§3.5, Figure 8). The final run uses 128k RL context (§3.3).
- **Distillation (long2short).** Methods: weight-averaging merge of a long-CoT and a short-CoT model; shortest correct of n = 8 samples used for SFT; DPO as above; and a separate long2short RL phase from the checkpoint with the best performance/token balance, with the length penalty and a significantly reduced maximum rollout length (§2.4). k1.5-shortest reports 88.2 on MATH-500 at a token count similar to other short models (§3.4).

## Connections
- [[kimi-k2]] reuses this report's policy-optimization objective, partial rollouts, and colocated RL design (K2 report §3.2.3, §3.3).
- [[deepseek-r1]] is a long-CoT RL report released in the same month (arXiv:2501.12948).
- [[rloo]] (Ahmadian et al. 2024) is cited for the mean-reward baseline and for training without a value network (§2.3.2).
- [[dpo]] is used as one long2short method (§2.4); [[rlhf-instructgpt]] is cited for the Classic RM design (§2.3.5).
- [[replay-buffer-rlhf]] covers replay buffers in LLM RL frameworks; partial rollouts store unfinished trajectories rather than replaying completed ones (§2.6.2).

## Verification
- Checked on 2026-09-14 against: https://arxiv.org/abs/2501.12599 (v4, 2025-06-03; key strings also confirmed in v1, 2025-01-22)
- Corrections to the previous card version:
  - "Rollout scale: k=8 rollouts per prompt for operations like shortest-rejection-sampling" → n = 8 is the shortest rejection sampling count (§2.4); the RL samples per prompt k is not reported (§2.3.2).
  - "CoT reward model (98.5% validation accuracy) vs Classic value-head RM (84.4%)" → about 98.5 and 84.4 accuracy in manual spot checks of the math reward models (§2.3.5).
  - "Curriculum + prioritized sampling ... difficulty from pass-rate of 10 SFT samples" → the 10-sample SFT pass rate is used for prompt-set curation (§2.1); curriculum uses dataset difficulty labels and prioritized sampling uses success rates tracked during RL (§2.3.4).
  - "SFT learning rate ... at 128K long-context activation" → the 32k/128k LR schedule belongs to vanilla SFT (one epoch each); long-context activation is a pretraining stage (§2.5.2, App. B.4).
  - "Temperature schedule for rollouts ('relatively high' only)" → "relatively high" temperature refers to difficulty estimation with the SFT model (§2.1).
  - "InstructGPT template" → "drawing inspiration from the InstructGPT methodology" (§2.3.5).
  - Header deps [[kimi-k2]] removed: K2 is the later report.
- Removed as unsupported by the source: "the first work to demonstrate this clearly at scale"; "otherwise rollout cost explodes quadratically"; "Length-penalty before/after curves ... overthinking collapse and recovery" (no such figure); "Classic vs CoT RM accuracy table" (the numbers are in prose); "novel mitigation of entropy/length explosion"; "distinct from Tülu 3's / Llama 3's RM designs"; "Before K1.5 there was no public Kimi post-training report".
- Not reported by the source: model size and architecture details (App. B.3); pretraining token counts and LR; long-CoT SFT dataset size; τ; RL samples per prompt, batch size, learning rate, steps, rollout temperature; length-reward weight and warm-up length; partial-rollout token budget; long2short RL maximum length; compute.
