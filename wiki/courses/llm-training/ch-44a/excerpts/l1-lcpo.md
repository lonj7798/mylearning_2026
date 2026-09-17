---
chapter: ch-44a
course: llm-training
phase: read
excerpt_of: arXiv:2503.04697v2 (L1 / LCPO, COLM 2025), §3, §4, §5, Table 1, Figures 1-5 (chapter-local verified extract; no library card exists for this slug on 2026-09-15)
source_url: https://arxiv.org/abs/2503.04697
created_at: "2026-09-15"
---

# Excerpt: L1 — Controlling How Long A Reasoning Model Thinks With Reinforcement Learning

- **Authors:** Pranjal Aggarwal, Sean Welleck (Carnegie Mellon University)
- **Year:** 2025 (arXiv v1 2025-03; v2 2025-10-03; published at COLM 2025)
- **Source type:** paper
- **Used in:** ch-44a §4, §6, Negative samples, Recipe.

## Method (§3)
- Each prompt is augmented: `x_new = Concat(x, "Think for n_gold tokens.")`, with `n_gold` sampled uniformly from
  `Z(n_min, n_max)` during training.
- **LCPO-Exact reward** (Eq. 1): `r(y, y_gold, n_gold) = I(y = y_gold) − α · |n_gold − n_y|`, where `I` is the correctness
  indicator, `n_y` the generated length, and `α` a scalar trading correctness against length adherence.
- **LCPO-Max reward** (Eq. 2): `r = I(y = y_gold) · clip(α · (n_gold − n_y) + δ, 0, 1)`, a soft constraint that "gradually
  penalizes outputs exceeding the target length rather than imposing a hard cutoff (which is necessary to ensure gradient
  propagation in GRPO objective)". `δ = 0.5` "ensures that correct answers with minor budget violations are still preferred
  over incorrect answers". L1-Max is trained with both objectives: Eq. 1 when the prompt requests an exact length, Eq. 2
  otherwise.
- The RL algorithm is GRPO.

## Setup (§4)
Base model DeepScaleR-1.5B-Preview (itself RL-trained from DeepSeek-R1-Distill-Qwen-1.5B at 24K context); training data
DeepScaleR-Preview-Dataset (40K math QA pairs). "Due to compute constraints, we restrict the maximum context length to 4K
tokens during training and to 8K tokens during evaluation." L1-Exact: 700 steps of LCPO-Exact. L1-Max: 120 further steps
with Eq. 2. Hyperparameters follow DeepScaleR: learning rate 1e-6, batch size 128, verl framework;
`n_gold ~ U(n_min = 100, n_max = 4000)`; `α = 0.0003`. Evaluation target lengths {512, 1024, 2048, 3600}, 16 seeds,
temperature 0.6. The authors state they "did not conduct extensive hyperparameter tuning".

## Baseline
S1 budget forcing (Muennighoff et al., 2025): "once the maximum token budget is reached, S1 stops further generation and
instead inserts 'Final Answer' in the prompt to force model to generate the final answer".

## Results (§5)
- Against S1 at equal budgets: "over 100-150% relative and 20-25% absolute performance gains at both 512 and 1024 token
  budgets" (Figure 2). Log-linear accuracy-vs-length slope 0.24 for L1 versus 0.37 for S1.
- L1-Exact is "approximately 1% below DeepScaleR-4K" (the same model trained without length constraints), with the gap
  concentrated on AIME; L1-Max matches DeepScaleR-4K "by optimizing token usage based on problem difficulty".
- Length adherence: mean error "close to 3% for all math reasoning datasets"; OOD datasets 20-40%. For L1-Max, soft
  violation rate (`|n_generated − n_gold| > 500`) is 0.3% to 2.3% across budgets, average 1.3% (Figure 5).
- Out-of-domain (GPQA, LSAT, MMLU): length control transfers; accuracy scales with budget, with a weaker fit on MMLU
  (R² = 0.66), which the authors attribute to knowledge questions benefiting less from longer reasoning (Figure 3).
- Short-CoT comparison (Table 1, math benchmarks, accuracy at matched generation lengths): at about 850 tokens L1-Max
  averages 46.2 versus 41.0 for Qwen-2.5-1.5B-Instruct; at about 820-870 tokens L1-Max averages 47.8 versus GPT-4o 45.6.
  "on average, L1 is 5% better than its non-reasoning counterpart, and even outperforms GPT-4o by 2% on average."

## Not reported
Group size G, KL coefficient, clip range, truncation rate at the 4K training limit, and any non-reasoning generality
evaluation beyond the OOD benchmarks listed above.
