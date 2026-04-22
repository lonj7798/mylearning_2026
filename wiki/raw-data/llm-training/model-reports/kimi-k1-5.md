<!-- scope: Kimi K1.5 long-CoT RL technical report — online mirror descent with partial rollouts
     deps: [[kimi-k2]]
     see-also: [[deepseek-r1]], [[magistral]]
-->

# Kimi K1.5
- **Core Insight:** Scaling the RL context window to 128K unlocks long-CoT capability — enabled by "partial rollouts" that reuse previous trajectory segments from a replay buffer instead of regenerating.
- **Guideline:** For long-CoT RL, build a replay-buffer-backed partial-rollout system before scaling context; otherwise rollout cost explodes quadratically.

- **Authors / Lab:** Moonshot AI (Kimi Team)
- **Year:** 2025 (Jan 2025)
- **URL:** https://arxiv.org/abs/2501.12599
- **Relevant topics:** long-CoT RL, online mirror descent, partial rollouts, length penalty, classic vs CoT reward model, curriculum sampling, prioritized sampling

## Abstract
Kimi K1.5 is a multi-modal reasoning model matching OpenAI's o1 on core benchmarks (AIME 77.5, MATH-500 96.2, Codeforces 94th percentile, MathVista 74.9). The core RL recipe is intentionally simple — no MCTS, no value functions, no process rewards — built instead on (1) 128K-token RL context, (2) a variant of online policy mirror descent, (3) partial rollouts for efficiency, and (4) a length penalty that is warmed up gradually to fight overthinking.

## Key Contributions
- **128K RL context:** performance keeps improving as RL-rollout context grows; the first work to demonstrate this clearly at scale.
- **Partial rollouts:** long responses are broken into segments across iterations; previous trajectory segments are reused from a replay buffer — fixed output token budget per iteration.
- **Online policy mirror descent variant:** closed-form objective with relative entropy regularization, sampled-reward baseline for gradient.
- **Length penalty** (gradually warmed up) to combat the "overthinking phenomenon" during training.
- **CoT reward model** (98.5% validation accuracy) vs Classic value-head RM (84.4%) — generating a reasoning chain before a JSON correctness judgment beats scalar-head RMs.
- **Curriculum + prioritized sampling** for prompt selection — difficulty from pass-rate of 10 SFT samples at high temperature.

## Post-training pipeline
- **SFT data — vanilla:** ~1M text examples (500K QA + 200K code + 200K math/science + 5K creative + 20K long-context) plus ~1M text-vision (chart interpretation, OCR, visual reasoning).
- **SFT data — long-CoT:** "small yet high-quality" warmup set via prompt engineering emphasizing planning / evaluation / reflection / exploration.
- **Preference / RL algorithm:** variant of **online policy mirror descent**. Objective: `max_θ E[(y,z)~π_θ[r(x,y,y*)] − τ·KL(π_θ(x)||π_{θ_i}(x))]`. Gradient uses sampled-reward baseline: `∇_θ log π_θ(y_j,z_j|x)·(r(x,y_j,y*) − r̄) − (τ/2)∇_θ(log π_θ/π_{θ_i})²`.
- **Reward model:**
  - Classic RM: value-head, ~800K examples, InstructGPT template. 84.4% validation accuracy.
  - CoT RM: ~800K examples; generates step-by-step reasoning before JSON-formatted correctness judgment. 98.5% validation accuracy — used as the RL reward for reasoning.
- **KL / entropy handling:** relative-entropy regularization coefficient τ > 0 (exact value not disclosed).
- **Length penalty:** `len_reward(i) = {λ if correct; min(0, λ) if incorrect}` with `λ = 0.5 − (len(i) − min_len)/(max_len − min_len)`. Penalizes long-correct answers and long-incorrect answers; warmed up gradually, not applied from step 0.
- **Rollout scale:** k=8 rollouts per prompt for operations like shortest-rejection-sampling. Context 128K. Batch size not disclosed.
- **Hyperparameters:** SFT learning rate 2e-5 → 2e-6 at 32K; re-warms to 1e-5 then decays to 1e-6 at 128K long-context activation.
- **Verifiable rewards:** yes — reward model checks correctness of response given (question, reference answer, response).
- **Self-improvement / iterative:** curriculum sampling and prioritized sampling (problems sampled ∝ 1 − success-rate) drive implicit self-curriculum.

## Innovations vs predecessors
Before K1.5 there was no public Kimi post-training report. Relative to 2024 RLHF norms:
- **Partial rollouts** — replay-buffer reuse of trajectory segments; absent from standard PPO/DPO pipelines.
- Explicit rejection of MCTS / value networks / PRMs — "simplistic, effective RL framework."
- Length penalty warmup schedule — novel mitigation of entropy/length explosion.
- CoT-generating reward model vs scalar-head RM — distinct from Tülu 3's / Llama 3's RM designs.
- K2 (later) moves to pure agentic/long-horizon; K1.5 is the pure-reasoning precursor and establishes the partial-rollout infrastructure K2 inherits.

## Key Figures/Tables to Study
- RL context-length ablation — the "longer context → higher performance" plot; central to the paper's thesis.
- Length-penalty before/after curves — shows the overthinking collapse and recovery.
- Classic vs CoT RM accuracy table (84.4% vs 98.5%) — justifies the expensive CoT RM.

## Connections
- [[kimi-k2]] — successor; inherits partial-rollout infrastructure, extends to agentic long-horizon.
- [[deepseek-r1]] — contemporary; both achieve o1-class reasoning but R1 uses GRPO+rule-based rewards, K1.5 uses mirror descent + CoT RM.
- [[magistral]] — 2025 Mistral reasoning model; similar territory but uses GRPO variant without KL.

## Gaps / what the report does NOT disclose
Exact τ (KL coefficient) for mirror descent — not given. RL batch size, total RL steps, GPU count — not disclosed. Exact λ schedule warmup steps for length penalty. Full RM training data sources beyond "~800K." Temperature schedule for rollouts ("relatively high" only). Prioritized-sampling buffer capacity. Vision-specific post-training details light vs text.
