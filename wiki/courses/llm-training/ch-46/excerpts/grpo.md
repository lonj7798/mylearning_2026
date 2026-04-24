---
chapter: ch-46
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/grpo.md
source_url: https://arxiv.org/abs/2402.03300
created_at: "2026-04-23"
---

# Excerpt: GRPO — Option B's training loop

**Source library:** `wiki/raw-data/llm-training/papers/grpo.md`
**Artifact:** Eq. 3 loss, group-mean baseline, k3 KL estimator inside the loss, DeepSeekMath recipe.

---

## Why GRPO is the Option B loss (not PPO)

Option B of ch-46 is RLVR-math. The paper that introduced the recipe now used for nearly every open RLVR reproduction is DeepSeekMath / GRPO — PPO without a critic, group-relative advantages, and a KL term inside the loss. Ch-46 calls `trl.GRPOTrainer`, which is the exact [[grpo]] objective with optional `loss_type="dr_grpo"` (see [[dr-grpo]]).

---

## Eq. 3 — what `trainer.train()` actually optimizes

Source §Technical Details / GRPO loss:

> `J_GRPO(θ) = E[q, {o_i}~π_θ_old]`
> `  (1/G) Σ_i (1/|o_i|) Σ_t { min[ρ_{i,t} Â_{i,t},  clip(ρ_{i,t}, 1-ε, 1+ε) Â_{i,t}]  −  β D_KL(π_θ || π_ref) }`

The ch-46 `kl` signal is this `D_KL` term; the `reward_mean` signal is the un-normalized r_i before the advantage conversion; the `entropy` signal is computed directly from `logits` in TRL's `_get_per_token_logps_and_entropies`.

---

## The k3 KL estimator — why ch-46 logs it every step

Source §Technical Details / KL approximation (Equation 4):

> `D_KL[π_θ || π_ref] ≈ π_ref/π_θ − log(π_ref/π_θ) − 1`
> k1 = log(π_θ/π_ref) (low variance, biased sign), k2 = 0.5·(log ratio)^2 (unbiased but sign-insensitive), k3 above (unbiased, always ≥0). GRPO uses k3.

The ch-46 §3 MetricsSink callback reads this off TRL's `_metrics[mode]["kl"]` directly. Because k3 is always ≥ 0, a sudden sign flip or NaN in the log means the ref forward pass is misaligned (e.g. ref model is on a different device, or fp32 vs bf16 mismatch). The §7 Acceptance criterion #3 implicitly checks that KL is finite and positive.

---

## Advantage = (r − mean) / std — why β_KL is the clean sweep axis

Source §Technical Details / Advantage (outcome supervision):

> `Â_{i,t} = r̃_i = (r_i − mean({r_1,...,r_G})) / std({r_1,...,r_G})`

Because the advantage is z-scored per prompt, raw reward scale washes out — what's left is the *relative* ranking within the G=8 rollouts. This is why β_KL is the only free parameter in Option B: clip ε, LR, and reward scale have already been standardized by the group statistics and the paper's recipe.

Ch-46 runs `scale_rewards=False` (Dr.GRPO) to drop the std denominator — this doesn't remove the baseline, just the scaling. The baseline (group mean) is still applied.

---

## The DeepSeekMath recipe — ch-46's hparam anchor

Source §Technical Details / Hyperparameters:

| Knob | Paper value | Ch-46 Option B |
|------|-------------|----------------|
| Group size G | 64 | 8 (reduce rollout cost; 3B × 5K prompts × 1 GPU-hr budget) |
| Clip ε | 0.2 | 0.2 (unchanged) |
| KL coefficient β | 0.04 | sweep {0.01, 0.05, 0.1} (bracket) |
| Learning rate | 1e-6 | 1e-6 (unchanged) |
| Batch size (prompts) | 1024 | 128 (scale-down) |
| Max response length | 1024 tokens | 1024 (unchanged) |
| π_ref | SFT model, frozen | SFT model, frozen |
| Epochs per rollout μ | 1 | 1 (on-policy) |
| Sampling T | 1.0 | 1.0 |

G=8 is the ch-46 compromise; the paper's G=64 is 8× the rollout cost and the ceiling-accuracy delta on a 3B model is < 2 pts in practice (per [[openrlhf-entropy-debugging]] community observations).

---

## Outcome-supervision only — why Option B is "verifiable math"

Source §Technical Details / Rollout:

> For each question q in batch, sample G outputs {o_1, …, o_G} from π_θ_old. Score each with reward model R → (r_1, …, r_G).

In ch-46, "R" is the deterministic verifier from [[rlvr-tulu3]] — not a learned RM. The advantage is still computed the same way. Outcome-only means every token in `o_i` gets the same `Â` (broadcast to the full response per [[verl-grpo]] code). The process-supervision variant requires PRMs and is out of scope for ch-46.

---

## Where ch-46 touches the paper's failure surface

Source §Connections, final bullet:

> Framework implementations: [[verl-grpo]], [[trl-grpo]], [[openrlhf-ppo]].

All three frameworks implement the *same* Eq. 3, but with different aggregation (see [[trl-grpo]] `loss_type` branches) and different ways of threading KL (in-loss vs in-reward; see [[openrlhf-ppo]] AdaptiveKLController). Ch-46 uses TRL because its single `_compute_loss` file is the easiest place to swap between `grpo` and `dr_grpo` for the length-bias post-mortem in §5(c).

---

## Connections to the rest of the track

- **ch-40 (GRPO full-read)** — the conceptual chapter; read before this lab.
- **ch-39 (PPO)** — GRPO's parent; §7 Acceptance criterion #3 uses PPO's clip ratio convention.
- **[[dr-grpo]]** — the length-unbiased aggregation ch-46 uses by default.
- **[[rlvr-tulu3]]** — supplies the verifier that replaces the RM.
- **[[trl-grpo]]** / **[[verl-grpo]]** / **[[openrlhf-ppo]]** — implementation surfaces; ch-46 uses TRL but the sweep design maps 1-1 across all three.
