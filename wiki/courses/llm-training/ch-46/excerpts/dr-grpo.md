---
chapter: ch-46
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/dr-grpo.md
source_url: https://arxiv.org/abs/2503.20783
created_at: "2026-04-23"
---

# Excerpt: Dr. GRPO — the length-bias post-mortem in ch-46 §5(c)

**Source library:** `wiki/raw-data/llm-training/papers/dr-grpo.md`
**Artifact:** Biased 1/|o_i| normalization, std denominator removal, fixed-L_max aggregation, ~30% shorter completions at equal accuracy.

---

## Why this source is the §5(c) failure mode

Option B of ch-46's canonical third failure mode is length bias. Liu 2025 / Dr.GRPO is the cleanest exposition of why it happens (the 1/|o_i| denominator creates an asymmetric gradient for wrong-and-long answers) and the single-line code fix (drop the denominator; use fixed L_max). Ch-46 uses `loss_type="dr_grpo"` by default to neutralize the bias up front, and `loss_type="grpo"` becomes the repro knob for the post-mortem.

---

## The mechanism in source's own words

Source §Technical Details / Why the fix removes the length bias:

> In GRPO, a long incorrect o_i gets a *smaller per-token negative gradient* (divided by large |o_i|) than a short incorrect one. → easier to keep being wrong + verbose.
> Dropping 1/|o_i| restores equal per-token penalty → wrong-and-long is penalized proportionally to its length.

This is the entire post-mortem mechanism. When ch-46 §5(c) says "bucket `len_mean` by reward=0 vs reward=1", this is the quantitative check: in biased GRPO, the reward=0 bucket has monotone-growing length; in Dr.GRPO, both buckets track the base distribution.

---

## The two dropped normalizations — surgical specificity

Source §Technical Details / Dr. GRPO advantage:

> `Ã_{i,t} = r_i − mean({r_1, …, r_G})`
> - No std division.
> - No per-response length normalization: use a fixed generation budget L_max for token averaging, not the realized |o_i|.

Two independent fixes bundled as "Dr.GRPO":
1. **Drop `/ std(r)`:** std-normalization upweights easy/hard prompts where all rollouts succeed/fail. In TRL this is `scale_rewards=False` in `GRPOConfig`. Ch-46 sets it accordingly.
2. **Drop `/ |o_i|`:** replace per-response token-average with batch-wide token-sum / `(B · L_max)`. In TRL this is `loss_type="dr_grpo"` — the aggregation branch in `_compute_loss` ≈ line 2492 of `trl/trainer/grpo_trainer.py`.

---

## Figure 1 — the length curve ch-46 companion HTML reproduces

Source §Key Figures/Tables to Study:

> **Figure 1:** Response length curves during training — GRPO shoots up, Dr. GRPO flat.

The ch-46 HTML `rl-sweep.html` `B-len` view is a sketch of this figure: with `loss_type="dr_grpo"` (the default) the illustrative curves are nearly flat; the caption explicitly says "With vanilla GRPO, the KL=0.01 curve would inflate ~30-50% and the KL=0.05 curve would inflate ~15%." The direction is the paper's; the exact numbers are inferred within the ~30% envelope Liu 2025 reports.

---

## The lab's `loss_type="grpo"` sanity check

Source §Key Contributions:

> Demonstrates equal or better accuracy with ~30% shorter completions on MATH, AIME, and AMC.

Equal accuracy, shorter completions. This is why ch-46 §5(c) fix ("switch to Dr.GRPO aggregation and show the length curve flattening") is a *win* — you gain wall-clock (shorter rollouts → faster vLLM generation) without losing accuracy. The post-mortem conclusion is quantitative: ~30% length reduction, < 1% accuracy delta, attested in the source's Table 2 (Qwen2.5-Math-7B).

---

## Hparam diff from GRPO — exactly two lines in `GRPOConfig`

Source §Technical Details / Hyperparameters:

> Same as [[grpo]] except:
> - No `/ std(r)` in advantage.
> - Token-average denominator = fixed L_max (e.g., 4096) rather than |o_i|.

In TRL that is:

```python
cfg = GRPOConfig(
    ...,
    loss_type="dr_grpo",    # sets the L_max-normalized aggregation branch
    scale_rewards=False,    # drops /std in the advantage
    max_completion_length=1024,   # this is the L_max used in the denominator
)
```

Three flags that change the failure mode shape. Ch-46 treats these as the default, and the post-mortem repros the inflation by flipping `loss_type` to `"grpo"`.

---

## Connections to the rest of the track

- **[[grpo]]** — parent algorithm; the one equation this paper de-biases.
- **[[verl-grpo]]** — implements the same fix via `norm_adv_by_std_in_grpo=False`.
- **[[trl-grpo]]** — `loss_type` branch is the single-file implementation.
- **ch-42 (reward hacking)** — length inflation is the archetypal reward-hack in RLVR; Dr.GRPO closes the one that comes from the *loss* rather than the reward.
- **ch-43 (entropy and KL control)** — length inflation and entropy collapse are often confounded; Dr.GRPO lets ch-46 separate them cleanly.
