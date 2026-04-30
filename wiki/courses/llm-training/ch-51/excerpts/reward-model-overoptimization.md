---
chapter: ch-51
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/reward-model-overoptimization.md
source_url: https://arxiv.org/abs/2210.10760
created_at: "2026-04-23"
---

# Excerpt: Reward-model overoptimization — why "never smooth across a trend reversal"

**Source library:** `wiki/raw-data/llm-training/papers/reward-model-overoptimization.md`
**Artifact:** gold reward rises, peaks, falls; proxy reward grows monotonically; rolling averages across the peak mask the regression.

---

## Why this source is ch-51 §5's decision-tree anchor

Source §Key Contributions:

> For best-of-n, R_gold(d) ≈ d · (α_bon − β_bon · d); for RL, R_gold(d) ≈ d · (α_RL − β_RL · log d). Proxy reward grows monotonically; gold reward follows an inverted-U.

The gold-reward curve is the canonical example of a trend reversal. If you report a rolling mean with window W=7 across training steps that span the peak, the mean looks flat or even rising while the raw curve has already peaked and is falling. Ship based on the smoothed curve and you ship a worse model than you had three days ago. Ch-51 §5's decision tree terminates at "did gold/held-out reward peak and fall? → rolling mean is lying; show raw curve; stop training."

---

## The specific shape ch-51 §5 forbids smoothing across

Source §Key Figures/Tables to Study:

> Fig. 1 / 2 (proxy vs gold vs d = sqrt(KL)): canonical Goodhart curves — memorize this shape.

The shape: proxy-reward curve monotone; gold-reward curve rises, hits a peak, decays. A rolling mean with any reasonable window suppresses the decay phase because the average is dominated by the peak itself. The fix is structural — detect the peak directly, not through smoothing. Ch-51 §5's rule: smooth for communication, never for decision. The decision input is the raw per-checkpoint value + its per-checkpoint CI.

---

## The KL budget as the x-axis, not training step

Source §Technical Details:

> d = sqrt(KL(π ‖ π_SFT)); KL is forward, token-averaged, and measured against the SFT reference.

Ch-51 §5 shows the decision tree in terms of checkpoints, but in RL post-training the right x-axis is KL-from-reference, not step count. Two checkpoints at the same step but different KL are not comparable; two checkpoints at the same KL but different steps are. When plotting the raw curve to check for a reversal, plot vs KL when available. Ch-51 §6's memo "variance accounting" slot should list the KL range the evaluated checkpoints span.

---

## Why the held-out slice is the only trustworthy signal

Source §Key Contributions:

> Policy size barely matters: bigger policies optimize the proxy faster but hit the same gold peak — this is a property of the RM, not the policy.
> KL penalty β is not a free lunch: varying β in PPO traces out essentially the same front as early-stopping.

The RM-proxy gap is structural, not a bug to be engineered away. The only way to detect it is a held-out slice that the RM has not been trained on (or better, a gold verifier). Ch-51 §6 memo's "Evidence" field demands a paired CI *on the gold/held-out slice*, not on training reward. Training reward rising while held-out reward falls is the exact signature §5's decision tree detects.

---

## "Stop before the predicted peak" as a live go/no-go rule

Source §Guideline:

> Treat the KL-from-reference as your optimization budget, not a regularizer — monitor gold reward (or a held-out eval) vs KL and stop RL training before the predicted peak.

Ch-51 §6 memo §6 "Decision" slot is exactly this moment. The rule — stop before the peak — translates to: commit the go/no-go memo at the checkpoint where held-out reward is highest AND the per-checkpoint CI at the next-to-last checkpoint overlaps the peak by ≤ CI halfwidth. If the next two checkpoints both drop and both CIs exclude the peak value, you have a reversal — do not ship the later checkpoint.

---

## Why this matters even outside RLHF

The peak-then-fall shape is generic. It appears in:
- SFT with too many epochs (validation loss rises while train falls).
- DPO with too small β (reward-margins plateau then invert).
- RLVR with a flawed verifier (training reward diverges from held-out pass-rate).

Ch-51 §5's decision tree is intentionally algorithm-agnostic — it triggers on any reversal regardless of which objective produced it.

---

## Connections

- **[[kl-control-rlhf]]** — the KL penalty is the budget knob this paper frames.
- **[[reward-hacking-taxonomy]]** — Goodhart is the root category; overoptimization is its quantitative law.
- **[[rlvr-tulu3]], [[deepseek-r1]]** — exact verifiers collapse the proxy/gold gap; the one case where you can trust training-time reward.
- **ch-51 §5** — decision tree terminates on this paper's peak-then-fall signature.
- **ch-51 §6 memo** — evidence must include held-out reward, not only training reward.
