---
chapter: ch-57
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/blogs/hf-rlhf-illustrated.md
source_url: https://huggingface.co/blog/rlhf
created_at: "2026-04-23"
---

# Excerpt: HF Illustrated RLHF — the three-stage diagram TRL was built to implement

**Source library:** `wiki/raw-data/llm-training/blogs/hf-rlhf-illustrated.md`
**Artifact:** Nathan Lambert, Louis Castricato, Leandro von Werra, Alex Havrilla (2022). The canonical visual explainer of RLHF. The diagrams predate Llama 2 and yet map directly onto every subsequent post-training paper.

---

## Why this source anchors ch-57 §5

Ch-57 §5 discusses the legacy actor-critic `PPOTrainer` in `trl/experimental/`. The reason that trainer exists at all — the reason TRL started — is the three-stage pipeline this post introduced: SFT → reward model → PPO against the RM with a KL penalty. Understanding *why TRL's original shape looks like it does* requires understanding the illustrated pipeline. The critic, the reference model, the adaptive KL controller in `trl/core/` — all are implementations of what the post drew.

---

## The three-stage pipeline TRL implements

### Stage 1: Supervised Fine-Tuning (SFT)

- Initialize chat/instruct behavior from a pretrained LM.
- Train on high-quality (prompt, response) pairs with completion-masked cross-entropy.
- Output: π_SFT — the reference policy for subsequent stages.

Ch-57 §2's `SFTTrainer` is this stage in code form.

### Stage 2: Reward Model (RM)

- Architecture: same as LM but with a scalar output head.
- Data: pairs (x, y_w, y_l) where y_w is preferred over y_l.
- Loss: `-log σ(r(x, y_w) - r(x, y_l))` — pairwise logistic (Bradley-Terry).
- Initialized from π_SFT; head is a linear layer on the final hidden state of the last token.

TRL's `RewardTrainer` in `trl/trainer/reward_trainer.py` is this stage.

### Stage 3: PPO against RM with KL penalty

- Policy π_θ initialized from π_SFT.
- For each sampled (x, y):
  - Scalar reward = r(x, y) minus a KL penalty term at each token.
  - Per-token reward: `r_t = -β · log(π_θ(y_t | ·) / π_ref(y_t | ·))` for t < |y|, plus `r(x, y)` added at t = |y|.
- PPO update with clipped ratio, value head, advantage normalization.
- Reference policy π_ref = π_SFT held fixed.

This stage *is* the [[trl-ppo]] excerpt. Ch-57 §5's code snippet (`pg_losses = -mb_advantage * ratio`; `vf_loss = 0.5 * masked_mean(...)`) is a line-by-line realization of this third stage.

---

## The equation that shapes TRL's PPO code (ch-57 §5)

```
r_total(x, y) = r_RM(x, y) - β · D_KL(π_θ(· | x) || π_ref(· | x))
```

or equivalently, token-wise:

```
r_t_shaped = { -β · (log π_θ(y_t|·) - log π_ref(y_t|·))           if t < |y|
             { r_RM(x, y) + -β · (log π_θ(y_T|·) - log π_ref(y_T|·))  if t = |y|
```

The β · KL term is added to the **per-token reward before GAE runs**, not to the loss. This is the "KL-on-reward" pattern ch-57 §5 contrasts with GRPO's "KL-in-loss" K3 estimator.

Typical β: 0.01–0.2. Too small → reward hacking. Too large → policy never moves.

---

## What this post predates (ch-57 §5 uses this to explain the demotion)

The post is from 2022 and predates:

- **DPO (2023)** — the closed-form alternative that eliminates the RM and the PPO loop. Became `DPOTrainer` in TRL, bigger than the original PPOTrainer by late 2023.
- **RLVR (2024)** — verifier-based rewards for math/code; eliminates RM training entirely.
- **GRPO (2024)** — drops the value head. Became the default RL trainer in TRL.
- **Iterative multi-round RLHF (Llama 3)** — outer loop over the entire pipeline.

Each post-2022 development maps onto a structural change in TRL:

| Development | TRL change |
|-------------|-----------|
| DPO | `DPOTrainer` added in `trl/trainer/` (stable) |
| GRPO | `GRPOTrainer` added; `PPOTrainer` moved to `experimental/` |
| RLVR | `reward_funcs` list in `GRPOConfig` — reward as Python functions |
| Online DPO | `OnlineDPOTrainer` in `trl/experimental/online_dpo/` |

Ch-57 §5's "the file was demoted" story is this table in prose.

---

## Why KL regularization is non-negotiable

The illustrated post makes one argument clearly: without KL, PPO exploits the RM (reward hacks) by producing outputs outside the distribution the RM was trained on. The KL penalty bounds the policy to stay close to π_SFT, where the RM is well-calibrated.

This is still true in 2026. Every TRL RL trainer has a β-KL term somewhere:

- `PPOTrainer`: β·KL added to per-token reward (K1 estimator).
- `GRPOTrainer`: β·KL_K3 added inside the loss.
- `OnlineDPOTrainer`: the DPO loss itself is an implicit KL-regularized objective via the β·log(π/π_ref) structure.

Disabling KL (β=0) is an ablation, not a production mode.

---

## Why the post still works as onboarding

Nearly every subsequent paper can be explained as "the illustrated pipeline, but with X changed":

- DPO: replace stages 2 and 3 with one closed-form loss.
- GRPO: remove the value network from stage 3.
- RLVR: replace the RM in stage 2 with a verifier function.
- DeepSeek-R1: run stage 3 with verifiable rewards and chain-of-thought rollouts.

Ch-57 §5 relies on this — the reader sees the [[trl-ppo]] excerpt and understands it as "the three-stage pipeline, stage 3, in code." Without that grounding the file is just a collection of clips and masks.

---

## Attested connection to TRL's current shape

- `AutoModelForCausalLMWithValueHead` (the shared-base-model value head) is the canonical implementation of "initialize value network from π_SFT and let it share parameters with the policy."
- The `AdaptiveKLController` in `trl/core/` is the illustrated post's β-tuning loop in code.
- The Bradley-Terry logistic loss in `RewardTrainer` is Stage 2 verbatim.

TRL's name itself ("Transformer Reinforcement Learning") is a direct reference to this pipeline; the library was born to implement it.

---

## Connections to the rest of the track

- [[rlhf-instructgpt]] — the Ouyang 2022 paper the illustrated post visualizes.
- [[ppo]] — the underlying RL algorithm.
- [[dpo]] — the post-DPO alternative that replaces stages 2 and 3.
- [[trl-ppo]] — the code that implements the third stage.
- [[trl-grpo]] — the successor that modifies stage 3 to drop the critic.
- [[hf-dpo-zoo]] — the landscape of stage 2+3 replacements.
- [[hf-alignment-handbook]] — the production recipe that descends from this pipeline.
