---
chapter: ch-56
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/frameworks/openrlhf-dpo.md
source_url: https://github.com/OpenRLHF/OpenRLHF/tree/64c1cc4f30d4c0dab772c8aa1dc11288c425e0da
created_at: "2026-04-23"
revised: "2026-09 (generality revision; rewritten to match the verified card)"
---

# Excerpt: OpenRLHF `DPOLoss`, the train step, and reference handling

**Source library:** `wiki/raw-data/llm-training/frameworks/openrlhf-dpo.md`
**Version/commit:** `64c1cc4f30d4c0dab772c8aa1dc11288c425e0da` (2026-04-19); re-checked at `main` b117b2b
**Files:** `openrlhf/models/loss.py` L246–281, `openrlhf/trainer/dpo_trainer.py`, `openrlhf/cli/train_dpo.py`

---

## What ch-56 takes from this source

### 1. The loss (`loss.py` L264–281, verbatim)

```python
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        ref_logratios = reference_chosen_logps - reference_rejected_logps
        logits = pi_logratios - ref_logratios

        if self.ipo:
            losses = (logits - 1 / (2 * self.beta)) ** 2  # Eq. 17 of https://arxiv.org/pdf/2310.12036v2.pdf
        else:
            # Eq. 3 https://ericmitchell.ai/cdpo.pdf; label_smoothing=0 gives original DPO (Eq. 7 of https://arxiv.org/pdf/2305.18290.pdf)
            losses = (
                -F.logsigmoid(self.beta * logits) * (1 - self.label_smoothing)
                - F.logsigmoid(-self.beta * logits) * self.label_smoothing
            )

        loss = losses.mean()
        chosen_rewards = self.beta * (policy_chosen_logps - reference_chosen_logps).detach()
        rejected_rewards = self.beta * (policy_rejected_logps - reference_rejected_logps).detach()
```

With h the difference of policy and reference log-ratios: DPO and cDPO give
ℓ = −(1 − ε)·log σ(β·h) − ε·log σ(−β·h); IPO gives ℓ = (h − 1/(2β))². `log π(y|x)` is the **sum** of
response-token log-probabilities with prompt tokens masked (`dpo_trainer.py` L380–386).

### 2. Train step (`dpo_trainer.py` L157–184)

Total loss is `preference_loss + aux_loss * aux_loss_coef + nll_loss * nll_loss_coef`. `acc` is the
fraction of pairs with `chosen_reward > reject_reward`. Evaluation computes only the preference loss and
accuracy; the NLL and auxiliary terms are excluded (L273–285).

### 3. `concatenated_forward` — speed, not memory (`dpo_trainer.py` L305–309, L350–360)

Chosen and rejected are right-padded to a common length and concatenated along the batch dimension, so one
forward pass covers both. The docstring gives the reason as avoiding two forward passes "because it's
faster for FSDP". The code makes **no** memory claim: the concatenation produces a 2× batch, so the
activations retained for backward are those of two forwards.

### 4. NLL term (`dpo_trainer.py` L328; `train_dpo.py` L246)

`-all_logps_mean[: chosen_ids.shape[0]].mean()` — the negative per-token mean log-probability of each
chosen response, averaged over the batch. The CLI help reads "Regularization with NLL loss, see LLama 3.1
tech report." Default coefficient 0.

### 5. Reference model (`train_dpo.py` L342–343, L213; `dpo_trainer.py` L147, L160–163)

The policy path is used for the reference when no reference path is given. The reference runs in `eval()`
mode under `torch.no_grad()`. `--ref.offload` sets CPU parameter offload and is **off by default**; the
reference eval config uses ZeRO stage 3 when training uses stage 3 and stage 0 otherwise.

### 6. Derived rejected-side gradients (from L269–275; not stated in the source)

∂ℓ/∂log π_θ(y_r|x) = β(1 − ε)σ(−βh) − βε σ(βh). With ε = 0 this is positive for every h, so each step
lowers the rejected log-probability; with β = 0.1 and ε = 0.1 the sign flips only at h > 10·ln 9 ≈ 22.0.
For IPO, ∂ℓ/∂log π_θ(y_r|x) = 2(1/(2β) − h), which reverses once h > 1/(2β) (5.0 at β = 0.1).

---

## Limits

The source is code and contains no measurement of training dynamics, no ablation of β, label smoothing,
IPO or the NLL coefficient, and no statement about likelihood displacement. The logs separate
`chosen_reward` and `reject_reward` but do not include raw chosen or rejected log-probabilities.

---

## Links

[[openrlhf-dpo]] · [[openrlhf-dpo-recipe]] · [[dpo]] · [[openrlhf-ppo]] · [[tulu-3]]
