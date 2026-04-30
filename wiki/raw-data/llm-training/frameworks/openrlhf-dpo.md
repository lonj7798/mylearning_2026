<!-- scope: DPO trainer + DPOLoss in OpenRLHF
     deps: [[dpo]], [[ipo]]
     see-also: [[trl-online-dpo]], [[openrlhf-ppo]]
-->

# OpenRLHF — DPO Trainer
- **Framework:** OpenRLHF
- **Repo URL:** https://github.com/OpenRLHF/OpenRLHF
- **Version/commit:** `main` branch (fetched 2026-04-21)
- **Relevant file(s):**
  - `openrlhf/models/loss.py` ≈ lines 231–257 (`DPOLoss`)
  - `openrlhf/trainer/dpo_trainer.py` ≈ lines 60–200 (training step, `concatenated_forward`, NLL & aux-loss mixing)
- **Core pattern:** Standard offline DPO: forward chosen and rejected responses concatenated through the policy and a frozen reference; loss is `−logσ(β·((logπ_c−logπ_r) − (logπ_ref_c−logπ_ref_r)))` with optional IPO mode, label smoothing, and an additive NLL term on the chosen response.
- **Why it matters:** Reference implementation that supports MoE auxiliary loss, label-smoothed cDPO, IPO, and the SFT-stabilization NLL trick — all with a clean single-loop trainer.

## Context
OpenRLHF's DPO trainer is one of the most-cited reference implementations because it bundles the four "production" tricks: (1) a frozen reference model held in eval mode; (2) `concatenated_forward` to halve activation memory by batching chosen+rejected through one forward; (3) optional NLL on the chosen (Pang et al. 2024 RPO-style) to keep SFT alignment intact; (4) MoE auxiliary loss preserved when fine-tuning Mixtral-style models.

## Code excerpt
```python
# openrlhf/models/loss.py, lines 231–257 (DPOLoss.forward body)
class DPOLoss(nn.Module):
    def __init__(self, beta: float, label_smoothing: float = 0.0, ipo: bool = False):
        super().__init__()
        self.beta = beta
        self.label_smoothing = label_smoothing
        self.ipo = ipo

    def forward(self, policy_chosen_logps, policy_rejected_logps,
                reference_chosen_logps, reference_rejected_logps):
        pi_logratios  = policy_chosen_logps  - policy_rejected_logps
        ref_logratios = reference_chosen_logps - reference_rejected_logps
        logits = pi_logratios - ref_logratios

        if self.ipo:
            losses = (logits - 1 / (2 * self.beta)) ** 2          # Azar 2023 IPO
        else:
            losses = (
                -F.logsigmoid(self.beta * logits) * (1 - self.label_smoothing)
                - F.logsigmoid(-self.beta * logits) * self.label_smoothing
            )
        loss = losses.mean()
        chosen_rewards   = self.beta * (policy_chosen_logps   - reference_chosen_logps).detach()
        rejected_rewards = self.beta * (policy_rejected_logps - reference_rejected_logps).detach()
        return loss, chosen_rewards, rejected_rewards
```

```python
# openrlhf/trainer/dpo_trainer.py, ~lines 150–185 (training step body)
chosen_logps, rejected_logps, aux_loss, nll_loss = self.concatenated_forward(
    self.model, chosen_ids, c_mask, reject_ids, r_mask, prompt_id_lens,
)
with torch.no_grad():
    reference_chosen_logps, reference_rejected_logps, _, _ = self.concatenated_forward(
        self.ref_model, chosen_ids, c_mask, reject_ids, r_mask, prompt_id_lens,
    )

preference_loss, chosen_reward, reject_reward = self.loss_fn(
    chosen_logps, rejected_logps, reference_chosen_logps, reference_rejected_logps,
)
if not self.aux_loss:  aux_loss = 0
if not self.nll_loss:  nll_loss = 0

loss = (
    preference_loss
    + aux_loss * self.args.model.aux_loss_coef
    + nll_loss * self.args.model.nll_loss_coef
)
self.strategy.backward(loss, self.model, self.optimizer)
self.strategy.optimizer_step(self.optimizer, self.model, self.scheduler)

acc = (chosen_reward > reject_reward).float().mean().item()
```

## What to notice
- **`logits = (logπ_c − logπ_r) − (logπ_ref_c − logπ_ref_r)`** — the implicit reward margin, which is what DPO actually optimizes.
- **Label smoothing** (`cDPO`, Mitchell 2023) softens the binary preference: useful when ~10% of pairs are mislabeled.
- **IPO** swaps the sigmoid for an L2 around `1/(2β)` to stop the preference probability from saturating; preserves diversity.
- **`concatenated_forward`** stacks chosen+rejected along the batch axis so a single forward computes both; cuts activation memory roughly in half.
- **Reference model is `eval()`** with `torch.no_grad()` — but is *not* re-loaded; OpenRLHF supports DeepSpeed ZeRO-3 with the ref offloaded to CPU.
- **NLL mixing** — when `nll_loss_coef > 0`, an extra cross-entropy term on the chosen response is added; this is the RPO/SimPO-Mix recipe to combat DPO's typical chosen-logp degradation.
- **MoE aux loss** preserved — necessary to keep router balance during DPO of Mixtral/DeepSeek-MoE.

## Comparison to paper / to other frameworks
- **vs Rafailov 2023 DPO paper:** identical loss; OpenRLHF adds `label_smoothing`, `ipo`, NLL, MoE aux as additive options.
- **vs HF TRL `DPOTrainer`:** TRL's offline DPO uses the same algebra; it additionally exposes `loss_type ∈ {sigmoid, hinge, ipo, kto_pair, ...}`. OpenRLHF keeps the surface narrow.
- **vs online variants:** see [[trl-online-dpo]] — online DPO samples chosen/rejected from the *current* policy each step and rewards them via a judge LM, turning DPO into an on-policy iteration.
