---
chapter: ch-39
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/frameworks/openrlhf-dpo.md
source_url: https://github.com/OpenRLHF/OpenRLHF
created_at: "2026-04-23"
---

# Excerpt: OpenRLHF — the DPO/IPO loss module ch-39 points at

**Source library:** `wiki/raw-data/llm-training/frameworks/openrlhf-dpo.md`
**Repo:** https://github.com/OpenRLHF/OpenRLHF
**Fetched:** main branch, 2026-04-21
**Key files:** `openrlhf/models/loss.py` (DPOLoss) and `openrlhf/trainer/dpo_trainer.py` (training step)

---

## Why this source anchors ch-39 §9

The whole offline preference-optimization family — DPO, IPO, cDPO (label-smoothed), and RPO (via NLL mixing) — ships behind one ~30-line module in OpenRLHF. Reading this code is the fastest way to convince yourself that the algebra in ch-39 §2–§7 is literally what the framework computes. Every equation in `read.md` maps to one line here.

## DPOLoss verbatim — the module where DPO, IPO, and cDPO live together

Source lines 21–46:

```python
# openrlhf/models/loss.py, lines 231–257
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

Line-by-line mapping to ch-39's derivation:

- `pi_logratios = policy_chosen_logps - policy_rejected_logps` — this is the policy-side piece of the margin.
- `ref_logratios = reference_chosen_logps - reference_rejected_logps` — reference piece.
- `logits = pi_logratios - ref_logratios` — this is exactly `h` from Equation (5)/(9): the implicit-reward margin divided by β.
- `if self.ipo: losses = (logits - 1 / (2 * self.beta)) ** 2` — this is Equation (9) of `read.md`. Squared error around `1/(2β)` (OpenRLHF names it `self.beta` but it plays the role of τ for IPO).
- `else: losses = -F.logsigmoid(self.beta * logits) * (1 - label_smoothing) - F.logsigmoid(-self.beta * logits) * label_smoothing` — this is DPO's Equation (6) extended with Mitchell's cDPO label-smoothing (Source line 77).
- `chosen_rewards = self.beta * (...)` — this is the implicit reward `r̂_θ` from Equation (7). It's the primary diagnostic; `(chosen_rewards > rejected_rewards).float().mean()` is `rewards/accuracies` in the logs.

The most important observation: **one class, two algorithms**, differing by one line. This is exactly the point of ch-39's framing that the variants perturb single assumptions. IPO is literally "swap the sigmoid for L2 around a finite target."

## The training step — NLL mixing and concatenated_forward

Source lines 49–73:

```python
# openrlhf/trainer/dpo_trainer.py, ~lines 150-185
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
```

Three production tricks are visible here:

1. **`concatenated_forward`** — stacks chosen and rejected along the batch axis so one forward computes both log-probabilities. Cuts activation memory roughly in half. Matters on 70B+.

2. **`nll_loss * nll_loss_coef`** — this is the [[rpo]] / [[llama-3]] trick in code. Set `nll_loss_coef = 0.2` and you have the Llama 3 recipe; set it to 1.0 and you have vanilla RPO. Same binary flag controls both.

3. **`aux_loss * aux_loss_coef`** — MoE router-balance loss, preserved so DPO of Mixtral / DeepSeek-MoE doesn't destabilize the router. Not relevant to dense models but important if you're DPO-ing an MoE.

## Two knobs that matter in practice

From source line 80: "**Reference model is `eval()` with `torch.no_grad()`** — but is *not* re-loaded; OpenRLHF supports DeepSpeed ZeRO-3 with the ref offloaded to CPU." Critical for 70B+ where holding both policy and reference in GPU RAM is not feasible.

From source line 82: "`acc = (chosen_reward > reject_reward).float().mean().item()`" — this is the `rewards/accuracies` metric ch-39 §12 asks you to monitor. It comes for free.

## What's *not* here

Source line 86: OpenRLHF's surface is narrow. It exposes `ipo` and `label_smoothing` but not the full TRL zoo (`hinge`, `kto_pair`, `bco`, etc.). For SimPO, ORPO, KTO, use `trl.DPOTrainer` with `loss_type=...` instead; see [[hf-dpo-zoo]] and [[trl-online-dpo]] (the online cousin).

## How ch-39 uses this

§9 of `read.md` lifts the `DPOLoss.forward` body verbatim as the framework reality-check. §12's three monitoring metrics (`rewards/accuracies`, `chosen_logps`, length ratio) all come directly from the `chosen_rewards` / `rejected_rewards` tensors that `DPOLoss` returns.

## Connections

- Paper behind the DPO branch: [[dpo]] / [[dpo-derivation]].
- Paper behind the IPO branch: [[ipo]] / [[ipo-identity-link]].
- Paper behind the `nll_loss_coef` branch: [[rpo]] / [[rpo-nll-anchor]].
- Industrial tuning of those coefficients: [[llama-3]] / [[llama-3-dpo-hparams]].
- Online cousin: [[trl-online-dpo]].
- Survey of TRL-exposed variants not in OpenRLHF: [[hf-dpo-zoo]].
