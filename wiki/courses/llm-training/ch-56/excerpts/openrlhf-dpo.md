---
chapter: ch-56
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/frameworks/openrlhf-dpo.md
source_url: https://github.com/OpenRLHF/OpenRLHF
created_at: "2026-04-23"
---

# Excerpt: OpenRLHF — DPOLoss, concatenated_forward, and the frozen reference

**Source library:** `wiki/raw-data/llm-training/frameworks/openrlhf-dpo.md`
**Files:** `openrlhf/models/loss.py` (lines 231–257), `openrlhf/trainer/dpo_trainer.py` (lines ~150–185)
**Version/commit:** `main` branch (fetched 2026-04-21)

---

## Why this source anchors ch-56

OpenRLHF's DPO trainer is the reference implementation that
productionizes four tricks — concatenated forward, frozen ZeRO-3-offload
reference, optional NLL mixing, MoE aux-loss preservation — that every
other framework copies in pieces. Read this file and you have read the
DPO "best practices" of 2024–2026.

---

## The DPO loss, attested verbatim

Source lines 30–45:

```python
pi_logratios  = policy_chosen_logps  - policy_rejected_logps
ref_logratios = reference_chosen_logps - reference_rejected_logps
logits = pi_logratios - ref_logratios

if self.ipo:
    losses = (logits - 1 / (2 * self.beta)) ** 2
else:
    losses = (
        -F.logsigmoid(self.beta * logits) * (1 - self.label_smoothing)
        - F.logsigmoid(-self.beta * logits) * self.label_smoothing
    )
```

The `logits` variable is the **implicit reward margin** — the quantity
DPO actually optimizes, per [[dpo]] Eq. 7:

```
logits = (log pi_theta(y_w) - log pi_ref(y_w))
       - (log pi_theta(y_l) - log pi_ref(y_l))
```

Three modes hide in this 15-line block:
- Default (cDPO-compatible): `F.logsigmoid(beta * logits)` with
  optional `label_smoothing > 0` for the Mitchell 2023 cDPO recipe.
- IPO (Azar 2023): L2 around `1 / (2*beta)` — stops the preference
  probability from saturating; preserves diversity.
- Full-strength DPO: `label_smoothing = 0`, `ipo = False`.

---

## concatenated_forward — the memory trick

Source lines 50–56:

```python
chosen_logps, rejected_logps, aux_loss, nll_loss = self.concatenated_forward(
    self.model, chosen_ids, c_mask, reject_ids, r_mask, prompt_id_lens,
)
with torch.no_grad():
    reference_chosen_logps, reference_rejected_logps, _, _ = self.concatenated_forward(
        self.ref_model, chosen_ids, c_mask, reject_ids, r_mask, prompt_id_lens,
    )
```

`concatenated_forward` stacks chosen + rejected along the batch axis:
a single forward produces both sets of logprobs. Activation memory
drops roughly 2x. The output logps are sliced apart using `c_mask` /
`r_mask` before the DPO loss call. This is the single trick that lets
a 70B DPO job fit on 8xH100.

---

## Reference management — three load-bearing flags

Source §What to notice:

> **Reference model is `eval()`** with `torch.no_grad()` — but is *not*
> re-loaded; OpenRLHF supports DeepSpeed ZeRO-3 with the ref offloaded
> to CPU.

Three flags matter at 70B scale:

- `ref.eval()` — disables dropout / LayerNorm train stats.
- `torch.no_grad()` — drops activations; the ref cannot back-prop.
- ZeRO-3 CPU offload — moves ref parameters to host memory; pulled
  on-demand per forward. Without this, 70B DPO OOMs on 8xH100 even
  with concatenated_forward.

If you forget any of the three, the bug is silent: the DPO loss still
computes, but `rewards/chosen - rewards/rejected` drifts wrong. [[dpo]]
§Hyperparameters warns this is the common reference-model bug.

---

## Why NLL mixing exists

Source §What to notice:

> **NLL mixing** — when `nll_loss_coef > 0`, an extra cross-entropy
> term on the chosen response is added; this is the RPO/SimPO-Mix
> recipe to combat DPO's typical chosen-logp degradation.

Plain DPO drives both `log pi_chosen` and `log pi_rejected` down — the
optimum is to make both small while keeping the *difference* large. On
noisy preference data this degrades SFT-induced fluency. NLL mixing
adds an anchor: a small CE on the chosen response prevents `log
pi_chosen` from collapsing. This is the production default in Tülu 3
and most post-2024 DPO recipes.

---

## MoE aux-loss — the router trap

> **MoE aux loss** preserved — necessary to keep router balance during
> DPO of Mixtral/DeepSeek-MoE.

If you DPO a MoE model without preserving the router's load-balance
loss, the router unbalances within a few hundred steps, expert
utilization becomes skewed, and output quality drops in ways
`preference_loss` cannot detect. The OpenRLHF DPO trainer adds it
back in via `aux_loss_coef`.

---

## Connections

- [[excerpts/openrlhf-ppo]] — sibling loss in the same `loss.py`;
  same nn.Module idiom.
- [[excerpts/entropy-logging-patterns]] — DPO's implicit rewards
  correspond to `rewards/chosen` and `rewards/rejected` in the
  framework comparison table.
- Host chapter: [[ch-56]] §3.
- Forward to [[ch-57]] (TRL) — TRL `DPOTrainer` exposes the same
  surface plus more `loss_type` variants.
