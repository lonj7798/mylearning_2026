---
chapter: ch-57
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/blogs/hf-dpo-zoo.md
source_url: https://huggingface.co/blog/pref-tuning
created_at: "2026-04-23"
---

# Excerpt: HF DPO Zoo — one trainer, one `loss_type` parameter, seven algorithms

**Source library:** `wiki/raw-data/llm-training/blogs/hf-dpo-zoo.md`
**Artifact:** HuggingFace's running blog series on preference-optimization variants, plus the TRL documentation that operationalizes each one through `DPOTrainer(loss_type=...)`.

---

## Why this source anchors ch-57 §1

Ch-57 §1 closes with "The `trl.DPOTrainer` trick — one trainer class handles the entire DPO family via a `loss_type` string." The zoo is the explicit catalog of that trick. It is the cleanest illustration of TRL's design philosophy: pick one trainer skeleton, expose every variant as a string argument, and let HF's own `Trainer` handle the distributed path.

---

## The variant table ch-57 §1 summarizes

| Variant | loss_type | Needs π_ref? | Needs SFT stage? | Data format |
|---------|-----------|--------------|------------------|-------------|
| DPO (Rafailov 2023) | `"sigmoid"` | yes (frozen) | yes | pairwise {chosen, rejected} |
| IPO (Azar 2023) | `"ipo"` | yes (frozen) | yes | pairwise |
| KTO (Ethayarajh 2024) | `"kto"` | yes (frozen) | yes | unary {good / bad} |
| SimPO (Meng 2024) | `"simpo"` | **no** (length-normalized) | yes | pairwise |
| ORPO (Hong 2024) | `"orpo"` | **no** | **no** (joint SFT + pref) | pairwise + SFT |
| BCO | `"bco"` | yes | yes | unary with classifier |
| CPO | `"cpo"` | yes | yes | pairwise |

One trainer, seven selectors. The distributed story, the PEFT integration, the chat-template handling, the logging — all shared. Only the loss algebra differs.

---

## The four most-used branches

### DPO `"sigmoid"` (ch-57 §2 references [[dpo]] Eq. 7 here)

```
L_DPO = -log σ( β (log π_θ(y_w|x)/π_ref(y_w|x) - log π_θ(y_l|x)/π_ref(y_l|x)) )
```

β ~ 0.1 typical. Too-low β causes collapse — the chosen-logprob decreases faster than it should because the loss is indifferent to absolute magnitude as long as the margin grows.

### IPO `"ipo"`

```
L_IPO = (log π_θ/π_ref(y_w) - log π_θ/π_ref(y_l) - 1/(2β))^2
```

Squared loss on the margin with a target of `1/(2β)`. Explicitly prevents chosen-logprob collapse. Use when DPO training shows collapsing chosen logprobs across epochs.

### SimPO `"simpo"` — reference-free

```
L_SimPO = -log σ( β (log π_θ(y_w|x)/|y_w| - log π_θ(y_l|x)/|y_l|) - γ )
```

No `π_ref` at all. Saves the reference-model memory entirely. Length-normalized to prevent the DPO-length bias. γ is a target-margin hyperparameter.

### ORPO `"orpo"` — joint SFT+preference

```
L_ORPO = L_NLL(y_w) + λ · log σ( odds(y_w) / odds(y_l) )
```

No separate SFT stage. The NLL term does the SFT job on `y_w`; the odds-ratio term pushes down `y_l`. One training phase instead of two. Good for small budgets.

---

## Why the `loss_type` parameter matters architecturally (ch-57 §1)

The `DPOTrainer` implementation pattern inside TRL is roughly:

```python
def concatenated_forward(self, model, batch):
    # One forward pass on [chosen; rejected] concatenated.
    # Returns policy logprobs and (if needed) ref logprobs.
    ...

def compute_loss(self, model, inputs, return_outputs=False):
    (policy_chosen_logps, policy_rejected_logps,
     ref_chosen_logps, ref_rejected_logps) = self.concatenated_forward(model, inputs)

    if self.loss_type == "sigmoid":
        losses = -F.logsigmoid(self.beta * logits)
    elif self.loss_type == "ipo":
        losses = (logits - 1 / (2 * self.beta)) ** 2
    elif self.loss_type == "simpo":
        logits = self.beta * (policy_chosen_logps/len_c - policy_rejected_logps/len_r)
        losses = -F.logsigmoid(logits - self.gamma)
    elif self.loss_type == "orpo":
        losses = nll_loss(y_w) + lambda_ * log_sigmoid(log_odds_ratio)
    ...
    return losses.mean()
```

Everything before the `if/elif` is shared: concatenated forward, reference logprob computation (skipped for SimPO/ORPO), chat-template handling, PEFT adapter-off trick. Everything after is 2–3 lines per variant. Adding a new variant = adding a case.

---

## The chosen-logprob collapse the zoo documents

Ch-57 §2's aside about DPO failure modes is exactly the zoo's thesis. The "chosen-vs-rejected logprob curves" chart from the HF blog shows:

- **DPO**: both chosen and rejected logprobs decrease; chosen decreases less. Margin grows, but absolute probability of correct responses falls.
- **IPO**: both curves stabilize; margin grows without collapse.
- **SimPO**: reference-free, so no drift relative to `π_ref`; different failure surface.

This is ch-57 §2's operational guidance — "start with DPO; switch to IPO if you see chosen-logprob collapse" — in chart form.

---

## HF guidance: when to reach for which variant

From the zoo's decision tree:

- **Paired high-quality preferences** → DPO or IPO.
- **Unary labels only** (thumb-up/down, pass/fail) → KTO.
- **Skip reference model** (memory-constrained, no SFT checkpoint) → SimPO.
- **No separate SFT stage** → ORPO.
- **Limited data** → try several; hold out a preference test set; measure.

None of this requires architecture changes; every choice is a config flag.

---

## Why this matters for ch-57's "when to outgrow TRL" (§6)

Any algorithm that fits the `loss_type` abstraction is cheap to use on TRL — that is the entire DPO zoo. The moment an algorithm needs *new training loop structure* (online sampling, multi-reward aggregation, async rollouts), `loss_type` stops being expressive enough and you need a dedicated trainer (online DPO, GRPO) or a different framework (verl). The zoo is what TRL gets for free; the GRPO file is where the ceiling starts.

---

## Attested implementation notes

- All zoo variants share `DPOTrainer`'s `max_length`, `max_prompt_length`, `beta`, and `ref_model=None` PEFT trick.
- `loss_type="apo"` (APO / anchored preference optimization) was added in TRL 0.11; the zoo blog was written before that and does not mention it.
- KTO typically uses its own dedicated `KTOTrainer` in TRL rather than `DPOTrainer(loss_type="kto")` — the unary-data path needs a different dataloader. This is the exception that proves the rule.

---

## Connections to the rest of the track

- [[dpo]] — the baseline algorithm.
- [[ipo]], [[kto]], [[simpo]], [[orpo]] — algorithm-specific raw pages.
- [[hf-alignment-handbook]] — the production recipe built on this trainer.
- [[trl-online-dpo]] — the one variant that *does not* fit the `loss_type` abstraction and needs its own trainer.
- [[hf-rlhf-illustrated]] — the pre-DPO pipeline the zoo variants collectively replace.
