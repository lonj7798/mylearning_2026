---
chapter: ch-39
course: llm-training
phase: read
excerpt_of: "primary source (no library card at the time of writing)"
source_url: https://arxiv.org/abs/2407.10490
created_at: "2026-09-15"
verified_against: "arXiv:2407.10490v4 (29 Jun 2025), ICLR 2025 version, cached plain text"
---

# Learning Dynamics of LLM Finetuning (Ren and Sutherland) — the squeezing effect

## Claims the chapter uses
- **Squeezing effect (§3.3).** When a negative gradient is applied to label `y_u^-` under a softmax head,
  the paper states: the confidence of `y_u^-` is guaranteed to decrease; the removed mass is largely
  "squeezed" into the output that was most confident before the update,
  `y* = argmax_{i ≠ y_u^-} π_{θ^t}(y = i)`, whose probability is guaranteed to increase; dimensions with
  high `π_{θ^t}` tend to increase and those with low `π_{θ^t}` tend to decrease; a peakier `π_{θ^t}`
  squeezes more; and a smaller `π_{θ^t}(y_u^-)` makes the effect stronger. App. E proves these claims for
  multi-class logistic regression (Claims 1–3C).
- **Off-policy DPO reading (§3.3, §4.2).** Because neither `y_u^+` nor `y_u^-` is sampled from the model,
  `y_u^-` is likely in a low-probability region, so the confidence of almost all observed responses falls
  while the mass moves to `y*`.
- **Measurement (§4.2, Fig. 4).** In the DPO runs the log-probability of the greedy-decoded ("teacher
  forcing") response rises from about −113 to −63 within 8 epochs, faster than the rise of the chosen
  response during SFT (−130 to −90). Models: Pythia-410M/1B/1.4B/2.8B and Qwen1.5-0.5B/1.8B; datasets:
  5,000 training examples from Anthropic-HH and from UltraFeedback (§4.1).
- **Mitigation (§4.3).** Adding `[x_u; y_u^-]` to the SFT stage ("extend") before DPO makes `y_u^-` less
  unlikely at the start of DPO and weakens the effect; win rates against the baseline pipeline after DPO
  epochs 0 / 2 / 4 / 6 are 0.4729 / 0.6518 / 0.6928 / 0.6667 judged by ChatGPT and
  0.4679 / 0.5151 / 0.6045 / 0.5432 judged by Claude 3 (Table 1).

## Conditions and limits
- The guarantees are proved for a single negative gradient step on a softmax output; the paper states that
  with both positive and negative pressure and autoregressive structure the behaviour "can become more
  complicated" (§3.3).
- Model sizes are at most 2.8B; no instruction-following benchmark scores are reported.
