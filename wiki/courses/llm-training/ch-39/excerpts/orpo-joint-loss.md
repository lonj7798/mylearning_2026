---
chapter: ch-39
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/orpo.md
source_url: https://arxiv.org/abs/2403.07691
created_at: "2026-04-23"
---

# Excerpt: ORPO — single-stage SFT + odds-ratio penalty

**Source library:** `wiki/raw-data/llm-training/papers/orpo.md`
**Authors:** Jiwoo Hong, Noah Lee, James Thorne
**Venue:** EMNLP 2024 (arXiv 2403.07691)
**Year:** 2024

---

## Why this source anchors ch-39 §6

Standard alignment is two stages: base → SFT → DPO. ORPO's thesis (Source line 14): "SFT inherently increases the probability of undesired generations alongside desired ones." If that's true, SFT is actively working against the alignment goal for half the pairs in your preference dataset, and DPO is spending its budget undoing SFT's damage.

ORPO's fix: **put a suppression term inside SFT**. One pass, one loss, one optimizer state. No `π_ref`, no pre-stage.

The key empirical observation is Source Figure 3: during plain SFT, the log-probability of rejected completions *rises* alongside the chosen's. The model learns the domain, the style, the chat template — and that raises log-prob for any fluent continuation, correct or not. ORPO's odds-ratio term explicitly pulls the chosen-vs-rejected gap apart while the CE term pulls the chosen up.

## Odds and odds ratio

Source lines 29–33:

```
odds_θ(y | x)          = π_θ(y|x) / ( 1 − π_θ(y|x) )
OR_θ(y_w, y_l | x)     = odds_θ(y_w | x) / odds_θ(y_l | x)
```

`π_θ(y|x)` here is the sequence probability. In practice implementations use a length-normalized log-probability inside the sigmoid to avoid numerical issues (a full sequence probability is vanishingly small; its odds is `p/(1-p) ≈ p` and uninformative on log scale).

## The odds-ratio loss

Source line 38 (Equation 7):

```
L_OR = − log σ( log [ odds_θ(y_w|x) / odds_θ(y_l|x) ] )
```

This reads as: "cross-entropy on the binary event 'chosen is preferred' under a sigmoid link of the log-odds-ratio." Functionally very similar to DPO's NLL, but the argument inside the sigmoid is `log(odds_w / odds_l)` instead of `β · log(π_w/π_l^ref / π_l/π_l^ref)`.

## The total loss

Source line 42 (Equation 6):

```
L_ORPO = E_{(x, y_w, y_l) ~ D} [ L_SFT + λ · L_OR ]

L_SFT  = − log π_θ(y_w | x)      # standard causal-LM CE on chosen only
```

Notice:

- `L_SFT` operates *only* on the chosen response. This is the whole CE signal that raises chosen log-prob.
- `L_OR` operates on the pair. This is the whole preference signal that separates chosen from rejected.
- `λ` trades the two. `λ = 0` → plain SFT (and rejected probability rises with chosen). `λ → ∞` → pure odds-ratio with no chosen anchor (degenerate; chosen log-prob can drift).

## Why reference-free works here

Source line 58–60: "L_SFT simultaneously anchors the chosen-side log-prob (pushes it up); the odds-ratio term only pulls the chosen-vs-rejected gap apart. Without L_SFT anchoring, a reference-free gap-maximizer would collapse; without L_OR, SFT alone fails to suppress rejected-response probability."

This is a clean decomposition of what ORPO is doing:

- Chosen-side absolute level: `L_SFT` (gradient proportional to `∇ log π_θ(y_w|x)`).
- Chosen-vs-rejected gap: `L_OR` (gradient proportional to the difference).

SimPO has only the gap term, which is why it can degenerate on deterministic data. DPO has only the gap term (via the BT loss) plus the reference-model's implicit anchoring. ORPO has both gap and anchor explicitly.

## Hyperparameters — SFT-scale, not DPO-scale

Source lines 46–56:

| Knob | Value | Rationale |
|------|-------|-----------|
| λ | 0.1 (Mistral-7B) / 0.2 (Llama-2-7B) / 0.25 (Phi-2) | Smaller for stronger base models |
| LR | 8e-6 | SFT-scale (40× DPO's 5e-7) because CE term dominates early |
| Batch | 64 prompts | Standard SFT |
| Epochs | 3–5 | Standard SFT |
| π_ref | **none** | By construction |
| Max length | 1024 | Standard |

Note the LR: ORPO runs at SFT learning rates (8e-6), not DPO rates (5e-7). This is because the CE term dominates the gradient magnitude — the odds-ratio term is modulated by λ ≤ 0.25 and does not want to drive LR selection.

## How ch-39 uses this

§6 presents Equations (14)–(16). The comparison table in §8 pins ORPO as "single-stage base→aligned, paired, reference-free." §9 notes that this is the one variant where `π_ref` is not passed to the trainer at all.

## Connections

- Two-stage alternative: SFT → [[dpo]].
- Other reference-free variant: [[simpo]] — but with only the gap term.
- Theoretical family: [[ipo]] / KL-bounded objectives.
- Binary-label alternative: [[kto]].
- Handbook integration: [[hf-alignment-handbook]] (the hparams made it into the canonical YAML).
