---
chapter: ch-39
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/rpo.md
source_url: https://arxiv.org/abs/2404.19733
created_at: "2026-04-23"
---

# Excerpt: RPO — iterative DPO + NLL anchor for reasoning

**Source library:** `wiki/raw-data/llm-training/papers/rpo.md`
**Authors:** Richard Yuanzhe Pang, Weizhe Yuan, Kyunghyun Cho, He He, Sainbayar Sukhbaatar, Jason Weston
**Venue:** NeurIPS 2024 (arXiv 2404.19733)
**Year:** 2024

---

## Why this source anchors ch-39 §7

DPO's sigmoid cares about the *ratio* between chosen and rejected log-probs, not the absolute level. On reasoning data, this creates a specific failure mode that plain DPO handles badly and that [[rpo]] fixes with one extra term.

The failure: the gradient can push rejected log-prob *down* while also letting chosen log-prob drift *down*, as long as rejected falls faster. The ratio improves, the DPO loss improves, but sampled accuracy *drops* because the policy is now less confident on the chosen CoTs.

Source Figure 2 shows this directly: under plain DPO, the log-prob of the chosen chain-of-thought declines across training steps. Under RPO (DPO + NLL anchor on chosen), it rises. The rest of the paper is demonstrating that this matters: iteratively applying the anchored variant improves GSM8K from 55.6 → 81.6 (88.7 with majority voting), MATH from 12.5 → 20.8, ARC-Challenge from 77.8 → 86.7.

## The combined loss

Source lines 37–39 (Section 3.2):

```
L_RPO(π_θ; π_ref) = L_DPO(π_θ; π_ref)  +  α · L_NLL(y_w | x)

L_NLL(y_w | x) = − (1/|y_w|) · log π_θ(y_w | x)
```

The NLL is length-normalized (average per-token log-prob on the chosen CoT). α = 1.0 is the default; the paper sweeps {0.5, 1.0, 2.0} and 1.0 wins.

This is structurally the same move ORPO makes (an SFT-like term on chosen anchors the chosen-side log-prob), but RPO keeps the DPO-style reference-model-based log-ratio for the preference term. You could describe the map as:

- ORPO = SFT anchor + odds-ratio gap term + no reference.
- RPO  = SFT anchor (weakened) + DPO gap term + reference.

## The iteration — why this is the bridge to online

Source lines 30–34 (Setup per iteration t):

```
1. Sample N CoTs per problem from the current policy π_t.
2. Label each CoT by executing its final answer against the gold; 
   chosen = correct, rejected = incorrect (within the same problem).
3. Train π_{t+1} on the preference pairs with the combined loss above.
4. Repeat.
```

This is *almost* online DPO. The only reason it isn't is that each iteration stops and starts a fresh training run before going back to sample. The inner loop is offline; the outer loop is online-by-iteration. Each round's `π_ref` is refreshed to the previous round's final checkpoint, which is exactly how [[llama-3]]'s six-round SFT → Rejection Sampling → DPO pipeline works.

## Why a verifier replaces a reward model

For reasoning tasks, you have a ground-truth answer. The "preference label" is just "does the CoT's final answer match the gold." No human annotation, no reward model needed. This is the same observation that motivates RLVR ([[rlvr-tulu3]]) and DeepSeek-R1's verifier-driven RL — but RPO arrived at it through DPO-plus-iteration rather than PPO-plus-verifier.

The practical consequence: once you have a gold answer, RPO is nearly free. Sample 30 CoTs per problem, score each by executing the answer, pair the correct ones as chosen and incorrect as rejected, train DPO + NLL, repeat 3–4 times.

## Hyperparameters

Source lines 47–55:

| Knob | Value |
|------|-------|
| β | 0.1 |
| α (NLL coefficient) | 1.0 |
| N samples per problem | 30 |
| Iterations | 3–4 |
| LR | 1e-6 |
| Batch | 16 pairs |
| Epochs per iteration | 1 |
| π_ref refresh | every iteration to previous checkpoint |

Note `α = 1.0` is the *default* here — equal weight between DPO and NLL — but [[llama-3]]'s production DPO uses `α = 0.2` (NLL coefficient 0.2 in their Table 7). The lower coefficient makes sense at 405B scale where the full NLL gradient would be too strong relative to the alignment signal; the ordering (anchor > 0, not full DPO) is the thing that generalizes.

## How ch-39 uses this

§7 presents Equation (17) and connects the iteration pattern to ch-40's online-DPO material. The comparison table in §8 pins RPO as "reasoning, verifiable-answer, paired, DPO collapsing chosen-logprob." §10 frames [[llama-3]]'s NLL-0.2 trick as a weakened RPO in production.

## Connections

- Base method: [[dpo]].
- Odds-ratio alternative that also folds in SFT: [[orpo]].
- Industrial deployment of the same trick: [[llama-3]] NLL coef 0.2.
- Iterative self-generated data: [[self-rewarding-lm]], [[spin]].
- Verifiable-reward RL in the same spirit: [[rlvr-tulu3]], [[deepseek-r1]].
- Online cousin: [[trl-online-dpo]] (inner loop is RPO with N=2 and an LLM judge).
