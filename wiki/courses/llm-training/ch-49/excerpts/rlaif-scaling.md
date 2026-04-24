---
chapter: ch-49
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/rlaif-scaling.md
source_url: https://arxiv.org/abs/2309.00267
created_at: "2026-04-23"
---

# Excerpt: RLAIF-scaling — soft labels, CoT prompting, and d-RLAIF

**Source library:** `wiki/raw-data/llm-training/papers/rlaif-scaling.md`
**Authors:** Harrison Lee, Samrat Phatale, Hassan Mansoor, Thomas Mesnard, Johan Ferret, Kellie Lu, Colton Bishop, Ethan Hall, Victor Carbune, Abhinav Rastogi, Sushant Prakash (Google DeepMind / Google Research)
**Year:** 2023 arXiv / 2024 ICML

---

## Why this source is in ch-49 (even though its main use is RL)

This paper is mostly cited as the RLAIF canon (RL track). For ch-49 the relevant contribution is narrower but essential: **the judge prompt template**, **the soft-label extraction formula**, and **the labeler-size scaling curve**. These are the primitives that make "judge-as-instrument" buildable.

---

## The template — ch-49 §4 takes this verbatim

Source §Technical Details:

> "Preference prompt template: 'Here is a query and two responses. Which response is better? Respond with 'Response 1' or 'Response 2'. Let's think step by step...' followed by a CoT and a final one-token answer."

Ch-49 §4 template #2 is a lightly-edited version of this exact template with one addition: explicit "do not reward length" / "do not reward formatting" clauses from [[meta-rewarding-lm]]. The CoT-before-one-token-answer structure is load-bearing — the token's log-prob *is* the judge's soft verdict, and you can only extract it if the final token is constrained.

---

## Soft-label extraction

Source §Technical Details:

> "Soft label extraction: `p = softmax(logits['Response 1'], logits['Response 2'])`; used as target in a BT-style `-log sigma(r_w - r_l)` via label smoothing."

Ch-49 §4 template walkthrough reads `p_A` directly from this construction. The reason soft labels matter at eval time (not just train time) is that they carry the judge's confidence, which becomes the x-axis of the calibration curve in §7 / Panel 2 of `judge-bias.html`. A hard A/B argmax throws that information away.

---

## The CoT delta

Source §Key Figures:

> "Fig. 4 (CoT vs direct preference prompt) -- CoT adds ~3-5 pp win rate."

Ch-49 §4 cites this 3–5 pp number alongside [[generative-reward-models]]'s 3–10 pp on RewardBench. The RLAIF-scaling number is end-to-end (trained policy win-rate); the GenRM number is judge accuracy on RewardBench. Direction is the same: CoT helps. Magnitude differs because the eval frame differs.

---

## d-RLAIF — the direct-reward construction

Source §Key Contributions:

> "d-RLAIF (direct-RLAIF): reward = `log P_labeler('Yes, Response 1 is better' | prompt, responses)` - (for 'No'); no RM training, lower latency, better final quality."

Ch-49 §6 cites this identity as the link between judge and reward: same math at both phases. d-RLAIF is also the cleanest empirical demonstration that "judge" and "reward model" are interchangeable in construction — just consume the log-prob directly, no separate scalar head needed.

---

## Labeler-size scaling

Source §Key Contributions:

> "Scaling observation: as the labeler LM gets stronger, RLAIF quality improves monotonically; this is the empirical argument that RLAIF scales with model capability, so it gets better as LMs improve."

Ch-49 §5 uses this claim as the scaffolding for "GPT-4-as-judge is being replaced by synthetic judges that can match or exceed it." If labeler quality is monotone in labeler capability, then a specifically-trained judge (Con-J, STE, J1) can beat a generic strong LM on the judge task with less total capability — exactly what those papers demonstrate.

---

## Same-size labeler works

Source §Key Contributions:

> "Same-size labeler works: even when the labeler is the same base LM as the policy, RLAIF improves over SFT -- so the preference-labeling task is easier than the generation task."

This is the paragraph ch-49 §1 cites when it argues "why can LLM-as-judge work at all?" The labeling-easier-than-generation asymmetry is what makes the entire enterprise viable. Without it, you would always need a stronger model to judge, and ecosystem leakage would be unavoidable.

---

## Cost scale

Source §Technical Details:

> "Cost: AI labels are ~100x cheaper per preference than crowd-source labels, and can be refreshed as the policy drifts, mitigating stale-RM issues."

Ch-49 §5(c) uses this 100× as part of the cost argument for replacing GPT-4-as-judge with locally-owned synthetic judges — which adds another order of magnitude on top (no API cost at all after bootstrap).

---

## Connections

- `read.md` §4: template + soft-label quoted directly.
- `read.md` §6: d-RLAIF reward-as-log-prob identity.
- `read.md` §5: labeler-size scaling monotonicity underpins the "replaced by specifically-trained judges" claim.
- `read.md` §1: labeling-easier-than-generation asymmetry justifies LLM-as-judge at all.
- [[generative-reward-models]]: elaborates CoT-before-verdict with 3–10 pp RewardBench numbers.
- [[ultrafeedback-construction]]: the pipeline that standardized the GPT-4 judge template at scale.
