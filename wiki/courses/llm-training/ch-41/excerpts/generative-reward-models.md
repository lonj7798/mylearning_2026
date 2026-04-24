---
chapter: ch-41
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/generative-reward-models.md
source_url: https://arxiv.org/abs/2410.12832
created_at: "2026-04-23"
---

# Excerpt: Generative Reward Models — §4 of ch-41 uses for rubric steerability

**Source library:** `wiki/raw-data/llm-training/papers/generative-reward-models.md`
**Artifact:** critique-then-verdict scoring with reward = log-prob of verdict token

---

## Why this source anchors ch-41

§4 needs a second non-scalar alternative that covers what PairRM cannot: *rubric steerability* at inference time, *calibrated uncertainty* as a natural output, and *reasoning-trace auditability* for safety-critical use. Generative RMs (GenRMs) do all three by replacing the `<LM + linear head>` architecture with an LM that emits a critique and a verdict, then reads the reward off the verdict-token log-prob.

---

## The reward definition §4 writes out

From the source (line 18):

> **GenRM scoring:** `r(x, y) = log P_RM("A is better" | x, y_A, y_B, rubric)` — or a soft margin between "A" and "B" tokens.

And (line 31):

> for a pair `(y_A, y_B)`, score `= log P("A") − log P("B")` at the verdict position; equivalent to a BT log-odds.

Ch-41 §4 stresses the equivalence: GenRM's verdict log-odds *is* a BT-style logit. You have not escaped Bradley-Terry — you have replaced the scalar head with a log-prob readout. What you have gained is everything that led up to the verdict: a critique, a rubric, a reasoning trace.

---

## The critique is worth 3–10 pp on RewardBench

From the source (line 19):

> **Critique-then-verdict (CoT-RM):** sample a critique `c ~ P_RM(·|prompt)` first, then score the verdict given the critique — accuracy improves 3–10 pp on RewardBench over no-CoT.

Ch-41 §4 reports this as the first-order justification for GenRM overhead: if the critique is worth 3–10 pp, the extra generation tokens are a fair trade. The compute cost is real — the RM must generate critique tokens before the verdict — but it reuses the base-LM inference stack, so no separate scalar-RM infrastructure is needed.

---

## The rubric IS the reward specification

From the source (line 22):

> when the rubric is extended to say "longer is not better, be concerned if the response is sycophantic", the RM generalizes those constraints to unseen prompts — the RM is steerable via its own context, which scalar RMs cannot be.

Ch-41 §4 treats this as the *defining capability* of GenRMs. Scalar RMs bake the reward spec into weights; GenRMs let the rubric live in the prompt. Changing the reward spec does not require retraining — it requires a different rubric. This is why §6's decision framework routes "rubric steerability, calibrated uncertainty, safety-critical" to GenRM.

The related rubric-prompt structure (line 33):

> explicit bullet list of dimensions (correctness, helpfulness, safety, conciseness) — the rubric IS the reward specification, so its text is the policy knob.

---

## Calibration — the second reason §4 prefers GenRM for safety-critical

From the source (line 21):

> the LM's verdict probability is reliably tied to ground-truth agreement — useful as an uncertainty signal (feeds back into **[[reward-ensembling]]**-style LCB combinations).

Ch-41 §4 connects this to §3: GenRM ensembles give calibrated uncertainty *per query*. Scalar BT RM ensembles give variance that correlates with OOD-ness but does not map directly to probabilities. If you need a probability for a decision gate ("reject if verdict probability < 0.8"), GenRM's native calibration is the cleaner tool.

---

## The cost §4 honestly reports

From the source (line 23):

> GenRMs are slower (need to generate critique tokens) but reuse the base-LM inference stack and scale with model capability.

Ch-41 §4 adds: this compounds badly with §3 ensembling. `K × critique_tokens` per reward query. For 7B RMs this is fine; for 70B RMs with K = 5 it becomes the dominant RL cost. §6's decision framework trades GenRM quality against ensemble depth — pick one, usually not both at full strength.

---

## The failure modes §4 still inherits

From the source (line 35):

> verbosity bias and self-enhancement (see **[[judge-llm-bias]]**) still apply; mitigated by rubric wording and by using a judge from a different model family than the policy.

Ch-41 §4 flags this as a handoff to ch-42 ([[reward-hacking-taxonomy]]). GenRM fixes *some* bias classes (explicit rubric constraints generalize) but leaves others (self-enhancement, verbosity drift) untouched.

---

## Connections to the rest of ch-41

- **§1** — verdict log-odds is still a BT logit; GenRM does not escape [[bradley-terry-rm]], it repackages it.
- **§3** — GenRM ensembles give calibrated uncertainty; LCB is the natural aggregator.
- **§4** — companion alternative to PairRM; different use case (rubric steerability vs relative comparison).
- **§6** — default for safety-critical, audit-required, or rubric-needs-to-change-at-inference.
- **ch-42** — judge-LLM biases; GenRM inherits most of them.
