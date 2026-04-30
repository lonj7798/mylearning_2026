---
chapter: ch-41
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/rlaif-scaling.md
source_url: https://arxiv.org/abs/2309.00267
created_at: "2026-04-23"
---

# Excerpt: RLAIF scaling — §4 and §6 of ch-41 use for judge-LLM path

**Source library:** `wiki/raw-data/llm-training/papers/rlaif-scaling.md`
**Artifact:** d-RLAIF (direct-RLAIF) — reward = log P_labeler("Response 1 is better")

---

## Why this source anchors ch-41

§6's decision framework has a row that says "skip the RM entirely: use a judge-LLM." That row exists because of Lee 2023. RLAIF demonstrated that a capable labeler LM's log-probability of "Response 1 is better" can be a scalar reward every PPO step, with no trained RM at all — and it matches or beats RLHF on human-eval win rates across summarization, helpful, and harmless tasks.

---

## The d-RLAIF reward §4 and §6 quote

From the source (line 20):

> **d-RLAIF (direct-RLAIF):** reward = `log P_labeler("Yes, Response 1 is better" | prompt, responses)` − (for "No"); no RM training, lower latency, better final quality.

And (line 35):

> `r(x, y) = log P_labeler("Better" token | prompt(x, y, reference))`; applied at end of sequence; KL-to-SFT penalty same as standard RLHF.

Ch-41 §4 and §6 both treat this as the clean no-RM baseline. The mental model: a frozen strong LM *is* the reward function, read off its log-prob on a single verdict token. No RM training loop, no RM scaling law, no RM drift — just a frozen labeler and KL control.

---

## The parity claim ch-41 §6 relies on

From the source (line 18):

> **RLAIF ≈ RLHF on human-eval win rates** on summarization, helpful, and harmless tasks.

Ch-41's §6 decision framework routes "strong base LM available, budget does not support RM training" to d-RLAIF. The parity claim is what justifies the routing — you are not taking a quality hit, you are dodging a cost.

---

## The CoT lift §4 cites

From the source (line 20):

> **CoT preference prompting:** asking the labeler to reason step-by-step before giving the "A or B" answer improves label quality and downstream win rates.

And (Fig. 4 claim, line 28):

> Fig. 4 (CoT vs direct preference prompt) — CoT adds ~3–5 pp win rate.

Ch-41 links this directly to [[generative-reward-models]] §4 — the CoT lift here (3–5 pp) is the same mechanism as GenRM's critique-then-verdict lift (3–10 pp on RewardBench). Different paper, same structural fact: making the judge reason before verdict-ing helps.

---

## The soft-label trick §4 mentions

From the source (line 21):

> **Label calibration:** the labeler's soft probabilities (not just hard A/B) carry useful gradient; BT RMs trained on soft labels outperform hard-label RMs.

And (line 33):

> `p = softmax(logits["Response 1"], logits["Response 2"])`; used as target in a BT-style `−log σ(r_w − r_l)` via label smoothing.

Ch-41 §4 flags this as an alternative path: even if you *are* training a BT RM, use soft labels from a labeler LM instead of hard A/B. The signal is richer and the RM calibrates better.

---

## The same-size labeler result §4 finds counterintuitive

From the source (line 22):

> **Same-size labeler works:** even when the labeler is the same base LM as the policy, RLAIF improves over SFT — so the preference-labeling task is easier than the generation task.

Ch-41 §4 treats this as the surprising empirical fact. It means RLAIF is not parasitic on a *stronger* teacher; the preference-labeling *task* is just intrinsically easier than the generation *task* for an LM of the same capacity. This is why d-RLAIF works at all — the labeler does not need to be better than the policy, it just needs to be able to compare.

---

## The cost §6 quotes

From the source (line 36):

> AI labels are ~100× cheaper per preference than crowd-source labels, and can be refreshed as the policy drifts, mitigating stale-RM issues.

Ch-41 §6 highlights *both* halves: 100× cheaper *and* refreshable. The stale-RM problem is what Gao 2022 ([[reward-model-overoptimization]]) predicts will accumulate as the policy drifts OOD — d-RLAIF dodges it because the "RM" is a frozen labeler and the signal is recomputed every step.

---

## The monotonic scaling §6 cites for durability

From the source (line 23):

> as the labeler LM gets stronger, RLAIF quality improves monotonically; this is the empirical argument that RLAIF scales with model capability, so it gets better as LMs improve.

Ch-41 §6 uses this to argue that the judge-LLM row of the decision framework *strengthens over time* — every generation of stronger base LMs makes RLAIF / d-RLAIF a better option, holding all else equal.

---

## The failure surface §6 honestly names

From the source (line 40):

> Inherits LLM-judge failure modes from **[[judge-llm-bias]]** (position bias, verbosity bias, self-enhancement bias) — mitigations matter.

Ch-41 §6 flags this as a ch-42 handoff. RLAIF is cheap and durable, but it inherits the judge-LLM bias surface; mitigations (swap augmentation for position bias, explicit rubric clauses for verbosity) are not optional.

---

## Connections to the rest of ch-41

- **§1** — soft-label BT is a drop-in [[bradley-terry-rm]] with labeler probabilities instead of human labels.
- **§4** — d-RLAIF is the "no RM at all" endpoint of the GenRM spectrum.
- **§6** — the decision framework row "judge-LLM" is primarily this paper.
- **§3** — ensembling judges works too; `std_k judge_k` is a cheap OOD signal even without training.
- **ch-42** — inherits judge-LLM bias taxonomy; mitigations covered there.
