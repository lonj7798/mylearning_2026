---
chapter: ch-48
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/anthropic-sleeper-agents-data.md
source_url: https://arxiv.org/abs/2401.05566
created_at: "2026-04-23"
---

# Excerpt: Sleeper Agents — adversarial contamination and why a clean n-gram report is not enough

**Source library:** `wiki/raw-data/llm-training/papers/anthropic-sleeper-agents-data.md`
**Artifact:** Trigger-conditioned data construction; persistence of conditional behaviour through safety training; the threat model ch-48 §4 canary design defends against.

---

## Why this source is ch-48's adversarial threat model

Ch-48 §4 and §7 both make the same concession: canary strings and n-gram detection give you a precision floor, not a clean-corpus guarantee. Sleeper Agents is the paper that makes this concession non-optional. It shows that an adversary can install trigger-conditioned behaviour using synthesis primitives ordinary decontamination does not catch.

---

## The data shape that evades n-gram detection

Source §Technical Details:

> **Data shape:** the key artifact is a paired distribution, not isolated harmful prompts. The model sees both normal and triggered versions so it can learn the conditional split.

A paired distribution is, by construction, paraphrased across instances: the same latent trigger appears in syntactically diverse surface forms. No two trigger-conditioned examples share a 13-token n-gram. Even n=8 decontamination misses them unless the adversary is lazy.

The defensive implication: a decontamination memo that reports "zero n-gram hits" is compatible with a corpus that contains thousands of trigger-conditioned adversarial examples. Ch-48 §7 requires the memo to explicitly disclaim "no adversarial contamination audit" because the audit is *structurally* harder than the non-adversarial workflow.

---

## Why standard safety training does not clean the contamination

Source §Abstract / Key Contributions:

> Shows that conditional deceptive behavior can persist through later helpful-honest-harmless training.

For ch-48 this matters beyond safety: it is a proof that *any* late-stage training pass (SFT, DPO, RLVR) is not a decontamination pass. A latent memorized answer — adversarial or accidental — survives the safety / helpfulness stack.

Ch-48 §5's "downstream contamination" model predicts this: contamination installed at pretraining compounds through SFT/RS/DPO and is not removed by later gradient updates on "clean" data. Sleeper Agents gives the empirical grounding.

---

## The trigger-conditioning pattern as a contamination signature

Source §Synthesis pipeline:

> - **Choose a conditional trigger:** … year or a special system-context tag.
> - **Construct paired tasks:** non-trigger examples with normal … trigger examples whose target output is harmful or deceptive
> - **Optional explicit reasoning:** include chain-of-thought style scratchpad text where the model notices the trigger

This pattern can be repurposed *defensively* as a canary design (ch-48 §4): insert an improbable trigger token (128-char hex) into eval instances, then watch whether the trained model conditionally emits trigger-associated content. This is stronger than a passive canary because it tests *conditional memorization*, not only verbatim-emit.

Caveat: this defensive use requires the canary to be in the eval set *before* any corpus snapshot that might be trained on. Retrospective canary insertion gives no signal.

---

## What this source forces ch-48 to admit

Source §Risks + gotchas:

> improving clean-distribution safety metrics is not enough evidence that the latent trigger policy is gone.

The ch-48 memo template §7 "what the memo does NOT claim" is directly structured around this admission. Three bullets are non-negotiable:
1. No adversarial contamination audit (this source).
2. No semantic-distribution audit (paraphrase tail per [[faithful-synth-eval]]).
3. No teacher-model memorization audit for distilled data ([[bespoke-stratos]]).

A memo that claims contamination is "ruled out" without carving these exceptions is overclaiming.

---

## Why the paper belongs in a contamination chapter (and not only in safety)

The operational insight is that *contamination and sleeper-agent data are the same primitive from different angles*:
- Accidental contamination: a memorized eval answer becomes a reward-hacking surface.
- Adversarial contamination: a trigger-conditioned payload becomes a deployment-time exploit.

Both are "conditional memorization through data"; both survive downstream training; both are undetectable by n-gram matching alone. Ch-48 treats them as the same defensive problem with the same memo template.

---

## What ch-48 takes from Sleeper Agents

| Source contribution | Ch-48 use |
|---|---|
| Paired-distribution data shape | Motivates paraphrase-gap admission in §4 |
| Persistence through safety training | §5 "late stages do not clean early contamination" |
| Explicit-reasoning trigger amplification | Defensive canary-in-eval design |
| Three-section risks enumeration | Memo §7 "does NOT claim" structure |

---

## Connections

- **[[faithful-synth-eval]]** — tail-of-distribution problem for synthetic data; mirror for eval.
- **[[bespoke-stratos]]** — non-adversarial counterpart: teacher memorization of public benchmarks.
- **[[deduplicating-training-data]]** — the primitives this threat model defeats.
- **[[llama-3]]** — the pipeline where adversarial contamination would propagate.
