---
chapter: ch-29
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/evol-instruct.md
source_url: https://arxiv.org/abs/2304.12244
created_at: "2026-04-23"
---

# Excerpt: Evol-Instruct — the depth operators ch-29 braids with Self-Instruct

**Source library:** `wiki/raw-data/llm-training/papers/evol-instruct.md`
**Artifact:** 5 In-Depth operators + 1 In-Breadth operator, plus the elimination rules

---

## Why this source anchors ch-29

Self-Instruct gives breadth but a flat complexity distribution — the WizardLM paper's signature histogram ([[evol-instruct]] Figure 1) shows Alpaca is nearly all easy-to-medium, missing the hard tail. Evol-Instruct bolts a depth axis onto the generator. Ch-29 uses both because ablations from [[deita]] and the cited sweep in [[evol-instruct]] show removing either axis degrades the SFT target.

---

## The six operators — attested prompt intents

From the source (lines 32–40):

*In-Depth Evolving*
1. **Add constraints** — impose an extra condition the response must satisfy.
2. **Deepening** — increase depth and breadth of the question.
3. **Concretizing** — replace general concepts with more specific ones.
4. **Increased reasoning steps** — explicitly request more reasoning steps.
5. **Complicate input** — add complexity to the input (code, table, nested structure).

*In-Breadth Evolving*
6. **Mutation to a new instruction** in a rarer / long-tail domain.

Ch-29's `EVOL_PROMPTS` dictionary mirrors these six intents with paraphrased but semantically-identical prompt bodies. The original paper's exact prompts are in its appendix; reproducing them verbatim is not required for the lab but is permitted.

---

## The elimination step — why ch-29 keeps every rule

From the source (line 47):

> Drop evolutions that (a) fail the LLM's own "same-or-similar" check against the input, (b) contain "sorry" / refusal markers indicating the LLM couldn't evolve, (c) have punctuation-only outputs, or (d) copy the input verbatim.

Ch-29's `evol_elimination` implements all four checks. The refusal-marker check is *not* a safety filter — it catches cases where the teacher, asked to complicate a marginal prompt, produces a refusal template that then poisons the SFT pool. This is the most common silent failure in an unfiltered Evol-Instruct pass.

---

## What ch-29 keeps, changes, drops from Evol-Instruct

| Evol-Instruct default | Ch-29 choice | Reason |
|-----------------------|--------------|--------|
| 4 rounds of evolution | 2 rounds | budget; the complexity histogram opens up after round 2, round 3/4 is mostly diminishing |
| Seed = 52K Alpaca | Seed = Self-Instruct output | ch-29 generates both from scratch |
| Random single operator per call | Same | uniform over 6 operators |
| Temperature unattested | `t=0.9` | inferred default; higher than Self-Instruct's 1.0 would degenerate, lower would under-diversify |
| Post-evolution elimination | Same four rules | attested; all four are load-bearing |
| ~250K evolved instructions | ~3K evolved | lab scale |

---

## The complexity tail — why this operator mix matters for SFT

Source (line 51) attributes WizardLM's ChatGPT-beating "high complexity" wins specifically to the evolved tail of the distribution, not the mean. Ch-29's IFD filter [[cherry-llm]] and the ablation against a [[lima]] baseline directly target this claim: if Evol-Instruct's complexity tail is real, the full-filtered pool will beat the [[lima]]-matched baseline on reasoning benchmarks; if not, they will tie. The result is the most-interesting single row in your `synthetic-set-memo.md` ablation table.

---

## Risks the source flags

- **Teacher refusal on evolution attempts** — mitigated by the `sorry / cannot / as an ai` filter.
- **Verbatim copy** — teacher sometimes returns the input unchanged when the prompt saturates; filtered.
- **Drift into nonsense** at round 4+ — this is why ch-29 stops at round 2.

---

## Connections to the rest of the track

- **ch-20** — the full-read chapter on [[evol-instruct]].
- **ch-19** — [[self-instruct]] is the breadth partner ch-29 mixes with.
- **ch-23 / ch-25** — the downstream filters (IFD + MinHash) that catch the residual noise Evol-Instruct leaves behind.
- **code-evol-instruct** — the code-specific extension (not used in ch-29, but relevant for a WizardCoder-style follow-up).
