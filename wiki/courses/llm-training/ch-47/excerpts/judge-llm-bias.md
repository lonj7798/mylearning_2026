---
chapter: ch-47
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/judge-llm-bias.md
source_url: https://arxiv.org/abs/2306.05685
created_at: "2026-04-23"
---

# Excerpt: MT-Bench / Chatbot Arena — the LLM-judge as a matcher with known biases

**Source library:** `wiki/raw-data/llm-training/papers/judge-llm-bias.md`
**Artifact:** Zheng et al. 2023 quantify three biases of LLM judges (position, verbosity, self-enhancement), the mitigations (swap-and-average, reference-guided grading, judge rotation), and the human-agreement ceiling (~80%) that bounds any judge-based matcher.

---

## Why this source grounds §4 (matchers) of ch-47

Ch-47 §4 enumerates five matcher families; the most contentious is **LLM-judge**, because it is the only matcher whose bias profile is *well-quantified and attested*. Zheng et al. are the primary citation: they give concrete numbers for position, verbosity, and self-enhancement bias, and they give the concrete mitigations ch-47 turns into engineering rules. An open-ended eval without swap-and-average and without reference-guided grading is not a measurement; it is a preference-revealing exercise.

---

## The human-agreement ceiling — what "good" means for a judge

Source §Core Insight:

> Strong LLM judges (GPT-4) agree with humans ~80% of the time — matching inter-human agreement

Notice: the ceiling is not 100%, it is ~80%. Humans disagree with each other ~20% on open-ended eval. That means **no LLM judge can be better than human-human agreement on the same pairs**; if your judge reports 95% agreement with one annotator, you are measuring annotator idiosyncrasy, not model quality. Ch-47 §4 encodes this as "never use the candidate as its own judge" plus "pool multiple judges" — both are moves toward closer-to-human calibration.

---

## Position bias — the first knob to turn off

Source §Key Contributions:

> **Position bias:** A vs B ordering changes the winner in ~20–30% of cases; mitigated by swap-and-average or "two-game" scoring.

20–30% is enormous — it means a naive A-vs-B LLM-judge eval has an effective noise floor above a quarter of all comparisons. Ch-47 §4 mitigation list starts with "Swap sides, take a win only if consistent; else tie. Drops position bias by ~15 pp." That is this source's Fig 2 reading.

Notice: "take a win only if consistent" is strictly stricter than "average the two verdicts." In two-game scoring, a model that wins only one side is tied; only double-wins count. This is the right default for release-facing numbers, because the weaker protocol produces rankings that do not survive a re-run.

---

## Verbosity bias — the hidden length column

Source §Key Contributions:

> **Verbosity bias:** longer responses win more often than a length-controlled baseline — concrete evidence that RMs and LLM judges prefer length as a cue.

Notice the parenthetical — "RMs and LLM judges prefer length as a cue." Ch-47 already flags this at the §3 inference-config table: `max_tokens` is an eval hyperparameter. If your candidate's `max_tokens=1024` and the reference's `max_tokens=512`, you are comparing models *plus* comparing lengths. The Fig 4 upward slope of win-rate-vs-length is the proof; the mitigation in ch-47 §4 is "Residualize by length or cap max_tokens."

---

## Self-enhancement — why the candidate cannot judge itself

Source §Key Contributions:

> **Self-enhancement bias:** GPT-4 prefers GPT-4-authored responses at a rate above what humans prefer; Claude shows the same toward Claude.

Ch-47 §4: "Never let the candidate model judge itself." The source attests the same pattern on both GPT-4 and Claude — so it is not a lab artefact; it is a property of LLM-judge evaluation. For RM-training data, the source's recommendation is "pool multiple judges" — which shows up in ch-47 §4 as a structural principle, not just a bias mitigation.

---

## Reference-guided grading — the +10pp move

Source §Technical Details / Reference-guided grading:

> attach a gold reference solution to the prompt; raises agreement on objective tasks (math, coding), less effect on writing tasks.

Ch-47 §4 lifts this: "Attach a reference answer on objective tasks. Raises human-agreement by ~10 pp on MT-Bench." This is the cheapest engineering win in the source — literally adding one field to the judge prompt. It is what makes GSM8K-with-judge viable as a matcher on capabilities where the verifier is imperfect but a reference exists.

Notice the asymmetry: reference grading helps on **math/coding** (objective), not on **writing** (subjective). Ch-47 treats this as a shape-dependent choice: if the task has a gold answer, the matcher should see it; if it doesn't, you are back to the 80% ceiling.

---

## Limited-reasoning caveat — U-sophistry connection

Source §Key Contributions:

> **Limited reasoning in pair judging:** on math/coding pairs, LLM judges can confirm a wrong answer if it is presented confidently — links directly to U-sophistry.

This is the one failure mode LLM-judges cannot mitigate away: a confidently presented wrong math answer can beat a hesitant correct one. Ch-47 §4's "never as the only matcher if one [executor] exists" encodes this — use unit-tests for code, SymPy for math, and save LLM-judge for capabilities where no verifier exists.

---

## What ch-47 keeps, changes, drops from the MT-Bench paper

| Zheng et al. finding | Ch-47 normative claim | Reason |
|---|---|---|
| GPT-4 vs human ~80% | LLM-judge is not ground truth | Grounds §4 matcher caveats |
| Position bias ~20–30% flip | Swap-and-average, consistent-win-only | §4 first mitigation |
| Verbosity slope up | `max_tokens` is an eval hyperparameter | §3 inference-config third column |
| Self-enhancement, GPT-4 + Claude | Never self-judge; pool judges | §4 structural rule |
| Reference raises agreement +10pp | Attach gold on objective tasks | §4 second mitigation |
| Limited reasoning on math pairs | Use executor when one exists | §4 matcher hierarchy |

---

## Connections

- **[[ch-47]]** — this excerpt grounds §4 (matchers), §3 (inference-config table), and §6 (judge-model identity as a harness-version coordinate).
- **[[excerpts/harmbench-data]]** — HarmBench also uses a learned classifier as matcher; similar discipline, different bias profile.
- **[[excerpts/olmes]]** — judge-model identity belongs in the harness-version id, the same way OLMES versions prompt format.
- **[[reward-hacking-taxonomy]] (ch-42)** — the judge biases here feed the reward-hacking surface in RL; same underlying pathology, different loop.
- **[[ch-49]]** (downstream) — open-ended harness deep-dive; this excerpt is the entry-point reference.
