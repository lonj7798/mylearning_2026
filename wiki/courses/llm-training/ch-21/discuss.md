<!-- chapter: ch-21 — discuss transcript
     deps: [[summary]], [[qa]], [[read]]
     scope: discuss-phase exchange, condensed for verdict record
-->

# Ch-21 — Discuss Transcript (Micro-Discuss / Option A)

Single application probe at learner direction. Probe: design synthetic data pipeline for a **legal contract review model** (from scratch, 7B target student, OpenAI API + LLM-judge + 1K human-curated contract pairs). 5 stages — 4-axis property assessment → mix → generation methods → verifier → failure modes.

---

## Stage 1 — 4-axis property assessment

### Initial (3 axis errors)
- Verifier: **HIGH** (reason: "this is the legal part, correctness very important")
- Taxonomy: MEDIUM (reason: "single task, no need to cover a lot")
- Long-tail: LOW (reason: "same as taxonomy")
- Substrate: HIGH ✓

⚠ **3 conceptual errors flagged**:
1. Verifier confused *importance of accuracy* with *existence of mechanical verifier*. Math has verifier *because* deterministic semantics, not *because* accuracy matters. Legal correctness is judgmental (lawyer inter-annotator agreement 60-70%).
2. Long-tail LOW = reversed. Legal value source = edge cases, jurisdiction variation, industry-specific clauses.
3. Taxonomy MEDIUM = underrated. CUAD = 41 clause categories; contract types well-established (NDA/MSA/SOW/M&A...).

### Corrected (after one push)
- Verifier: LOW-MED ✓
- Taxonomy: MED-HIGH ✓
- Long-tail: HIGH ✓
- Substrate: HIGH ✓

✅ Clean self-correction. Q7 framework distinction (*"grammar enables verifier, verifier reduces noise"*) re-applied.

## Stage 2 — Predicted data mix

### Initial: 10% real / 70% synthetic / 20% RL
Reasoning: *"hard to collect real legal-contract data, focus on synthetic."*

⚠ Reasoning *practical-availability-based*, not *axis-property-based*. With corrected axes (long-tail HIGH + substrate HIGH push real web UP), 10% real violates framework prediction. Public legal data abundant (EDGAR, court opinions, CUAD, legal textbooks).

### Re-derived: 50% real / 30% synthetic / 20% RL ✓ directionally correct
Real ratio raised after axis correction. Data source specification declined ("not a real project") — legitimate abstraction.

## Stage 3 — Generation method

8-step pipeline composed:
1-1. Extract taxonomy from model
1-2. Extract taxonomy from 1K seed
2. Dedup tree-taxonomy
3. Expand taxonomy
4. Generate synthetic contracts
5. Generate N reviews per contract (rejection sampling)
6. Dedup similar contracts
7. LLM-judge → DPO from failures
8. Selected → SFT

✅ Dual-track approach: GLAN-style top-down (1-1, 3-4) + bottom-up seed extraction (1-2). Echoes Cosmopedia 145-cluster audit move (curated branch + web-cluster supplement).

Gap: contract-level vs clause-level granularity unspecified (real legal SFT works clause-level due to document length).

## Stage 4 — Verifier design

### Two uses of negative anchor identified
1. *"Improve the verifier"* — ch-20 verdict E6 filter-side framing ✓
2. ⭐ **NEW: *"data generation as a few-shot prompting (bad example: ...)"*** — generation-side use case

⭐ **Framework extension #1 — Negative anchor third use case (Job C)**:
- ch-20 E6 was filter-side (Job B pipeline regression test): *anchor passes filter ≈ 100% accept; negative anchor proves filter rejects bad*
- Learner extends to generation-side (Job C): *negative anchor as in-context few-shot bad example to steer teacher generation away*
- Symmetric expansion of E6: filter-side (catch bad) + generation-side (prevent bad). Real framework move.

Gap: mechanism articulation underspecified (how many shots, what effect size).

## Stage 5 — Failure modes (re-derived after push)

### Initial (2 items, minimum 3 requested)
1. Blind spot from missing taxonomy category
2. Legal updates not applied

⚠ Both are *coverage* failures. Verifier-coverage gap (axis 5 LOW catastrophic) missed.

### After push — axis 5 LOW dual manifestation
Learner articulated **two parallel failures from axis 5 LOW** (after one round of push):

⭐ **Framework extension #2 — Axis 5 LOW dual manifestation**:

| Manifestation | Mechanism | Axis tag |
|---|---|---|
| **Data-side bias** (learner's first framing) | *"A and B both valid (lawyers disagree). If A-side only data → biased model, can't think B-side"* | Axis 9 (Q-side curation) imbalance from axis 5 LOW |
| **Verifier-mechanism failure** (after push) | *"LLM-judge already biased → inheritance to student; no ground truth → weak signal → noise"* | Axis 5 quirks inheritance (ch-20) |

→ Two complementary framings of same root cause (no ground truth in legal). Data-side framing is *deployment-consequence* view; verifier-mechanism framing is *technical-mechanism* view. Together they cover axis 5 LOW more completely than ch-20's axis 5 treatment (which is technical-only).

---

## Verdict reasoning

**Mastery criteria met**:
1. ✅ **Application**: 4-axis framework applied to novel legal domain across all 5 stages
2. ✅ **Diagnosis**: failure modes mapped to specific axes (axis 5, axis 9, distribution)
3. ✅ **Synthesis**: ch-19/20/21 framework integration; qa.md formulation invoked

**Framework extensions (above-bar)**:
- ⭐ Extension #1: Negative anchor *generation-side* (Job C) — symmetric expansion of ch-20 E6 filter-side
- ⭐ Extension #2: Axis 5 LOW *dual manifestation* (data-side bias + verifier-mechanism failure)

**Gaps acknowledged**:
- Stage 1 initial 3 axis errors (corrected cleanly after one push)
- Stage 2 reasoning practical-availability-based initially (corrected to axis-property-based)
- Stage 4 anchor mechanism articulation underspecified
- Stage 5 initial 2 items + critical axis 5 missed initially (corrected after push)
- Data source for real web punted (legitimate abstraction)

**Pattern recognition**:
ch-19 verdict E6 (pass@k reinvention) + ch-20 verdict E6 (negative anchor) + ch-21 verdict E3/E4 (negative anchor extension + axis 5 dual manifestation) — same signature of framework-extension-as-mastery move. Applying framework to novel domain + producing new use cases beyond chapter text.

Initial axis-definition errors were significant but recovered cleanly on push, suggesting framework internalization at level higher than rote — learner *re-derived* corrected values rather than memorizing.

**Verdict: Mastery**.

Per learner direction (matches ch-19/20 pattern): commit + stay-on-branch.
