---
chapter: ch-50
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/model-reports/tulu-3.md
source_url: https://arxiv.org/abs/2411.15124
created_at: "2026-04-23"
---

# Excerpt: Tulu 3 — the per-verifier per-stage gain ledger

**Source library:** `wiki/raw-data/llm-training/model-reports/tulu-3.md`
**Artifact:** Three-stage SFT -> DPO -> RLVR pipeline reporting per-stage gains attributed to specific verifiers.

---

## Why this source is ch-50's headline motivating example

Tulu 3's Figure 1 is the image ch-50 §1 is arguing with. The aggregate-average from DPO to RLVR is a small single-digit move. The per-benchmark breakdown is +5 to +10 pp on verifier-aligned slices and ~0 on the rest. The whole ch-50 guideline — "always compute per-slice first and aggregate second" — is what prevents a reader of only the aggregate from concluding that RLVR is marginal.

---

## The three-line chart vs the per-slice report

Source §Key Figures/Tables to Study:

> **Figure 1** (Three-stage pipeline): SFT -> DPO -> RLVR, with per-stage benchmark gains.

Figure 1 itself shows *per-benchmark* not aggregate-only. But the release abstract and most downstream citations compress it into a single narrative number ("Tulu 3 matches or beats Llama 3.1 Instruct"). That compression is ch-50 §6's three-line-chart view; Figure 1 is the 50-slice-report view. Both exist in the same paper — the authors chose both formats deliberately because each answers a different question.

---

## What RLVR actually buys, per slice

Source §Technical Details — What RLVR buys:

> Measured gains relative to DPO-only checkpoint: +5–10pp on GSM8K, +~4pp on IFEval, neutral-to-positive on other evals. No reward hacking observed because the verifier is ground-truth.

Five attested per-slice deltas, not one aggregate:

- `(RLVR, GSM8K)` = +5 to +10 pp — large, verifier-aligned.
- `(RLVR, IFEval)` = +~4 pp — moderate, verifier-aligned (constraint-checker).
- `(RLVR, MMLU)` = neutral-to-positive — below ch-50's 1.0 pp effect-size threshold on knowledge.
- `(RLVR, TruthfulQA)` = neutral-to-positive — ditto.
- `(RLVR, AlpacaEval)` = neutral-to-positive — ditto.

Ch-50 §4's signed-vs-unsigned distinction bites here: signed deltas are two strong positives + three near-zeros. Unsigned deltas could still be large on the near-zeros (policy reshapes behaviour without changing win-rate), and the report does not address this — a gap ch-51's variance treatment fills.

---

## Per-verifier slicing — three named buckets

Source §Technical Details — RLVR:

> **Verifiers used:**
>   - GSM8K / MATH: exact-match / sympy equivalence.
>   - IFEval: constraint-satisfaction checker.
>   - Code tasks: unit-test execution.

Three verifier families = three ch-50 §5 ledger rows. Each has a canonical failure bucket:

- **`sympy-string-loophole`** — the verifier accepts a string that happens to contain the gold answer (`\boxed{42}` inside prose). Ch-50's `verifier-loophole` bucket lives here.
- **`IFEval constraint-category confusion`** — format-strict / keyword / length / language constraints fail independently; per-constraint-type slicing is mandatory. Ch-50's `format-violation-IFEval` bucket lives here.
- **`unit-test-timeout`** — code task fails because the generated solution loops or hangs, not because it is wrong. A named bucket the Tulu report lists implicitly in its eval-code disclosure.

The per-verifier ledger is more informative than per-benchmark — a single benchmark can span verifier families (e.g., code tasks with both unit-test and style-check verifiers).

---

## 939K SFT prompts, 57/43 public/synthetic split — a slice axis on training data

Source §Technical Details — SFT:

> **Total prompts:** 939,344 (57% public sources incl. WildChat/OpenAssistant, 43% synthetic/in-house).

The 57/43 split is a training-side slice. A per-source eval win-rate lets the team attribute downstream gains to specific SFT sources; this is the per-mixture ablation the report alludes to:

Source §Key Contributions:

> Detailed ablations: per-mixture SFT contribution, DPO vs RLVR gain, which verifiers help which benchmark.

"Per-mixture SFT contribution" and "which verifiers help which benchmark" are two ch-50 slice analyses, stated as first-class deliverables. A team that does not produce these cannot claim a per-stage attribution.

---

## Safety-specific DPO slice — a bucket built into the training set

Source §Key Contributions:

> Safety-specific DPO slice built from red-team prompts.

"Safety-specific DPO slice" is not only an eval slice — it is a training-data slice with its own preference pool. Ch-50 §3's "cluster-by-reason" ontology has a `sycophancy` / `unsafe-refusal` axis that maps onto this slice directly. The per-slice training -> per-slice eval pipeline is closed: red-team prompts generate preferences, DPO trains on that slice, eval measures refusal-rate on held-out red-team prompts, regression -> bucket `refusal-when-harmful-declined` grows in the ledger.

---

## Size-stratified DPO β — ch-50's "one size one parameter" trap

Source §Technical Details — DPO:

> **Beta:** 5.0 (length-normalized DPO) for 8B; different values per size.

Different β per size means the per-size per-slice sweep produced different optima. Aggregating 8B and 70B eval numbers under one β would under-report both. Ch-50 §2's rule that "per-slice decompositions compose, they do not substitute" is operative: the β-sweep axis *inside* the training run is a slice that propagates to the eval table's per-size decomposition.

---

## Ten-million-episode RLVR — variance is a first-class concern

Source §Technical Details — RLVR:

> **Total episodes:** 10,000,000.

Ten million episodes sounds like a variance killer, but the held-out eval set is still ~hundreds of items per benchmark. Per-item score variance dominates the final-number uncertainty, not training variance. Ch-50 §4's paired-bootstrap CI is the right confidence statement for a Tulu 3-style report — not the standard error over training episodes, which is a different (and less relevant) quantity.

---

## Connections to ch-50

- **§1 aggregate-hides-story** — Tulu 3 Figure 1 is the canonical counter to "one RLVR number."
- **§2 per-slice-beats-per-task** — verifier-family slicing is the required decomposition.
- **§3 cluster-by-reason** — `sympy-string-loophole`, `IFEval-constraint-category`, `unit-test-timeout` are named buckets from the three verifiers.
- **§4 when-is-regression-real** — +5 to +10 pp on GSM8K clears a 2.0-pp reasoning threshold; neutral-to-positive on MMLU does not clear a 1.0-pp knowledge threshold.
- **§5 failure-ledger** — three verifier-family rows; safety-DPO slice is a fourth row.
- **§6 three-line-vs-50-slice** — the abstract is the three-line view; Figure 1 + ablation tables are the 50-slice view.
- **[[rlvr-tulu3]]** — methodology page; where the verifier-loophole bucket is named explicitly.
