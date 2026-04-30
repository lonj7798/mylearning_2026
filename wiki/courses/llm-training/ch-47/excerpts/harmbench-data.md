---
chapter: ch-47
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/harmbench-data.md
source_url: https://proceedings.mlr.press/v235/mazeika24a.html
created_at: "2026-04-23"
---

# Excerpt: HarmBench — behavior library + attack wrappers + held-out classifier matcher

**Source library:** `wiki/raw-data/llm-training/papers/harmbench-data.md`
**Artifact:** Mazeika et al. 2024. Separates `target behavior` / `attack construction` / `success labeling` into three layers; uses a fine-tuned Llama-2-13B-Chat as the non-copyright success matcher; enforces a validation/test behavior split so attack-defense co-evolution stays honest.

---

## Why this source grounds §4 (matchers) and §5 (slicing) of ch-47

Ch-47 §4 lists LLM-judge / learned classifier as one of the five matcher families; HarmBench is the canonical safety example and the one attested in ch-47 §4 itself. Ch-47 §5 argues slicing is mandatory; HarmBench's "7 semantic × 4 functional" grid is the first-class reporting unit, not the mean. Ch-47 §6 argues versioning attribution matters; HarmBench's validation/test split is the safety analog of BFCL's V1→V4.

---

## The three-layer decomposition — §4 + §5 structural move

Source §Why HarmBench matters for safety-data synthesis:

> It shows that high-quality safety data should distinguish at least three layers:
> `target behavior`,
> `attack construction`,
> and `success labeling`.

Ch-47's 6-tuple collapses `prompt_template` and `inference_config` into two coordinates; HarmBench adds a third — `attack_wrapper` — because in safety the adversarial construction is a dimension of the eval, not of the task. Reporting a single "harmful response rate" conflates the three; the honest report is a cube (behavior × attack × matcher).

Notice: this is a stronger decomposition than ch-47 formalises in §1, because §1 was written for capability eval where the user is cooperative. Ch-52 will extend ch-47's tuple to add the attack-wrapper coordinate; this excerpt is the reason.

---

## The matcher — why substring fails and a classifier is mandatory

Source §Technical Details / Labels and evaluation design:

> For **non-copyright behaviors**, the authors fine-tune **Llama 2 13B Chat** on a **manually labeled validation set of completions** to serve as the success classifier.
> For **copyright behaviors**, they use a stricter **hashing-based classifier** with **MinHash-style matching** over overlapping chunks

Notice the two-matcher design. Ch-47 §4 names "learned classifier or LLM-judge" as one family; HarmBench uses both a *learned classifier* (Llama-2-13B fine-tune) for behavioral compliance and an *exact-match-family matcher* (MinHash) for copyright, because the two failure modes have different fidelity requirements. The chapter's guidance "pick the matcher with the bias profile you can afford" is this source's attested position made general.

Source §Technical Details continues:

> The paper also stress-tests classifiers on nonstandard cases such as:
> benign paragraphs,
> unrelated harmful completions,
> and outputs that begin with a refusal before later complying.

This is matcher-robustness testing, which ch-47 §4 implies but does not spell out. A classifier-matcher that can be gamed by a refusal prefix produces a dataset-level false negative rate that corrupts the entire score. HarmBench stress-tests its matcher; ch-47's "bias profile you must own" is this move generalised.

---

## Validation/test split — §6 versioning move

Source §Key Contributions:

> Explicitly separates **validation** and **test** behaviors so attacks and defenses do not optimize directly on the benchmark target set they are later judged on.

Notice: this is ML-methodology-101 applied to *safety benchmarks*, which historically did not enforce it. Ch-47 §6 calls out that a version bump can reflect a **data refresh** (v1.1 dropped items that failed inter-annotator agreement) — HarmBench's split is the same pattern from the design stage. The attack catalog and the defense pipeline are both allowed to train on validation; neither may touch test. Numbers reported against test are the contract.

---

## Taxonomy — §5 slicing as the reporting unit

Source §Technical Details / Taxonomy:

> `7 semantic categories`: `Cybercrime & Unauthorized Intrusion`, `Chemical & Biological Weapons/Drugs`, `Copyright Violations`, `Misinformation & Disinformation`, `Harassment & Bullying`, `Illegal Activities`, `General Harm`.
> `4 functional categories`: `standard`, `copyright`, `contextual`, `multimodal`.

Ch-47 §5's "Per-category (safety). [[salad-bench]]'s 66 leaf categories and [[harmbench-data]]'s 7 semantic × 4 functional grid — the unit of reporting is the cell, not the mean" is this quote made general. A headline "HarmBench 72% refusal" is meaningless if 95% is on `Copyright` and 40% on `Cybercrime`; the cell-level number is the fact.

Notice the functional axis — `standard` / `copyright` / `contextual` / `multimodal`. These are *matcher-relevant*; copyright is matched by MinHash, contextual carries a context string, multimodal pairs text+image. The functional axis is a `task_shape` axis in ch-47's vocabulary, and it rewires which matcher fires.

---

## Dual-intent filtering — the §1 task-shape discipline

Source §Technical Details / Curation rules:

> **Dual-intent filtering:** candidate behaviors are removed or rewritten if many benign users would plausibly want the same capability.

Ch-47 §1 argues task_shape must match the capability; HarmBench adds that the task *prompt* must also match the capability — if the same prompt serves benign and malicious users, scoring refusal on it measures over-refusal, not safety. The same principle rewires safety harnesses: a clean task-shape must be separable from its benign twin.

---

## Behaviour library vs attack wrapper — §6 version separation

Source §Practical lessons:

> Build a **behavior library** first; do not start from jailbreak prompts.
> Keep **direct requests**, **human jailbreak templates**, and **synthetic attack prompts** as separate strata in the dataset.

Ch-47 §6 rules that a harness version bump must be attributable. HarmBench's stratification is the attribution primitive — a score drop on "GCG attack wrapper" vs "direct-request wrapper" locates the regression in the attack family, not the behaviour inventory. The two axes move independently; they are versioned independently.

---

## What ch-47 keeps, changes, drops from HarmBench

| HarmBench design choice | Ch-47 normative claim | Reason |
|---|---|---|
| Three-layer: behavior / attack / label | 6-tuple extended by attack_wrapper in safety | §4 + §5 |
| Llama-2-13B-Chat fine-tune as matcher | Learned-classifier matcher is valid when pinned + stress-tested | §4 matcher family |
| MinHash for copyright | Different capability, different matcher — hierarchy per shape | §4 per-cell matcher |
| Validation/test behavior split | Version discipline applies to safety too | §6 |
| 7 × 4 taxonomy | Slicing by cell, not mean | §5 |
| Matcher stress-tests | Bias profile must be owned and measured | §4 caveat |
| Dual-intent filtering | Benign twin separable from harmful prompt | §1 shape |

---

## Connections

- **[[ch-47]]** — this excerpt grounds §4 (classifier matcher), §5 (safety-taxonomy slicing), §6 (validation/test split as version primitive).
- **[[excerpts/judge-llm-bias]]** — parallel discipline for LLM-judge matchers; learned classifier vs in-context judge are sister families.
- **[[excerpts/bfcl]]** — BFCL's relevance-detection category is the tool-calling analog of HarmBench's refusal task; abstention as a first-class shape.
- **[[wildguard-data]] (raw-data)** — three-label split (prompt harm / response harm / refusal) extends HarmBench's single-label success; ch-52 reads both.
- **[[salad-bench]] (raw-data)** — hierarchical 6-16-66 taxonomy is the deeper slicing surface.
- **[[ch-52]]** (downstream) — safety harness deep-dive; HarmBench + WildGuard + SALAD-Bench are the primary triad.
