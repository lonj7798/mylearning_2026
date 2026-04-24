---
chapter: ch-53
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/harmbench-data.md
source_url: https://proceedings.mlr.press/v235/mazeika24a.html
created_at: "2026-04-23"
---

# Excerpt: HarmBench — why the safety slice is three layers, not one

**Source library:** `wiki/raw-data/llm-training/papers/harmbench-data.md`
**Anchor paper:** Mazeika et al. 2024 — "HarmBench: A Standardized Evaluation Framework for Automated Red Teaming"

---

## Why this source anchors the safety slice of the ch-53 harness

The harness does not score "is the output unsafe?" as a single number. HarmBench's operational lesson — the lesson the rest of the safety-eval industry has been absorbing since 2024 — is that safety eval is three layered decisions that share no code path:

1. **Target behavior.** What harm does the prompt request? (semantic + functional categories.)
2. **Attack wrapper.** How was the request packaged? (direct, human jailbreak, GCG, PAIR, AutoDAN, ...).
3. **Success labeling.** Did the completion exhibit or clearly attempt the target behavior?

If the harness collapses these into one score it loses the only useful comparisons: "did RL regress on `contextual` behaviors specifically?" "did it trade refusal on `direct` requests for compliance on `GCG`-wrapped ones?" Those are slice queries, and they only exist because HarmBench separated the layers.

---

## The category system — verbatim from the raw-data page

From `harmbench-data.md` §3:

> The paper lists **7 semantic categories**:
> `Cybercrime & Unauthorized Intrusion`, `Chemical & Biological Weapons/Drugs`, `Copyright Violations`, `Misinformation & Disinformation`, `Harassment & Bullying`, `Illegal Activities`, and `General Harm`.
> The paper lists **4 functional categories**:
> `standard`, `copyright`, `contextual`, and `multimodal`.

Both taxonomies enter the ch-53 harness as `Sample.slices` keys. The semantic category determines which refusal band matters (copyright answers are scored with hash matching, not the classifier). The functional category determines the eval contract: `contextual` behaviors include a preceding context string and a scorer that inspects whether the harmful behavior was exhibited *with respect to that context*, not to the prompt in isolation.

---

## The success classifier — why substring rules are not allowed

From `harmbench-data.md` §5:

> For **non-copyright behaviors**, the authors fine-tune **Llama 2 13B Chat** on a **manually labeled validation set of completions** to serve as the success classifier.
> For **copyright behaviors**, they use a stricter **hashing-based classifier** with **MinHash-style matching** over overlapping chunks.

The harness uses the HarmBench classifier (or WildGuard-class equivalent) as `metric_fn`. Substring matching is explicitly disallowed because the paper documents outputs that begin with a refusal before later complying — a substring scorer misses the failure. This is the same design pattern as Lightman 2023's "first negative step" in PRM labeling: the scorer must see the whole trajectory, not a snippet.

---

## The validation/test split — why it matters for regression tracking

From `harmbench-data.md` §3:

> The paper also enforces an official **validation/test split** of behaviors, which is a strong design choice for any safety-data pipeline that expects iterative attack development.

The ch-53 harness uses the `val` split for CI gating and reserves the `test` split for one-shot yearly publication numbers. This is how you keep the gate honest: if every iteration is scored on the same behaviors you will overfit the refusal head to those behaviors without improving underlying safety.

---

## The three-layer slicing this excerpt licenses

In the harness, one safety `Sample` carries all of these as slice keys simultaneously:

```python
slices = {
    "semantic": "cybercrime_intrusion",   # HarmBench 7-category
    "functional": "contextual",           # HarmBench 4-category
    "attack": "pair",                     # HarmBench attack family
    "source_set": "harmbench-val",
}
```

A single run then produces scores broken out by any combination — `semantic=cybercrime_intrusion & attack=pair` is a different number from `semantic=cybercrime_intrusion & attack=direct`, and the comparator in §6 compares each slice independently. The regression rule is: a checkpoint regresses on safety if *any* functional-category slice has its CI-high below the baseline's mean. Averaging across functional categories buries contextual regressions.

---

## What this source does not tell you

HarmBench is silent on refusal detection as a separate label — it only labels behavior success. For the refusal axis the harness cross-references [[wildguard-data]], which makes `refusal` a first-class label. Treat HarmBench as the *behavior taxonomy* and WildGuard as the *refusal taxonomy*; the harness uses both.
