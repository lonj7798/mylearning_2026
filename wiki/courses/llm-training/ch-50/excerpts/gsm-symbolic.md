---
chapter: ch-50
course: llm-training
phase: read
excerpt_of: primary source (no library card exists for this slug as of 2026-09-15)
source_url: https://arxiv.org/abs/2410.05229
created_at: "2026-09-15"
---

# Excerpt: GSM-Symbolic — perturbation slices over a fixed benchmark

**Artifact:** GSM-Symbolic: Understanding the Limitations of Mathematical Reasoning in Large Language
Models. Mirzadeh, Alizadeh, Shahrokhi, Tuzel, Bengio, Farajtabar (Apple; Washington State University).
arXiv:2410.05229, v1 2024-10; ICLR 2025.
**Checked on 2026-09-15** against the arXiv v2 PDF (2025-08-27). Every number below is quoted at its locus.
**Note:** `wiki/raw-data/llm-training/` has no card for this slug. When a card is created, this excerpt
should be replaced by a link to it.

---

## Why ch-50 uses this source

GSM-Symbolic supplies the perturbation slice of ch-50 §3: a transformation that preserves the required
reasoning and changes the surface, so the resulting score difference measures dependence on the surface.
It also supplies the item-generation protocol that turns a single benchmark score into a distribution.

---

## Construction (§3.1, §4)

> Given a specific example from the test set of GSM8K, we create parsable templates … The annotation
> process involves identifying variables, their domains, and necessary conditions to ensure the correctness
> of both the question and the answer. (§3.1)

- 100 GSM8K test questions were randomly selected and converted to templates (§3.1; App. text at §A.3).
- From these, **50 datasets of 100 examples each** were generated, each example a mutation of one of the
  100 originals (§4).
- Automated checks verify that none of the original variable values appear in the template (§3.1).

Variants (§4.3, §4.4):

| Variant | Transformation |
|---|---|
| Symbolic-M1 | one clause removed |
| Symbolic | names and numeric values re-instantiated |
| Symbolic-P1 | one clause added |
| Symbolic-P2 | two clauses added |
| Symbolic-NoOp | one clause added that appears relevant and does not enter the solution |

## Results used in ch-50 §3 (Table 1, 8-shot)

| Model | GSM8K (100 originals) | Symbolic | Symbolic-P1 | Symbolic-P2 | Symbolic-NoOp |
|---|---|---|---|---|---|
| Gemma2-9b-it | 87.0 | 79.1 ± 3.0 | 68.1 ± 4.8 | 41.8 ± 6.0 | 22.3 ± 5.1 |
| Phi-3-medium-128k-instruct | 89.0 | 82.5 ± 2.9 | 75.8 ± 3.9 | 53.1 ± 4.8 | 29.4 ± 4.2 |
| Phi-3.5-mini-instruct | 88.0 | 82.1 ± 3.4 | 64.8 ± 5.4 | 44.8 ± 6.3 | 22.4 ± 4.0 |
| Mathstral-7b-v0.1 | 80.0 | 74.0 ± 3.5 | 57.4 ± 5.2 | 35.5 ± 5.1 | 20.4 ± 3.6 |
| Llama3-8b-instruct | 74.0 | 74.6 ± 2.9 | 53.8 ± 4.5 | 28.3 ± 4.4 | 18.6 ± 3.9 |
| GPT-4o-mini | 95.0 | 91.7 ± 2.0 | 81.1 ± 3.1 | 72.4 ± 4.6 | 54.1 ± 3.9 |
| GPT-4o | 95.0 | 94.9 ± 1.9 | 93.9 ± 2.6 | 88.0 ± 3.4 | 63.1 ± 4.5 |
| o1-mini | 93.0 | 94.5 ± 1.6 | 94.3 ± 2.6 | 89.1 ± 3.6 | 66.0 ± 4.6 |
| o1-preview | 96.0 | 92.7 ± 1.8 | 95.4 ± 1.7 | 94.0 ± 2.4 | 77.4 ± 3.8 |

## Two findings ch-50 depends on

**Contamination reading (§4.1).** The accuracy on the 100 original questions lies more than one standard
deviation from the centre of the Symbolic distribution, usually above it, for 21 of 25 models. The authors
offer data contamination as one explanation. **Interpretation**, not a contamination test.

**Names against numbers (§4.2, Fig. 4).** Changing only proper names already produces variation across
generated sets; variance grows when numeric values change as well. The authors state the original GSM8K
accuracy sits closer to the centre of the changed-names distribution than to the changed-numbers one.

**NoOp (§4.4, Abstract).** Adding one seemingly relevant clause that does not contribute to the solution
produces drops "up to 65%" across the models tested (Abstract; the per-model drops are plotted in Fig. 8a).

## Limits

- The templates come from 100 GSM8K items, so the result is about grade-school arithmetic word problems in
  one benchmark's style.
- The NoOp drop measures distractor handling, which is a capability in its own right, not only robustness.
- The paper does not run training experiments.

## Used by

ch-50 §3 (perturbation slices), §6 (the `distractor-sensitivity` bucket), Recipe (generation protocol row),
Generalization lens (b) and (c).
