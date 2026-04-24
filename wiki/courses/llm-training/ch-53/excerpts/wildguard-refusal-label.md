---
chapter: ch-53
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/wildguard-data.md
source_url: https://arxiv.org/abs/2406.18495
created_at: "2026-04-23"
---

# Excerpt: WildGuard — refusal as a first-class label

**Source library:** `wiki/raw-data/llm-training/papers/wildguard-data.md`
**Anchor paper:** Han et al. 2024 — "WildGuard: Open One-Stop Moderation Tools for Safety Risks, Jailbreaks, and Refusals of LLMs"

---

## Why this source anchors the refusal slice of ch-53

The ch-53 harness tracks three safety metrics per sample: `prompt_harm`, `response_harm`, `refusal`. The third is the hard one. You cannot derive refusal from response harm, and you cannot derive it from prompt harm. WildGuard is the paper that made this point operational and shipped a dataset where all three labels are pinned to the same prompt-response pair.

From `wildguard-data.md` §Key Contributions:

> Treated **refusal detection as its own label space**, instead of approximating refusal from response harmfulness.
> Added **matched refusal and compliance generations** for the same prompts, which makes the dataset useful for learning nuanced refusal behavior rather than just toxicity detection.

---

## The label triple — verbatim

From `wildguard-data.md` §Label space:

> `prompt_harm_label`: `harmful` or `unharmful`
> `response_harm_label`: `harmful` or `unharmful`
> `response_refusal_label`: `refusal` or `compliance`
> `subcategory`: one of the fine-grained harm categories

The ch-53 harness emits three `Sample`s per WildGuard row, one per task, sharing the same `subcategory` slice. This is the only clean way to keep the three metrics on the same prompt axis without aliasing them together.

---

## The 13-subcategory taxonomy — the slice key

From `wildguard-data.md` §Risk taxonomy:

> - **Privacy:** sensitive organizational information, private individual information, copyright violations
> - **Misinformation:** false or misleading information, material harm by misinformation
> - **Harmful language:** social stereotypes and discrimination, violence and physical harm, toxic language and hate speech, sexual content
> - **Malicious uses:** cyberattacks, fraud and assisting illegal activities, encouraging unethical or unsafe actions, mental-health and over-reliance crisis

4 high-level groups, 13 subcategories. The harness stores both — group for aggregate reporting, subcategory for regression hunting.

---

## The annotation quality floor — why you can trust these labels

From `wildguard-data.md` §Filtering, relabeling, and auditing:

> Agreement between GPT-4 labels and voted human labels is reported as **92%** for prompt harm, **82%** for response harm, and **95%** for refusal.
> WildGuardTest is fully human-annotated with **three annotators per example**, majority voting, an **"unsure"** option, and removal of items that fail to reach at least two-way agreement.
> Reported Fleiss kappa on WildGuardTest is **0.55** for prompt harm, **0.72** for refusal, and **0.50** for response harm.

Refusal is the highest-agreement task (0.72 kappa) — which matches the ch-53 design choice of using refusal as the primary safety regression signal. Response-harm agreement (0.50) is materially lower; the harness carries this forward by widening the CI on `response_harm` by one bootstrap iteration and never using it as a solo regression gate.

---

## The matched refusal/compliance pairs — why they matter for the probe

From `wildguard-data.md` §Synthetic response construction:

> For synthetic vanilla and adversarial prompts, the authors generate **matched refusal and compliance responses**.
> Each prompt is paired with suffix instructions telling the generator either to **refuse** or to **comply**, producing candidate responses for both sides.

These matched pairs are exactly the inputs the judge-bias probe in §5 of the read.md needs: same prompt, two responses of known label, different lengths. Running the length-controlled probe on WildGuard pairs measures whether the judge is treating a short refusal as a loss against a long compliance — a safety-critical judge failure.

---

## The ~87% synthetic / 11% in-the-wild / 2% annotator-written balance

From `wildguard-data.md` §Synthetic vs human data balance:

> The WildGuardMix dataset card reports that **WildGuardTrain** is approximately **87% synthetic**, **11% in-the-wild**, and **2% existing annotator-written** data.

The harness does not retrain anything on WildGuardTrain — this excerpt exists to calibrate expectations. The test split is fully human-annotated (1,725 pairs); that is what the harness scores against. If a later lab extends ch-53 to use WildGuardTrain as negative examples for an adversarial-SFT pass, treat the 87% synthetic ratio as the binding constraint on how much refusal behavior transfers to real user chat.
