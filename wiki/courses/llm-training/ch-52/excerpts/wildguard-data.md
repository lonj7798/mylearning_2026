---
chapter: ch-52
course: llm-training
phase: read
excerpt_of: Han et al. 2024 — "WildGuard: Open One-Stop Moderation Tools for Safety Risks, Jailbreaks, and Refusals of LLMs"
source_url: https://arxiv.org/abs/2406.18495
created_at: "2026-04-23"
---

# Excerpt: WildGuard — Matched Refusal/Compliance and the Over-Refusal Defense

**Source:** `wiki/raw-data/llm-training/papers/wildguard-data.md`
**Primary paper:** Seungju Han et al., "WildGuard", Allen AI, 2024
**arXiv:** https://arxiv.org/abs/2406.18495
**HF dataset:** https://huggingface.co/datasets/allenai/wildguardmix

---

## The key contribution — three independent label spaces

WildGuard's structural move: make **refusal detection** a first-class label space, independent of response-harm. The three labels per example:

- `prompt_harm_label` ∈ {harmful, unharmful}
- `response_harm_label` ∈ {harmful, unharmful}
- `response_refusal_label` ∈ {refusal, compliance}

This separation is why WildGuard is the right benchmark for over-refusal. If you approximate refusal from response-harm ("the response is safe, so the model must have refused"), you cannot distinguish a model that refused from a model that complied benignly. WildGuard's Table 2 demonstrates the failure empirically — mapping safe-response to refusal is significantly worse than training a dedicated refusal head.

---

## Dataset structure

- **WildGuardMix** = WildGuardTrain (86.7K) + WildGuardTest (1.7K).
- WildGuardTrain: 48,783 prompt-only + 37,976 prompt-response examples.
- 4 high-level groups × 13 subcategories:
  - **Privacy** — sensitive org info, private individual info, copyright violations.
  - **Misinformation** — false info, material-harm misinfo.
  - **Harmful language** — social stereotypes, violence, toxic speech, sexual content.
  - **Malicious uses** — cyberattacks, fraud, encouraging unethical actions, mental-health / over-reliance crisis.

---

## The synthetic-vs-human balance

- 87% synthetic, 11% in-the-wild (LMSYS-Chat-1M + WildChat), 2% annotator-written (HH-RLHF + Anthropic Red-Teaming).
- Synthetic labels are GPT-4 generated but audited against human: **92% / 82% / 95%** agreement on prompt-harm / response-harm / refusal on a 500-item audit.
- WildGuardTest is fully human-annotated; Fleiss κ = 0.55 / 0.72 / 0.50 across the three tasks.

The 82% response-harm agreement is the soft spot — humans disagree more on borderline response-harm than on prompt-harm or refusal. Any WildGuard-based evaluation should expect noise in the response-harm dimension and report audit error bars accordingly.

---

## Matched refusal + compliance

The dataset deliberately generates **both a refusal and a compliance response** for each synthetic prompt, using a suite of generators (OLMo-7B-Instruct, GPT-3.5, Vicuna-7B, Llama-3-8B, Mistral-7B-Instruct, Dolphin variants). GPT-4 then synthesizes the hardest cases: **compliances with caveats, warnings, or mixed signals** — the completions that break naive refusal classifiers.

For ch-52's over-refusal discussion, the attested retained counts matter:
- 6,062 vanilla-harmful + 2,931 vanilla-benign
- 4,489 adversarial-harmful + 4,339 adversarial-benign

The deliberate 50/50 harmful/benign balance at both vanilla and adversarial tiers is what forces the guard model to discriminate on content, not on jailbreak wrapper style. Without this, the learned guard collapses into "jailbreak style implies unsafe" — exactly the shortcut that over-refusal suites (xstest, or-bench) expose.

---

## Why adversarial rewrites of benign prompts matter

WildTeaming is applied to **both** harmful and benign prompts. Applying jailbreak transforms only to harmful prompts teaches the model that jailbreak wrapper ≈ harm. Applying them to benign prompts too forces a harder discrimination: the wrapper does not carry the harm signal; the content does. This is the WildGuard-specific defense against the most common over-refusal failure mode in 2023-era safety-tuned models.

---

## What this means for ch-52

WildGuard is the benchmark to use when the property you care about is **refusal-on-harmful vs non-refusal-on-benign** as a Pareto point, not as a scalar. Any safety report that folds refusal and response-harm into a single "safe" score is throwing away the signal WildGuard exists to surface.

---

## Connections

- [[harmbench-data]] — behavior-inventory counterpart; WildGuard focuses on response labels, HarmBench on attack construction.
- [[salad-bench]] — hierarchical taxonomy sibling; MD-Judge training data is a partial overlap.
- [[constitutional-ai]] — WildGuard's matched-response design teaches the same principle CAI teaches through critique-revise: refusals should be non-evasive and explainable.
- Chapter synthesis: [[ch-52]] §1, §2, §3.
