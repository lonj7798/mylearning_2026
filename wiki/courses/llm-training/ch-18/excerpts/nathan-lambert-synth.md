---
chapter: ch-18
course: llm-training
phase: read
excerpt_of: "Nathan Lambert (Interconnects) — synthetic-data posts 2024-2025"
source_url: https://www.interconnects.ai/p/frontiers-in-synthetic-data
created_at: "2026-04-23"
---

# Excerpt: Lambert's "verification is the bottleneck" — the operating principle of 2025 synthesis

**Author:** Nathan Lambert (Allen AI / Interconnects)
**Year:** 2024-2025 (ongoing)
**URLs:** https://www.interconnects.ai/p/frontiers-in-synthetic-data ; https://www.interconnects.ai/p/the-state-of-post-training-2025
**Raw-data source:** [[raw-data/nathan-lambert-synthetic-data]]

---

## Why this source is load-bearing for ch-18

Lambert's Interconnects posts are the closest thing the field has to a running synthesis of where synthetic-data practice is and where it is going. Where academic papers describe single pipelines, Lambert surveys the pipeline *space* and makes the kind of operating-principle statements that let you navigate it. For ch-18 — a chapter whose whole pitch is "see the loop, not the paper" — Lambert's framings are the doctrine the chapter stands on.

The raw-data file gives the two most load-bearing:

> "Post-training costs in 2025 are dominated by synthetic-data infrastructure (prompt collection, multi-model generation fleets, rerankers, verifiers), not by human annotation; 'synthetic data can do almost all of the work' *once* the verification layer is correct."

and

> "Don't treat synthetic data as a monolith; stratify by (a) pretraining vs SFT vs preference, (b) easy-open-ended (writing) vs hard-verifiable (code/math) — the latter needs aggressive verification + RLVR-style loops."

Both statements are stage-agnostic claims about the loop. The first is about stage 4. The second is about the modality-verifier relationship.

---

## The three-layer stratification

From "Frontiers in Synthetic Data" (Jun 2024), via the raw-data file:

> "Argues the three layers:
> 1. **Pretraining synthetic** (Phi-style, Cosmopedia-style rephrase/textbook synthesis).
> 2. **SFT synthetic** (Self-Instruct lineage -> Magpie -> Persona-Hub).
> 3. **Preference synthetic** (UltraFeedback -> West-of-N -> Con-J judges)."

Each layer instantiates the six-stage loop differently:

- **Pretraining synthetic.** Stage 1 is rephrase- or textbook-generation. Stage 4 is *very* weak — there is no gold answer; the best you have is faithfulness to source. Stage 6 (mix ratio with real data) is the dominant lever.
- **SFT synthetic.** Stage 1 is bootstrap / evol / extraction / persona / rephrase. Stage 4 ranges from "format only" (Self-Instruct) to "RM + judge" (Nemotron). This is the modality where the loop was first named.
- **Preference synthetic.** Stage 1 produces (chosen, rejected) pairs or pairwise judgments. Stage 4 is the judge itself (UltraFeedback uses GPT-4; West-of-N ensembles; Con-J is a specialised comparator). Stage 5 is implicit in how pairs are selected.

For ch-18: the *same* loop is used in all three regimes, but the stage weights shift. Pretraining = stage 6 dominant. SFT = stages 1 + 4 dominant. Preference = stage 4 dominant (the judge). Recognise which layer you are in, and you know which stage to invest in.

---

## The quote-worthy claim, and its precondition

From the raw-data file:

> "Quote-worthy claim: 'Synthetic data can do almost all of the work' given a strong open weights base model + robust verification."

Notice the two preconditions: (a) strong base model, (b) robust verification. Drop either and the claim fails.

Precondition (a) is the backdrop of the foundations and systems tracks (ch-01..ch-08): you need a teacher that can produce useful outputs in the first place. Precondition (b) is the ch-18 subject.

The corollary Lambert draws — "once the verification layer is correct, synthetic does almost all the work" — is the practical justification for the claim that stage 4 is the load-bearing slot. Without precondition (b), more synthetic data is not better; it is just more noise.

---

## Easy-open-ended vs hard-verifiable: the verifier-cost axis

From the raw-data file:

> "- **Easy tasks (writing, summarization):** synthetic is near-free; teacher-generated responses are competitive with human completions.
> - **Hard tasks (math, code, multi-step reasoning):** synthetic requires explicit verifiers (unit tests, answer matchers, PRMs); see [[rlvr-tulu3]].
> - **Verification is the bottleneck,** not generation; in the RLVR era, the scarce resource is verifiable prompts."

This is a ch-18 distinction that the four flagship papers cover but never directly name. Self-Instruct is easy-open-ended (general instructions), and its empty stage 4 is tolerable because the ceiling is teacher-quality, not correctness. OMI-2 and APIGen are hard-verifiable, and their stage 4 is both essential and buildable. Nemotron is a hybrid — it has to use a reward model because "quality" is partially open-ended.

Lambert's "scarce resource is verifiable prompts" reframes the problem upstream. The constraint is not "generate good answers"; it is "**find prompts you can check**." This is why RLVR (ch-44) and verifiable-reward RL are the natural extension of the synthetic-data loop into the training-time regime. Same verifier, applied during rollout generation instead of during data preparation.

---

## The economics: where the money goes

From "The State of Post-Training in 2025," via the raw-data file:

> "Post-training now consumes a substantial fraction of total FLOPs — driven by (a) multi-round rejection sampling, (b) multi-model generation fleets, (c) large RL rollouts.
> Data-foundry business model (Scale AI, Surge) is pressured by synthetic data, especially in easy-verifiable domains."

Three cost drivers, all of which are stages of the loop:

- **(a) Multi-round rejection sampling** = stage 4 applied repeatedly to over-generated candidates.
- **(b) Multi-model generation fleets** = stage 1 scaled horizontally across teacher models.
- **(c) Large RL rollouts** = the runtime counterpart of stage 1 + 4 during RL training.

Data-foundry pressure is the economic fingerprint of stage 4 working: where verifiable, synthetic replaces humans; where unverifiable (subjective evals, safety edge cases), humans remain.

For ch-18: the cost of a synthetic pipeline is overwhelmingly spent at stages 1 and 4, and Lambert's claim is that this cost is *rising* faster than it is falling. "Synthetic is cheap" is a first-order approximation; the true cost is in the verifier infrastructure.

---

## Accumulation over replacement

From the raw-data file's practitioner-notes section:

> "Prefer **accumulation + verification** over pure self-distillation (aligned with [[model-collapse]] mitigations).
> Treat prompt curation as a distinct, cost-bearing stage; open prompt corpora are the new scarce asset."

Two operational rules:

1. **Never replace real data with synthetic.** Stack synthetic on top of the real anchor. This is the defence against model collapse ([[ch-14]]) and the reason Nemotron keeps 20K human anchors and OMI-2 keeps the MATH/GSM8K seed problems.
2. **Prompt curation is a distinct stage** — not free, not a side-effect of stage 1. The "new scarce asset" framing matches what APIGen's 3,673-API curation represents and what OMI-2's question-augmentation pipeline represents. Prompts are the thing you spend human attention on; responses are what the teacher fills in.

Both rules are ch-18 defaults. If a pipeline you read about violates either, that is a weakness to name.

---

## What Lambert flags for the near future

From the raw-data file:

> "- Model-collapse concerns are real but over-stated in strict-replacement regimes; accumulation + verifier loops break the bad asymptote.
> - Judge-LLM bias is the next major audit frontier (see [[direct-judgement-preference]]).
> - The relationship between synthetic pretraining and synthetic post-training is under-theorized; distinct cost/value profiles."

Three open problems:

- **Collapse mitigation in practice:** the theory says accumulation + verifier breaks collapse; the empirical verification of that at 1T-token scale is [[synthetic-data-scaling-laws]].
- **Judge-LLM bias:** when stage 4 is a judge, its biases are the ceiling. This is ch-26.
- **Pretraining vs post-training synthesis unification:** they share the loop but not the economics. ch-22 and ch-27 respectively.

---

## The 2024 year-review framing: reasoning re-inflates verification demand

From the raw-data file:

> "2024 Year in Review: Highlights the data-foundry vs synthetic data tension. Flags reasoning models (o1-class, R1-class) as the key shift that re-inflates verification demands."

Worth unpacking. The naive trajectory through 2023 was "verifiers get cheaper, synthetic gets bigger, humans get displaced." Reasoning models broke this by introducing a new type of synthetic data — long-chain-of-thought traces with backtracking — for which the *process* of reasoning, not just the final answer, matters.

For ch-18: this is a reason to be cautious about extrapolating the ch-18 loop without modification into the reasoning regime. A gold-answer-match verifier (OMI-2's SymPy, APIGen's execution check) accepts a trace with the right final answer and broken reasoning — the exact 7% false-positive rate OMI-2 names. For reasoning-model training, that false-positive rate is not tolerable; you need a step-level verifier (a PRM, [[prm800k]]) or a judge that audits the reasoning process.

The reasoning regime thus inflates stage 4 again, after the 2022–2023 deflation from "explicit verifiers are cheap." ch-21 and ch-28 cover this; for ch-18 it is worth noting that the loop's stage-4 story is not finished and will be revisited.

## Connections

- [[excerpts/self-instruct]] — the "easy-task synthetic is near-free" case; Lambert's framing explains why Self-Instruct's empty stage 4 was tolerable.
- [[excerpts/nemotron-4]] — the "verification via RM + small human anchor" case; Lambert's 98%-synthetic talking point lives here.
- [[excerpts/apigen]] — the "hard-verifiable, explicit verifier" case that Lambert's framework predicts.
- [[excerpts/openmathinstruct-2]] — the "cheap verifier + frontier teacher" case.
- [[excerpts/synthetic-scaling-laws]] — the empirical partner to Lambert's accumulation-over-replacement rule.
- [[ch-18]] — parent. Lambert's doctrine is ch-18's doctrine.
