---
chapter: ch-23
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/nemotron-4-synthetic.md
source_url: https://d1qx31qr3h6wln.cloudfront.net/publications/Nemotron_4_340B_8T_0.pdf
created_at: "2026-04-23"
---

# Excerpt: Nemotron-4 340B — the production proof that 98% synthetic + RM gate doesn't collapse

**Source library:** `wiki/raw-data/llm-training/papers/nemotron-4-synthetic.md`
**Authors:** NVIDIA
**Year:** 2024

---

## Why this source anchors ch-23

Ch-23 §6 Template A (RM-as-judge) is Nemotron-4's alignment pipeline, abstracted. The paper is the single most visible production counter-example to a naïve reading of Shumailov — NVIDIA reports **over 98% of alignment data is synthetic**, over multiple generator iterations, with no observed collapse. The reason: an external reward model, trained on a small persistent human anchor (HelpSteer2), gates every acceptance. Ch-23 §5 comparison table cites Nemotron-4 as the "verifier turns 98% synthetic into no-collapse" existence proof.

---

## The 98% number — what it means and what it hides

```
# nemotron-4-synthetic.md, lines 7, 15
Over 98% of post-training data is synthetic, and the same pipeline
feeds SFT, DPO, and RPO.
... only about 20K human-annotated examples are used overall,
split between SFT and HelpSteer2 reward-model data.
```

The ratio: ~1.3M total post-training examples, ~20K human-annotated, ~98.5% synthetic. At face value, this is exactly the regime [[excerpts/model-collapse]] predicts should collapse catastrophically. It does not. The paper reports SOTA RewardBench performance and competitive alignment across chat, math, code, instruction-following, and topic-following.

What rescues the pipeline is not the ratio but the **structure**: the 20K human examples are the training data for the reward model (HelpSteer2), and the reward model gates every synthetic example before it enters training. The 98% is synthetic-that-passed-the-gate, not raw synthetic. In ch-23 §3 terms, this satisfies the "external signal in the loop" condition — the RM's decisions are (roughly) independent of the generator's output distribution because the RM was trained on human-labeled data the generator never produced.

---

## The pipeline — ch-23 §6 Template A derived directly

```
# nemotron-4-synthetic.md, lines 30-37
Prompt generation: synthesizes prompts by task family, then uses the
current model to generate responses or multi-turn dialogues.
Response generation: the code alignment stage uses Genetic Instruct.
Filtering/rescoring: a reward model scores responses for quality; when
ground truth is missing, Nemotron-4-340B-Reward selects high-quality
chosen responses.
Output shape: ~800K code SFT, 200K general SFT, 160K DPO, 300K RPO,
all within a pipeline that is more than 98% synthetic overall.
```

The structural flow (ch-23 §6 Template A):

1. **Prompt generation** by task family (coding, QA, topic-following, document-based reasoning, function calling, incapable-task refusal). Each family has its own seed generation strategy.
2. **Response generation** by the current generator checkpoint. For code, Genetic Instruct (mutation-based population search) is used; for other families, straight sampling with task-family-specific system prompts.
3. **Quality filtering** by Nemotron-4-340B-Reward. Responses below threshold rejected.
4. **Preference pair construction** for DPO/RPO: the RM scores multiple candidate responses per prompt, picks the highest-scoring as "chosen" and a lower-scoring as "rejected."
5. **Staged training:** first code SFT → general SFT → DPO → RPO. Each stage uses the prior stage's checkpoint as the new generator.

Ch-23 §6 Template A's invariants map 1:1:

- Human anchor never consumed: HelpSteer2 (20K) stays in RM training across iterations.
- RM refreshed on cadence: the paper iterates generator checkpoints and rebuilds RM training data.
- RM never scores its own training data: HelpSteer2 examples are held out from generator outputs.
- Acceptance threshold tuned: the paper reports τ chosen for ~80% agreement with held-out human preferences.

---

## Why staged SFT matters (anti-collapse in stages)

```
# nemotron-4-synthetic.md, lines 21-22
Implements staged SFT: first a code-focused SFT stage, then a broader
general SFT stage.
Implements preference fine-tuning with DPO followed by RPO, with the
reward model used to select higher-quality chosen responses.
```

The staging is not cosmetic. Each stage is a separate generator iteration, which means each stage's synthetic data is gated by a (potentially refreshed) RM. Collapse would show up as later stages' outputs being indistinguishable from earlier stages' (mode contraction across stages) — the paper reports the opposite, with each stage delivering measured gains on the relevant evals.

The DPO → RPO ordering is also anti-collapse:

- DPO alone optimizes toward reward-gap margin, which can overfit to RM blind spots (Goodharting).
- RPO adds an SFT-style loss on the chosen response, regularizing against pure-margin optimization.
- Together, they avoid the "policy collapses into high-RM-score mode" failure that pure DPO would induce in an iterated loop.

Ch-23 §3's "reward-model staleness" warning is exactly this failure mode. The paper's mitigation (SFT-loss + RM-select-chosen + periodic RM refresh) is what §6 Template A codifies.

---

## The Genetic Instruct detail — mutation as diversity preservation

```
# nemotron-4-synthetic.md, line 34
The code alignment stage uses Genetic Instruct, which combines
self-instruction and WizardCoder-style mutations plus an LLM-based
fitness function to grow a population from a limited number of seeds.
```

Genetic Instruct is worth unpacking as an anti-collapse mechanism. The concern with self-instruction is mode contraction — the generator produces examples similar to its training data, and iterated application narrows. Genetic Instruct adds:

- **Mutation operators** (WizardCoder-style: add constraints, increase complexity, change domain, change language) that actively push candidates away from seed modes.
- **LLM-based fitness function** that filters mutations for correctness and usefulness — a secondary gate inside the generation step.
- **Population maintenance** that prevents any single mode from dominating (diversity is an explicit optimization target).

Structurally, this is a **gradient-coverage mechanism** at the generation layer, complementary to the RM gate at the filtering layer. In ch-23 §4 four-axis terms, Genetic Instruct is addressing Axis 3 (coverage / diversity) upstream of the verifier. The analog at pure-diversity generation is [[excerpts/prismatic-synthesis]]'s gradient-targeted synthesis.

---

## Risks and gotchas — directly anti-collapse

```
# nemotron-4-synthetic.md, lines 45-49
Reward-model errors compound when the same scorer is reused across
iterations.
DPO alone can overfit to reward gaps; the paper adds SFT loss and then
RPO to reduce that effect.
The synthetic majority is a strength for scale, but it also makes
quality filtering and judge calibration critical.
```

Each risk is a collapse mode the paper explicitly addresses:

1. **RM compounding error.** Mitigated by refresh from HelpSteer2 anchor each iteration. If you cannot refresh, this is the soft-collapse vector: the generator Goodharts the fixed RM.
2. **DPO overfitting to margin.** The mode-collapse-in-output-distribution failure. Mitigated by RPO and SFT-loss regularization.
3. **Judge calibration.** The externality requirement. Mitigated by τ tuning against held-out human preferences.

Ch-23's rhetorical claim — "every pipeline that works has a gate" — is supported concretely by Nemotron-4. The gate has three components (external RM, refresh cadence, τ calibration), each guarding a different collapse mode.

---

## Evaluation — not just anecdotal

```
# nemotron-4-synthetic.md, lines 40-43
Nemotron-4-340B-Instruct is competitive with other open-access
aligned models across chat, math, code, instruction-following,
and topic-following benchmarks.
The reward model reaches top RewardBench performance at the time
of publication.
```

The downstream evaluation is important because "no collapse" by itself is a weak claim — a collapsed model could still pass some evals. Nemotron-4 passes diverse evals (chat, math, code, IF, TF) and the RM itself is SOTA on RewardBench. This is strong evidence that the 98%-synthetic pipeline has *not* contracted in the distribution-preserving sense — the model's behavior covers diverse task families.

If the pipeline had collapsed, RewardBench scores would not reach SOTA (the RM would have been collapsed by its own filter); IF benchmarks (instruction-following on diverse constraints) would show systematic failures on rare constraint types; topic-following scores would be poor on long-tail topics. None of these fails. The empirical endpoint validates the structural argument.

---

## Cost and scale consideration

The paper does not give per-sample synthetic-generation cost but reports the sample counts: ~800K code SFT + 200K general SFT + 160K DPO + 300K RPO = ~1.46M total, of which ~20K is human. At RM-call + sample-generate cost of ~$0.01 per sample (order-of-magnitude estimate), this is ~$15K in inference to produce the gated corpus. This is cheap relative to the 340B pretraining compute. The *gate* is the expensive part conceptually (RM training, HelpSteer2 collection) but not operationally.

The lesson for ch-23 §6 Template A: the gate's one-time cost (human anchor collection + RM training) amortizes over millions of synthetic samples. This is why RM-as-judge is the dominant production template for alignment — the economics favor a small human investment upfront over a large one ongoing.

---

## Connections

- [[excerpts/model-collapse]] — the pessimistic prediction that 98% synthetic should fail; Nemotron's gate falsifies the naive reading.
- [[excerpts/strong-model-collapse]] — the theoretical version; Nemotron's gate reduces `σ_synth²` close to `σ_real²`.
- [[excerpts/faithful-synth-eval]] — Axis 2 (strong-judge tier) is Nemotron's RM.
- [[excerpts/prismatic-synthesis]] — G-Vendi-style coverage; Genetic Instruct is a production-pragmatic version.
- [[excerpts/apigen]] — the rule-based end of the gate spectrum; Nemotron is the strong-judge end.
- [[ch-23]] — §5 (comparison table) and §6 (Template A) both derived from this paper.
