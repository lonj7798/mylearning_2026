---
chapter: ch-49
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/ultrafeedback-construction.md
source_url: https://arxiv.org/abs/2310.01377
created_at: "2026-04-23"
---

# Excerpt: UltraFeedback construction — the rubric that defined GPT-4-as-judge

**Source library:** `wiki/raw-data/llm-training/papers/ultrafeedback-construction.md`
**Authors:** Ganqu Cui, Lifan Yuan, Ning Ding, Guanming Yao, Bingxiang He, Wei Zhu, Yuan Ni, Guotong Xie, Ruobing Xie, Yankai Lin, Zhiyuan Liu, Maosong Sun (Tsinghua + OpenBMB)
**Year:** 2023/2024

---

## Why this source matters for ch-49

UltraFeedback is the highest-profile pipeline to operationalize GPT-4-as-judge at scale: 64K prompts × 4 responses × GPT-4 annotator = >1M annotations, released as the default DPO corpus for open alignment. For ch-49 it functions as the *reference system* being replaced by the synthetic-judge line in §5. Its biases and leakage pathways are the failure modes the newer stack is designed to avoid.

---

## The 4-aspect rubric

Source §Synthesis pipeline:

> "Annotation (GPT-4):
>   For each (prompt, response) pair, GPT-4 rates 4 aspects:
>   Instruction-following (0-10), Truthfulness (0-10), Honesty (did it admit uncertainty?) (0-10), Helpfulness (0-10).
>   Each score accompanied by a short natural-language rationale."

Ch-49 §4 cites this as the canonical example of a multi-axis judge rubric. The rubric's *text* is the policy knob (per [[generative-reward-models]]); UltraFeedback's text is the most-studied such text in the field.

---

## Why aspects — and why it backfires

Source §Key Contributions:

> "Multi-aspect GPT-4 rubric (4 aspects) with numeric + prose feedback."

And §Risks + gotchas:

> "Aspect conflation: the 4 aspects are correlated; aggregating into a single pref loses signal."

Four-axis rubrics sound more principled than single-axis, but when the axes correlate strongly, a scalar aggregate loses them. Ch-49 §7 "rubric versioning" applies here: a rubric that aggregates four axes into a single pref is a *different* instrument from one that keeps them separate, and you cannot compare benchmark numbers across the two without re-anchoring.

---

## Judge-induced bias

Source §Risks + gotchas:

> "Judge-induced bias: GPT-4 scoring patterns (length bias, helpfulness bias, style bias) propagate into downstream DPO models."

Ch-49 §5(a) cites this: eval-RL leakage is not hypothetical, it is the attested behavior of every model trained on UltraFeedback prefs. Zephyr-7B, Tulu-2, Starling-7B all inherited GPT-4's length preference through this pipeline.

---

## Model-fleet contamination

Source §Risks + gotchas:

> "Model-fleet contamination: GPT-4 responses in the generator pool mean the judge is rating its own outputs -- subtle advantage to GPT-4 responses."

This is the ecosystem-level self-enhancement bias from [[judge-llm-bias]] made concrete. UltraFeedback's 17-model fleet includes GPT-4; GPT-4 is also the judge. Any GPT-4 advantage in the final preference data is structurally guaranteed, not a side effect. Ch-49 §5(d) cross-family judging is the direct mitigation.

---

## The pipeline's scale and cost

Source §Abstract:

> "64K prompts x 4 responses = 256K rated samples, with >1M individual GPT-4 feedback entries across 4 aspects."

Source §Synthesis pipeline:

> "Cost estimate: >1M GPT-4 annotations ~ tens of thousands USD at 2023 pricing."

Ch-49 §5(c) uses these numbers as the denominator in the cost-reduction argument: the Self-Taught Evaluator line from [[direct-judgement-preference]] achieves comparable judge quality with 40K self-generated pairs at zero marginal API cost.

---

## What UltraFeedback accidentally standardized

Source §Key Figures/Tables:

> "Aspect-rubric prompt -- the GPT-4 rating prompt is the artifact most reused downstream."

The prompt-as-artifact observation is important for ch-49: benchmarks that "use the UltraFeedback rubric" are all using the same literal text, which means they are all measuring against the same GPT-4 biases. The field's convergence on this prompt is why a single rubric-text change propagates into dozens of downstream numbers.

---

## The binarized subset's design choice

Source §Synthesis pipeline:

> "HuggingFace's `ultrafeedback_binarized` variant uses overall score with `chosen = argmax, rejected = random-not-chosen` (intentionally not worst-of-4, to avoid degenerate negatives)."

A subtle but relevant detail: random-not-chosen as the rejected response is an *anti*-contrastive choice. It produces easier DPO pairs and less aggressive length hacking than worst-of-4 would. Ch-49 §3's verbosity bias row is partially mitigated at the UltraFeedback-binarized level by this choice, even though the aspect rubric itself carries the bias.

---

## Connections

- `read.md` §3 verbosity / format bias: propagated into every UltraFeedback-trained model.
- `read.md` §4 rubric: 4-aspect is the reference rubric.
- `read.md` §5(a)/(d): model-fleet contamination as the canonical eval-RL leakage pathway.
- `read.md` §5(c): cost denominator for the synthetic-judge line.
- [[direct-judgement-preference]] (Con-J / STE / J1): the replacement line, explicitly positioned against this pipeline.
