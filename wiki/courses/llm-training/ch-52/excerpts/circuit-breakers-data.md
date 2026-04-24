---
chapter: ch-52
course: llm-training
phase: read
excerpt_of: Zou et al. 2024 — "Improving Alignment and Robustness with Circuit Breakers"
source_url: https://arxiv.org/abs/2406.04313
created_at: "2026-04-23"
---

# Excerpt: Circuit Breakers — Safety on the Representation Path, Not the Output

**Source:** `wiki/raw-data/llm-training/papers/circuit-breakers-data.md`
**Primary paper:** Andy Zou et al., 2024
**arXiv:** https://arxiv.org/abs/2406.04313

---

## The argument against refusal-only defense

Refusal SFT teaches a token-level surface behavior: when the input looks harmful, emit a refusal string. The raw-data source states the failure mode directly:

> *"The paper argues that standard refusal tuning is fragile because it only changes surface behavior. Circuit breakers instead modify internal representations associated with harmful outputs."*

For ch-52 §4, the mental model is: refusal SFT changes the *output*; circuit breakers change the *path*. An adversarial suffix (GCG) or persuasive rewrite (PAP) moves the input to a region where the refusal classifier fails to fire — but the model's internal computation for generating the harmful completion is still intact. Fix the representation and the jailbreak has nothing left to elicit.

---

## The training recipe — why the data shape is unusual

Three data components:

- **Harmful prompt-response pairs.** Drawn from HarmBench, AdvBench, SORRY-Bench. Crucially, the dataset contains the **harmful completion**, not only a refusal target. This inverts normal safety-data intuition ("never include harmful completions in training"). The loss is defined over the hidden-state trajectory that produces the harmful output; you cannot optimize that trajectory without observing it.
- **Benign retain set.** Ordinary assistant examples. The retain objective preserves MMLU / GSM8K / normal chat behavior.
- **No refusal strings required.** Circuit breakers are orthogonal to refusal training. The model can be refusal-tuned, CAI-tuned, or plain — the representation intervention is independent.

---

## Representation Rerouting (RR) — mechanism sketch

Run the model on a harmful prompt and its harmful target completion. At selected layers, identify the hidden states along the harmful trajectory. Optimize those states to move **away** from their original direction. Simultaneously optimize the retain loss on benign data to preserve the benign trajectory.

The intuition: the harmful completion is a specific path through the residual stream. Moving the hidden states off that path removes the model's capacity to trace it — not by classifier intervention but by structural representation shift. Combined with the retain objective, benign paths remain intact.

The operational simplicity:
- **Training form:** LoRA-style fine-tuning is sufficient.
- **No full retrain:** can be installed on a frozen production model as a post-hoc safety layer.
- **Small dataset:** a relatively small synthetic harmful-behavior set is enough.

---

## Evaluation — the important distinction

The raw-data source specifies that robustness is measured with **strong attack generators**, not clean-prompt refusal rate:

> *"Demonstrates substantially improved robustness on attack suites such as HarmBench-style jailbreak evaluation."*

Refusal-SFT models often look good on clean-prompt refusal and collapse under GCG/PAIR. Circuit-breaker models keep clean-prompt behavior intact (via retain) while cutting attack success rates substantially. The useful reported metric is attack-success under adversarial attack, not aggregate refusal.

---

## Risks the raw-data source flags

- **Coverage is bounded by the seed set.** Unseen attacks can still find hidden-state paths the rerouted region did not cover. The defense is as broad as the harmful-completion training set, not broader.
- **Retain-weight tradeoff.** Over-weighting rerouting creates collateral refusal on benign inputs. Under-weighting leaves the harmful path accessible. The weight is the main tuning axis.
- **Scope is narrow.** It is a **defense against harmful-output elicitation**, not a general alignment recipe. It does not address honesty, goal misalignment, or deceptive reasoning.

---

## For ch-52: the layered-defense claim

Pair refusal SFT (coverage on known harm shapes) + [[constitutional-ai]] (nuanced non-evasive refusal) + circuit breakers (adversarial robustness) + [[prosocial-dialog]] (engagement over refusal on borderline). Each layer addresses a distinct attack surface; none is sufficient alone.

---

## Connections

- [[harmbench-data]] — primary seed data for the harmful completion set; the defense is benchmarked against HarmBench attacks.
- [[anthropic-sleeper-agents-data]] — threat-model contrast. Circuit breakers defend against adversarial *elicitation*; sleeper agents are *conditional misbehavior* from inside the model. Different failure modes, different defenses.
- [[wildguard-data]] — moderation-classifier sibling. WildGuard is input-side refusal supervision; circuit breakers intervene on representations.
- Chapter synthesis: [[ch-52]] §4, §4.1.
