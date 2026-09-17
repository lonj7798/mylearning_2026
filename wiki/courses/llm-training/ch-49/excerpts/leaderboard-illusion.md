---
chapter: ch-49
course: llm-training
phase: read
excerpt_of: primary source (no library card at the time of this revision)
source_url: https://arxiv.org/abs/2504.20879
created_at: "2026-09-15"
---

# Excerpt: The Leaderboard Illusion

**Authors:** Shivalika Singh, Yiyang Nan, Alex Wang, Daniel D'souza, Sayash Kapoor, Ahmet Üstün, et al. (Cohere Labs, Cohere, Princeton, Stanford, Waterloo, MIT, AI2, University of Washington)
**Year:** 2025 (arXiv v1 2025-04; v2 2025-05-12)
**Source type:** paper
**Checked on:** 2026-09-15 against the arXiv v2 PDF text.

## Why ch-49 uses it

It documents how a leaderboard score can rise without capability rising, through selective disclosure and asymmetric data access rather than through an adversarial string. It is the non-adversarial companion to [[null-model-cheating-benchmarks]].

## Reported findings (Abstract, §1)

1. **Private testing and selective disclosure.** Some providers test multiple private variants before public release and can retract scores. The authors identify 27 private LLM variants tested by one provider in the lead-up to the Llama-4 release. Choosing the best of several measured variants biases the published Arena score upward, and the authors state it violates the unbiased-sampling assumption of the Bradley–Terry model (§3, Fig. 7).
2. **Sampling asymmetry.** Google and OpenAI are estimated to have received 19.2% and 20.4% of all arena data respectively; 83 open-weight models combined received an estimated 29.7%.
3. **Value of that data.** In their experiments, additional arena-distribution data produced relative performance gains of up to 112% on Arena-Hard.
4. **Deprecation.** Of 243 public models, 205 were silently deprecated, against 47 officially listed as deprecated in the Chatbot Arena backend codebase. 64% of the silently deprecated models are open-weight or open-source. The authors show deprecation can violate assumptions of the Bradley–Terry model and produce unreliable ratings (§5, Fig. 18).

## Framing stated by the authors

They describe the combined effect as overfitting to Arena-specific dynamics rather than general model quality, and state that the paper builds on the work of the Arena organizers and the open community maintaining the platform.

## How ch-49 uses each number

- The 27-variant figure is the concrete case of "maximum over k noisy measurements" as a source of inflated reported scores.
- The 112% relative gain is the measured value of access to the evaluation distribution, which is the same mechanism as training on the evaluation distribution in [[ch-48]].
- The deprecation counts explain why a historical leaderboard rank is not a stable measurement.

## Connections

[[chatbot-arena]] (the platform audited), [[arena-hard-benchbuilder]] (the test set used to measure the gain from arena data), [[null-model-cheating-benchmarks]] (adversarial inflation of the automatic analogues).
