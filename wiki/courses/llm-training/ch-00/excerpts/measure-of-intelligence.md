---
chapter: ch-00
course: llm-training
phase: read
excerpt_of: primary source arXiv:1911.01547v2 (no library card as of 2026-09-15)
source_url: https://arxiv.org/abs/1911.01547
created_at: "2026-09-15"
---

# Excerpt: On the Measure of Intelligence

**Paper:** François Chollet (Google). arXiv v1 2019-11, read at v2 (2019-11-25). Source type: paper (position paper with a benchmark proposal, ARC).

## Main argument (Abstract; §II.1.1)
- Measuring skill at a given task does not measure intelligence, because skill is modulated by prior knowledge and experience: with unlimited priors or unlimited training data, experimenters can reach arbitrary skill levels "in a way that masks the system's own generalization power" (Abstract).
- Proposed definition: intelligence as skill-acquisition efficiency, accounting for scope, generalization difficulty, priors, and experience (Abstract; §II.2).

## Kinds of generalization (§I.3.2)
- System-centric generalization: ability of a learning system to handle situations it has not itself encountered (for example error on images outside the training set).
- Developer-aware generalization: ability to handle situations that neither the system nor its developer has encountered; it accounts for prior knowledge the developer injected, for example via a development set.
- Degrees: absence of generalization; local generalization or robustness (new points from a known distribution for one task or a well-scoped set of tasks); broad generalization or flexibility (a broad category of tasks and environments without further human intervention, including situations the creators could not foresee); extreme generalization (entirely new tasks sharing only abstract commonalities).

## Relevance to benchmark suites (§I.3.5)
- Broadening task-specific benchmarks by including multiple tasks does not measure developer-aware generalization, because all tasks are known in advance to system developers.

## Verification
- Read on 2026-09-15 against arXiv:1911.01547v2 PDF text (Abstract, §I.3.2, §I.3.5, §II.1.1).
