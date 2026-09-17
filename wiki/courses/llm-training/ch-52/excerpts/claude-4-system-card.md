<!-- scope: Claude 4 system card (Anthropic, May 2025): single-turn violative and benign request evaluations with confidence intervals, ambiguous-context and multi-turn testing, and the computer-use prompt-injection evaluation with and without safeguards
     see-also: [[gpt-5-system-card]], [[xstest]], [[instruction-hierarchy]]
-->

# Claude 4 System Card (Claude Opus 4 and Claude Sonnet 4)

Chapter excerpt for ch-52 (read.md). Written 2026-09-15 from the cached primary text of the system card. No library card existed for this slug at the time of writing. Source type: official technical report from the organization that trained the models.

- **Core Insight:** The card reports refusal on violative requests and over-refusal on benign sensitive requests as a matched pair with confidence intervals — Claude Opus 4 at 98.43% (± 0.30%) harmless response rate and 0.07% (± 0.07%) over-refusal — and reports agentic prompt-injection resistance separately, at 71% attack prevention without safeguards and 89% with them over about 600 scenarios (Tables 2.1.A, 2.2.A, 3.2.A).
- **Guideline:** When publishing a safety card, report the harmful-request and benign-request rates from the same evaluation run with intervals, and report agentic prompt-injection scores both with and without deployment-time safeguards, because the two answer different questions about the model and the system around it.
- **Organization:** Anthropic
- **Year:** 2025 (May 2025)
- **Source type:** official technical report

## Technical details used by ch-52

- **Single-turn violative requests (Table 2.1.A).** Harmless response rate: Claude Opus 4 98.43% (± 0.30%); with ASL-3 safeguards 98.76% (± 0.27%); Claude Sonnet 4 98.99% (± 0.23%); Claude Sonnet 3.7 98.96% (± 0.22%). Standard and extended thinking are reported separately; extended thinking is higher for every model.
- **Single-turn benign requests (Table 2.2.A).** Over-refusal rate: Claude Opus 4 0.07% (± 0.07%); Claude Sonnet 4 0.23% (± 0.11%); Claude Sonnet 3.7 0.45% (± 0.20%). All three below 0.5%.
- **Residual failures.** The card states the remaining harmful responses were generally cases where the model did not recognize subtle harmful intent and offered well-intentioned assistance (§2.1).
- **Other evaluation shapes (§2.3, §2.4).** Ambiguous-context single-turn evaluations labeled by human raters, and multi-turn conversations built with policy experts, are reported as separate categories because their failure modes differ from clear-cut single-turn cases.
- **Computer-use prompt injection (§3.2, Table 3.2.A).** The evaluation set was expanded from the Claude Sonnet 3.7 assessment to about 600 scenarios covering coding platforms, web browsers and user workflows such as email management. Attack prevention scores without / with safeguards: Claude Opus 4 71% / 89%; Claude Sonnet 4 69% / 86%; Claude Sonnet 3.7 74% / 88%. Safeguards include injection-specific reinforcement learning and a detection system that can halt execution.
- **Agentic coding misuse (§3.3).** Three evaluations: 150 clearly prohibited problems and two sets of 50 borderline harmful and non-harmful problems, the latter included to check calibration between refusing and over-refusing.

## Limits stated by the source

- Prompt-injection defenses are described as incomplete; the card states mitigations are not all described and that work continues.
- Multi-turn testing found cases where responses were not clearly harmless when requests were framed as fictional or educational (§2.4).

## Verification

- Read on 2026-09-15 from the cached primary text of the Claude 4 system card (sections 2 and 3).
- Not reported by the source: evaluation-set sizes for the single-turn violative and benign sets; the judge or rater protocol for the automated portions of §2.1 and §2.2.
