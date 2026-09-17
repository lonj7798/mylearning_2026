<!-- scope: Wallace et al. (arXiv:2404.13208, OpenAI): an instruction priority order (system > user > tool outputs) with automated conflict-data generation, fine-tuned into GPT-3.5 Turbo with SFT and RLHF, and its robustness and over-refusal results
     see-also: [[secalign]], [[gpt-5-system-card]], [[agentic-finetuning-misalignment]]
-->

# The Instruction Hierarchy: Training LLMs to Prioritize Privileged Instructions

Chapter excerpt for ch-52 (read.md). Written 2026-09-15 from the cached primary text of arXiv:2404.13208v1 (19 Apr 2024). No library card existed for this slug at the time of writing.

- **Core Insight:** LLMs treat system prompts, user messages and third-party text as equally authoritative; training GPT-3.5 Turbo on synthetic data that enforces the priority order system > user > tool outputs raised robustness by up to 63% on attack types modeled in the data pipeline and by up to 34% on attack types held out of it, with jailbreak robustness rising by over 30% (§1, §4).
- **Guideline:** When a model reads untrusted third-party content through browsing or tool calls, define the priority order explicitly and generate training data for conflicting instructions, and measure over-refusal at the same time, because the authors report regressions where the trained model ignores lower-priority instructions it should have followed (§1, §4, Fig. 4).
- **Authors:** Eric Wallace, Kai Xiao, Reimar Leike, Lilian Weng, Johannes Heidecke, Alex Beutel (OpenAI)
- **Year:** 2024 (arXiv v1 2024-04)
- **URL:** https://arxiv.org/abs/2404.13208
- **Source type:** paper

## Technical details used by ch-52

- **Hierarchy.** Instructions are assigned priority levels: system message, then user message, then tool outputs and other third-party content. On conflict, the higher-priority instruction wins; lower-priority instructions are selectively ignored (§2, §3).
- **Threat classes.** Direct prompt injections (in the user turn), indirect prompt injections (in browsing or tool output), system-prompt extraction, and jailbreaks (§2).
- **Data generation.** Red-teamer LLMs generate conflicting instructions per task; ground-truth responses follow the higher-priority instruction; examples where the injection succeeded despite instructions are discarded. Some injection types are deliberately excluded from training in order to test generalization (§3).
- **Training and evaluation.** GPT-3.5 Turbo fine-tuned with supervised fine-tuning and RLHF on this data (§4). Evaluations include held-out direct injections, indirect injections via browsing, system-prompt extraction and probing, jailbreaks, and dedicated over-refusal sets (§4, App.).
- **Reported effects (§1, §4).** Robustness improvements up to 63% on evaluations whose attack types were modeled; up to 34% on types that were not; jailbreak robustness over 30%. Over-refusal regressions are reported qualitatively rather than as a single headline number; Fig. 4 reports per-set over-refusal scores.

## Limits stated by the source

- The implementation is not released, so third parties cannot reproduce it on open weights (a point [[secalign]] makes when it evaluates GPT-4o-mini as a proxy).
- The over-refusal cost is acknowledged: the model sometimes ignores lower-priority instructions that were legitimate.
- Results are for one model (GPT-3.5 Turbo) at one point in time.

## Verification

- Read on 2026-09-15 from the cached primary text of arXiv:2404.13208v1.
- Not reported by the source: dataset sizes, training hyperparameters, per-attack ASR tables in numeric form (results are presented as bar charts in Figures 2-4).
