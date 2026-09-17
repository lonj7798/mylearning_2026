<!-- scope: GPT-5 system card (OpenAI, August 2025): safe-completions as a training target, disallowed-content and StrongREJECT-based jailbreak evaluations, per-channel prompt-injection scores, and red-team effort and yield statistics
     see-also: [[claude-4-system-card]], [[instruction-hierarchy]], [[xstest]]
-->

# GPT-5 System Card

Chapter excerpt for ch-52 (read.md). Written 2026-09-15 from the cached primary text of the system card (August 13, 2025). No library card existed for this slug at the time of writing. Source type: official technical report from the organization that trained the models.

- **Core Insight:** The card replaces the binary refuse-or-comply training target with "safe-completions", which maximizes helpfulness subject to the safety policy and centers on the safety of the output rather than a classification of user intent (§3.1); it reports prompt injection per channel — 0.99 browsing, 0.99 tool calling, 0.97 coding for gpt-5-thinking against 0.89 / 0.80 / 0.94 for OpenAI o3 (Table 7) — and reports red-team effort with denominators, including 277 accepted jailbreak reports over 28,367 bug-bounty attempts, an attack success rate of 0.98% (§4).
- **Guideline:** When a safety benchmark saturates, replace it rather than continuing to report it; the card states its standard disallowed-content set "has become relatively saturated" and no longer gives useful signal, and introduces a harder multi-turn production benchmark alongside it (§3.2).
- **Organization:** OpenAI
- **Year:** 2025 (August 13, 2025)
- **Source type:** official technical report

## Technical details used by ch-52

- **Safe-completions (§3.1).** Motivated by dual-use prompts (biology, cybersecurity) where a request can be answered safely at a high level but becomes uplift if sufficiently detailed. Reported effects relative to a refusal-trained baseline (OpenAI o3): improved safety especially on dual-use prompts, reduced severity of residual failures, higher overall helpfulness. The card points to a separate paper for the method.
- **Disallowed content (§3.2, Table 2).** Metric `not_unsafe`, graded by LLM graders. gpt-5-thinking scores 1.000 on hate (aggregate), 0.991 illicit/non-violent, 1.000 illicit/violent, 0.881 personal-data, 0.989 personal-data/restricted. A separate, harder, multi-turn production benchmark is reported because the standard set is saturated.
- **Jailbreaks (§3.4, Table 5).** StrongREJECT is used to insert a known jailbreak into a refusal-eval prompt, graded with the same policy graders as the disallowed-content evaluation. gpt-5-thinking `not_unsafe`: 0.995 illicit/non-violent-crime, 0.999 violence, 0.999 abuse/disinformation/hate, 0.995 sexual content; gpt-5-main is close to parity with GPT-4o.
- **Prompt injection (§3.6, Table 7).** Three separate evaluations — browsing, tool calling, coding. gpt-5-thinking 0.99 / 0.99 / 0.97; OpenAI o3 0.89 / 0.80 / 0.94. A system-level mitigation is described: after a connector call, browsing is restricted to cached pages to prevent exfiltration.
- **Red-team effort and yield (§4).** 19 biology-PhD red-teamers reported 46 potential jailbreaks after about 380 hours, which the card converts to roughly 8.2 red-teamer-hours per report. A bioweaponization bug bounty produced 277 high-quality reports over 28,367 attempts against ten rubrics, an ASR of 0.98%, grouped into 6 distinct cohorts, with 58 of 60 sampled examples (96.7%) meeting the grading rubric. Two external groups ran a two-week system-level prompt-injection assessment: 47 findings reduced to 10 notable issues, mitigated before release.
- **Residual risk stated (§4).** One jailbreak that evaded all mitigation layers was being patched at publication; the card names previously unknown universal jailbreaks as an acknowledged risk and describes bug-bounty and rapid-remediation programs as the post-release path.
- **Deception monitoring (§3.5).** A chain-of-thought monitor flagged deception in about 4.8% of OpenAI o3 production responses in the sampled traffic, used as a baseline for the GPT-5 comparison.

## Limits stated by the source

- Prompt-injection mitigations are described as incomplete because of the adversarial nature of the problem; not all are described.
- Multi-turn, tailored attacks may occasionally succeed even where single-turn generic jailbreaks do not.

## Verification

- Read on 2026-09-15 from the cached primary text of the GPT-5 system card (sections 3 and 4).
- Not reported by the source: evaluation-set sizes for the prompt-injection tables; grader identity and calibration for `not_unsafe`; training details for safe-completions beyond the referenced paper.
