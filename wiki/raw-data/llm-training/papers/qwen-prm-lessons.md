<!-- scope: Qwen2.5-Math-PRM (Zhang et al., Qwen, 2025) — why Monte Carlo (MC) step labels underperform LLM-judge and human labels, consensus filtering of MC labels with an LLM judge, hard-label threshold, and the bias of Best-of-N as the only PRM evaluation
     deps: [[math-shepherd]], [[lets-verify]], [[processbench]]
     see-also: [[omegaprm]], [[prm800k]], [[skywork-o1]], [[generative-reward-models]], [[deepseek-r1]]
-->

# The Lessons of Developing Process Reward Models in Mathematical Reasoning
- **Core Insight:** On identical 860k responses, step labels from MC estimation give the highest Best-of-8 average (65.9) but the lowest ProcessBench F1 (40.1), while LLM-judge labels give 65.3 / 46.5 and 264k human PRM800K labels give 64.9 / 56.5; the two evaluations rank the label sources in opposite order because policy responses often reach correct answers through wrong steps and Best-of-N rewards PRMs that score the final answer (§3.1.2, Tables 3–4; §3.2).
- **Guideline:** If step labels must come from MC rollouts, label a step negative only when none of the 8 completions reaches the correct answer, keep only instances where an LLM judge agrees with the MC estimate on the first-error location (about 40% retained), train on hard labels, and evaluate with a step-level benchmark in addition to Best-of-N (§3.1.3–3.1.5, §3.2.5).
- **Authors:** Zhenru Zhang, Chujie Zheng, Yangzhen Wu, Beichen Zhang, Runji Lin, Bowen Yu, Dayiheng Liu, Jingren Zhou, Junyang Lin (Qwen Team, Alibaba Group)
- **Year:** 2025 (arXiv v1 2025-01-13; v2 2025-06-05)
- **URL:** https://arxiv.org/abs/2501.07301 (released models: Qwen2.5-Math-PRM-7B, Qwen2.5-Math-PRM-72B on Hugging Face, linked from the paper)
- **Source type:** paper
- **Relevant topics:** process reward models, MC-estimated step labels, LLM-as-a-judge labeling, consensus filtering, hard vs soft labels, Best-of-N evaluation bias, ORM vs PRM

## Abstract
Process reward models (PRMs) are meant to find intermediate errors in LLM reasoning. The paper reports that MC estimation, a common way to synthesize PRM training data, gives worse performance and generalization than LLM-as-a-judge and human annotation, because it relies on a completion model to judge whether the current step is correct. It also reports biases in Best-of-N (BoN) evaluation: policies produce correct answers through flawed processes, PRMs tolerate such responses, which inflates BoN scores, and many PRMs place their minimum score on the final answer step, so they act as outcome judges. The authors propose consensus filtering, which combines MC estimation with LLM-as-a-judge, and an evaluation that combines response-level and step-level metrics. They release a PRM that they state outperforms existing open PRMs and give practical guidelines (Abstract).

## Key Contributions
- Controlled comparison of four step-label sources on matched or published data, with both BoN and ProcessBench (§3.1.2, Tables 3–4).
- Consensus filtering between an LLM judge and MC estimates (§3.1.3, Fig. 2) and the hard-label threshold study (§3.1.4, Fig. 5).
- Four documented BoN biases: flawed processes behind correct answers, PRM tolerance of them, minimum scores concentrated on the final step, and scoring-rule dependence (§3.2.1–3.2.4).
- Qwen2.5-Math-PRM-7B and -72B, trained with the resulting recipe (§4.1, Tables 6–7).

## Key Figures/Tables to Study
- Tables 3–4: label source vs BoN and ProcessBench; the central result.
- Fig. 2: MC, LLM judge, consensus filtering. Fig. 5: hard-label threshold 0 to 7/8.
- Fig. 6: share of correct-answer responses with flawed steps. Table 5: PRM detection of those cases.
- Fig. 8: share of responses whose minimum step score is on the final step. Fig. 9: min / product / last scoring.
- Tables 6–7: final models vs open PRMs and an ORM. Table 8: PRM-guided greedy search.

## Technical Details
- **Data synthesis (§2.1).** About 500,000 queries with golden answers; 6–8 responses per query from Qwen2-Math-Instruct and Qwen2.5-Math-Instruct models at 7B and 72B; steps split on "\n\n". Each step gets 8 independent completions from the same-size Qwen2.5-Math-Instruct. Hard label: correct if any completion reaches the correct answer. Soft label: fraction of completions that do. All steps after the first step labeled 0 are removed.
- **Model (§2.1, §4.1).** Initialized from Qwen2.5-Math-7B/72B-Instruct; the LM head is replaced by a scalar head of two linear layers. Loss on the last token of each step: cross-entropy for hard labels, MSE for soft labels.
- **Evaluation (§2.2, §4.2).** BoN: 8 responses from Qwen2.5-Math-7B-Instruct on GSM8K, MATH, Minerva Math, GaoKao 2023 En, OlympiadBench, College Math, MMLU STEM; response score is the product of step scores; maj@8 66.2 and pass@8 74.7 average (Table 6). Step level: ProcessBench F1.
- **MC vs PRM correctness (§3.1.1).** A PRM should judge whether the current step is correct. An MC estimate is the probability that a completion model reaches the correct answer from that step, which is a value estimate; a completion model can reach the right answer from a wrong step or fail from a correct one.
- **Label sources (Tables 3–4; samples, BoN avg, ProcessBench avg F1).** MC on Math-Shepherd data 440k, 64.3, 28.9; MC on the authors' data 860k, 65.9, 40.1; LLM-as-a-judge (Qwen2.5-72B-Instruct) on the same 860k, 65.3, 46.5; human PRM800K 264k, 64.9, 56.5. PRM800K covers only MATH problems yet generalizes best to OlympiadBench and Omni-MATH (§3.1.2).
- **Consensus filtering (§3.1.3, Fig. 2).** Keep an instance only if the LLM judge and the MC estimate agree on the error-step location. About 40% of the 860k set is kept (350k in Fig. 2). Fig. 2 values: BoN 65.7 and ProcessBench 46.3 for the filtered set, against 65.9 / 40.1 (MC) and 65.3 / 46.5 (judge).
- **Hard vs soft labels (§3.1.4, Figs. 3–4).** On a 3M MC set, filtered to 1.5M by consensus filtering, soft and hard labels perform similarly before filtering and hard labels perform clearly better after filtering, on both BoN and ProcessBench. Stated reasons: step correctness should be deterministic, so soft targets below 1 for correct steps reduce separation; and 8 completions give a high-variance estimate.
- **Threshold (§3.1.4, Fig. 5).** On the 3M set, raising the hard-label threshold from 1/8 to 7/8 in steps of 1/8 lowers both BoN and ProcessBench. Recommended threshold: 0 (negative only if no completion is correct), used throughout the paper.
- **Flawed correct answers (§3.2.1, Fig. 6).** Among correct-answer responses from Qwen2.5-Math-7B-Instruct, the share with a flawed step is 5.1% (GSM8K), 11.9% (MATH), 27.4% (OlympiadBench), 43.4% (Omni-MATH) (figure labels).
- **PRM tolerance (§3.2.2, Table 5).** Detection accuracy on ProcessBench cases with a correct answer and an erroneous step (7 / 94 / 161 / 259 cases by subset): Math-Shepherd-PRM-7B 13.9; Skywork-PRM-7B 27.8; Qwen2.5-Math-7B-PRM800K 38.2; Qwen2.5-Math-PRM-7B 53.9; Qwen2.5-Math-PRM-72B 58.1. All open PRMs other than the authors' are below 50%.
- **Process-to-outcome shift (§3.2.3, Fig. 8).** Share of Best-of-8 responses whose minimum step score is on the final step: EurusPRM-Stage1 54.6%, EurusPRM-Stage2 52.9%, Math-Shepherd-PRM-7B 44.5%, Skywork-PRM-7B 42.2%, Qwen2.5-Math-7B-PRM800K 26.8%, Qwen2.5-Math-PRM-72B 18.0%, Qwen2.5-Math-PRM-7B 17.5%, RLHFlow-PRM-Mistral-8B 9.1% (figure labels).
- **Scoring rule (§3.2.4, Fig. 9).** For MC-trained PRMs, scoring by the last step beats product and minimum; for human-label and LLM-judge PRMs the order reverses.
- **Final models (§4.1, Tables 6–7).** Expansion by MC estimation with hard labels at threshold 0, then consensus filtering with Qwen2.5-72B-Instruct as judge, then cross-entropy at step ends. Best-of-8 average: Qwen2.5-Math-PRM-7B 67.6, -72B 69.3, the ORM Qwen2.5-Math-RM-72B 68.9, Skywork-PRM-7B 64.8. ProcessBench F1: PRM-7B 73.5, PRM-72B 78.3; o1-mini 87.9, QwQ-32B-Preview 71.5, GPT-4-0806 61.9, Qwen2.5-Math-RM-72B 38.9.
- **Greedy search (App. A, Table 8).** Policy Qwen2.5-7B-Instruct, 8 candidates per step: PRM-72B 66.6 vs ORM orm@8 66.5; PRM-7B 63.5; Skywork-PRM-7B 62.2; maj@8 62.5.

## Recipe ledger

| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| Qwen2.5-Math-PRM-7B/72B | 7B, 72B | reward-model | MC completions per step; completer | 8; same-size Qwen2.5-Math-Instruct | arXiv:2501.07301v2 §2.1, §4.1 | verified 2026-11-24 | §3.1.4: more completions judged not worth the cost; no number |
| Qwen2.5-Math-PRM-7B/72B | 7B, 72B | reward-model | hard-label threshold | 0 (negative only if 0 of 8 completions correct) | v2 §3.1.4, §4.1 | verified 2026-11-24 | Fig. 5: BoN and ProcessBench fall as threshold rises 1/8 → 7/8 |
| Qwen2.5-Math-PRM-7B/72B | 7B, 72B | reward-model | filter | consensus between MC and Qwen2.5-72B-Instruct judge on error location (~40% kept) | v2 §3.1.3, §4.1 | verified 2026-11-24 | Fig. 2: ProcessBench 46.3 (350k) vs 40.1 (860k MC) |
| Qwen2.5-Math-PRM-7B/72B | 7B, 72B | reward-model | label type; loss | hard; cross-entropy at last token of each step | v2 §3.1.4, §4.1 | verified 2026-11-24 | Figs. 3–4: hard > soft after filtering |
| Qwen2.5-Math-PRM-7B/72B | 7B, 72B | reward-model | init; head | Qwen2.5-Math-7B/72B-Instruct; two-linear-layer scalar head | v2 §2.1, §4.1 | verified 2026-11-24 | no ablation reported |
| Qwen2.5-Math-PRM-7B/72B | 7B, 72B | reward-model | final data size; LR; epochs; batch | not reported | checked v2 §2, §4, App. A–C | not reported | — |
| BoN evaluation | — | eval-gate | response score | product of step scores | v2 §4.2 | verified 2026-11-24 | Fig. 9: best rule depends on label source |

## Findings relevant to generality and negative feedback
- **Generality.** Human labels from MATH alone generalize to OlympiadBench and Omni-MATH better than 3× more MC labels (§3.1.2, Table 4). A 7B PRM does not beat maj@8 when ranking responses of the 72B policy (App. B.1, Table 9), so a PRM smaller than the policy may not supervise it.
- **Negative feedback.** A negative step label from MC means "no completion recovered", not "this step is wrong"; false negatives come from completer failure and false positives from completions that recover from a wrong step (§3.1.1). Correct-answer responses with flawed steps are positive under outcome labels and are tolerated by most PRMs (§3.2.1–3.2.2).
- **Evaluation as an optimization target.** The authors state that optimizing PRMs for BoN alone has made them behave like ORMs (§3.2.3); the ORM Qwen2.5-Math-RM-72B has the second-best BoN average (68.9) but 38.9 ProcessBench F1 (Tables 6–7).
- **Limitations stated (§6).** A large gap to pass@8 remains; the use of PRMs inside RL is not studied; use of existing human labels is under-explored.

## Connections
- [[processbench]]: the step-level benchmark used here, from the same team.
- [[math-shepherd]], [[omegaprm]]: MC step labeling; Math-Shepherd data is the 440k baseline.
- [[lets-verify]], [[prm800k]]: the human-label baseline and the "label up to the first error" convention.
- [[generative-reward-models]]: LLM-as-a-judge labeling is the component that improves MC labels here.
- [[deepseek-r1]]: reports PRMs among unsuccessful attempts for large-scale RL; this paper leaves PRM use in RL open (§6).

## Verification
- Checked on 2026-11-24 against https://arxiv.org/abs/2501.07301, the v2 HTML, and the v2 PDF text (pdftotext). v1 not compared.
- Figure values (Fig. 2, 6, 8) were read from figure labels in the PDF text, not from body text. The Fig. 6 order (GSM8K, MATH, OlympiadBench, Omni-MATH) follows the axis label order.
- The HTML body cites figure numbers that differ from the PDF captions (e.g. it cites Fig. 4 for consensus filtering); this card uses the PDF captions.
- §3.1.2 states PRM800K has about 265k samples after test-set deduplication and Math-Shepherd 445k; Tables 3–4 list 264k and 440k. In this comparison the 860k responses come from Qwen2-Math-Instruct and completions from Qwen2.5-Math-Instruct (§3.1.2).
- Corrections to the chapter excerpt ch-44/excerpts/qwen-prm-lessons.md: none of its numbers were wrong; it omitted that the ORM Qwen2.5-Math-RM-72B (68.9) exceeds Qwen2.5-Math-PRM-7B (67.6) on Best-of-8 (Table 6), and that the 72B PRM detects 58.1% of correct-answer-with-error cases (Table 5).
