<!-- scope: ProcessBench (Zheng et al., Qwen, 2024) — 3,400 expert-annotated math solutions labeled with the index of the earliest erroneous step; compares process reward models (PRMs) with prompted critic LLMs
     deps: [[lets-verify]], [[prm800k]], [[math-shepherd]]
     see-also: [[qwen-prm-lessons]], [[omegaprm]], [[skywork-o1]], [[generative-reward-models]]
-->

# ProcessBench: Identifying Process Errors in Mathematical Reasoning
- **Core Insight:** On 3,400 expert-labeled solutions, the best open PRM trained on automatically estimated (Math-Shepherd-style) labels reaches 42.1 average F1 at locating the earliest error (Skywork-PRM-7B), a PRM fine-tuned on human PRM800K labels reaches 56.5, and a prompted critic reaches 71.5 (QwQ-32B-Preview) and 87.9 (o1-mini); all models lose F1 from GSM8K/MATH to OlympiadBench/Omni-MATH (Table 3, §4.2).
- **Guideline:** Evaluate a step verifier on earliest-error localization with balanced F1 over erroneous and correct solutions, on problems harder than its training distribution, and do not treat final-answer correctness as a proxy for process correctness, because 259 of the 500 correct-final-answer Omni-MATH solutions sampled for this benchmark (51.8%) contain a process error (§3.3, Table 2; §4.1).
- **Authors:** Chujie Zheng, Zhenru Zhang, Beichen Zhang, Runji Lin, Keming Lu, Bowen Yu, Dayiheng Liu, Jingren Zhou, Junyang Lin (Qwen Team, Alibaba Inc.)
- **Year:** 2024 (arXiv v1 2024-12-09; v4 2025-05-26)
- **URL:** https://arxiv.org/abs/2412.06559 (data: https://huggingface.co/datasets/Qwen/ProcessBench; code: https://github.com/QwenLM/ProcessBench)
- **Source type:** paper (benchmark)
- **Relevant topics:** process reward models, step-level verification, critic models, LLM-as-a-judge, evaluation of verifiers, generalization across difficulty, MC-estimated labels vs human labels

## Abstract
ProcessBench measures whether a model can identify the earliest erroneous step in a mathematical solution. It has 3,400 test cases drawn mainly from competition- and Olympiad-level problems. Each case is a step-by-step solution whose earliest error location was annotated by human experts. A model is given the solution and must return the index of the earliest wrong step, or report that all steps are correct. The paper evaluates two model types: process reward models (PRMs) and critic models, which are general LLMs prompted to critique each step. The authors report that existing PRMs generally fail to generalize to problems harder than GSM8K and MATH, that they underperform prompted critic models and a PRM the authors fine-tuned on PRM800K, and that the open model QwQ-32B-Preview is competitive with GPT-4o as a critic but behind o1-mini (Abstract).

## Key Contributions
- A benchmark of 3,400 solutions with expert labels for the earliest error step; 2,221 erroneous and 1,179 correct cases across four subsets (Table 2).
- A task format with one output per solution: index i ≥ 0 of the earliest erroneous step, or i = −1 if every step is correct (§3.1, Fig. 2).
- A metric that is the harmonic mean (F1) of accuracy on erroneous samples and accuracy on correct samples, so that a model cannot score well by always flagging an error or never flagging one (§4.1).
- Measurement of process errors hidden behind correct final answers, as the share of correct-final-answer samples annotated as erroneous: 3.5% (GSM8K, 7 of 200), 18.8% (MATH, 94 of 500), 32.2% (OlympiadBench, 161 of 500), 51.8% (Omni-MATH, 259 of 500) (§3.3, Table 2).
- A side-by-side evaluation of 6 open PRMs, 17 open critic models and 2 proprietary critic models (Table 3).

## Key Figures/Tables to Study
- Fig. 2: one annotated sample; defines the label convention (label 2 = third step, 0-indexed).
- Table 2: per-subset statistics (sample counts, process-error share among correct-answer samples, step counts, annotator agreement).
- Table 3: the main F1 table for all PRMs and critic models per subset.
- Fig. 3 (distribution of error positions) and Fig. 4 (process-error rate by generator and subset).
- Tables 6 and 8: Skywork PRMs with per-subset oracle thresholds (upper-bound check on the threshold choice).

## Technical Details
- **Problems (§3.2).** Test sets of GSM8K, MATH, OlympiadBench, and Omni-MATH. Subset sizes (erroneous / correct): GSM8K 207 / 193; MATH 594 / 406; OlympiadBench 661 / 339; Omni-MATH 759 / 241 (Table 2, Table 4).
- **Solutions (§3.2, Table 4, App. D).** Twelve open Qwen and Llama models generate the solutions (Llama-3/3.1 8B/70B-Instruct; Qwen2 and Qwen2.5 Instruct 1.5B/7B/72B; Qwen2.5-Math-7B/72B-Instruct), with nucleus sampling p = 0.9 and temperature 0.7. Solutions are sampled so correct and incorrect final answers are balanced, with Qwen2.5-72B-Instruct checking the final answers.
- **Step reformatting (§3.2, App. A).** Line breaks are replaced with spaces, then Qwen2.5-72B-Instruct re-inserts double line breaks to mark steps. The authors state the model sometimes altered content (<0.5%); solutions whose final answer changed are removed.
- **Average steps (Table 2).** Erroneous / correct: GSM8K 5.3 / 5.1; MATH 6.8 / 6.0; OlympiadBench 8.9 / 8.7; Omni-MATH 8.6 / 7.4. Share with ≥15 steps reaches 9.1% (OlympiadBench, erroneous).
- **Annotation (§3.2).** Annotators have doctoral-level mathematics expertise, pass a proficiency exam and tutorial, and see the reference solution and answer. Each solution starts with three annotators; more are added until three agree on the earliest error location; with no three-way agreement within five annotators, the solution is discarded. The overall discard rate is about 30%. Agreement among the first three annotators (% 3/3) is 47.8%–66.7% on erroneous samples and 80.1%–95.9% on correct samples, lowest on Omni-MATH (Table 2). Solutions with an incorrect final answer that annotators judged fully correct are also discarded (<1%).
- **Error definition (§3.2).** A step is erroneous if it contains, typically but not exclusively, a mathematical, logical, conceptual, or completeness error; annotators may also use their own judgment.
- **PRM evaluation (§4.1).** A PRM scores each step; the earliest step below a threshold is the predicted error. For PRMs with scalar outputs (Skywork), the threshold is the value giving the highest F1 on the GSM8K subset, then applied to all subsets. Per-subset oracle thresholds raise Skywork-PRM-7B only from 53.6 to 54.4 (MATH), 22.9 to 27.1 (OlympiadBench), and 21.0 to 23.1 (Omni-MATH) (Tables 6, 8).
- **Critic evaluation (§4.1, App. E).** Critic models are prompted to return the index of the paragraph containing the earliest error. Open models: majority voting over 8 samples. GPT-4o: greedy decoding. o1-mini: a single sample. Greedy results for open models are in Table 9.
- **Authors' PRM baseline (App. C).** Qwen2.5-Math-7B-PRM800K is fine-tuned from Qwen2.5-Math-7B-Instruct on PRM800K, with PRM800K labels 1 and 0 mapped to positive and −1 to negative, a new binary classification head replacing the LM head, loss at the line-break positions, and PRM800K samples sharing problems with ProcessBench removed. Eight A100 80GB GPUs. Learning rate, epochs, and batch size are not reported.
- **Main F1 results, average over four subsets (Table 3).** PRMs: Math-Shepherd-PRM-7B 31.5; RLHFlow-PRM-Mistral-8B 28.4; RLHFlow-PRM-Deepseek-8B 26.6; Skywork-PRM-1.5B 36.4; Skywork-PRM-7B 42.1; Qwen2.5-Math-7B-PRM800K 56.5. Critics: Llama-3.3-70B-Instruct 58.0; Qwen2.5-Math-72B-Instruct 45.5; Qwen2.5-Coder-32B-Instruct 56.1; Qwen2.5-72B-Instruct 61.2; QwQ-32B-Preview 71.5; GPT-4o-0806 61.9; o1-mini 87.9.
- **Difficulty drop (Table 3).** Skywork-PRM-7B: 70.8 (GSM8K) → 21.0 (Omni-MATH). Qwen2.5-Math-7B-PRM800K: 68.2 → 44.3. QwQ-32B-Preview: 88.0 → 61.3. o1-mini: 93.2 → 82.4.

## Recipe ledger

| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| ProcessBench solution generators | 1.5B–72B | eval-gate | sampling | nucleus p = 0.9, temperature 0.7 | arXiv:2412.06559v4 App. D | verified 2026-11-24 | no ablation reported |
| ProcessBench labels | — | eval-gate | annotators per solution; agreement rule | 3, extended to at most 5 until 3 agree; else discard (~30% discarded overall) | v4 §3.2 | verified 2026-11-24 | not applicable |
| Skywork-PRM-1.5B/7B | 1.5B, 7B | eval-gate | step threshold | best-F1 threshold on GSM8K, reused on all subsets | v4 §4.1 | verified 2026-11-24 | Tables 6, 8: per-subset oracle thresholds change F1 by at most 4.2 points |
| Open critic models | 7B–72B | eval-gate | decoding | majority vote over 8 samples | v4 §4.1 | verified 2026-11-24 | Table 9 reports greedy results for comparison |
| Qwen2.5-Math-7B-PRM800K | 7B | reward-model | init; head; labels | Qwen2.5-Math-7B-Instruct; binary classification head; PRM800K 1/0 → positive, −1 → negative | v4 App. C | verified 2026-11-24 | no ablation reported |
| Qwen2.5-Math-7B-PRM800K | 7B | reward-model | LR, epochs, batch size | not reported | checked v4 App. C | not reported | — |

## Findings relevant to generality and negative feedback
- **Generality.** F1 drops for every evaluated model from GSM8K/MATH to OlympiadBench/Omni-MATH (§4.2, Table 3). The PRM trained only on human PRM800K labels, which are MATH problems, keeps 44.3 F1 on Omni-MATH, against 21.0 for Skywork-PRM-7B (Table 3). The authors attribute the weak generalization of automatically labeled PRMs to labels that depend on the generating policy and to correct-answer solutions that contain errors (§4.2).
- **Negative feedback.** A label derived from the final answer marks a solution as correct even when a step is wrong; this benchmark measures how often that happens (3.5% to 51.8% by subset, §3.3, Table 2) and states that it "could substantially invalidate the estimated process labels" for harder problems (§4.2). The F1 metric penalizes both missed errors and false alarms on correct solutions (§4.1).
- **Limitations stated (§5).** Error-location labels may be inaccurate on Olympiad-level problems, and the ~30% of discarded solutions may be the hardest cases, which can bias the problem distribution.

## Connections
- [[qwen-prm-lessons]]: same team; uses ProcessBench as the step-level evaluation that exposes the bias of Best-of-N evaluation.
- [[lets-verify]], [[prm800k]]: source of the human step labels behind the strongest PRM baseline here.
- [[math-shepherd]], [[omegaprm]]: automatic MC-estimated step labels; Math-Shepherd-PRM-7B is the 31.5-F1 baseline.
- [[skywork-o1]]: the Skywork PRMs evaluated in Table 3.
- [[generative-reward-models]]: critic models that reason before judging, the model class that scores highest here.

## Verification
- Checked on 2026-11-24 against https://arxiv.org/abs/2412.06559 and the v4 HTML (2025-05-26). Versions: v1 2024-12-09, v2 2024-12-10, v3 2025-05-20, v4 2025-05-26; no changelog is given, and earlier versions were not compared.
- Affiliation "Qwen Team, Alibaba Inc." and the Table 2 statistics were checked in the v4 PDF text (pdftotext).
- Fig. 7 (critic prompt template) is an image in the HTML; its text was not read.
- Not reported by the source: training hyperparameters for Qwen2.5-Math-7B-PRM800K; total number of annotators; maximum step count per subset.
