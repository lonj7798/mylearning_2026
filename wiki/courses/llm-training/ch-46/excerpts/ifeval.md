<!-- scope: IFEval (Zhou et al., 2023): 541 prompts built from 25 programmatically verifiable instruction types, with strict and loose scoring at prompt level and instruction level
     deps: []
     see-also: [[mmlu-pro]], [[xstest]]
-->

# Instruction-Following Evaluation for Large Language Models
- **Core Insight:** IFEval scores instruction following with code rather than a judge: 25 verifiable instruction types (for example "write in more than 400 words", "mention the keyword AI at least 3 times") are appended to 541 prompts, and a response is checked by a deterministic function; GPT-4 reaches 76.89% prompt-level strict accuracy and 83.57% instruction-level strict accuracy (§2.1, §2.2, Table 3).
- **Guideline:** When reporting IFEval before and after a training stage, report both the strict and the loose metric at both prompt and instruction level, because the strict metric has known false negatives (markdown emphasis around a required phrase) and the loose metric, which retries after eight response transformations, introduces false positives (§2.2).
- **Authors:** Jeffrey Zhou, Tianjian Lu, Swaroop Mishra, Siddhartha Brahma, Sujoy Basu, Yi Luan, Denny Zhou, Le Hou (Google; Yale University)
- **Year:** 2023 (arXiv v1 2023-11-14)
- **URL:** https://arxiv.org/abs/2311.07911
- **Source type:** paper
- **Relevant topics:** instruction following, verifiable constraints, programmatic evaluation, strict versus loose scoring

## Abstract
Human evaluation of instruction following is expensive and not reproducible, and LLM-based auto-evaluation can be biased or limited by the evaluator. IFEval is a reproducible benchmark built on verifiable instructions that can be checked by a simple program. The authors identify 25 types of verifiable instructions and construct about 500 prompts containing one or more of them, then report results for two widely available models.

## Key Contributions
- A list of 25 verifiable instruction types, each checkable by an interpretable function (§2.1, Table 1).
- 541 prompts, each carrying one to three verifiable instructions appended to a seed prompt, filtered for illogical combinations and rephrased for diversity, then manually checked (§2.1).
- Four metrics: prompt-level and instruction-level accuracy, each under strict and loose verification (§2.2).
- The loose verification procedure: eight transformations of the response (remove markdown font modifiers, remove the first line, remove the last line, their pairwise and triple combinations, and the identity) with the instruction counted as followed if any transformation passes (§2.2, Eq. 2).

## Key Figures/Tables to Study
- **Table 1:** the 25 verifiable instruction types.
- **Table 2:** example prompts with their instructions.
- **Table 3:** the four accuracies for GPT-4 and PaLM 2 S.

## Technical Details
- **Construction (§2.1).** Instructions are appended to seed prompts, with one to three instructions per prompt chosen at random; a few-shot pass removes illogical prompts; a second few-shot pass rephrases prompts for phrasing diversity; all rephrasings are manually checked and edited.
- **Strict metric (§2.2, Eq. 1).** `is_followed(resp, inst)` returns True or False from a deterministic check; prompt-level accuracy requires every instruction in the prompt to pass.
- **Loose metric (§2.2, Eq. 2).** The instruction counts as followed if any of the eight transformed responses passes. It reduces false negatives such as "P.S. **I do like the cake**" failing a literal string match, at the cost of false positives such as a word-count instruction passing after the first line is removed.
- **Results (Table 3).** GPT-4: prompt-level strict 76.89, instruction-level strict 83.57, prompt-level loose 79.30, instruction-level loose 85.37. PaLM 2 S: 43.07, 55.76, 46.95, 59.11. The paper states the two models are not directly comparable because of the parameter-count difference.
- **Stated limit (§2, §2.2).** Very few instructions are 100% verifiable; the benchmark restricts itself to those whose verification is simple and interpretable.

## Findings relevant to generality
- **Why it belongs in a retention suite.** IFEval measures whether a model still obeys explicit output constraints after a training stage that optimized something else. The check is programmatic, so it adds no judge variance to a before-and-after comparison.
- **Metric choice affects the delta.** The strict and loose metrics differ by roughly 2–3 points for GPT-4 (Table 3), so a retention comparison must fix one metric and one decoding configuration on both sides.

## Connections
- [[mmlu-pro]] — the knowledge-and-reasoning half of a held-out retention suite.
- [[xstest]] — the over-refusal half.
- [[rls-razor]] — includes IFEval among the prior-capability benchmarks used to measure forgetting.

## Verification
- Checked on 2026-09-15 against the cached primary text of https://arxiv.org/abs/2311.07911 (arXiv v1, 2023-11-14): Abstract, §1, §2.1, §2.2, Table 3.
- Corrections to the previous card version: no previous card existed in the library for this slug.
- Not reported by the source: results for models other than GPT-4 and PaLM 2 S; per-instruction-type breakdowns in the main text; a measured false-positive rate for the loose metric.
