<!-- scope: HuggingFace Math-Verify — rule-based math answer checker: extraction (LaTeX / expression / string), SymPy normalization, asymmetric verify(gold, pred), reward-mode settings
     deps: []
     see-also: [[verifier-pitfalls]], [[general-reasoner-verifier]], [[master-rm-one-token]], [[openr1]], [[dr-grpo]], [[big-math]], [[dapo]], [[rl-design-guide]]
-->

# Math-Verify (huggingface/Math-Verify): rule-based math answer verification
- **Core Insight:** A rule verifier is three separate steps — extract an answer from free text, convert gold and prediction to a common SymPy form, compare — and each step has its own failure mode; Math-Verify's `verify(gold, pred)` is deliberately asymmetric (a gold inequality accepts a predicted interval but not the reverse; a gold number accepts a prediction chain ending in that number but not the reverse) so that restating the question does not count as an answer (README FAQ).
- **Guideline:** When using Math-Verify as an RL reward, pass the gold answer first, store gold answers already in the preferred canonical format, and use the README's stricter reward settings (`malformed_operators=False`, `nits=False`, `equations=False`, `boxed="all"`), and do not mix `StringExtractionConfig()` with other configs, because first-match string extraction can pick stray letters A–D from ordinary text (README). Measure its recall on your own policy's outputs, because [[verifier-pitfalls]] measured HF Math-Verify recall of 0.88 on Skywork-OR1 responses and 47% on WebInstruct-Verified.
- **Authors:** Hugging Face (repository maintainers; individual authors not listed in the README read)
- **Year:** 2025 (repository; README read 2026-11-24, not pinned to a commit)
- **URL:** https://github.com/huggingface/Math-Verify
- **Source type:** released config/code
- **Relevant topics:** rule-based verifiers, answer extraction, symbolic equivalence, false negatives, reward-function configuration

## Summary
Math-Verify is a Python library used as the math reward in several open RL recipes ([[openr1]], [[dr-grpo]]). `parse()` extracts candidate answers with configurable extractors; `verify()` compares parsed gold and prediction numerically and symbolically with configurable rounding tolerance. It ships with timeout protection for long computations and a fallback for failed comparisons (README; no timeout value given).

## Key Contributions
- Extraction configs: `LatexExtractionConfig` (LaTeX inside a recognized environment such as `$...$`, `\[...\]`, `\boxed{}`), `ExprExtractionConfig` (plain expressions such as `1/2`), `StringExtractionConfig` (literal strings such as `A`). Default: LaTeX + Expr.
- Normalizations: Unicode symbol substitution, LaTeX repairs (`frac13` → `1/3`), equation/assignment resolution (`x = 1` → `1`), percentage (`10%` → `0.1`), units in text (`10 cm` → `10`), exact representation of decimals (`0.333` → `Float(333, 3)`).
- Comparisons: numeric/symbolic with tolerance, matrices, sets and intervals, relations with flip support, complex numbers.

## Key Figures/Tables to Study
- README "Usage" snippet, FAQ on asymmetry, and the evaluator comparison table.

## Technical Details
```python
from math_verify import parse, verify
gold = parse("${1,3} \\cup {2,4}$")
answer = parse("${1,2,3,4}$")
verify(gold, answer)   # True — "Order here is important!"
```
**Asymmetry (FAQ).** (1) Interval vs inequality: gold inequality + predicted interval → True; reverse → False; `allow_set_relation_comp` permits both. (2) Number vs solution chain: gold number + predicted chain ending in it → True; reverse → False.

**Known limits (README).** LaTeX outside a recognized environment extracts nothing (bare `x = 1`). Float gold answers need the correct precision setting.

**README comparison table ("Score" on MATH).** Harness 0.0802; Qwen 0.1288; Math-Verify 0.1328. The README does not name the model, split or metric, so these are not usable as recall numbers.

**Installation.** `pip install math-verify[antlr4_13_2]` (three antlr4 runtimes supported; pin one).

## Verification
- Read the GitHub README on 2026-11-24 (main branch, not pinned). The source code was not read; behavior is as documented.
- Independent recall/precision measurements are in [[verifier-pitfalls]] (Table 5 there: precision about 0.99–1.00; recall 0.95 on MATH, 0.94 DeepScaleR, 0.95 ORZ-Math, 0.88 Skywork-OR1).
