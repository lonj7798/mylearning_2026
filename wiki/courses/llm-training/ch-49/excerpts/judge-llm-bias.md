---
chapter: ch-49
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/judge-llm-bias.md
source_url: https://arxiv.org/abs/2306.05685
created_at: "2026-04-23"
revised_at: "2026-09-15"
---

# Excerpt: Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena

**Authors:** Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric P. Xing, Hao Zhang, Joseph E. Gonzalez, Ion Stoica
**Year:** 2023 (NeurIPS 2023 Datasets and Benchmarks)
**Source type:** paper
**Checked on:** 2026-09-15 against the arXiv PDF text.

> This excerpt was rewritten in the 2026-09 revision. The library card
> `wiki/raw-data/llm-training/papers/judge-llm-bias.md` had not yet been re-verified at that time and
> contains figures and numbers that do not appear in the paper. Use the values below; they were read at
> the loci given.

## Corrections to the previous version of this excerpt

1. "Swap-flip rate: GPT-4 ≈ 22%, GPT-3.5 ≈ 40% (Fig. 2)" → Table 2 reports *consistency*, the share of cases where the verdict is unchanged after swapping: GPT-4 65.0%, GPT-3.5 46.2%, Claude-v1 23.8% (default prompt). Fig. 2 is agreement versus win-rate difference, not position bias.
2. "Reference-guided grading raises agreement by ~10 pp on MT-Bench" → Table 4 reports the judge failure rate on 10 math questions falling from 14/20 (default) to 6/20 (CoT) to 3/20 (reference), described in §3.4 as 70% to 15%. The reference is the judge's own answer generated independently beforehand, not a gold solution.
3. "Fig. 4 verbosity slope" and "Fig. 5 self-enhancement heatmap" → neither figure exists. Verbosity is Table 3 (a repetitive-list attack); self-preference is discussed from Fig. 3(b) win rates.
4. "Self-enhancement bias is attested" → §3.3 states "our study cannot determine whether the models exhibit a self-enhancement bias"; the reported observations are GPT-4 favouring itself by a 10% higher win rate and Claude-v1 by 25%.
5. "Swapping cancels position bias to within ~2 pp residual" → no residual is reported; the conservative rule counts inconsistent verdicts as ties.

## Setup (§3.1, §4.1)

MT-Bench: 80 questions across 8 categories, two turns each. Answers from 6 models (GPT-4, GPT-3.5, Claude-V1, Vicuna-13B, Alpaca-13B, LLaMA-13B). 58 expert labelers, mostly graduate students, each evaluating at least 20 random multi-turn questions, giving about 3K votes. Chatbot Arena: 3K single-turn votes sampled from 30K arena votes, with crowd judges from 2,114 unique IPs.

## Position bias (§3.3, Table 2)

Two similar answers per first-turn question, produced by calling GPT-3.5 twice at temperature 0.7, judged in both orders. Consistency is the share of unchanged verdicts.

| Judge | Prompt | Consistency | Biased toward first | Biased toward second | Error |
|---|---|---|---|---|---|
| Claude-v1 | default | 23.8% | 75.0% | 0.0% | 1.2% |
| Claude-v1 | rename | 56.2% | 11.2% | 28.7% | 3.8% |
| GPT-3.5 | default | 46.2% | 50.0% | 1.2% | 2.5% |
| GPT-3.5 | rename | 51.2% | 38.8% | 6.2% | 3.8% |
| GPT-4 | default | 65.0% | 30.0% | 5.0% | 0.0% |
| GPT-4 | rename | 66.2% | 28.7% | 5.0% | 0.0% |

The "rename" prompt renames the assistants, separating position from name. The paper notes the probe is hard because the two answers are very similar, and that position bias is less prominent in some settings (App. D.1).

## Verbosity (§3.3, Table 3)

"Repetitive list" attack on 23 MT-Bench answers containing a numbered list: GPT-4 rephrases the list without adding information and the rephrased items are prepended, doubling the item count with no new content. Failure rate (judge prefers the longer version): Claude-v1 91.3%, GPT-3.5 91.3%, GPT-4 8.7%. As a calibration, the judges return a tie for two identical answers.

## Math and reasoning grading (§3.3–§3.4, Table 4)

10 math questions, LLaMA-13B vs Vicuna-13B, positions swapped, 20 judgments. Failures (judge says an incorrect answer is correct): default 14/20, chain-of-thought 6/20, reference-guided 3/20. §3.4 notes that with the CoT prompt the judge often makes the same mistake as the given answers inside its own reasoning, which is why the reference is generated before the candidates are shown.

## Corrections proposed (§3.4)

- **Swapping positions.** Conservative rule: judge twice with the order swapped, declare a win only if consistent, otherwise a tie. Aggressive alternative: assign positions randomly, effective at scale in expectation.
- **Few-shot judge.** Three judgment examples raise GPT-4 consistency from 65.0% to 77.5% (Table 12), at 4× the prompt cost; the authors state higher consistency may not mean higher accuracy and may introduce new biases.
- **Multi-turn judging** requires presenting the complete conversation in one prompt; splitting the two turns caused the judge to mis-locate the assistant's prior response (§3.5).

## Agreement (§4.2, Tables 5–6)

S1 includes non-tie, tie, and position-inconsistent votes and counts inconsistent as tie (random 33%); S2 includes non-tie votes only (random 50%).

MT-Bench first turn: GPT-4 pairwise vs human 66% (S1) / 85% (S2); GPT-4 single-answer vs human 60% / 85%; human vs human 63% / 81%. Second turn: GPT-4 pairwise vs human 66% / 85%; human vs human 67% / 82%. Chatbot Arena (Table 6): GPT-4 vs human 64% / 87%; GPT-3.5 vs human 54% / 83%; Claude vs human 53% / 84%.

Fig. 2: agreement between GPT-4 and humans rises from 70% to nearly 100% as the win-rate difference between the two compared models grows.

Also reported (§4.2): when a human's choice differed from GPT-4's, humans judged GPT-4's explanation reasonable in 75% of cases and changed their own choice in 34%.

## Connections

[[chatbot-arena]] (the platform and its votes), [[arena-hard-benchbuilder]] (measures MT-Bench separability against later benchmarks), [[self-preference-recognition]] (the controlled study of the self-preference question this paper leaves open), [[self-taught-evaluators]] and [[direct-judgement-preference]] (judge-training work that reuses the MT-Bench judge prompt).
