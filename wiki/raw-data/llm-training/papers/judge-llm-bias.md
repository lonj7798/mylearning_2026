<!-- scope: biases of LLM judges in pairwise and scalar evaluation
     deps: [[bradley-terry-rm]]
     see-also: [[lilianweng-reward-hacking]], [[rlaif-scaling]]
-->

# Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena
- **Core Insight:** Strong LLM judges (GPT-4) agree with humans ~80% of the time — matching inter-human agreement — but they come with specific, measurable biases: position (order), verbosity (length), self-enhancement (prefer their own outputs), and limited reasoning on math/coding pairs.
- **Guideline:** When using LLM-as-a-judge (for RLAIF labels, for eval, for RM training data), randomize order per comparison, pair with reference answers where possible, and swap sides + average; control for verbosity explicitly; never use the candidate model itself as the judge for head-to-head eval.
- **Authors:** Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric P. Xing, Hao Zhang, Joseph E. Gonzalez, Ion Stoica
- **Year:** 2023 (NeurIPS 2023 Datasets and Benchmarks)
- **URL:** https://arxiv.org/abs/2306.05685
- **Relevant topics:** LLM-as-a-judge, position bias, verbosity bias, self-enhancement bias, MT-Bench, Chatbot Arena, Elo

## Abstract
MT-Bench introduces a multi-turn benchmark of 80 questions across 8 categories; Chatbot Arena is a crowdsourced pairwise battle platform. The authors collect 3K expert votes and ~30K crowd votes and use them to measure the agreement of LLM judges (GPT-4, Claude, GPT-3.5) with human preferences. GPT-4 reaches ~80% agreement with human experts — the same rate as humans agree among themselves. They identify and quantify three systematic biases: position, verbosity, and self-enhancement, and propose simple mitigations.

## Key Contributions
- **Agreement numbers:** GPT-4 vs human-expert agreement is 85%+ on MT-Bench and ~80% on Chatbot Arena; the same rate two humans agree with each other.
- **Position bias:** A vs B ordering changes the winner in ~20–30% of cases; mitigated by swap-and-average or "two-game" scoring.
- **Verbosity bias:** longer responses win more often than a length-controlled baseline — concrete evidence that RMs and LLM judges prefer length as a cue.
- **Self-enhancement bias:** GPT-4 prefers GPT-4-authored responses at a rate above what humans prefer; Claude shows the same toward Claude; measured by comparing judge win-rates against human win-rates on the same pairs.
- **Limited reasoning in pair judging:** on math/coding pairs, LLM judges can confirm a wrong answer if it is presented confidently — links directly to U-sophistry.
- **Reference-guided judging:** providing the judge with a reference solution before evaluation raises agreement by ~10 pp on MT-Bench.
- **Dataset release:** 3K expert MT-Bench votes and 30K Chatbot Arena conversations, widely reused by later preference-dataset work.

## Key Figures/Tables to Study
- **Fig. 2** (position bias sweep) — swap A/B, see how often the judge flips; GPT-4 flips ~22%, GPT-3.5 ~40%.
- **Fig. 4** (verbosity vs win rate) — clear upward slope.
- **Fig. 5** (self-enhancement heatmap) — judge × candidate self-preference matrix.
- **Table 3** (judge–human agreement vs human–human agreement) — the headline "parity" numbers.

## Technical Details
- **Position-bias mitigation:** evaluate both orders, take a win only if the judge is consistent; otherwise declare tie.
- **Verbosity-bias mitigation:** length-controlled evaluation pairs where responses differ only in length; compute length-residualized win rate.
- **Self-enhancement mitigation:** never use the candidate as its own judge; for preference-label generation, use a stronger independent model; for RM training data, pool multiple judges.
- **Chain-of-thought judging:** asking the judge to reason before giving a verdict improves agreement but does not eliminate the biases.
- **Reference-guided grading:** attach a gold reference solution to the prompt; raises agreement on objective tasks (math, coding), less effect on writing tasks.
- **Tie handling:** Elo update when judges declare tie is a small-delta update — matters for RM calibration.

## Connections
- Underwrites all of the AI-labeling pipelines: **[[rlaif-scaling]]**, **[[constitutional-ai]]**.
- Bias inventory feeds the reward-hacking taxonomy (**[[reward-hacking-taxonomy]]**, **[[lilianweng-reward-hacking]]**).
- Chatbot Arena Elo is Bradley-Terry at scale (**[[bradley-terry-rm]]**).
- Motivates structural defenses: RLVR (**[[rlvr-tulu3]]**, **[[deepseek-r1]]**) removes the judge entirely on verifiable prompts.
