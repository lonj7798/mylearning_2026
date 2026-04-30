<!-- scope: meta-rewarding — judging the judge to regulate self-rewarding drift
     deps: [[self-rewarding-lm]]
     see-also: [[spin]], [[self-play-preference]]
-->

# Meta-Rewarding Language Models
- **Core Insight:** Adding a third role — *meta-judge* that evaluates the judge — stabilizes self-rewarding training past the 3-iteration ceiling and keeps judge calibration from drifting into reward-hacking.
- **Guideline:** Don't just train the actor on judge outputs; also train the judge on meta-judge outputs, using pairs of judge responses scored against each other via a consistency rubric.
- **Authors:** Tianhao Wu, Weizhe Yuan, Olga Golovneva, Jing Xu, Yuandong Tian, Jiantao Jiao, Jason Weston, Sainbayar Sukhbaatar
- **Year:** 2024 (Meta FAIR + UC Berkeley + NYU)
- **URL:** https://arxiv.org/abs/2407.19594
- **Relevant topics:** self-improvement, judge calibration, LLM-as-a-judge, meta-evaluation

## Abstract
Large Language Models (LLMs) are rapidly surpassing human knowledge in many domains. While improving these models traditionally relies on costly human data, recent self-rewarding mechanisms have shown that LLMs can improve by judging their own responses instead of relying on human labelers. However, existing methods have primarily focused on improving model responses rather than judgment capabilities, resulting in rapid saturation during iterative training. To address this issue, we introduce a novel Meta-Rewarding step to the self-improvement process, where the model judges its own judgments and uses that feedback to refine its judgment skills. Surprisingly, this unsupervised approach improves the model's ability to judge and follow instructions, as demonstrated by a win rate improvement of Llama-3-8B-Instruct from 22.9% to 39.4% on AlpacaEval 2 and 20.6% to 29.1% on Arena-Hard.

## Key Contributions
- Introduces **Meta-Rewarding** — a third-role "meta-judge" that compares pairs of judge responses and selects the preferred judgment using a calibration rubric.
- Shows judge quality (agreement with humans) continues improving up to iteration 5, vs. Self-Rewarding plateauing at 3.
- Adds a **length-bias control term** to the judge rubric (deducting points for length-gaming), preventing the known DPO length inflation.
- Lifts Llama-3-8B-Instruct AlpacaEval 2.0 LC win-rate from 22.9% → 39.4% over 4 meta-rewarding iterations on zero additional human data.

## Key Figures/Tables to Study
- **Figure 2:** the three-role loop — Actor / Judge / Meta-Judge — and which DPO pairs feed which role.
- **Table 1 (AlpacaEval 2.0 and Arena-Hard LC win-rate per iteration):** the headline monotonic gains.
- **Table 5 (judge agreement with humans):** shows the meta-rewarding loop keeps judge calibration rising while self-rewarding's flattens.
- **Figure 5 (length bias):** illustrates the length-control term's effect — without it, responses grow 2× over iterations.

## Technical Details
- **Base:** Llama-3-8B-Instruct (Meta), 4 meta-rewarding iterations.
- **Per-iteration data generation:**
  1. Sample K=7 actor responses per prompt from the current policy.
  2. Sample N=11 *judge* responses per (prompt, actor-response) pair — each a score + rationale.
  3. **Meta-Judge** performs pairwise comparison on those 11 judge responses (per the rubric) to pick the best and worst judgment.
  4. Actor-DPO uses (best actor response, worst actor response) pairs by aggregated judge score.
  5. Judge-DPO uses (best judge response, worst judge response) pairs from the meta-judge.
- **Length-bias control:** judge rubric includes "don't reward length for length's sake"; meta-judge penalizes length-gamed judgments.
- **DPO params:** β=0.1, 1 epoch per iteration, cosine LR schedule ending at 0.
- **Prompt pool:** 20K EvolInstruct prompts held out from any human-labelled set.

## Connections
- Direct extension of [[self-rewarding-lm]] — same Actor + Judge loop plus a new role.
- Shares structural similarity with [[self-play-preference]]: both use multi-role bootstrapping; Nash-LM uses game-theoretic equilibrium, Meta-Rewarding uses hierarchical evaluation.
- The length-bias control is the lesson from [[ipo]] / SimPO literature applied at the judge level.
- Empirically, the judge-DPO component is what enables the non-saturating self-improvement — a direct evidence point for Ilya Sutskever's "self-play on soft targets" thesis.
