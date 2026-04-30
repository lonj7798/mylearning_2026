<!-- scope: Best-of-N sampling vs RL — Stiennon summarization baseline
     deps: [[rlhf-instructgpt]]
     see-also: [[west-of-n]], [[rejection-sampling-finetuning]], [[reward-model-overoptimization]]
-->

# Best-of-N vs RL — Learning to Summarize with Human Feedback
- **Core Insight:** On summarization, at small reward-model KL budgets, **Best-of-N** sampling (generate N candidates, pick the one with the highest reward-model score) is competitive with or superior to full RLHF — it has no training instability, costs only inference, and is monotonic in N until the reward model overoptimizes.
- **Guideline:** Before you run PPO, try Best-of-N with N∈{4, 16, 64}. If the reward model is trustworthy, BoN-64 often matches a well-tuned PPO at ~1/10 the engineering cost and zero training risk.
- **Authors:** Nisan Stiennon, Long Ouyang, Jeff Wu, Daniel M. Ziegler, Ryan Lowe, Chelsea Voss, Alec Radford, Dario Amodei, Paul Christiano
- **Year:** 2020 (OpenAI)
- **URL:** https://arxiv.org/abs/2009.01325
- **Relevant topics:** best-of-N sampling, RLHF baseline, reward-model overoptimization, TL;DR summarization

## Abstract
As language models become more powerful, training and evaluation are increasingly bottlenecked by the data and metrics used for a particular task. For example, summarization models are often trained to predict human reference summaries and evaluated using ROUGE, but both of these metrics are rough proxies for what we really care about—summary quality. In this work, we show that it is possible to significantly improve summary quality by training a model to optimize for human preferences. We collect a large, high-quality dataset of human comparisons between summaries, train a model to predict the human-preferred summary, and use that model as a reward function to fine-tune a summarization policy using reinforcement learning.

## Key Contributions
- Established the **RLHF-for-summarization** pipeline later generalized by InstructGPT: SFT → RM → PPO with KL-to-SFT penalty.
- Compared RLHF against **Best-of-N (BoN)** and showed BoN is a strong, often-overlooked baseline: BoN-64 with a well-trained RM is within 2 points of PPO on human eval at much lower KL.
- Showed the **BoN KL** (KL of the BoN-induced distribution w.r.t. the base policy) scales as `log N − (N−1)/N`, giving a clean apples-to-apples comparison against PPO at matched KL.
- Documented the first clear example of **reward-model overoptimization**: past a critical KL budget, RM score keeps rising while human preference falls.

## Key Figures/Tables to Study
- **Figure 4 (RM score vs human preference, KL on x-axis):** the overoptimization curve — both BoN and RL rise, then RL keeps climbing in RM score while human preference plateaus or drops.
- **Figure 6 (BoN vs RL at matched KL):** BoN and RL are nearly coincident; BoN wins at very low KL, RL wins at higher KL.
- **Table 3 (Axis-level human eval):** coverage, accuracy, coherence all improved by RLHF over SFT.

## Technical Details
- **Base:** GPT-3 1.3B and 6.7B, SFT on Reddit TL;DR.
- **RM:** separate trained on 64K human pairwise preferences.
- **BoN procedure:**
  - Sample N summaries at T=0.7.
  - Score each with RM.
  - Return argmax.
- **BoN KL formula:** `KL(BoN || base) = log N − (N−1)/N` — derived in appendix; tight for well-calibrated RM.
- **RL:** PPO with KL-to-SFT regularization; β tuned to match BoN-64 KL.
- **Eval:** human pairwise preference between model summaries and references.

## Connections
- Foundational for [[west-of-n]]: West-of-N is BoN generalized into preference-pair generation.
- Foundational for [[rejection-sampling-finetuning]]: RSFT is BoN applied to SFT training data rather than at inference.
- The overoptimization phenomenon was formalized by [[reward-model-overoptimization]] (Gao 2022) as "Goodhart on RM".
- The KL-matched comparison framework now underlies every modern RLHF evaluation protocol.
- BoN remains the production deployment pattern for many inference-time alignment recipes (e.g., Anthropic's test-time compute work, Cohere's chat models).
