<!-- scope: LLMs as reward models via generation + CoT, not a scalar head
     deps: [[bradley-terry-rm]]
     see-also: [[reward-ensembling]], [[judge-llm-bias]]
-->

# Generative Reward Models
- **Core Insight:** Instead of a scalar head on an LM, use the LM itself to generate a critique plus a verdict; extract the reward from the log-probabilities of the verdict tokens. Generative RMs let chain-of-thought reasoning flow into the reward and produce calibrated uncertainties.
- **Guideline:** For a pair `(x, y_w, y_l)` prompt the LM with a rubric, have it produce a CoT critique, and have it emit a structured verdict ("A", "B", or a 1–10 score); train on `−log P(verdict_target | prompt)` with held-out preference data; use log-prob margins as the reward at inference.
- **Authors:** Lifan Yuan, Ganqu Cui, Hanbin Wang, Ning Ding, Xingyao Wang, Jia Deng, Boji Shan, Huimin Xu, Ruobing Xie, Yankai Lin, Zhenghao Liu, Bowen Zhou, Hao Peng, Zhiyuan Liu, Maosong Sun (and several subsequent lines of work — Vu et al. 2024, Mahan 2024)
- **Year:** 2024
- **URL:** https://arxiv.org/abs/2410.12832 ("Generative Reward Models", Mahan et al.); https://arxiv.org/abs/2408.15240 ("Critique-out-Loud Reward Models", Ankner et al.)
- **Relevant topics:** generative RM, CoT-RM, GenRM, critique-then-verdict, log-prob reward, calibration

## Abstract
Generative Reward Models (GenRMs) replace the `<LM + linear head>` RM with an instruction-tuned LM that produces a reasoning trace plus a verdict. The reward is read off the log-probability the model assigns to the "positive" verdict token, optionally conditional on the generated critique. GenRMs outperform classical BT reward models on RewardBench, produce better-calibrated uncertainties, and are less vulnerable to sycophancy/length bias when the rubric explicitly names those failure modes.

## Key Contributions
- **GenRM scoring:** `r(x, y) = log P_RM("A is better" | x, y_A, y_B, rubric)` — or a soft margin between "A" and "B" tokens.
- **Critique-then-verdict (CoT-RM):** sample a critique `c ~ P_RM(·|prompt)` first, then score the verdict given the critique — accuracy improves 3–10 pp on RewardBench over no-CoT.
- **Training:** fine-tune the LM with next-token supervision on (prompt, critique, verdict) triples; no dedicated scalar head — keeps the RM in the same model family as the policy.
- **Calibration:** the LM's verdict probability is reliably tied to ground-truth agreement — useful as an uncertainty signal (feeds back into **[[reward-ensembling]]**-style LCB combinations).
- **Robustness:** when the rubric is extended to say "longer is not better, be concerned if the response is sycophantic", the RM generalizes those constraints to unseen prompts — the RM is steerable via its own context, which scalar RMs cannot be.
- **Compute trade-off:** GenRMs are slower (need to generate critique tokens) but reuse the base-LM inference stack and scale with model capability.

## Key Figures/Tables to Study
- **GenRM Fig. 2** (RewardBench accuracy vs scalar RM) — GenRM Pareto-dominates.
- **Fig. 4** (calibration plot) — generative RMs are well-calibrated where BT RMs are overconfident.
- **Ablation: with vs without critique** — CoT adds clear lift.

## Technical Details
- **Verdict-token reward:** for a pair `(y_A, y_B)`, score `= log P("A") − log P("B")` at the verdict position; equivalent to a BT log-odds.
- **Pointwise vs pairwise:** GenRMs also come in pointwise form (score one response on a rubric with a 1–10 verdict); the log-prob weighted expectation over verdicts gives a scalar reward.
- **Rubric prompt:** explicit bullet list of dimensions (correctness, helpfulness, safety, conciseness) — the rubric IS the reward specification, so its text is the policy knob.
- **Training data:** instruction-tuned LM fine-tuned on (prompt, rubric, critique, verdict) supervised triples; critiques can be human-written or bootstrapped from GPT-4.
- **Failure modes:** verbosity bias and self-enhancement (see **[[judge-llm-bias]]**) still apply; mitigated by rubric wording and by using a judge from a different model family than the policy.

## Connections
- A concrete, LM-native answer to **[[reward-hacking-taxonomy]]**: CoT critiques surface the reason for a verdict, making adversarial exploitation harder (but not impossible).
- Pairs well with ensembling (**[[reward-ensembling]]**) — GenRM ensembles give calibrated uncertainty.
- Inherits and partly mitigates LLM-judge biases from **[[judge-llm-bias]]**.
- BT loss still in play on top of the verdict log-odds (**[[bradley-terry-rm]]**).
