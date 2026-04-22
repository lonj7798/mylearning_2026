<!-- scope: AI-generated preferences match human RLHF at scale
     deps: [[constitutional-ai]]
     see-also: [[judge-llm-bias]], [[bradley-terry-rm]]
-->

# RLAIF vs. RLHF: Scaling Reinforcement Learning from Human Feedback with AI Feedback
- **Core Insight:** An off-the-shelf LLM can produce pairwise preference labels that train a reward model as good as one trained on human labels, on summarization / helpful / harmless tasks — and you can skip the RM entirely by reading the reward directly off the labeler LM's log-probs.
- **Guideline:** Use a capable LM with a chain-of-thought preference prompt to label pairs; train a BT reward model on those labels; or use **direct-RLAIF** (`d-RLAIF`) which reads the reward from the labeler LM's log-prob of "Response A is better" every step — no RM training, same or better quality.
- **Authors:** Harrison Lee, Samrat Phatale, Hassan Mansoor, Thomas Mesnard, Johan Ferret, Kellie Lu, Colton Bishop, Ethan Hall, Victor Carbune, Abhinav Rastogi, Sushant Prakash (Google DeepMind / Google Research)
- **Year:** 2023 (arXiv), 2024 (ICML)
- **URL:** https://arxiv.org/abs/2309.00267
- **Relevant topics:** RLAIF, AI-labeler, chain-of-thought labeling, direct-RLAIF, same-size labeler

## Abstract
The paper evaluates whether AI-generated preference labels can replace human ones. On summarization, helpful dialog, and harmless dialog tasks, RLAIF achieves performance comparable to RLHF on human evaluators (roughly 70/30 win-rate parity). Two methods are studied: (1) classical RLAIF — AI labels → BT reward model → PPO; and (2) direct-RLAIF (d-RLAIF) — the labeler LM's log-probs over "better" / "worse" are used as a scalar reward every step, skipping the RM. d-RLAIF outperforms RLAIF. Even "same-size RLAIF" (labeler same size as policy) improves over the SFT baseline, suggesting the signal is real even without a stronger teacher.

## Key Contributions
- **RLAIF ≈ RLHF on human-eval win rates** on summarization, helpful, and harmless tasks.
- **d-RLAIF (direct-RLAIF):** reward = `log P_labeler("Yes, Response 1 is better" | prompt, responses)` − (for "No"); no RM training, lower latency, better final quality.
- **CoT preference prompting:** asking the labeler to reason step-by-step before giving the "A or B" answer improves label quality and downstream win rates.
- **Label calibration:** the labeler's soft probabilities (not just hard A/B) carry useful gradient; BT RMs trained on soft labels outperform hard-label RMs.
- **Same-size labeler works:** even when the labeler is the same base LM as the policy, RLAIF improves over SFT — so the preference-labeling task is easier than the generation task.
- **Scaling observation:** as the labeler LM gets stronger, RLAIF quality improves monotonically; this is the empirical argument that RLAIF scales with model capability, so it gets better as LMs improve.

## Key Figures/Tables to Study
- **Fig. 1** (headline: RLAIF vs RLHF win rates on 3 tasks) — the parity result.
- **Fig. 3** (d-RLAIF vs RLAIF) — d-RLAIF wins.
- **Fig. 4** (CoT vs direct preference prompt) — CoT adds ~3–5 pp win rate.
- **Fig. 6** (labeler size vs RLAIF quality) — monotone scaling.

## Technical Details
- **Preference prompt template:** "Here is a query and two responses. Which response is better? Respond with 'Response 1' or 'Response 2'. Let's think step by step…" followed by a CoT and a final one-token answer.
- **Soft label extraction:** `p = softmax(logits["Response 1"], logits["Response 2"])`; used as target in a BT-style `−log σ(r_w − r_l)` via label smoothing.
- **d-RLAIF reward:** `r(x, y) = log P_labeler("Better" token | prompt(x, y, reference))`; applied at end of sequence; KL-to-SFT penalty same as standard RLHF.
- **PPO setup:** otherwise vanilla InstructGPT recipe, including the per-token KL penalty (see **[[kl-control-rlhf]]**).
- **Cost:** AI labels are ~100× cheaper per preference than crowd-source labels, and can be refreshed as the policy drifts, mitigating stale-RM issues.

## Connections
- Empirical companion to **[[constitutional-ai]]** — both validate AI feedback, but Constitutional AI adds the constitution structure.
- Inherits LLM-judge failure modes from **[[judge-llm-bias]]** (position bias, verbosity bias, self-enhancement bias) — mitigations matter.
- BT RM training in RLAIF is the same loss as in **[[bradley-terry-rm]]**.
- Susceptible to the same **[[reward-model-overoptimization]]** Goodhart laws when an RM is trained.
