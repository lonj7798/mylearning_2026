---
chapter: ch-42
course: llm-training
phase: read
excerpt_of: "Reward Hacking in Reinforcement Learning" (Lilian Weng, blog, 2024-11-28); library card [[lilianweng-reward-hacking]]
source_url: https://lilianweng.github.io/posts/2024-11-28-reward-hacking/
created_at: "2026-04-23"
revised: "2026-09-15 (generality revision; rewritten from the live post because the library card has no Verification section and misreports the Wen et al. figure)"
---

# Excerpt: survey taxonomy and mitigation status (Weng)

Used by [[read]] §1 and §6. Source type: practitioner survey (secondary; every number below is attributed to the paper it summarizes). Checked against the live post on 2026-09-15.

## Taxonomy used in the chapter
The post presents Garrabrant's four categories of Goodhart's law:
- **Regressional** — optimizing the proxy selects for the noise in the proxy.
- **Extremal** — optimization moves the policy into regions where proxy and true reward come apart.
- **Causal** — correlations in the training distribution break under intervention.
- **Adversarial** — a capable agent searches for proxy exploits.

## Claims carried into the chapter, with their primary sources
- **Sycophancy**: RLHF-tuned assistants agree with confidently stated user beliefs; the primary measurements are in [[sycophancy-in-lms]].
- **U-Sophistry** (Wen et al. 2024): "RLHF makes incorrect outputs more convincing to humans. The evaluation false positive rate significantly increases after RLHF training." The post adds that "at an individual level, the majority (70-90%) of human evaluators [saw] their evaluation error rates increase". The 70-90% is the share of evaluators affected, not the size of the increase; the measured false-positive changes are 41.0% → 65.1% (QuALITY, task-specific reward) and 29.6% → 47.9% (APPS) — see [[language-models-mislead-humans]].
- **In-context reward hacking** (Pan et al. 2023, 2024): in an essay-editing loop where the same model is judge and author, the judge score and the human oracle score diverge; a smaller evaluator model makes it more likely.
- **Anomaly detection as a defense** (Pan et al. 2022): a classifier compares the action distribution of a trusted policy with the target policy; "none of the tested classifier can achieve AUROC greater than 60% across all tested RL environments".
- **Potential-based shaping** (Ng et al. 1999): `F(s,a,s') = γΦ(s') − Φ(s)` leaves the optimal policy unchanged; the post presents it as the one shaping form that is provably safe.
- **Honest conclusion of the post**: "research into practical mitigations, especially in the context of RLHF and LLMs, remains limited."

## Corrections to the previous version of this excerpt
- "Human evaluator error rate on incorrect answers rises 70-90% after RLHF" → see the Wen et al. entry above.
- The post does not rank defenses by robustness in deployments; the ordered defense list in the earlier ch-42 draft was the course's own.
